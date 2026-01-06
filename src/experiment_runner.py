"""
Main Experiment Runner for Green Code Refactoring.
Orchestrates: Dataset -> Prompt -> LLM -> Patch -> Measurement.
Features: 
- Hunter Parser: Finds patches in verbose LLM output.
- Super Fuzzy Matcher: Robustly applies patches ignoring whitespace/newline diffs.
- Integrated Measurement.
"""
import sys
import os
import re
import json
import logging
import subprocess
import time
import tempfile
import shutil
from typing import Optional, Dict, List, Any
from pathlib import Path

# --- CONFIGURAZIONE PATH ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "scripts"))

from src.llm_clients.client_manager import ClientManager
from src.prompt_templates.template_manager import PromptTemplateManager
from src.prompt_templates.base_template import PromptStrategy, ProblemStatementType, PromptContext
from openai import OpenAI

# Import misuratore originale
try:
    from scripts.measure_instance import SWEPerfMeasurer
    from src.measurement.collector import MetricsCollector
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import measurement scripts. {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GreenExperimentRunner:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset()
        self.client_manager = ClientManager() 
        self.template_manager = PromptTemplateManager()
        self.measurer_tool = SWEPerfMeasurer(dataset_path, country_code="ESP")

    def _load_dataset(self) -> List[Dict]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list): return data
        elif isinstance(data, dict) and "instances" in data: return data["instances"]
        else: raise ValueError(f"Unknown dataset structure")

    def _get_instance(self, instance_id: str) -> Dict[str, Any]:
        for item in self.dataset:
            if item.get("instance_id") == instance_id: return item
        raise ValueError(f"Instance {instance_id} not found")

    def _detect_running_model(self) -> str:
        try:
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
            models = client.models.list()
            if models.data: return models.data[0].id
        except Exception: pass
        return "active_model"

    def _extract_patch_content(self, content: str) -> str:
        """Estrae la patch pulendo il testo dell'LLM."""
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        idx_search = content.find("<<<<<<< SEARCH")
        if idx_search != -1:
            # Mantieni un buffer prima del primo blocco per intercettare il nome del file
            return content[max(0, idx_search - 500):]
            
        # Fallback diff standard
        idx_diff = content.find("diff --git")
        if idx_diff != -1:
            return content[idx_diff:]
            
        # Fallback markdown code block
        code_blocks = re.findall(r'```(?:diff|python)?\s*(.*?)```', content, re.DOTALL)
        if code_blocks: return max(code_blocks, key=len).strip()
            
        return content.strip()

    def _find_target_file(self, repo_path: Path, search_block: str, candidate_files: List[str]) -> Optional[str]:
        # Cerca il file che contiene la prima riga non vuota del blocco
        search_lines = [l.strip() for l in search_block.splitlines() if l.strip()]
        if not search_lines: return None
        
        signature = search_lines[0]
        for fname in candidate_files:
            fpath = repo_path / fname
            if fpath.exists():
                if signature in fpath.read_text(): return fname
        return None

    def _perform_fuzzy_replace(self, repo_path: Path, rel_path: str, search_lines: List[str], replace_lines: List[str]) -> bool:
        """
        SUPER FUZZY MATCHER:
        Confronta il contenuto ignorando spazi bianchi e righe vuote.
        """
        target_file = repo_path / rel_path
        if not target_file.exists():
            # Ricerca file se path errato
            candidates = list(repo_path.rglob(os.path.basename(rel_path)))
            if candidates: target_file = candidates[0]
            else: return False
                
        original_lines = target_file.read_text().splitlines(keepends=True)
        
        # 1. Normalizza Search Block (lista di stringhe pulite)
        norm_search = [l.strip() for l in search_lines if l.strip()]
        if not norm_search: return False 

        # 2. Crea Mappa del File Originale (contenuto_pulito -> indice_reale)
        file_map = []
        for idx, line in enumerate(original_lines):
            stripped = line.strip()
            if stripped:
                file_map.append((stripped, idx))
        
        # 3. Cerca la sequenza
        search_len = len(norm_search)
        match_start_idx = -1
        
        for i in range(len(file_map) - search_len + 1):
            # Prendi una finestra di righe dal file mappa
            window = [item[0] for item in file_map[i : i + search_len]]
            if window == norm_search:
                match_start_idx = i
                break
        
        if match_start_idx != -1:
            # Trovato! Recupera gli indici reali
            real_start_line = file_map[match_start_idx][1]
            real_end_line = file_map[match_start_idx + search_len - 1][1]
            
            # Prepara il blocco di sostituzione (aggiungi newline se mancano)
            final_replace = [l + '\n' if not l.endswith('\n') else l for l in replace_lines]
            
            # Applica modifica
            new_content = (
                original_lines[:real_start_line] + 
                final_replace + 
                original_lines[real_end_line + 1:]
            )
            target_file.write_text("".join(new_content))
            logger.info(f"✅ Applied Patch to {rel_path} (Fuzzy Match)")
            return True
            
        logger.warning(f"❌ Search block mismatch in {rel_path}")
        return False

    def _apply_patch_logic(self, repo_path: Path, raw_content: str, candidate_files: List[str]) -> bool:
        patch_content = self._extract_patch_content(raw_content)
        logger.info(f"\n📄 PATCH EXTRACTED ({len(patch_content)} chars)")

        if "<<<<<<< SEARCH" in patch_content:
            logger.info("🔧 Processing SEARCH/REPLACE blocks...")
            blocks = patch_content.split('<<<<<<< SEARCH')
            changes_count = 0
            
            # Il testo prima del primo blocco contiene indizi sul file
            context_text = blocks[0]
            
            for i in range(1, len(blocks)):
                block = blocks[i]
                if "=======" not in block or ">>>>>>> REPLACE" not in block: continue
                
                search_part, rest = block.split('=======', 1)
                replace_part, next_context = rest.split('>>>>>>> REPLACE', 1)
                
                # 1. Cerca nome file nel testo precedente (ultime righe)
                target_file = None
                lines_check = context_text.strip().splitlines()[-10:]
                for line in reversed(lines_check):
                    clean = line.strip().replace('###', '').replace('File:', '').strip()
                    # Rimuovi backticks se presenti `path/to/file.py`
                    clean = clean.replace('`', '')
                    if clean in candidate_files or (clean.endswith('.py') and '/' in clean):
                        target_file = clean
                        break
                
                # 2. Se non trovato, usa auto-detection dal contenuto
                if not target_file:
                    target_file = self._find_target_file(repo_path, search_part, candidate_files)
                
                if target_file:
                    if self._perform_fuzzy_replace(repo_path, target_file, search_part.splitlines(), replace_part.splitlines()):
                        changes_count += 1
                else:
                    logger.warning(f"⚠️ Could not identify file for block {i}")
                
                context_text = next_context # Aggiorna contesto per il prossimo blocco

            if changes_count > 0: return True

        # Fallback Git Apply
        logger.info("🔧 Trying Git Apply fallback...")
        patch_file = repo_path / "llm_gen.patch"
        with open(patch_file, 'w') as f: f.write(patch_content)
        
        # Prova p0 e p1
        for arg in ["-p0", "--ignore-space-change", "--ignore-whitespace"]:
             res = subprocess.run(["git", "apply", arg, "llm_gen.patch"], cwd=repo_path, capture_output=True)
             if res.returncode == 0:
                 logger.info(f"✅ Git apply success ({arg})")
                 return True
        
        logger.error("❌ All patch application methods failed.")
        return False

    def run_experiment(self, instance_id: str, strategy: PromptStrategy):
        logger.info(f"🚀 STARTING EXPERIMENT: {instance_id}")
        model_name = self._detect_running_model()
        instance = self._get_instance(instance_id)
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            logger.info("📥 Cloning repository...")
            repo_path = self.measurer_tool.setup_repository(instance, temp_dir, instance['base_commit'])
            
            # Identify target files from patch
            files_dict = {}
            for line in instance.get("patch", "").splitlines():
                if line.startswith("--- a/"): 
                    fname = line[6:].strip()
                    p = repo_path / fname
                    if p.exists(): files_dict[fname] = p.read_text()
            candidates = list(files_dict.keys())

            # Prompt
            test_cmd = f"pytest {' '.join(instance['efficiency_test'])}"
            ctx = PromptContext(
                problem_statement_type=ProblemStatementType.ORACLE,
                problem_description=f"Optimize energy usage. Tests: {test_cmd}",
                code_files=files_dict,
                test_command=test_cmd,
                target_functions=candidates
            )
            prompt = self.template_manager.generate_prompts(ctx, strategy)
            
            logger.info("📤 Querying LLM...")
            client = self.client_manager.get_client(model_name)
            response = client.generate(prompt, temperature=0.0, max_tokens=8192) # Max tokens increased
            
            logger.info("🔧 Applying Patch...")
            if not self._apply_patch_logic(repo_path, response.content, candidates):
                self._save_results(instance_id, strategy, model_name, {"error": "Patch Failed"}, response, temp_dir)
                return

            logger.info("📦 Installing Dependencies...")
            python_path, conda_env = self.measurer_tool.install_dependencies(
                repo_path, instance['repo'], instance['version'], instance['base_commit']
            )
            
            if not python_path:
                self._save_results(instance_id, strategy, model_name, {"error": "Build Failed"}, response, response.content)
                return

            logger.info("⚡ Measuring Energy...")
            collector = MetricsCollector(instance_id=instance_id, country_code="ESP")
            baseline = collector.measure_baseline(duration=2)
            results = []
            for test in instance['efficiency_test']:
                logger.info(f"   Running test: {test}")
                cmd = f"cd {repo_path} && {python_path} -m pytest '{repo_path}/{test}' -v"
                res = collector.measure_test_execution(test_command=cmd, repetitions=1)
                res['test_name'] = test
                results.append(res)
                
            self._save_results(instance_id, strategy, model_name, {"baseline": baseline, "tests": results}, response, response.content)
            if conda_env: self.measurer_tool.cleanup_conda_env(conda_env)

        except Exception as e:
            logger.error(f"Experiment Failed: {e}", exc_info=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _save_results(self, instance_id, strategy, model, measurements, llm_response, patch):
        output_dir = "results/experiments"
        os.makedirs(output_dir, exist_ok=True)
        report = {
            "instance_id": instance_id, "strategy": strategy.value, "model": model,
            "timestamp": time.time(), "metrics": measurements, "patch": str(patch)
        }
        fname = f"{output_dir}/{instance_id}_{strategy.value}.json"
        with open(fname, 'w') as f: json.dump(report, f, indent=2)
        logger.info(f"💾 Results saved to {fname}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--strategy", required=True, choices=["ZERO_SHOT", "COT", "LDB"])
    parser.add_argument("--dataset", default="data/processed/swe_perf_reduced.json")
    args = parser.parse_args()
    
    runner = GreenExperimentRunner(args.dataset)
    runner.run_experiment(args.instance, PromptStrategy[args.strategy])