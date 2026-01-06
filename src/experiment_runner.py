"""
Main Experiment Runner for Green Code Refactoring.
Orchestrates: Dataset -> Prompt -> LLM -> Patch -> Measurement.
Features: 
- "Hunter Parser" to find patches inside chatty LLM responses.
- Fuzzy Matcher to apply patches even with whitespace mismatches.
- Integration with SWEPerfMeasurer.
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
from typing import Optional, Dict, List, Any, Union
from pathlib import Path

# --- CONFIGURAZIONE PATH ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "scripts"))

# Import nostri moduli
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
        """
        HUNTER PARSER: Scansiona il testo per trovare la patch, ignorando le chiacchiere.
        """
        # 1. Rimuovi i tag di pensiero <think>
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # 2. Cerca l'inizio di un blocco SEARCH/REPLACE
        search_marker = "<<<<<<< SEARCH"
        diff_marker = "diff --git"
        
        start_idx = -1
        idx_search = content.find(search_marker)
        idx_diff = content.find(diff_marker)
        
        # Trova il primo marcatore valido
        if idx_search != -1 and (idx_diff == -1 or idx_search < idx_diff):
            start_idx = idx_search
            # Cerca di risalire al nome del file (spesso è nella riga sopra o due sopra)
            # Prendi un buffer di 200 char prima del match
            pre_buffer_start = max(0, start_idx - 200)
            return content[pre_buffer_start:] 
        elif idx_diff != -1:
            start_idx = idx_diff
            return content[start_idx:]
            
        # 3. Fallback: Se è wrappato in markdown python/diff senza marker specifici
        code_blocks = re.findall(r'```(?:diff|python)?\s*(.*?)```', content, re.DOTALL)
        if code_blocks:
            # Ritorna il blocco più lungo
            return max(code_blocks, key=len).strip()
            
        return content.strip()

    def _find_target_file(self, repo_path: Path, search_block: str, candidate_files: List[str]) -> Optional[str]:
        # Normalizza rimuovendo spazi vuoti per il confronto
        def clean(s): return [l.strip() for l in s.splitlines() if l.strip()]
        search_lines = clean(search_block)
        if not search_lines: return None
        
        for fname in candidate_files:
            fpath = repo_path / fname
            if fpath.exists():
                content = fpath.read_text()
                # Controllo rapido: la prima riga del blocco esiste nel file?
                if search_lines[0] in content:
                    return fname
        return None

    def _perform_file_replace(self, repo_path: Path, rel_path: str, search_lines: List[str], replace_lines: List[str]) -> bool:
        target_file = repo_path / rel_path
        if not target_file.exists():
            # Ricerca euristica
            candidates = list(repo_path.rglob(os.path.basename(rel_path)))
            if candidates: target_file = candidates[0]
            else:
                logger.warning(f"❌ File not found: {rel_path}")
                return False
                
        content_lines = target_file.read_text().splitlines(keepends=True)
        search_stripped = [l.strip() for l in search_lines]
        search_len = len(search_stripped)
        
        if search_len == 0: return False

        # Scansione Fuzzy (ignora whitespace)
        match_index = -1
        for i in range(len(content_lines) - search_len + 1):
            chunk = content_lines[i : i + search_len]
            chunk_stripped = [l.strip() for l in chunk]
            
            if chunk_stripped == search_stripped:
                match_index = i
                break
        
        if match_index != -1:
            # Costruisci le nuove linee
            final_replace_lines = [l + '\n' if not l.endswith('\n') else l for l in replace_lines]
            new_content_lines = content_lines[:match_index] + final_replace_lines + content_lines[match_index + search_len:]
            target_file.write_text("".join(new_content_lines))
            logger.info(f"✅ Applied Fuzzy Patch to {rel_path}")
            return True
            
        logger.warning(f"❌ Search block mismatch in {rel_path}")
        return False

    def _apply_search_replace_patch(self, repo_path: Path, patch_content: str, candidate_files: List[str] = []) -> bool:
        logger.info("🔧 Detected SEARCH/REPLACE format. Parsing...")
        
        # Splitto per blocchi, ma mantengo un po' di contesto prima del SEARCH per trovare il filename
        raw_blocks = patch_content.split('<<<<<<< SEARCH')
        if len(raw_blocks) < 2: return False
        
        applied_count = 0
        
        # Iteriamo sui split. 
        # block[0] è testo inutile (o filename del primo blocco).
        # block[1] è "content \n ======= \n content \n >>>>>>> REPLACE"
        
        # Il filename del blocco N si trova alla fine del blocco N-1
        previous_text = raw_blocks[0]
        
        for i in range(1, len(raw_blocks)):
            block_body = raw_blocks[i]
            
            # Parsing del corpo del blocco
            if "=======" not in block_body or ">>>>>>> REPLACE" not in block_body:
                logger.warning(f"⚠️ Malformed block {i}, skipping.")
                previous_text = block_body
                continue
                
            search_part, rest = block_body.split('=======', 1)
            replace_part, after_replace = rest.split('>>>>>>> REPLACE', 1)
            
            search_lines = search_part.splitlines()
            replace_lines = replace_part.splitlines()
            
            # Cerca il nome del file nelle ultime righe del testo precedente
            # Cerca pattern: "path/to/file.py" o "### path/to/file.py"
            target_file = None
            lines_to_check = previous_text.strip().splitlines()[-5:] # Guarda ultime 5 righe
            
            for line in reversed(lines_to_check):
                clean_line = line.strip().replace('###', '').strip()
                if clean_line.endswith('.py'): # Euristica semplice
                    target_file = clean_line
                    break
            
            # Se non trovato, usa auto-detection
            if not target_file:
                search_txt = "\n".join(search_lines)
                target_file = self._find_target_file(repo_path, search_txt, candidate_files)
                if target_file: logger.info(f"🔎 Auto-detected file: {target_file}")
            
            if target_file:
                if self._perform_file_replace(repo_path, target_file, search_lines, replace_lines):
                    applied_count += 1
            else:
                logger.warning(f"⚠️ Could not identify target file for block {i}")
            
            previous_text = after_replace # Il testo dopo il REPLACE diventa il preambolo del prossimo

        return applied_count > 0

    def _apply_patch(self, repo_path: Path, raw_content: str, candidate_files: List[str]) -> bool:
        patch_content = self._extract_patch_content(raw_content)
        
        logger.info(f"\n📄 PREVIEW EXTRACTED PATCH (First 300 chars):\n{'-'*30}\n{patch_content[:300]}\n{'-'*30}")

        # 1. Prova SEARCH/REPLACE
        if "<<<<<<< SEARCH" in patch_content:
            if self._apply_search_replace_patch(repo_path, patch_content, candidate_files): return True
            
        # 2. Prova Git Apply
        patch_file = repo_path / "llm_gen.patch"
        with open(patch_file, 'w') as f: f.write(patch_content)
        
        # Standard
        res = subprocess.run(["git", "apply", "--ignore-space-change", "--ignore-whitespace", "llm_gen.patch"], cwd=repo_path, capture_output=True, text=True)
        if res.returncode == 0:
            logger.info("✅ Git apply success")
            return True
            
        # P0 Fallback
        res = subprocess.run(["git", "apply", "-p0", "--ignore-space-change", "--ignore-whitespace", "llm_gen.patch"], cwd=repo_path, capture_output=True, text=True)
        if res.returncode == 0:
            logger.info("✅ Git apply (-p0) success")
            return True

        logger.error(f"❌ Patching failed.")
        return False

    def run_experiment(self, instance_id: str, strategy: PromptStrategy):
        logger.info(f"🚀 STARTING EXPERIMENT: {instance_id}")
        model_name = self._detect_running_model()
        instance = self._get_instance(instance_id)
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            logger.info("📥 Cloning repository...")
            repo_path = self.measurer_tool.setup_repository(instance, temp_dir, instance['base_commit'])
            
            # Context Extraction
            files_dict = {}
            target_files = set()
            for line in instance.get("patch", "").splitlines():
                if line.startswith("--- a/"): target_files.add(line[6:].strip())
            for tf in target_files:
                p = repo_path / tf
                if p.exists(): files_dict[tf] = p.read_text()
            candidate_files_list = list(files_dict.keys())

            test_cmd = f"pytest {' '.join(instance['efficiency_test'])}"
            ctx = PromptContext(
                problem_statement_type=ProblemStatementType.ORACLE,
                problem_description=f"Optimize energy usage. Tests: {test_cmd}",
                code_files=files_dict,
                test_command=test_cmd,
                target_functions=candidate_files_list
            )
            prompt = self.template_manager.generate_prompts(ctx, strategy)
            
            logger.info("📤 Querying LLM...")
            client = self.client_manager.get_client(model_name)
            response = client.generate(prompt, temperature=0.2)
            
            logger.info("🔧 Applying Patch...")
            if not self._apply_patch(repo_path, response.content, candidate_files_list):
                self._save_results(instance_id, strategy, model_name, {"error": "Patch Failed"}, response, temp_dir)
                return

            logger.info("📦 Installing Dependencies...")
            python_path, conda_env = self.measurer_tool.install_dependencies(
                repo_path, instance['repo'], instance['version'], instance['base_commit']
            )
            
            if not python_path: return

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
            "timestamp": time.time(), "metrics": measurements, "patch": str(patch) # Save raw
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