"""
Main Experiment Runner for Green Code Refactoring.
Orchestrates: Dataset -> Prompt -> LLM -> Patch -> Measurement.
Integrates with existing SWEPerfMeasurer for robust environment handling.
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

    def _clean_response(self, content: str) -> str:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        code_blocks = re.findall(r'```(?:diff|python)?\s*(.*?)```', content, re.DOTALL)
        if code_blocks:
            for block in code_blocks:
                if "<<<<<<< SEARCH" in block: return block.strip()
            for block in code_blocks:
                if "diff --git" in block or "--- a/" in block: return block.strip()
            return max(code_blocks, key=len).strip()
        return content.strip()

    def _find_target_file(self, repo_path: Path, search_block: str, candidate_files: List[str]) -> Optional[str]:
        search_lines = [l.strip() for l in search_block.splitlines() if l.strip()]
        if not search_lines: return None
        for fname in candidate_files:
            fpath = repo_path / fname
            if fpath.exists():
                content = fpath.read_text()
                if search_lines[0] in content: # Quick check
                    return fname
        return None

    def _apply_search_replace_patch(self, repo_path: Path, patch_content: str, candidate_files: List[str] = []) -> bool:
        logger.info("🔧 Detected SEARCH/REPLACE format. Applying manually...")
        blocks = patch_content.split('<<<<<<< SEARCH')
        if len(blocks) < 2: return False
        
        applied_count = 0
        lines = patch_content.splitlines()
        current_file = None
        search_lines = []
        replace_lines = []
        mode = "TEXT"
        
        for line in lines:
            stripped = line.strip()
            if mode == "TEXT":
                if stripped.startswith("###") or (stripped.endswith(".py") and "/" in stripped):
                    current_file = stripped.replace("###", "").strip()
                if stripped == "<<<<<<< SEARCH":
                    mode = "SEARCH"
                    search_lines = []
            elif mode == "SEARCH":
                if stripped == "=======":
                    mode = "REPLACE"
                    replace_lines = []
                else:
                    search_lines.append(line)
            elif mode == "REPLACE":
                if stripped == ">>>>>>> REPLACE":
                    mode = "TEXT"
                    target = current_file
                    if not target:
                        search_block = "\n".join(search_lines)
                        target = self._find_target_file(repo_path, search_block, candidate_files)
                        if target: logger.info(f"🔎 Auto-detected target file: {target}")
                    
                    if target and self._perform_file_replace(repo_path, target, search_lines, replace_lines):
                        applied_count += 1
                    current_file = None
                else:
                    replace_lines.append(line)
        return applied_count > 0

    def _perform_file_replace(self, repo_path: Path, rel_path: str, search_lines: List[str], replace_lines: List[str]) -> bool:
        """
        Esegue la sostituzione Fuzzy (ignorando whitespace) riga per riga.
        """
        target_file = repo_path / rel_path
        if not target_file.exists():
            candidates = list(repo_path.rglob(os.path.basename(rel_path)))
            if candidates: target_file = candidates[0]
            else:
                logger.warning(f"❌ File not found: {rel_path}")
                return False
                
        content_lines = target_file.read_text().splitlines(keepends=True)
        
        # Normalizziamo le righe di ricerca per il confronto (strip whitespace)
        search_stripped = [l.strip() for l in search_lines]
        search_len = len(search_stripped)
        
        match_index = -1
        
        # Scansione del file
        for i in range(len(content_lines) - search_len + 1):
            # Estraiamo un blocco di righe dal file
            chunk = content_lines[i : i + search_len]
            chunk_stripped = [l.strip() for l in chunk]
            
            # Confronto Fuzzy
            if chunk_stripped == search_stripped:
                match_index = i
                break
        
        if match_index != -1:
            # Costruiamo il nuovo contenuto
            # Aggiungiamo newline alle righe di rimpiazzo se mancano
            final_replace_lines = [l + '\n' if not l.endswith('\n') else l for l in replace_lines]
            
            new_content_lines = (
                content_lines[:match_index] + 
                final_replace_lines + 
                content_lines[match_index + search_len:]
            )
            
            target_file.write_text("".join(new_content_lines))
            logger.info(f"✅ Applied Fuzzy Patch to {rel_path}")
            return True
            
        logger.warning(f"❌ Search block mismatch in {rel_path} (Even with fuzzy match)")
        # Debug: Stampa cosa cercava vs cosa c'è
        # logger.info(f"Wanted: {search_stripped[:3]}...")
        return False

    def _apply_patch(self, repo_path: Path, raw_content: str, candidate_files: List[str]) -> bool:
        patch_content = self._clean_response(raw_content)
        
        logger.info(f"\n📄 PREVIEW PATCH RECEIVED (First 500 chars):\n{'-'*40}\n{patch_content[:500]}\n{'-'*40}")

        if "<<<<<<< SEARCH" in patch_content:
            if self._apply_search_replace_patch(repo_path, patch_content, candidate_files): return True
            
        patch_file = repo_path / "llm_gen.patch"
        with open(patch_file, 'w') as f: f.write(patch_content)
        
        res = subprocess.run(["git", "apply", "--ignore-space-change", "--ignore-whitespace", "llm_gen.patch"], cwd=repo_path, capture_output=True, text=True)
        if res.returncode == 0:
            logger.info("✅ Git apply success")
            return True
            
        res = subprocess.run(["git", "apply", "-p0", "--ignore-space-change", "--ignore-whitespace", "llm_gen.patch"], cwd=repo_path, capture_output=True, text=True)
        if res.returncode == 0:
            logger.info("✅ Git apply (-p0) success")
            return True

        logger.error(f"❌ Patching failed. Last git error: {res.stderr.strip()}")
        return False

    def run_experiment(self, instance_id: str, strategy: PromptStrategy):
        logger.info(f"🚀 STARTING EXPERIMENT: {instance_id}")
        model_name = self._detect_running_model()
        instance = self._get_instance(instance_id)
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            logger.info("📥 Cloning repository...")
            repo_path = self.measurer_tool.setup_repository(instance, temp_dir, instance['base_commit'])
            
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
            "timestamp": time.time(), "metrics": measurements, "patch": self._clean_response(str(patch))
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