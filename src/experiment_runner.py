"""
Main Experiment Runner for Green Code Refactoring.
Orchestrates: Dataset -> Prompt -> LLM (vLLM) -> Patch -> Measurement.
"""
import sys
import os
import json
import logging
import subprocess
import time
from typing import Optional, Dict, List, Any, Union
from pathlib import Path

# --- FIX IMPORT PATH ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -----------------------

from src.llm_clients.client_manager import ClientManager
from src.prompt_templates.template_manager import PromptTemplateManager
from src.prompt_templates.base_template import PromptStrategy, ProblemStatementType, PromptContext

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GreenExperimentRunner:
    def __init__(self, dataset_path: str, repo_base_dir: str = "repositories"):
        self.dataset_path = dataset_path
        self.repo_base_dir = repo_base_dir
        self.dataset_content = self._load_dataset()
        self.client_manager = ClientManager() 
        self.template_manager = PromptTemplateManager()

    def _load_dataset(self) -> Union[List[Dict], Dict]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        with open(self.dataset_path, 'r') as f:
            return json.load(f)

    def _get_instance_data(self, instance_id: str) -> Dict[str, Any]:
        data = self.dataset_content
        if isinstance(data, dict) and "instances" in data:
            target_list = data["instances"]
        elif isinstance(data, list):
            target_list = data
        else:
            raise ValueError("Unknown dataset structure.")

        for item in target_list:
            if item.get("instance_id") == instance_id:
                return item
        raise ValueError(f"Instance {instance_id} not found in dataset")

    def _checkout_base_commit(self, repo_name: str, base_commit: str):
        """
        Assicura che la repo esista e sia al commit corretto.
        SE MANCA, LA CLONA AUTOMATICAMENTE.
        """
        repo_path = os.path.join(self.repo_base_dir, repo_name)
        
        # --- AUTO CLONE FIX ---
        if not os.path.exists(repo_path):
            logger.info(f"📥 Repository {repo_name} missing. Cloning into {repo_path}...")
            os.makedirs(os.path.dirname(repo_path), exist_ok=True)
            try:
                subprocess.run(
                    ["git", "clone", f"https://github.com/{repo_name}.git", repo_path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Git Clone Failed: {e.stderr.decode()}")
                raise
        # ----------------------

        logger.info(f"♻️ Resetting {repo_name} to base commit {base_commit}")
        try:
            subprocess.run(["git", "reset", "--hard"], cwd=repo_path, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "clean", "-fd"], cwd=repo_path, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "checkout", base_commit], cwd=repo_path, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            logger.error(f"Git Checkout Error: {e}")
            raise

    def _get_oracle_context_files(self, instance: Dict[str, Any], repo_path: str) -> Dict[str, str]:
        gold_patch = instance.get("patch", "")
        files_content = {}
        lines = gold_patch.split('\n')
        target_files = set()
        for line in lines:
            if line.startswith("--- a/"):
                target_files.add(line[6:].strip())
            elif line.startswith("+++ b/"):
                target_files.add(line[6:].strip())

        if not target_files:
            logger.warning("⚠️ No target files found in patch.")
        
        for rel_path in target_files:
            full_path = os.path.join(repo_path, rel_path)
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    files_content[rel_path] = f.read()
        return files_content

    def _apply_patch(self, repo_path: str, patch_content: str) -> bool:
        patch_file = os.path.join(repo_path, "llm_gen.patch")
        clean_patch = patch_content.replace("```diff", "").replace("```python", "").replace("```", "").strip()
        
        with open(patch_file, 'w') as f:
            f.write(clean_patch)
            
        result = subprocess.run(
            ["git", "apply", "--ignore-space-change", "--ignore-whitespace", "llm_gen.patch"],
            cwd=repo_path, capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Patch applied successfully!")
            return True
        else:
            logger.error(f"❌ Patch failed: {result.stderr}")
            return False

    def measure_energy(self, instance_id: str) -> Dict[str, Any]:
        """
        Lancia la misurazione.
        NOTA: Attualmente measure_instance.py usa una copia pulita.
        Dovremo modificarlo nel prossimo passo per usare la nostra copia modificata.
        """
        logger.info("⚡ Running Measurement...")
        # Per ora lanciamo lo script standard, sapendo che misurerà la baseline originale
        # Questo ci serve solo per verificare che il flusso funzioni.
        cmd = [
            "python", "scripts/measure_instance.py",
            "--instance", instance_id,
            "--dataset", self.dataset_path,
            "--output", "data/measurements_llm"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            meas_path = Path(f"data/measurements_llm/{instance_id}/measurements.json")
            if meas_path.exists():
                with open(meas_path, 'r') as f:
                    return json.load(f)
            return {"error": "Measurement output missing"}
        except subprocess.CalledProcessError as e:
            logger.error(f"Measurement failed: {e.stderr}")
            return {"error": str(e)}

    def run_experiment(self, instance_id: str, strategy: PromptStrategy, model_alias: str = "active_model"):
        logger.info(f"🚀 STARTING EXPERIMENT: {instance_id}")
        
        instance = self._get_instance_data(instance_id)
        repo_name = instance['repo']
        base_commit = instance['base_commit']
        
        # 1. Checkout (Scarica se manca)
        self._checkout_base_commit(repo_name, base_commit)
        repo_path = os.path.join(self.repo_base_dir, repo_name)
        
        # 2. Prompt
        files_dict = self._get_oracle_context_files(instance, repo_path)
        test_cmd = f"pytest {' '.join(instance.get('efficiency_test', []))}"
        
        ctx = PromptContext(
            problem_statement_type=ProblemStatementType.ORACLE,
            problem_description=f"Optimize energy for: {test_cmd}",
            code_files=files_dict,
            test_command=test_cmd,
            target_functions=list(files_dict.keys())
        )
        
        prompt = self.template_manager.generate_prompts(ctx, strategy)
        
        # 3. LLM
        logger.info("📤 Asking LLM...")
        client = self.client_manager.get_client(model_alias)
        response = client.generate(prompt, temperature=0.2)
        
        # 4. Patch
        patch = self.template_manager.extract_code(response.content, strategy, ProblemStatementType.ORACLE)
        self._apply_patch(repo_path, patch)
        
        # 5. Measure
        results = self.measure_energy(instance_id)
        self._save_results(instance_id, strategy, model_alias, results, response, patch)

    def _save_results(self, instance_id, strategy, model, measurements, llm_response, patch):
        output_dir = "results/experiments"
        os.makedirs(output_dir, exist_ok=True)
        report = {
            "instance_id": instance_id, "strategy": strategy.value, "model": model,
            "timestamp": time.time(), "measurements": measurements,
            "patch": patch
        }
        fname = f"{output_dir}/{instance_id}_{strategy.value}.json"
        with open(fname, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"💾 Report saved: {fname}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--strategy", required=True, choices=["ZERO_SHOT", "COT", "LDB"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repo_dir", default="repositories")
    args = parser.parse_args()
    
    runner = GreenExperimentRunner(args.dataset, repo_base_dir=args.repo_dir)
    runner.run_experiment(args.instance, PromptStrategy[args.strategy])