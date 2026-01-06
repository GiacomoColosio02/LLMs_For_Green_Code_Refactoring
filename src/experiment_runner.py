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
from typing import Optional, Dict, List, Any
from pathlib import Path

# --- FIX IMPORT PATH ---
# Aggiungiamo la root del progetto al Python Path per poter importare 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -----------------------

from src.llm_clients.client_manager import ClientManager
from src.prompt_templates.template_manager import PromptTemplateManager
from src.prompt_templates.base_template import PromptStrategy, ProblemStatementType, PromptContext

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GreenExperimentRunner:
    def __init__(self, dataset_path: str, repo_base_dir: str = "repositories"):
        """
        Args:
            dataset_path: Path to swe_perf_green_k1.json
            repo_base_dir: Directory where repositories are cloned (data/repos)
        """
        self.dataset_path = dataset_path
        self.repo_base_dir = repo_base_dir
        self.dataset = self._load_dataset()
        
        # Managers
        self.client_manager = ClientManager() # Connects to localhost:8000
        self.template_manager = PromptTemplateManager()

    def _load_dataset(self) -> Dict[str, Any]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}. Did you run the creation script?")
            
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
            return data

    def _get_instance_data(self, instance_id: str) -> Dict[str, Any]:
        """Finds the specific instance in the dataset."""
        if isinstance(self.dataset, list):
            for item in self.dataset:
                if item.get("instance_id") == instance_id:
                    return item
        elif isinstance(self.dataset, dict):
            if instance_id in self.dataset:
                return self.dataset[instance_id]
        
        raise ValueError(f"Instance {instance_id} not found in dataset")

    def _checkout_base_commit(self, repo_name: str, base_commit: str):
        """Resets the repository to the clean base state."""
        repo_path = os.path.join(self.repo_base_dir, repo_name)
        if not os.path.exists(repo_path):
            # Se la repo non c'è, proviamo a crearla (opzionale) o diamo errore
            raise FileNotFoundError(f"Repository {repo_name} not found in {self.repo_base_dir}. Please run setup scripts first.")
        
        logger.info(f"♻️ Resetting {repo_name} to base commit {base_commit}")
        # Git reset hard e clean per assicurarsi che non ci siano residui
        subprocess.run(["git", "reset", "--hard"], cwd=repo_path, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "clean", "-fd"], cwd=repo_path, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "checkout", base_commit], cwd=repo_path, check=True, stdout=subprocess.DEVNULL)

    def _get_oracle_context_files(self, instance: Dict[str, Any], repo_path: str) -> Dict[str, str]:
        """
        ORACLE STRATEGY: Identifies which files to feed the LLM based on Gold Patch.
        """
        gold_patch = instance.get("patch", "")
        files_content = {}
        
        # Simple parser to find filenames in git diff
        lines = gold_patch.split('\n')
        target_files = set()
        for line in lines:
            if line.startswith("--- a/"):
                fname = line[6:].strip()
                target_files.add(fname)
            elif line.startswith("+++ b/"):
                fname = line[6:].strip()
                target_files.add(fname)

        if not target_files:
            logger.warning("⚠️ Could not extract target files from patch. Context might be empty!")
        
        # Read content
        for rel_path in target_files:
            full_path = os.path.join(repo_path, rel_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        files_content[rel_path] = f.read()
                except Exception as e:
                    logger.error(f"Error reading {rel_path}: {e}")
            else:
                logger.warning(f"Target file {rel_path} not found on disk.")
        
        return files_content

    def _apply_patch(self, repo_path: str, patch_content: str) -> bool:
        """Attempts to apply the LLM generated patch."""
        patch_file = os.path.join(repo_path, "llm_gen.patch")
        
        # Pulizia base del markdown
        clean_patch = patch_content.replace("```diff", "").replace("```", "").strip()
        
        with open(patch_file, 'w') as f:
            f.write(clean_patch)
            
        # Tentativo 1: Git Apply standard
        result = subprocess.run(
            ["git", "apply", "--ignore-space-change", "--ignore-whitespace", "llm_gen.patch"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Patch applied successfully!")
            return True
        else:
            logger.error(f"❌ Patch application failed: {result.stderr}")
            return False

    def measure_energy(self, instance_id: str) -> Dict[str, Any]:
        """
        Calls measure_instance.py to get the real consumption.
        """
        logger.info("⚡ Running Energy Measurement...")
        cmd = [
            "python", "scripts/measure_instance.py",
            "--instance", instance_id,
            "--dataset", self.dataset_path,
            "--output", "data/measurements_llm" 
        ]
        
        try:
            # Eseguiamo e aspettiamo
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Measurement script finished.")
            
            # Cerchiamo il file JSON prodotto
            meas_path = Path(f"data/measurements_llm/{instance_id}/measurements.json")
            if meas_path.exists():
                with open(meas_path, 'r') as f:
                    return json.load(f)
            else:
                logger.error(f"Measurement JSON not found at {meas_path}")
                return {"error": "Measurement output missing"}
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Measurement script failed: {e.stderr}")
            logger.error(f"Stdout: {e.stdout}")
            return {"error": str(e)}

    def run_experiment(
        self, 
        instance_id: str, 
        strategy: PromptStrategy,
        model_alias: str = "active_model"
    ):
        """
        MAIN EXECUTION FLOW
        """
        logger.info(f"🚀 STARTING EXPERIMENT: {instance_id} | Strategy: {strategy.value}")
        
        # 1. Setup Data & Repo
        try:
            instance = self._get_instance_data(instance_id)
        except ValueError as e:
            logger.error(str(e))
            return

        repo_name = instance['repo']
        base_commit = instance['base_commit']
        repo_path = os.path.join(self.repo_base_dir, repo_name)
        
        try:
            self._checkout_base_commit(repo_name, base_commit)
        except Exception as e:
            logger.error(f"Failed to checkout repo: {e}")
            return
        
        # 2. Build Context (Oracle)
        files_dict = self._get_oracle_context_files(instance, repo_path)
        
        if not files_dict:
            logger.error("No context files found. Aborting.")
            return

        test_list = instance.get('efficiency_test', [])
        test_cmd = f"pytest {' '.join(test_list)}"
        
        ctx = PromptContext(
            problem_statement_type=ProblemStatementType.ORACLE,
            problem_description=f"Optimize energy for tests: {test_cmd}",
            code_files=files_dict,
            test_command=test_cmd,
            target_functions=list(files_dict.keys())
        )
        
        # 3. Generate Prompt & Call LLM
        if strategy == PromptStrategy.LDB:
            self._run_ldb_loop(ctx, instance_id, model_alias, repo_path)
            return
        
        # Standard Flow
        try:
            prompt = self.template_manager.generate_prompts(ctx, strategy)
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            return

        logger.info("📤 Sending request to LLM...")
        client = self.client_manager.get_client(model_alias)
        
        try:
            # Temperature bassa per codice deterministico
            response = client.generate(prompt, temperature=0.2)
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return
        
        # 4. Extract & Apply
        code_patch = self.template_manager.extract_code(response.content, strategy, ProblemStatementType.ORACLE)
        
        applied = self._apply_patch(repo_path, code_patch)
        
        if not applied:
            logger.error("Experiment Failed at Patch Application phase. Saving result as failure.")
            self._save_results(instance_id, strategy, model_alias, {"error": "Patch application failed"}, response, code_patch)
            return
        
        # 5. Measure
        results = self.measure_energy(instance_id)
        
        # 6. Save Experiment Result
        self._save_results(instance_id, strategy, model_alias, results, response, code_patch)

    def _run_ldb_loop(self, context, instance_id, model_alias, repo_path):
        """Special Loop for LDB Strategy"""
        logger.info("🔄 Entering LDB Iterative Loop")
        
        # Iteration 0
        prompt = self.template_manager.generate_prompts(context, PromptStrategy.ZERO_SHOT)
        client = self.client_manager.get_client(model_alias)
        response = client.generate(prompt, temperature=0.2)
        patch = self.template_manager.extract_code(response.content, PromptStrategy.ZERO_SHOT, ProblemStatementType.ORACLE)
        
        max_iter = 2
        for i in range(max_iter + 1):
            logger.info(f"--- LDB Iteration {i} ---")
            
            if not self._apply_patch(repo_path, patch):
                logger.warning("Patch failed to apply in loop. Stopping.")
                break
                
            measurements = self.measure_energy(instance_id)
            feedback_str = self.template_manager.format_ldb_feedback(measurements)
            
            logger.info(f"Feedback: {feedback_str[:100]}...") # Log solo inizio
            
            if "[TARGET MET]" in feedback_str or i == max_iter:
                self._save_results(instance_id, PromptStrategy.LDB, model_alias, measurements, response, patch)
                break
            
            debug_prompt = self.template_manager.generate_ldb_debug_prompt(context, patch, feedback_str)
            response = client.generate(debug_prompt, temperature=0.2)
            patch = self.template_manager.extract_code(response.content, PromptStrategy.LDB, ProblemStatementType.ORACLE)

    def _save_results(self, instance_id, strategy, model, measurements, llm_response, patch):
        """Saves a JSON report of the run."""
        output_dir = "results/experiments"
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            "instance_id": instance_id,
            "strategy": strategy.value,
            "model": model,
            "timestamp": time.time(),
            "measurements": measurements,
            "llm_output_meta": {
                "latency": getattr(llm_response, 'latency_seconds', 0),
                "tokens": getattr(llm_response, 'total_tokens', 0)
            },
            "generated_patch": patch
        }
        
        fname = f"{output_dir}/{instance_id}_{strategy.value}.json"
        with open(fname, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"💾 Experiment Report saved to {fname}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, help="Instance ID (e.g. astropy__astropy-123)")
    parser.add_argument("--strategy", required=True, choices=["ZERO_SHOT", "COT", "LDB", "SELF_COLLABORATION"])
    parser.add_argument("--dataset", default="data/swe_perf_reduced.json") # Default al reduced se il green non c'è ancora
    
    args = parser.parse_args()
    
    strat_enum = PromptStrategy[args.strategy]
    
    runner = GreenExperimentRunner(args.dataset)
    runner.run_experiment(args.instance, strat_enum)