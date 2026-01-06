"""
Main Experiment Runner for Green Code Refactoring.
Orchestrates: Dataset -> Prompt -> LLM (vLLM) -> Patch -> Measurement.
"""
import os
import json
import logging
import subprocess
import time
from typing import Optional, Dict, List, Any
from pathlib import Path

# Imports from our modules
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
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
            # If dataset is a list (standard JSONL converted), verify format
            # If it's a dict (instance_id -> data), keep as is.
            # Assuming SWE-perf structure usually list or dict.
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
            raise FileNotFoundError(f"Repository {repo_name} not found in {self.repo_base_dir}. Please download it first.")
        
        logger.info(f"♻️ Resetting {repo_name} to base commit {base_commit}")
        subprocess.run(["git", "reset", "--hard"], cwd=repo_path, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "clean", "-fd"], cwd=repo_path, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "checkout", base_commit], cwd=repo_path, check=True, stdout=subprocess.DEVNULL)

    def _get_oracle_context_files(self, instance: Dict[str, Any], repo_path: str) -> Dict[str, str]:
        """
        ORACLE STRATEGY: Identifies which files to feed the LLM.
        It parses the 'patch' field in the dataset to see which files were modified
        in the gold solution.
        """
        gold_patch = instance.get("patch", "")
        files_content = {}
        
        # Simple parser to find filenames in git diff
        # Format: "--- a/path/to/file.py\n+++ b/path/to/file.py"
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
            logger.warning("Could not extract target files from patch. Using heuristics/fallback?")
        
        # Read content
        for rel_path in target_files:
            full_path = os.path.join(repo_path, rel_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        files_content[rel_path] = f.read()
                except Exception as e:
                    logger.error(f"Error reading {rel_path}: {e}")
        
        return files_content

    def _apply_patch(self, repo_path: str, patch_content: str) -> bool:
        """Attempts to apply the LLM generated patch."""
        patch_file = os.path.join(repo_path, "llm_gen.patch")
        
        # Clean patch content (remove markdown code blocks if present)
        clean_patch = patch_content.replace("```diff", "").replace("```", "").strip()
        
        with open(patch_file, 'w') as f:
            f.write(clean_patch)
            
        # Try applying with git apply
        # We try strict first, then with --reject or --ignore-space-change if needed
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
            # Try to just write the file if it's a full file replacement (LDB sometimes does this)
            # For now, return False
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
            # We run it and wait
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # The script saves a JSON file. We need to find and read it.
            # Usually data/measurements_llm/{instance_id}/measurements.json
            meas_path = Path(f"data/measurements_llm/{instance_id}/measurements.json")
            if meas_path.exists():
                with open(meas_path, 'r') as f:
                    return json.load(f)
            else:
                logger.error("Measurement JSON not found after execution.")
                return {"error": "Measurement output missing"}
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Measurement failed: {e.stderr}")
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
        instance = self._get_instance_data(instance_id)
        repo_name = instance['repo']
        base_commit = instance['base_commit']
        repo_path = os.path.join(self.repo_base_dir, repo_name)
        
        self._checkout_base_commit(repo_name, base_commit)
        
        # 2. Build Context (Oracle)
        # TODO: Add switch for Realistic (BM25) later
        files_dict = self._get_oracle_context_files(instance, repo_path)
        
        # Construct PromptContext object
        # Tests list format: "pytest path/to/test.py::test_func"
        test_cmd = f"pytest {' '.join(instance['efficiency_test'])}"
        
        ctx = PromptContext(
            problem_statement_type=ProblemStatementType.ORACLE,
            problem_description=f"Optimize energy for tests: {test_cmd}",
            code_files=files_dict,
            test_command=test_cmd,
            target_functions=list(files_dict.keys()) # For oracle, we target files we found
        )
        
        # 3. Generate Prompt & Call LLM
        # Handle LDB separately because it's a loop
        if strategy == PromptStrategy.LDB:
            return self._run_ldb_loop(ctx, instance_id, model_alias, repo_path)
        
        # Standard Flow (Zero-Shot / CoT / Self-Collab)
        # For Self-Collab, template manager handles the turn logic internally? 
        # Actually no, we implemented the turns in Manager. 
        # Let's start with Single Turn (Zero-Shot/CoT) for simplicity of this first version.
        
        prompt = self.template_manager.generate_prompts(ctx, strategy)
        
        client = self.client_manager.get_client(model_alias)
        logger.info("📤 Sending request to LLM...")
        response = client.generate(prompt, temperature=0.2) # Low temp for coding
        
        # 4. Extract & Apply
        code_patch = self.template_manager.extract_code(response.content, strategy, ProblemStatementType.ORACLE)
        
        applied = self._apply_patch(repo_path, code_patch)
        
        if not applied:
            logger.error("Experiment Failed at Patch Application")
            return
        
        # 5. Measure
        results = self.measure_energy(instance_id)
        
        # 6. Save Experiment Result
        self._save_results(instance_id, strategy, model_alias, results, response, code_patch)

    def _run_ldb_loop(self, context, instance_id, model_alias, repo_path):
        """Special Loop for LDB Strategy"""
        logger.info("🔄 Entering LDB Iterative Loop")
        
        # Iteration 0: Zero Shot
        prompt = self.template_manager.generate_prompts(context, PromptStrategy.ZERO_SHOT)
        client = self.client_manager.get_client(model_alias)
        response = client.generate(prompt, temperature=0.2)
        patch = self.template_manager.extract_code(response.content, PromptStrategy.ZERO_SHOT, ProblemStatementType.ORACLE)
        
        # Loop (max 2 iterations for energy saving)
        max_iter = 2
        for i in range(max_iter + 1):
            logger.info(f"--- LDB Iteration {i} ---")
            
            # Apply
            if not self._apply_patch(repo_path, patch):
                logger.warning("Patch failed to apply. Stopping LDB.")
                break
                
            # Measure
            measurements = self.measure_energy(instance_id)
            
            # Check Results (Did we improve?)
            feedback_str = self.template_manager.format_ldb_feedback(measurements)
            logger.info(f"Feedback:\n{feedback_str}")
            
            # If "TARGET MET" or last iteration, stop
            if "[TARGET MET]" in feedback_str or i == max_iter:
                self._save_results(instance_id, PromptStrategy.LDB, model_alias, measurements, response, patch)
                break
            
            # Else: Generate Debug Prompt
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
                "latency": llm_response.latency_seconds,
                "tokens": llm_response.total_tokens
            },
            "generated_patch": patch
        }
        
        fname = f"{output_dir}/{instance_id}_{strategy.value}.json"
        with open(fname, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"💾 Experiment Report saved to {fname}")

# CLI Entry Point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, help="Instance ID (e.g. astropy__astropy-123)")
    parser.add_argument("--strategy", required=True, choices=["ZERO_SHOT", "COT", "LDB", "SELF_COLLABORATION"])
    parser.add_argument("--dataset", default="data/swe_perf_green_k1.json")
    
    args = parser.parse_args()
    
    # Map string to Enum
    strat_enum = PromptStrategy[args.strategy]
    
    runner = GreenExperimentRunner(args.dataset)
    runner.run_experiment(args.instance, strat_enum)