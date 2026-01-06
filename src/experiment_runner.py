"""
Main Experiment Runner for Green Code Refactoring.
Orchestrates: Dataset -> Prompt -> LLM -> Patch -> Measurement (via SWEPerfMeasurer).
"""
import sys
import os
import re
import json
import logging
import subprocess  # <--- MANCAVA QUESTO!
import time
import tempfile
import shutil
from typing import Optional, Dict, List, Any, Union
from pathlib import Path

# --- CONFIGURAZIONE PATH ---
# Aggiungiamo la root e la cartella scripts per importare i moduli necessari
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "scripts"))

# Importiamo i nostri moduli
from src.llm_clients.client_manager import ClientManager
from src.prompt_templates.template_manager import PromptTemplateManager
from src.prompt_templates.base_template import PromptStrategy, ProblemStatementType, PromptContext
from openai import OpenAI

# Importiamo il misuratore originale (senza modificarlo)
try:
    from measure_instance import SWEPerfMeasurer
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
        
        # Inizializziamo il misuratore originale per sfruttare le sue utility (clone, install)
        self.measurer_tool = SWEPerfMeasurer(dataset_path, country_code="ESP")

    def _load_dataset(self) -> List[Dict]:
        """Carica il dataset Reduced (Lista di Dict)."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
            
        # Gestione robusta della struttura
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "instances" in data:
            return data["instances"]
        else:
            raise ValueError(f"Unknown dataset structure in {self.dataset_path}")

    def _get_instance(self, instance_id: str) -> Dict[str, Any]:
        for item in self.dataset:
            if item.get("instance_id") == instance_id:
                return item
        raise ValueError(f"Instance {instance_id} not found")

    def _detect_running_model(self) -> str:
        """Rileva automaticamente il modello vLLM attivo."""
        try:
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
            models = client.models.list()
            if models.data:
                name = models.data[0].id
                logger.info(f"✅ Detected Model: {name}")
                return name
        except Exception:
            logger.warning("⚠️ Could not detect model. Using 'active_model'")
        return "active_model"

    def _clean_patch_content(self, content: str) -> str:
        """Pulisce la risposta di DeepSeek per estrarre la patch."""
        # 1. Rimuovi i pensieri <think>
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # 2. Cerca blocchi di codice
        code_blocks = re.findall(r'```(?:diff|python)?\s*(.*?)```', content, re.DOTALL)
        if code_blocks:
            # Prende il blocco più lungo che sembra una patch
            candidates = [b for b in code_blocks if "diff --git" in b or ("<<<<<<< SEARCH" in b)]
            if candidates:
                return max(candidates, key=len).strip()
            return max(code_blocks, key=len).strip()
            
        return content.strip()

    def _apply_patch_to_repo(self, repo_path: Path, patch_content: str) -> bool:
        """Applica la patch nella cartella specificata."""
        patch_file = repo_path / "llm_gen.patch"
        clean_patch = self._clean_patch_content(patch_content)
        
        with open(patch_file, 'w') as f:
            f.write(clean_patch)
            
        # Tentativo 1: Git Apply standard
        res = subprocess.run(
            ["git", "apply", "--ignore-space-change", "--ignore-whitespace", "llm_gen.patch"],
            cwd=repo_path, capture_output=True, text=True
        )
        if res.returncode == 0:
            return True
            
        # Tentativo 2: Patch rejection allow (per formati imprecisi)
        logger.warning(f"Git apply failed ({res.stderr.strip()}), trying robust mode...")
        res = subprocess.run(
            ["git", "apply", "--reject", "--whitespace=fix", "llm_gen.patch"],
            cwd=repo_path, capture_output=True, text=True
        )
        return res.returncode == 0

    def run_experiment(self, instance_id: str, strategy: PromptStrategy):
        logger.info(f"🚀 STARTING EXPERIMENT: {instance_id}")
        
        # 1. Preparazione Dati
        model_name = self._detect_running_model()
        instance = self._get_instance(instance_id)
        
        # Creiamo una cartella temporanea per tutto il processo
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            # 2. Clonazione Repo (Uso logica originale di measure_instance)
            # Scarica la repo pulita al commit BASE
            logger.info("📥 Cloning repository (Base Commit)...")
            repo_path = self.measurer_tool.setup_repository(
                instance, temp_path, instance['base_commit']
            )
            
            # 3. Prompting e LLM
            # Per l'Oracle context, leggiamo i file dalla repo appena clonata
            files_dict = {}
            # Parsing semplice della patch originale per trovare i file target
            target_files = set()
            for line in instance.get("patch", "").splitlines():
                if line.startswith("--- a/"): target_files.add(line[6:].strip())
                elif line.startswith("+++ b/"): target_files.add(line[6:].strip())
            
            for tf in target_files:
                p = repo_path / tf
                if p.exists():
                    files_dict[tf] = p.read_text()

            test_cmd = f"pytest {' '.join(instance['efficiency_test'])}"
            ctx = PromptContext(
                problem_statement_type=ProblemStatementType.ORACLE,
                problem_description=f"Optimize energy usage. Tests: {test_cmd}",
                code_files=files_dict,
                test_command=test_cmd,
                target_functions=list(files_dict.keys())
            )
            
            prompt = self.template_manager.generate_prompts(ctx, strategy)
            
            logger.info("📤 Querying LLM...")
            client = self.client_manager.get_client(model_name)
            response = client.generate(prompt, temperature=0.2)
            
            # 4. Applicazione Patch
            logger.info("🔧 Applying LLM Patch...")
            patch_success = self._apply_patch_to_repo(repo_path, response.content)
            
            if not patch_success:
                logger.error("❌ Patch Application Failed. Aborting measurement.")
                self._save_results(instance_id, strategy, model_name, {"error": "Patch Failed"}, response, temp_dir)
                return

            # 5. Installazione Dipendenze (Uso logica originale)
            logger.info("📦 Installing Dependencies...")
            python_path, conda_env = self.measurer_tool.install_dependencies(
                repo_path, instance['repo'], instance['version'], instance['base_commit']
            )
            
            if not python_path:
                logger.error("❌ Dependency Installation Failed.")
                return

            # 6. Misurazione (Custom Loop usando collector originale)
            logger.info("⚡ Measuring Energy (LLM Solution)...")
            collector = MetricsCollector(instance_id=instance_id, country_code="ESP")
            
            # Baseline (Idle)
            baseline = collector.measure_baseline(duration=2)
            
            results = []
            for test in instance['efficiency_test']:
                logger.info(f"   Running test: {test}")
                # Costruiamo il comando come fa measure_instance.py
                cmd = f"cd {repo_path} && {python_path} -m pytest '{repo_path}/{test}' -v"
                res = collector.measure_test_execution(test_command=cmd, repetitions=1)
                res['test_name'] = test
                results.append(res)
                
            # 7. Salvataggio
            final_data = {
                "baseline": baseline,
                "tests": results,
                "status": "success"
            }
            self._save_results(instance_id, strategy, model_name, final_data, response, response.content)
            
            # Cleanup Env
            if conda_env:
                self.measurer_tool.cleanup_conda_env(conda_env)

        except Exception as e:
            logger.error(f"Experiment Failed: {e}", exc_info=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _save_results(self, instance_id, strategy, model, measurements, llm_response, patch):
        output_dir = "results/experiments"
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            "instance_id": instance_id,
            "strategy": strategy.value,
            "model": model,
            "timestamp": time.time(),
            "metrics": measurements,
            "patch": self._clean_patch_content(str(patch))
        }
        
        fname = f"{output_dir}/{instance_id}_{strategy.value}.json"
        with open(fname, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"💾 Results saved to {fname}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--strategy", required=True, choices=["ZERO_SHOT", "COT", "LDB"])
    # Default al Reduced Dataset come richiesto
    parser.add_argument("--dataset", default="data/processed/swe_perf_reduced.json")
    
    args = parser.parse_args()
    
    runner = GreenExperimentRunner(args.dataset)
    runner.run_experiment(args.instance, PromptStrategy[args.strategy])