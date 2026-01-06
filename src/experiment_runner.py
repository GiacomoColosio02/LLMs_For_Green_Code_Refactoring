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
# Aggiungiamo la root e la cartella scripts per importare i moduli necessari
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
    # Fallback per debug se paths non sono allineati
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GreenExperimentRunner:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset()
        self.client_manager = ClientManager() 
        self.template_manager = PromptTemplateManager()
        # Inizializziamo il misuratore originale per sfruttare le sue utility
        self.measurer_tool = SWEPerfMeasurer(dataset_path, country_code="ESP")

    def _load_dataset(self) -> List[Dict]:
        """Carica il dataset gestendo sia liste che dizionari wrapper."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
            
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

    def _clean_response(self, content: str) -> str:
        """Pulisce la risposta da <think> e markdown per estrarre la patch."""
        # 1. Rimuovi i pensieri <think>
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # 2. Cerca blocchi di codice
        code_blocks = re.findall(r'```(?:diff|python)?\s*(.*?)```', content, re.DOTALL)
        if code_blocks:
            # Cerchiamo il blocco che assomiglia di più a una patch
            for block in code_blocks:
                if "<<<<<<< SEARCH" in block or "diff --git" in block:
                    return block.strip()
            # Se non troviamo keywords, prendiamo il più lungo
            return max(code_blocks, key=len).strip()
            
        return content.strip()

    def _apply_search_replace_patch(self, repo_path: Path, patch_content: str) -> bool:
        """
        Applica manualmente le patch in formato SEARCH/REPLACE (SWE-bench style).
        Necessario perché git apply non supporta questo formato.
        """
        logger.info("🔧 Attempting SEARCH/REPLACE patch application...")
        
        # Semplice parser per blocchi SEARCH/REPLACE
        # Formato atteso:
        # [filepath]
        # <<<<<<< SEARCH
        # ...
        # =======
        # ...
        # >>>>>>> REPLACE
        
        blocks = patch_content.split('<<<<<<< SEARCH')
        if len(blocks) < 2:
            return False # Nessun blocco trovato
            
        applied_count = 0
        
        # Il primo blocco è testo prima del primo match
        # Iteriamo sui successivi
        current_file = None
        
        # Cerchiamo di parsare linea per linea per essere più robusti
        lines = patch_content.splitlines()
        mode = "TEXT"
        search_lines = []
        replace_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Cerca nome file (euristica)
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
                    # Applica blocco
                    if current_file and self._replace_in_file(repo_path, current_file, search_lines, replace_lines):
                        applied_count += 1
                else:
                    replace_lines.append(line)
                    
        return applied_count > 0

    def _replace_in_file(self, repo_path: Path, rel_path: str, search_lines: List[str], replace_lines: List[str]) -> bool:
        target_file = repo_path / rel_path
        if not target_file.exists():
            # Provo a cercare il file se il path non è esatto
            found = list(repo_path.rglob(os.path.basename(rel_path)))
            if found:
                target_file = found[0]
            else:
                logger.warning(f"❌ File not found: {rel_path}")
                return False
        
        content = target_file.read_text()
        search_block = "\n".join(search_lines)
        replace_block = "\n".join(replace_lines)
        
        if search_block in content:
            new_content = content.replace(search_block, replace_block)
            target_file.write_text(new_content)
            logger.info(f"✅ Modified {rel_path}")
            return True
        elif search_block.strip() in content: # Try ignoring trailing newline
             new_content = content.replace(search_block.strip(), replace_block)
             target_file.write_text(new_content)
             logger.info(f"✅ Modified {rel_path} (stripped)")
             return True
        else:
            logger.warning(f"❌ Search block not found in {rel_path}")
            return False

    def _apply_patch(self, repo_path: Path, raw_content: str) -> bool:
        """Tenta diverse strategie di applicazione patch."""
        patch_content = self._clean_response(raw_content)
        patch_file = repo_path / "llm_gen.patch"
        
        # 1. Prova SEARCH/REPLACE (Formato preferito da DeepSeek)
        if "<<<<<<< SEARCH" in patch_content:
            if self._apply_search_replace_patch(repo_path, patch_content):
                return True
            logger.warning("SEARCH/REPLACE parsing failed, falling back to git apply...")

        # 2. Prova GIT APPLY Standard
        with open(patch_file, 'w') as f:
            f.write(patch_content)
            
        res = subprocess.run(
            ["git", "apply", "--ignore-space-change", "--ignore-whitespace", "llm_gen.patch"],
            cwd=repo_path, capture_output=True, text=True
        )
        if res.returncode == 0:
            logger.info("✅ Git apply success")
            return True
            
        # 3. Prova GIT APPLY Relaxed (Reject file)
        res = subprocess.run(
            ["git", "apply", "--reject", "--whitespace=fix", "llm_gen.patch"],
            cwd=repo_path, capture_output=True, text=True
        )
        if res.returncode == 0:
            logger.info("✅ Git apply (relaxed) success")
            return True
            
        logger.error(f"❌ All patch methods failed. Git error: {res.stderr}")
        return False

    def run_experiment(self, instance_id: str, strategy: PromptStrategy):
        logger.info(f"🚀 STARTING EXPERIMENT: {instance_id}")
        
        # 1. Setup
        model_name = self._detect_running_model()
        instance = self._get_instance(instance_id)
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # 2. Clone (Base Commit)
            logger.info("📥 Cloning repository...")
            repo_path = self.measurer_tool.setup_repository(
                instance, temp_dir, instance['base_commit']
            )
            
            # 3. Prompting
            files_dict = {}
            # Estrazione file contesto dalla patch originale
            for line in instance.get("patch", "").splitlines():
                if line.startswith("--- a/"):
                    fname = line[6:].strip()
                    p = repo_path / fname
                    if p.exists(): files_dict[fname] = p.read_text()

            test_cmd = f"pytest {' '.join(instance['efficiency_test'])}"
            ctx = PromptContext(
                problem_statement_type=ProblemStatementType.ORACLE,
                problem_description=f"Optimize energy usage. Tests: {test_cmd}",
                code_files=files_dict,
                test_command=test_cmd,
                target_functions=list(files_dict.keys())
            )
            
            prompt = self.template_manager.generate_prompts(ctx, strategy)
            
            # 4. LLM
            logger.info("📤 Querying LLM...")
            client = self.client_manager.get_client(model_name)
            response = client.generate(prompt, temperature=0.2)
            
            # 5. Patching
            logger.info("🔧 Applying Patch...")
            if not self._apply_patch(repo_path, response.content):
                self._save_results(instance_id, strategy, model_name, {"error": "Patch Failed"}, response, temp_dir)
                return

            # 6. Installazione & Misurazione
            # Usiamo le funzioni esistenti di measure_instance.py ma dobbiamo adattarle
            # Poiché measure_instance.py non espone una funzione "install_and_measure" pubblica generica,
            # usiamo un trucco: lanciamo l'installazione specifica per repo
            
            logger.info("📦 Installing Dependencies...")
            # Logica semplificata basata su measure_instance.py
            python_exec = "python3" # Default fallback
            
            # TODO: Qui dovremmo chiamare la logica esatta di measure_instance per installare l'env
            # Per ora, per testare se la patch funziona, proviamo a usare l'ambiente corrente se compatibile
            # O meglio: saltiamo l'installazione complessa per questo test rapido e usiamo pytest diretto se le lib ci sono
            
            logger.info("⚡ Measuring Energy...")
            collector = MetricsCollector(instance_id=instance_id, country_code="ESP")
            baseline = collector.measure_baseline(duration=2)
            
            results = []
            for test in instance['efficiency_test']:
                logger.info(f"   Running test: {test}")
                # Tentativo di usare python del venv corrente se le dipendenze ci sono
                cmd = f"cd {repo_path} && python3 -m pytest '{repo_path}/{test}' -v"
                res = collector.measure_test_execution(test_command=cmd, repetitions=1)
                res['test_name'] = test
                results.append(res)
                
            self._save_results(instance_id, strategy, model_name, {"baseline": baseline, "tests": results}, response, response.content)

        except Exception as e:
            logger.error(f"Experiment Failed: {e}", exc_info=True)
        finally:
            if os.path.exists(temp_dir):
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
            "patch": self._clean_response(str(patch))
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