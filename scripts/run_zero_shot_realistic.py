"""
Specific Experiment Runner: ZERO SHOT + REALISTIC SETTING.
Logic:
1. Trigger: Failing Test.
2. Context Anchor: The Test Code itself.
3. Retrieval: Scan repo and find Top-5 files semantically related to the Test Code.
4. LLM: Asks to fix the issue given the mix of relevant/irrelevant files.
"""
import sys
import os
import re
import json
import logging
import subprocess
import time
import tempfile
import random
from pathlib import Path
from collections import Counter

# --- SETUP PATH ---
# Aggiusta il path per importare src se lo script è in scripts/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.llm_clients.client_manager import ClientManager
from src.prompt_templates.zero_shot_realistic import ZeroShotRealisticTemplate
from src.prompt_templates.base_template import PromptStrategy, ProblemStatementType, PromptContext
from scripts.measure_instance import SWEPerfMeasurer
from src.measurement.collector import MetricsCollector
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ZS_Realistic")

class ZeroShotRealisticRunner:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset()
        self.client_manager = ClientManager() 
        self.template = ZeroShotRealisticTemplate()
        self.measurer_tool = SWEPerfMeasurer(dataset_path, country_code="ESP")

    def _load_dataset(self):
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else data["instances"]

    def _get_instance(self, instance_id: str):
        for item in self.dataset:
            if item.get("instance_id") == instance_id: return item
        raise ValueError(f"Instance {instance_id} not found")

    # --- REALISTIC RETRIEVAL LOGIC ---
    
    def _generate_repo_map(self, repo_path: Path) -> str:
        """Crea l'albero dei file (Repository Map)."""
        tree = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))]
            level = root.replace(str(repo_path), '').count(os.sep)
            indent = ' ' * 4 * level
            tree.append(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                if f.endswith('.py'):
                    tree.append(f"{subindent}{f}")
        return "\n".join(tree[:100]) + "\n... (truncated)" # Tronca se troppo lungo

    def _tokenize(self, text: str) -> set:
        """Estrae token significativi (nomi funzioni, variabili) dal codice."""
        # Rimuove commenti e stringhe semplici, tiene identificatori
        return set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text))

    def _simulated_retrieval(self, repo_path: Path, test_paths: List[str]) -> Dict[str, str]:
        """
        Simula BM25: Usa i token del Test Code come query per trovare i Top-5 file nella repo.
        """
        logger.info("🕵️ Running Simulated Retrieval (Anchor: Test Code)...")
        
        # 1. Costruisci la Query (Tokens dai Test Files)
        query_tokens = set()
        test_content_map = {}
        
        for tp in test_paths:
            full = repo_path / tp
            if full.exists():
                content = full.read_text(errors='ignore')
                test_content_map[tp] = content
                query_tokens.update(self._tokenize(content))
        
        # Stopwords pythoniche per ridurre falsi positivi
        stopwords = {'def', 'class', 'self', 'import', 'from', 'in', 'if', 'else', 'return', 'assert', 'test', 'none', 'true', 'false', 'and', 'or'}
        query_tokens -= stopwords
        
        # 2. Rank dei file della repo
        scores = {}
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):
                    rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                    if rel_path in test_paths: continue # Non ritrovare se stesso
                    
                    try:
                        f_content = (repo_path / rel_path).read_text(errors='ignore')
                        f_tokens = self._tokenize(f_content)
                        # Score = Intersezione dei token (Semplificazione di BM25)
                        score = len(query_tokens.intersection(f_tokens))
                        if score > 0:
                            scores[rel_path] = score
                    except: pass
        
        # 3. Prendi Top-5
        top_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"🔎 Retrieved Files: {[t[0] for t in top_files]}")
        
        # 4. Costruisci il contesto finale
        context_files = test_content_map.copy() # Anchor (Test)
        for fname, _ in top_files:
            context_files[fname] = (repo_path / fname).read_text(errors='ignore')
            
        return context_files

    # --- PATCH APPLICATION (Robust) ---
    def _extract_patch(self, raw_response: str) -> str:
        # Pulisce <think> e markdown
        clean = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
        if "<<<<<<< SEARCH" in clean:
            return clean[max(0, clean.find("<<<<<<< SEARCH")-200):]
        diff_match = re.search(r'```(?:diff)?(.*?)```', clean, re.DOTALL)
        return diff_match.group(1).strip() if diff_match else clean

    def _fuzzy_apply(self, repo_path: Path, patch_content: str, context_files: List[str]):
        """Super Fuzzy Matcher Logic (Internal implementation shortcut)"""
        # (Qui inseriamo la logica robusta che abbiamo sviluppato prima)
        # Per brevità copio la logica essenziale di sync
        logger.info("🔧 Applying Patch...")
        patch_file = repo_path / "llm_gen.patch"
        with open(patch_file, "w") as f: f.write(patch_content)
        
        # Prova 1: Git standard
        res = subprocess.run(["git", "apply", "--ignore-space-change", "--ignore-whitespace", "llm_gen.patch"], cwd=repo_path, capture_output=True)
        if res.returncode == 0: return True
        
        # Prova 2: Git p0
        res = subprocess.run(["git", "apply", "-p0", "--ignore-space-change", "--ignore-whitespace", "llm_gen.patch"], cwd=repo_path, capture_output=True)
        if res.returncode == 0: return True
        
        logger.warning("Git failed. Would use Python Fuzzy Matcher here (implemented in main runner).")
        return False

    def run(self, instance_id: str):
        logger.info(f"🚀 START ZS_REALISTIC: {instance_id}")
        instance = self._get_instance(instance_id)
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # 1. Setup Repo
            logger.info("📥 Cloning...")
            repo_path = self.measurer_tool.setup_repository(instance, temp_dir, instance['base_commit'])
            
            # 2. Realistic Context Construction
            # A. Test Anchor (dal campo efficiency_test)
            test_list = instance['efficiency_test'] # ['path/to/test.py::Class::func']
            test_files = list(set([t.split("::")[0] for t in test_list]))
            
            # B. Retrieval (Top-5 + Anchor)
            retrieved_context = self._simulated_retrieval(repo_path, test_files)
            
            # C. Repo Map
            repo_map = self._generate_repo_map(repo_path)
            
            # 3. Prompting
            ctx = PromptContext(
                problem_statement_type=ProblemStatementType.REALISTIC,
                problem_description=instance.get('problem_statement_realistic', "Optimize tests."),
                code_files=retrieved_context,
                test_command=f"pytest {' '.join(test_list)}",
                target_functions=list(retrieved_context.keys()),
                repo_map=repo_map
            )
            
            prompt = self.template.generate_prompt(ctx)
            
            # 4. LLM
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
            logger.info("📤 Sending prompt to LLM...")
            response = client.chat.completions.create(
                model="active_model", # Auto-detect o hardcoded
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8192
            )
            llm_output = response.choices[0].message.content
            logger.info(f"📝 Response received ({len(llm_output)} chars).")

            # 5. Patch & Measure
            patch = self._extract_patch(llm_output)
            if self._fuzzy_apply(repo_path, patch, list(retrieved_context.keys())):
                logger.info("⚡ Running Measurement...")
                # ... Qui chiameresti il tuo codice di installazione e misura ...
                # Per ora salviamo l'output
                self._save(instance_id, llm_output, patch, "Success")
            else:
                self._save(instance_id, llm_output, patch, "Failed Patch")

        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            shutil.rmtree(temp_dir)

    def _save(self, instance_id, response, patch, status):
        os.makedirs("results/zs_realistic", exist_ok=True)
        with open(f"results/zs_realistic/{instance_id}.json", "w") as f:
            json.dump({"instance": instance_id, "status": status, "patch": patch, "full_response": response}, f, indent=2)
        logger.info(f"💾 Saved to results/zs_realistic/{instance_id}.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    
    runner = ZeroShotRealisticRunner(args.dataset)
    runner.run(args.instance)