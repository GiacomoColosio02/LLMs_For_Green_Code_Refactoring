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
import shutil
from pathlib import Path
from typing import Dict, List, Any

# --- SETUP PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORT ---
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

    def _detect_running_model(self) -> str:
        try:
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
            models = client.models.list()
            if models.data: return models.data[0].id
        except Exception: pass
        return "active_model"

    # --- REALISTIC HELPERS (Repo Map & Retrieval) ---
    
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
        # Tronca se troppo lungo per risparmiare token
        return "\n".join(tree[:150]) + ("\n... (truncated)" if len(tree) > 150 else "")

    def _tokenize(self, text: str) -> set:
        """Estrae token significativi (nomi funzioni, variabili) dal codice."""
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
        stopwords = {'def', 'class', 'self', 'import', 'from', 'in', 'if', 'else', 'return', 'assert', 'test', 'none', 'true', 'false', 'and', 'or', 'pytest'}
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
        
        # 4. Costruisci il contesto finale (Anchor + Retrieved)
        context_files = test_content_map.copy() 
        for fname, _ in top_files:
            context_files[fname] = (repo_path / fname).read_text(errors='ignore')
            
        return context_files, list(context_files.keys())

    # --- PATCH ENGINE (Robust) ---
    def _extract_patch_content(self, content: str) -> str:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        idx_search = content.find("<<<<<<< SEARCH")
        if idx_search != -1: return content[max(0, idx_search - 500):]
        idx_diff = content.find("diff --git")
        if idx_diff != -1: return content[idx_diff:]
        code_blocks = re.findall(r'```(?:diff|python)?\s*(.*?)```', content, re.DOTALL)
        if code_blocks: return max(code_blocks, key=len).strip()
        return content.strip()

    def _perform_fuzzy_replace(self, repo_path: Path, rel_path: str, search_lines: List[str], replace_lines: List[str]) -> bool:
        target_file = repo_path / rel_path
        if not target_file.exists(): return False
        original_lines = target_file.read_text().splitlines(keepends=True)
        norm_search = [l.strip() for l in search_lines if l.strip()]
        if not norm_search: return False 

        file_map = []
        for idx, line in enumerate(original_lines):
            stripped = line.strip()
            if stripped: file_map.append((stripped, idx))
        
        search_len = len(norm_search)
        match_start_idx = -1
        
        for i in range(len(file_map) - search_len + 1):
            window = [item[0] for item in file_map[i : i + search_len]]
            if window == norm_search:
                match_start_idx = i; break
        
        if match_start_idx != -1:
            real_start = file_map[match_start_idx][1]
            real_end = file_map[match_start_idx + search_len - 1][1]
            final_replace = [l + '\n' if not l.endswith('\n') else l for l in replace_lines]
            new_content = original_lines[:real_start] + final_replace + original_lines[real_end + 1:]
            target_file.write_text("".join(new_content))
            logger.info(f"✅ Applied Patch to {rel_path} (Fuzzy Match)")
            return True
        return False

    def _apply_patch_logic(self, repo_path: Path, raw_content: str, candidate_files: List[str]) -> bool:
        patch_content = self._extract_patch_content(raw_content)
        if "<<<<<<< SEARCH" in patch_content:
            logger.info("🔧 Processing SEARCH/REPLACE blocks...")
            blocks = patch_content.split('<<<<<<< SEARCH')
            changes_count = 0
            context_text = blocks[0]
            for i in range(1, len(blocks)):
                block = blocks[i]
                if "=======" not in block or ">>>>>>> REPLACE" not in block: continue
                search_part, rest = block.split('=======', 1)
                replace_part, next_context = rest.split('>>>>>>> REPLACE', 1)
                
                target_file = None
                lines_check = context_text.strip().splitlines()[-15:]
                for line in reversed(lines_check):
                    clean = line.strip().replace('###', '').replace('File:', '').replace('`', '').strip()
                    if clean in candidate_files or (clean.endswith('.py') and '/' in clean):
                        target_file = clean; break
                
                if not target_file: # Find in candidates by content signature
                    search_l = [l.strip() for l in search_part.splitlines() if l.strip()]
                    if search_l:
                        sig = search_l[0]
                        for cand in candidate_files:
                            if (repo_path / cand).exists() and sig in (repo_path / cand).read_text():
                                target_file = cand; break

                if target_file and self._perform_fuzzy_replace(repo_path, target_file, search_part.splitlines(), replace_part.splitlines()):
                    changes_count += 1
                context_text = next_context
            if changes_count > 0: return True

        patch_file = repo_path / "llm_gen.patch"
        with open(patch_file, 'w') as f: f.write(patch_content)
        for arg in ["-p0", "--ignore-space-change", "--ignore-whitespace"]:
             res = subprocess.run(["git", "apply", arg, "llm_gen.patch"], cwd=repo_path, capture_output=True)
             if res.returncode == 0:
                 logger.info(f"✅ Git apply success ({arg})"); return True
        return False

    # --- MAIN FLOW ---
    def run(self, instance_id: str):
        logger.info(f"🚀 START ZS_REALISTIC: {instance_id}")
        instance = self._get_instance(instance_id)
        model_name = self._detect_running_model()
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            logger.info("📥 Cloning repository...")
            repo_path = self.measurer_tool.setup_repository(instance, temp_dir, instance['base_commit'])
            
            # --- REALISTIC CONTEXT ---
            # 1. Anchor: Test Files
            test_list = instance['efficiency_test'] 
            # Esempio: "tests/test_x.py::func" -> "tests/test_x.py"
            test_files = list(set([t.split("::")[0] for t in test_list]))
            
            # 2. Retrieval: Anchor + Top 5
            context_files, candidates = self._simulated_retrieval(repo_path, test_files)
            
            # 3. Repo Map
            repo_map = self._generate_repo_map(repo_path)
            
            # 4. Prompt
            test_cmd = f"pytest {' '.join(test_list)}"
            # Usa la descrizione Realistica se presente, altrimenti fallback
            desc = instance.get('problem_statement_realistic', f"Optimize failing tests: {test_cmd}")
            
            ctx = PromptContext(
                problem_statement_type=ProblemStatementType.REALISTIC,
                problem_description=desc,
                code_files=context_files,
                test_command=test_cmd,
                target_functions=candidates,
                repo_map=repo_map
            )
            
            prompt = self.template.generate_prompt(ctx)
            
            # 5. LLM
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
            logger.info("📤 Querying LLM (Context includes Noise)...")
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8192
            )
            llm_output = response.choices[0].message.content
            logger.info(f"📝 Response received ({len(llm_output)} chars).")

            # 6. Apply & Measure
            logger.info("🔧 Applying Patch...")
            if self._apply_patch_logic(repo_path, llm_output, candidates):
                logger.info("📦 Installing & Measuring...")
                python_path, conda_env = self.measurer_tool.install_dependencies(
                    repo_path, instance['repo'], instance['version'], instance['base_commit']
                )
                if python_path:
                    collector = MetricsCollector(instance_id=instance_id, country_code="ESP")
                    baseline = collector.measure_baseline(duration=2)
                    results = []
                    for test in instance['efficiency_test']:
                        logger.info(f"   Running test: {test}")
                        cmd = f"cd {repo_path} && {python_path} -m pytest '{repo_path}/{test}' -v"
                        res = collector.measure_test_execution(test_command=cmd, repetitions=1)
                        res['test_name'] = test
                        results.append(res)
                    self._save(instance_id, llm_output, "Success", results)
                    if conda_env: self.measurer_tool.cleanup_conda_env(conda_env)
                else:
                    self._save(instance_id, llm_output, "Build Failed", None)
            else:
                self._save(instance_id, llm_output, "Patch Failed", None)

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _save(self, instance_id, response, status, metrics):
        os.makedirs("results/zs_realistic", exist_ok=True)
        data = {"instance": instance_id, "status": status, "metrics": metrics, "full_response": response}
        with open(f"results/zs_realistic/{instance_id}.json", "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"💾 Saved to results/zs_realistic/{instance_id}.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    
    runner = ZeroShotRealisticRunner(args.dataset)
    runner.run(args.instance)