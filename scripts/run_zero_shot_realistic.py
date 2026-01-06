"""
Specific Experiment Runner: ZERO SHOT + REALISTIC SETTING.
Hybrid Engine:
- Brain: Realistic Retrieval (32k Context, System Prompts).
- Hands: "Old" Robust Patching Engine (Fuzzy Match + Git Apply Fallback).
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
logger = logging.getLogger("ZS_Realistic_Hybrid")

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
            if models.data:
                model_name = models.data[0].id
                logger.info(f"✅ Detected vLLM Model: {model_name}")
                return model_name
        except Exception as e:
            logger.warning(f"⚠️ Could not detect model name: {e}")
        return "active_model"

    # --- REALISTIC HELPERS (Context Builder) ---
    
    def _generate_repo_map(self, repo_path: Path) -> str:
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
        return "\n".join(tree[:150]) + ("\n... (truncated)" if len(tree) > 150 else "")

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text))

    def _simulated_retrieval(self, repo_path: Path, test_paths: List[str]) -> Dict[str, str]:
        """Retrieval Logic (32k Optimized)."""
        logger.info("🕵️ Running Simulated Retrieval...")
        
        query_tokens = set()
        test_file_contents = {}
        for tp in test_paths:
            full = repo_path / tp
            if full.exists():
                content = full.read_text(errors='ignore')
                test_file_contents[tp] = content
                query_tokens.update(self._tokenize(content))
        
        stopwords = {'def', 'class', 'self', 'import', 'from', 'in', 'if', 'else', 'return', 'assert', 'test', 'none', 'true', 'false', 'and', 'or', 'pytest'}
        query_tokens -= stopwords
        
        scores = {}
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):
                    rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                    if rel_path in test_paths: continue 
                    try:
                        f_content = (repo_path / rel_path).read_text(errors='ignore')
                        f_tokens = self._tokenize(f_content)
                        score = len(query_tokens.intersection(f_tokens))
                        if score > 0: scores[rel_path] = score
                    except: pass
        
        MAX_CONTEXT_CHARS = 85000 
        current_chars = 0
        final_context = {}
        
        # 1. Test Files
        TEST_LIMIT = 25000 
        for tp, content in test_file_contents.items():
            if len(content) > TEST_LIMIT:
                final_context[tp] = content[:TEST_LIMIT] + "\n... [TRUNCATED]"
                current_chars += TEST_LIMIT
            else:
                final_context[tp] = content
                current_chars += len(content)
        
        # 2. Retrieved Files
        top_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        retrieved_list = []
        for fname, _ in top_files:
            if len(retrieved_list) >= 15: break
            if current_chars >= MAX_CONTEXT_CHARS: break
            try:
                content = (repo_path / fname).read_text(errors='ignore')
                if current_chars + len(content) < MAX_CONTEXT_CHARS:
                    final_context[fname] = content
                    current_chars += len(content)
                    retrieved_list.append(fname)
                else:
                    remaining = MAX_CONTEXT_CHARS - current_chars
                    if remaining > 2000:
                        final_context[fname] = content[:remaining] + "\n... [TRUNCATED]"
                        current_chars += remaining
                        retrieved_list.append(fname)
            except: pass
            
        logger.info(f"🔎 Added Retrieved Source Files ({len(retrieved_list)}): {retrieved_list}")
        return final_context, list(final_context.keys())

    # --- ROBUST PATCH ENGINE (PORTED FROM OLD RUNNER) ---
    
    def _extract_patch_content(self, content: str) -> str:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        idx_search = content.find("<<<<<<< SEARCH")
        if idx_search != -1:
            return content[max(0, idx_search - 500):]
        idx_diff = content.find("diff --git")
        if idx_diff != -1:
            return content[idx_diff:]
        code_blocks = re.findall(r'```(?:diff|python)?\s*(.*?)```', content, re.DOTALL)
        if code_blocks: return max(code_blocks, key=len).strip()
        return content.strip()

    def _find_target_file(self, repo_path: Path, search_block: str, candidate_files: List[str]) -> str:
        """Cerca il file giusto scansionando i candidati per la firma del codice."""
        search_lines = [l.strip() for l in search_block.splitlines() if l.strip()]
        if not search_lines: return None
        signature = search_lines[0] # La prima riga del blocco SEARCH deve esistere
        
        for fname in candidate_files:
            fpath = repo_path / fname
            if fpath.exists():
                # Ignoriamo le virgolette per il match
                file_content = fpath.read_text().replace('"', "'")
                sig_norm = signature.replace('"', "'")
                if sig_norm in file_content: 
                    return fname
        return None

    def _perform_fuzzy_replace(self, repo_path: Path, rel_path: str, search_lines: List[str], replace_lines: List[str]) -> bool:
        target_file = repo_path / rel_path
        if not target_file.exists(): return False
        original_lines = target_file.read_text().splitlines(keepends=True)
        
        # Normalize: strip whitespace + normalize quotes
        norm_search = [l.strip().replace('"', "'") for l in search_lines if l.strip()]
        if not norm_search: return False 

        file_map = []
        for idx, line in enumerate(original_lines):
            stripped = line.strip().replace('"', "'")
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
        
        # --- STRATEGIA 1: Blocchi SEARCH/REPLACE ---
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
                # Cerca nome file nelle righe precedenti
                lines_check = context_text.strip().splitlines()[-20:]
                for line in reversed(lines_check):
                    clean = line.strip().replace('###', '').replace('File:', '').replace('`', '').strip()
                    if clean in candidate_files or (clean.endswith('.py') and '/' in clean):
                        target_file = clean; break
                
                # Se non trovato, usa Auto-Detection (Logic Old Runner)
                if not target_file:
                    target_file = self._find_target_file(repo_path, search_part, candidate_files)
                
                if target_file:
                    if self._perform_fuzzy_replace(repo_path, target_file, search_part.splitlines(), replace_part.splitlines()):
                        changes_count += 1
                else:
                    logger.warning(f"⚠️ Could not identify target file for block {i}")
                
                context_text = next_context
            
            if changes_count > 0: return True

        # --- STRATEGIA 2: Git Apply Fallback (Quella che salvava la situazione) ---
        logger.info("🔧 Trying Git Apply fallback...")
        patch_file = repo_path / "llm_gen.patch"
        with open(patch_file, 'w') as f: f.write(patch_content)
        
        # Proviamo diversi argomenti per git apply
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
            
            # Context
            test_list = instance['efficiency_test'] 
            test_files = list(set([t.split("::")[0] for t in test_list]))
            context_files, candidates = self._simulated_retrieval(repo_path, test_files)
            repo_map = self._generate_repo_map(repo_path)
            
            test_cmd = f"pytest {' '.join(test_list)}"
            desc = instance.get('problem_statement_realistic', f"Optimize failing tests: {test_cmd}")
            
            ctx = PromptContext(
                problem_statement_type=ProblemStatementType.REALISTIC,
                problem_description=desc,
                code_files=context_files,
                test_command=test_cmd,
                target_functions=candidates,
                repo_map=repo_map
            )
            
            user_prompt = self.template.generate_prompt(ctx)
            system_prompt = (
                "You are an expert Green Software Engineer.\n"
                "Optimize the provided code for energy efficiency.\n"
                "Use SWE-bench format (<<<<<<< SEARCH / ======= / >>>>>>> REPLACE).\n"
                "Provide actual code blocks immediately."
            )

            client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
            logger.info("📤 Querying LLM...")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=4096 
            )
            llm_output = response.choices[0].message.content
            logger.info(f"📝 Response received ({len(llm_output)} chars).")

            logger.info("🔧 Applying Patch...")
            if self._apply_patch_logic(repo_path, llm_output, candidates):
                logger.info("📦 Installing & Measuring...")
                python_path, conda_env = self.measurer_tool.install_dependencies(
                    repo_path, instance['repo'], instance['version'], instance['base_commit']
                )
                if python_path:
                    logger.info("⚡ Measuring Energy...")
                    collector = MetricsCollector(instance_id=instance_id, country_code="ESP")
                    results = []
                    try:
                        baseline = collector.measure_baseline(duration=2)
                    except Exception as e:
                        logger.warning(f"⚠️ Baseline failed: {e}")
                        baseline = None

                    for test in instance['efficiency_test']:
                        logger.info(f"   Running test: {test}")
                        cmd = f"cd {repo_path} && {python_path} -m pytest '{repo_path}/{test}' -v"
                        try:
                            res = collector.measure_test_execution(test_command=cmd, repetitions=1)
                            res['test_name'] = test
                            results.append(res)
                            time.sleep(2)
                        except Exception as e:
                            logger.error(f"❌ Measurement failed for {test}: {e}")
                            results.append({'test_name': test, 'error': str(e), 'energy_joules': None})

                    self._save(instance_id, llm_output, "Success", results)
                    if conda_env: self.measurer_tool.cleanup_conda_env(conda_env)
                else:
                    self._save(instance_id, llm_output, "Build Failed", None)
            else:
                logger.error("❌ No patch applied.")
                logger.info(f"Preview Response:\n{llm_output[:1000]}")
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