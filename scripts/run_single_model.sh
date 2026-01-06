"""
Specific Experiment Runner: ZERO SHOT + REALISTIC SETTING (AWQ High Context).
Logic:
1. Trigger: Failing Test.
2. Context Anchor: The Test Code (Full or slightly truncated).
3. Retrieval: Scan repo and find Top-N files (up to ~40k tokens).
4. Prompting: Uses System Prompt to enforce SWE-bench format on high context models.
5. Robustness: Handles measurement crashes and ensures Patch Output.
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
logger = logging.getLogger("ZS_Realistic_AWQ")

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

    # --- REALISTIC HELPERS ---
    
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
        # Con 60k token possiamo permetterci una mappa dettagliata (300 righe)
        return "\n".join(tree[:300]) + ("\n... (truncated)" if len(tree) > 300 else "")

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text))

    def _simulated_retrieval(self, repo_path: Path, test_paths: List[str]) -> Dict[str, str]:
        """
        Retrieval Tuned for AWQ (Huge Context).
        Budget: ~160k characters (approx 40-50k tokens).
        Strategy: Load Test Code -> Load relevant Source Code until full.
        """
        logger.info("🕵️ Running Simulated Retrieval (Ultra Context Mode)...")
        
        # A. Analizza Test Code
        query_tokens = set()
        test_file_contents = {}
        for tp in test_paths:
            full = repo_path / tp
            if full.exists():
                content = full.read_text(errors='ignore')
                test_file_contents[tp] = content
                query_tokens.update(self._tokenize(content))
        
        stopwords = {'def', 'class', 'self', 'import', 'from', 'in', 'if', 'else', 'return', 'assert', 'test', 'none', 'true', 'false', 'and', 'or', 'pytest', 'request', 'response'}
        query_tokens -= stopwords
        
        # B. Rank Repo Files
        scores = {}
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):
                    rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                    if rel_path in test_paths: continue 
                    try:
                        f_content = (repo_path / rel_path).read_text(errors='ignore')
                        f_tokens = self._tokenize(f_content)
                        # Score semplice: numero di token condivisi
                        score = len(query_tokens.intersection(f_tokens))
                        if score > 0: scores[rel_path] = score
                    except: pass
        
        # C. Costruisci Contesto (Max 160k chars)
        MAX_CONTEXT_CHARS = 160000 
        current_chars = 0
        final_context = {}
        
        # 1. Carica TEST FILES (Priorità 1)
        # Tronchiamo solo se il file è assurdamente enorme (>40k chars)
        TEST_LIMIT = 40000 
        for tp, content in test_file_contents.items():
            if len(content) > TEST_LIMIT:
                final_context[tp] = content[:TEST_LIMIT] + "\n... [TRUNCATED]"
                current_chars += TEST_LIMIT
                logger.warning(f"⚠️ Truncated massive test file {tp} to {TEST_LIMIT} chars")
            else:
                final_context[tp] = content
                current_chars += len(content)
        
        logger.info(f"✅ Added Test Anchors size: {current_chars} chars")

        # 2. Carica RETRIEVED FILES (Priorità 2 - Fill remaining space)
        top_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        retrieved_list = []
        
        for fname, _ in top_files:
            # Limite di sicurezza: non più di 20 file sorgente per non confondere troppo
            if len(retrieved_list) >= 20: break 
            
            if current_chars >= MAX_CONTEXT_CHARS:
                break

            try:
                content = (repo_path / fname).read_text(errors='ignore')
                if current_chars + len(content) < MAX_CONTEXT_CHARS:
                    final_context[fname] = content
                    current_chars += len(content)
                    retrieved_list.append(fname)
                else:
                    # Se il file non ci sta intero, proviamo a troncarlo se è importante
                    remaining = MAX_CONTEXT_CHARS - current_chars
                    if remaining > 2000: # Se c'è spazio per un pezzo decente
                        final_context[fname] = content[:remaining] + "\n... [TRUNCATED TO FIT CONTEXT]"
                        current_chars += remaining
                        retrieved_list.append(fname)
                        logger.info(f"⚠️ Truncated {fname} to fit remaining context.")
                    else:
                        logger.info(f"⚠️ Skipping {fname} (Context Full).")
            except: pass
            
        logger.info(f"🔎 Added Retrieved Source Files ({len(retrieved_list)}): {retrieved_list}")
        return final_context, list(final_context.keys())

    # --- PATCH ENGINE ---
    def _extract_patch_content(self, content: str) -> str:
        # Pulisci tag di pensiero (DeepSeek specific)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        if "<<<<<<< SEARCH" in content:
            return content[max(0, content.find("<<<<<<< SEARCH") - 500):]
            
        code_blocks = re.findall(r'```(?:diff|python)?\s*(.*?)```', content, re.DOTALL)
        if code_blocks:
            for block in code_blocks:
                if "<<<<<<< SEARCH" in block or "diff --git" in block:
                    return block
            return max(code_blocks, key=len).strip()
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
            
            for i in range(1, len(blocks)):
                block = blocks[i]
                if "=======" not in block or ">>>>>>> REPLACE" not in block: continue
                search_part, rest = block.split('=======', 1)
                replace_part, next_context = rest.split('>>>>>>> REPLACE', 1)
                
                target_file = None
                # Cerca indizio nel testo precedente
                lines_check = patch_content[:patch_content.find(block)].strip().splitlines()[-30:]
                
                for line in reversed(lines_check):
                    clean = line.strip().replace('###', '').replace('File:', '').replace('`', '').strip()
                    if clean in candidate_files or (clean.endswith('.py') and '/' in clean):
                        target_file = clean; break
                
                # Se non trova file nel testo, cerca nel contenuto
                if not target_file:
                    search_l = [l.strip() for l in search_part.splitlines() if l.strip()]
                    if search_l:
                        sig = search_l[0]
                        for cand in candidate_files:
                            if (repo_path / cand).exists() and sig in (repo_path / cand).read_text():
                                target_file = cand; break

                if target_file and self._perform_fuzzy_replace(repo_path, target_file, search_part.splitlines(), replace_part.splitlines()):
                    changes_count += 1
            
            if changes_count > 0: return True

        # Fallback diff standard
        if "diff --git" in patch_content:
             patch_file = repo_path / "llm_gen.patch"
             with open(patch_file, 'w') as f: f.write(patch_content)
             res = subprocess.run(["git", "apply", "-p0", "--ignore-space-change", "--ignore-whitespace", "llm_gen.patch"], cwd=repo_path, capture_output=True)
             if res.returncode == 0:
                 logger.info("✅ Git apply success"); return True

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
            test_list = instance['efficiency_test'] 
            test_files = list(set([t.split("::")[0] for t in test_list]))
            
            # Retrieval (High Context)
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
            
            # Generazione Prompt Base
            user_prompt = self.template.generate_prompt(ctx)
            
            # SYSTEM PROMPT: Enforcement Forte del Formato
            system_prompt = (
                "You are an expert Green Software Engineer optimizing Python code for energy efficiency.\n"
                "Your goal is to optimize the provided code to reduce energy consumption.\n\n"
                "### OUTPUT FORMAT (MANDATORY):\n"
                "You MUST use the SWE-bench format for code changes. Do not use standard Markdown code blocks for the solution.\n"
                "Use this exact format:\n\n"
                "path/to/file.py\n"
                "<<<<<<< SEARCH\n"
                "... original code lines ...\n"
                "=======\n"
                "... optimized code lines ...\n"
                ">>>>>>> REPLACE\n\n"
                "Do NOT provide extensive explanations. Focus on generating the code patch."
            )

            client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
            logger.info("📤 Querying LLM...")
            
            # Con 60k di contesto, lasciamo 4k per l'output
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
                # Logghiamo l'inizio della risposta per debugging
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