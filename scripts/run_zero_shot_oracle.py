"""
Specific Experiment Runner: ZERO SHOT + ORACLE SETTING.
Logic:
1. Context: Extracts EXACT target files from the gold patch (Oracle Mode).
2. Prompting: System Prompt enforcement + DeepSeek/Qwen optimization.
3. Patching: Ultra-Robust Hybrid Engine (Matches the Realistic logic exactly).
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

from src.llm_clients.client_manager import ClientManager
from src.prompt_templates.zero_shot_oracle import ZeroShotOracleTemplate
from src.prompt_templates.base_template import PromptStrategy, ProblemStatementType, PromptContext
from scripts.measure_instance import SWEPerfMeasurer
from src.measurement.collector import MetricsCollector
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ZS_Oracle_Hybrid")

class ZeroShotOracleRunner:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset()
        self.client_manager = ClientManager() 
        self.template = ZeroShotOracleTemplate()
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
        except Exception: pass
        return "active_model"

    # --- ULTRA ROBUST PATCH ENGINE (Identical to Realistic) ---
    def _extract_patch_content(self, content: str) -> str:
        # 1. Clean Thinking Tags
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # 2. Prefer Explicit Search Block
        if "<<<<<<< SEARCH" in content:
            return content[max(0, content.find("<<<<<<< SEARCH") - 500):]
            
        # 3. Fallback to Diff or Markdown
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
        
        if "<<<<<<< SEARCH" in patch_content:
            logger.info("🔧 Processing SEARCH/REPLACE blocks with Smart Auto-Correct...")
            blocks = patch_content.split('<<<<<<< SEARCH')
            changes_count = 0
            
            for i in range(1, len(blocks)):
                block = blocks[i]
                if "=======" not in block or ">>>>>>> REPLACE" not in block: continue
                search_part, rest = block.split('=======', 1)
                replace_part, next_context = rest.split('>>>>>>> REPLACE', 1)
                
                # 1. Identify File from Content
                target_file = None
                lines_check = patch_content[:patch_content.find(block)].strip().splitlines()[-30:]
                for line in reversed(lines_check):
                    clean = line.strip().replace('###', '').replace('File:', '').replace('`', '').strip()
                    if clean in candidate_files or (clean.endswith('.py') and '/' in clean):
                        target_file = clean; break
                
                # 2. Attempt 1: Stated File
                success = False
                if target_file:
                    logger.info(f"👉 Attempting patch on stated file: {target_file}")
                    if self._perform_fuzzy_replace(repo_path, target_file, search_part.splitlines(), replace_part.splitlines()):
                        success = True
                        changes_count += 1
                
                # 3. Attempt 2: Smart Scan (Oracle usually has few candidates, so scanning is fast)
                if not success:
                    # Fallback logic: check signature in known files
                    search_l = [l.strip() for l in search_part.splitlines() if l.strip()]
                    if search_l:
                        sig = search_l[0].replace('"', "'")
                        for cand in candidate_files:
                            if cand == target_file: continue
                            path = repo_path / cand
                            if path.exists() and sig in path.read_text().replace('"', "'"):
                                logger.info(f"🎉 Smart Fix! Found matching code in {cand}")
                                if self._perform_fuzzy_replace(repo_path, cand, search_part.splitlines(), replace_part.splitlines()):
                                    changes_count += 1
                                    success = True
                                break

                if not success:
                    logger.error(f"❌ Failed to find matching code for block {i} in any file.")

            if changes_count > 0: return True

        # Fallback to Git Apply
        logger.info("🔧 Trying Git Apply fallback...")
        patch_file = repo_path / "llm_gen.patch"
        with open(patch_file, 'w') as f: f.write(patch_content)
        for arg in ["-p0", "--ignore-space-change", "--ignore-whitespace"]:
             res = subprocess.run(["git", "apply", arg, "llm_gen.patch"], cwd=repo_path, capture_output=True)
             if res.returncode == 0:
                 logger.info(f"✅ Git apply success ({arg})"); return True
        return False

    # --- MAIN FLOW ---
    def run(self, instance_id: str):
        logger.info(f"🚀 START ZS_ORACLE: {instance_id}")
        instance = self._get_instance(instance_id)
        model_name = self._detect_running_model()
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            logger.info("📥 Cloning repository...")
            repo_path = self.measurer_tool.setup_repository(instance, temp_dir, instance['base_commit'])
            
            # ORACLE CONTEXT: Extract ONLY target files from patch (Gold Standard)
            files_dict = {}
            for line in instance.get("patch", "").splitlines():
                if line.startswith("--- a/"): 
                    fname = line[6:].strip()
                    p = repo_path / fname
                    if p.exists(): files_dict[fname] = p.read_text()
            candidates = list(files_dict.keys())
            
            test_cmd = f"pytest {' '.join(instance['efficiency_test'])}"
            desc = f"Optimize the energy efficiency of the repository. Focus on these tests: {test_cmd}"
            
            ctx = PromptContext(
                problem_statement_type=ProblemStatementType.ORACLE,
                problem_description=desc,
                code_files=files_dict, 
                test_command=test_cmd,
                target_functions=candidates
            )
            
            user_prompt = self.template.generate_prompt(ctx)
            
            system_prompt = (
                "You are an expert Green Software Engineer.\n"
                "Your goal is to optimize the provided code for energy efficiency.\n"
                "CRITICAL: You MUST output the code changes using the SWE-bench format:\n"
                "<<<<<<< SEARCH\n"
                "... original code ...\n"
                "=======\n"
                "... optimized code ...\n"
                ">>>>>>> REPLACE\n\n"
                "Do NOT provide only explanations. Provide the actual code block immediately."
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
        os.makedirs("results/zs_oracle", exist_ok=True)
        data = {"instance": instance_id, "status": status, "metrics": metrics, "full_response": response}
        with open(f"results/zs_oracle/{instance_id}.json", "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"💾 Saved to results/zs_oracle/{instance_id}.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    
    runner = ZeroShotOracleRunner(args.dataset)
    runner.run(args.instance)