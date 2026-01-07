"""
Specific Experiment Runner: ZERO SHOT + ORACLE SETTING.
Logic:
1. Context: Extracts EXACT target files from the gold patch.
2. Limits: Enforces strict 60k char limit to prevent vLLM 32k context errors.
3. Patching: Ultra-Robust Hybrid Engine.
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
logger = logging.getLogger("ZS_Oracle_Safe")

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

    # --- SAFE CONTEXT LOADER ---
    def _load_context_files(self, repo_path: Path, patch_content: str) -> Dict[str, str]:
        """Carica i file dalla patch rispettando il limite di sicurezza."""
        files_dict = {}
        MAX_CONTEXT_CHARS = 60000 # Safety limit for 32k tokens
        current_chars = 0
        
        # Identifica file target
        target_files = []
        for line in patch_content.splitlines():
            if line.startswith("--- a/"):
                fname = line[6:].strip()
                if (repo_path / fname).exists():
                    target_files.append(fname)
        
        target_files = list(set(target_files)) # Deduplicate
        
        for fname in target_files:
            if current_chars >= MAX_CONTEXT_CHARS:
                logger.warning(f"⚠️ Context limit reached. Skipping {fname}")
                continue
                
            try:
                content = (repo_path / fname).read_text(errors='ignore')
                # Se il singolo file è enorme, troncalo
                if len(content) > 30000 and len(target_files) > 1:
                    content = content[:30000] + "\n... [TRUNCATED FOR SAFETY] ..."
                
                if current_chars + len(content) < MAX_CONTEXT_CHARS:
                    files_dict[fname] = content
                    current_chars += len(content)
                else:
                    remaining = MAX_CONTEXT_CHARS - current_chars
                    if remaining > 1000:
                        files_dict[fname] = content[:remaining] + "\n... [TRUNCATED]"
                        current_chars += remaining
            except Exception as e:
                logger.warning(f"Failed to read {fname}: {e}")
                
        return files_dict

    # --- ULTRA ROBUST PATCH ENGINE (Same as Realistic) ---
    def _extract_patch_content(self, content: str) -> str:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        if "<<<<<<< SEARCH" in content:
            return content[max(0, content.find("<<<<<<< SEARCH") - 500):]
        code_blocks = re.findall(r'```(?:diff|python)?\s*(.*?)```', content, re.DOTALL)
        if code_blocks:
            return max(code_blocks, key=len).strip()
        return content.strip()

    def _perform_fuzzy_replace(self, repo_path: Path, rel_path: str, search_lines: List[str], replace_lines: List[str]) -> bool:
        target_file = repo_path / rel_path
        if not target_file.exists(): return False
        original_lines = target_file.read_text().splitlines(keepends=True)
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
            return True
        return False

    def _apply_patch_logic(self, repo_path: Path, raw_content: str, candidate_files: List[str]) -> bool:
        patch_content = self._extract_patch_content(raw_content)
        if "<<<<<<< SEARCH" in patch_content:
            blocks = patch_content.split('<<<<<<< SEARCH')
            changes_count = 0
            for i in range(1, len(blocks)):
                block = blocks[i]
                if "=======" not in block or ">>>>>>> REPLACE" not in block: continue
                search_part, rest = block.split('=======', 1)
                replace_part, next_context = rest.split('>>>>>>> REPLACE', 1)
                
                target_file = None
                lines_check = patch_content[:patch_content.find(block)].strip().splitlines()[-30:]
                for line in reversed(lines_check):
                    clean = line.strip().replace('###', '').replace('File:', '').replace('`', '').strip()
                    if clean in candidate_files or (clean.endswith('.py') and '/' in clean):
                        target_file = clean; break
                
                if not target_file:
                    search_l = [l.strip() for l in search_part.splitlines() if l.strip()]
                    if search_l:
                        sig = search_l[0].replace('"', "'")
                        for cand in candidate_files:
                            path = repo_path / cand
                            if path.exists() and sig in path.read_text().replace('"', "'"):
                                target_file = cand; break

                if target_file and self._perform_fuzzy_replace(repo_path, target_file, search_part.splitlines(), replace_part.splitlines()):
                    changes_count += 1
            if changes_count > 0: return True

        patch_file = repo_path / "llm_gen.patch"
        with open(patch_file, 'w') as f: f.write(patch_content)
        for arg in ["-p0", "--ignore-space-change", "--ignore-whitespace"]:
             res = subprocess.run(["git", "apply", arg, "llm_gen.patch"], cwd=repo_path, capture_output=True)
             if res.returncode == 0: return True
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
            
            # Use Safe Context Loader
            files_dict = self._load_context_files(repo_path, instance.get("patch", ""))
            candidates = list(files_dict.keys())
            
            if not candidates:
                logger.error("❌ No target files found in patch!")
                return

            test_cmd = f"pytest {' '.join(instance['efficiency_test'])}"
            ctx = PromptContext(
                problem_statement_type=ProblemStatementType.ORACLE,
                problem_description=f"Optimize energy usage. Tests: {test_cmd}",
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

            # Save result immediately (run_batch_all will handle the rest)
            self._save(instance_id, llm_output, "Success", None)

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            self._save(instance_id, "", "Error", None) # Save error state
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