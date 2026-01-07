"""
BATCH ORCHESTRATOR & MEASUREMENT ENGINE (ULTIMATE EDITION)
FIXED: Added 'import tempfile' explicitly.
FEATURE: Immediate feedback on patch generation success/failure.
"""

import sys
import os
import json
import logging
import subprocess
import time
import shutil
import numpy as np
import argparse
import tempfile 
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# --- CONFIGURAZIONE PATH ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import moduli interni
from scripts.measure_instance import SWEPerfMeasurer
from src.measurement.collector import MetricsCollector

# --- CONFIGURAZIONE GLOBALE ---
INPUT_DATASET = PROJECT_ROOT / "data/processed/swe_perf_reduced_test.json"
OUTPUT_DATASET = PROJECT_ROOT / "data/processed/Zeroshot_Green_Dataset_Test.json"
LOG_FILE = "batch_full_process.log"
TIMEOUT_GEN = 600  # 10 min timeout per generazione patch

# Metriche da collezionare
METRICS_MAP = {
    'green': [
        'cpu_energy_joules', 'gpu_energy_joules', 'total_energy_joules', 
        'power_watts', 'carbon_grams', 'energy_efficiency'
    ],
    'efficiency': [
        'duration_seconds', 'cpu_usage_mean_percent', 'cpu_usage_peak_percent',
        'ram_usage_mean_mb', 'ram_usage_peak_mb', 'gpu_temperature_mean_celsius'
    ]
}
AGGREGATIONS = ['mean', 'std', 'min', 'max']

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GreenOrchestrator")

# ==============================================================================
#  CLASSE: GreenMeasurementEngine
# ==============================================================================
class GreenMeasurementEngine:
    def __init__(self, original_dataset_path: Path, output_file: Path):
        self.output_file = output_file
        self.measurer = SWEPerfMeasurer(str(original_dataset_path), country_code="ESP")
        
        # Load Original Data per metadati (repo url, commit, tests)
        with open(original_dataset_path, 'r') as f:
            raw = json.load(f)
            items = raw if isinstance(raw, list) else raw.get('instances', [])
            self.original_data = {item['instance_id']: item for item in items}

    def _calculate_aggregates(self, raw_runs: List[Dict]) -> Dict[str, float]:
        if not raw_runs: return {}
        agg_results = {}
        all_keys = METRICS_MAP['green'] + METRICS_MAP['efficiency']
        
        values = {k: [] for k in all_keys}
        for run in raw_runs:
            for k in all_keys:
                val = run.get(k)
                if val is None and k == 'duration_seconds': val = run.get('runtime_seconds')
                if val is None and k == 'total_energy_joules': val = run.get('energy_joules')
                if val is not None: values[k].append(float(val))
        
        for k, vals in values.items():
            if vals:
                agg_results[f"{k}_mean"] = float(np.mean(vals))
                agg_results[f"{k}_std"] = float(np.std(vals))
                agg_results[f"{k}_min"] = float(np.min(vals))
                agg_results[f"{k}_max"] = float(np.max(vals))
        return agg_results

    def measure_single_instance(self, instance_id: str, patch_content: str, strategy: str) -> Optional[Dict]:
        temp_dir = Path(tempfile.mkdtemp(prefix="green_meas_"))
        final_metrics = {} 
        
        try:
            meta = self.original_data.get(instance_id)
            if not meta:
                logger.error(f"❌ [MetaData Missing] ID: {instance_id}")
                return None

            test_list = meta.get('efficiency_test', [])
            if not test_list:
                logger.warning(f"⚠️ [No Tests] ID: {instance_id} ha lista test vuota.")
                return None

            # --- FASE 1: BASELINE ---
            logger.info(f"   🔹 [Base] Cloning & Installing {instance_id}...")
            repo_path = self.measurer.setup_repository(meta, temp_dir, meta['base_commit'])
            python_path, conda_env = self.measurer.install_dependencies(
                repo_path, meta['repo'], meta['version'], meta['base_commit']
            )
            
            if not python_path:
                logger.error(f"❌ [Build Failed] Base environment creation failed.")
                return None

            logger.info(f"   🔹 [Base] Measuring {len(test_list)} tests...")
            collector = MetricsCollector(instance_id=instance_id, country_code="ESP")
            
            valid_base_tests = set()
            for test in test_list:
                cmd = f"cd {repo_path} && {python_path} -m pytest '{repo_path}/{test}' -v"
                runs = []
                try:
                    res = collector.measure_test_execution(test_command=cmd, repetitions=1)
                    if res and res.get('return_code') == 0:
                        runs.append(res)
                        valid_base_tests.add(test)
                except: pass
                
                if runs:
                    if test not in final_metrics: final_metrics[test] = {}
                    final_metrics[test]['base'] = self._calculate_aggregates(runs)

            if conda_env: self.measurer.cleanup_conda_env(conda_env)
            shutil.rmtree(temp_dir)
            
            if not valid_base_tests:
                logger.error("❌ [Base Failed] Nessun test originale è passato.")
                return None

            # --- FASE 2: HEAD ---
            temp_dir = Path(tempfile.mkdtemp(prefix="green_meas_head_"))
            logger.info(f"   🔸 [Head] Setup Patched Environment...")
            repo_path = self.measurer.setup_repository(meta, temp_dir, meta['base_commit'])
            
            patch_file = repo_path / "llm_generated.patch"
            with open(patch_file, 'w') as f: f.write(patch_content)
            
            proc = subprocess.run(
                ["git", "apply", "-p0", "--ignore-space-change", "--ignore-whitespace", "llm_generated.patch"],
                cwd=repo_path, capture_output=True, text=True
            )
            
            if proc.returncode != 0:
                logger.error(f"❌ [Patch Apply Failed] Git stderr: {proc.stderr[:200]}...")
                shutil.rmtree(temp_dir)
                return None

            python_path, conda_env = self.measurer.install_dependencies(
                repo_path, meta['repo'], meta['version'], meta['base_commit']
            )
            
            if not python_path: return None

            logger.info(f"   🔸 [Head] Measuring {len(valid_base_tests)} tests...")
            success_count = 0
            for test in test_list:
                if test not in valid_base_tests: continue
                
                cmd = f"cd {repo_path} && {python_path} -m pytest '{repo_path}/{test}' -v"
                runs = []
                try:
                    res = collector.measure_test_execution(test_command=cmd, repetitions=1)
                    if res and res.get('return_code') == 0:
                        runs.append(res)
                        success_count += 1
                except: pass
                
                if runs:
                    final_metrics[test]['head'] = self._calculate_aggregates(runs)

            if conda_env: self.measurer.cleanup_conda_env(conda_env)
            
            if success_count > 0: return final_metrics
            else: return None

        except Exception as e:
            logger.error(f"💥 Critical Error in measurement loop: {e}", exc_info=True)
            return None
        finally:
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir, ignore_errors=True)

    def update_dataset(self, results_dirs: List[Path], strategies: List[str]):
        dataset = {"metadata": {}, "instances": []}
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r') as f: dataset = json.load(f)
            except: pass
        
        existing_ids = {f"{i['instance_id']}_{i['_green_metadata']['strategy']}" for i in dataset['instances']}
        
        dataset['metadata'] = {
            "name": "ZeroShot Green Dataset Test",
            "last_updated": datetime.now().isoformat(),
            "metrics": METRICS_MAP
        }

        for folder, strat in zip(results_dirs, strategies):
            if not folder.exists(): continue
            files = list(folder.glob("*.json"))
            logger.info(f"📂 Scanning {strat}: Found {len(files)} files.")
            
            for idx, fpath in enumerate(files):
                try:
                    res_data = json.load(open(fpath))
                    inst_id = res_data.get('instance')
                    
                    if res_data.get('status') != "Success": continue

                    unique_key = f"{inst_id}_{strat}"
                    if unique_key in existing_ids:
                        logger.info(f"⏩ [{idx+1}/{len(files)}] Skipping {inst_id} ({strat}) - Already measured")
                        continue

                    logger.info(f"\n⚡ [{idx+1}/{len(files)}] MEASURING {inst_id} ({strat})...")
                    metrics = self.measure_single_instance(inst_id, res_data.get('full_response', ''), strat)
                    
                    if metrics:
                        meta = self.original_data.get(inst_id, {})
                        new_entry = {
                            "instance_id": inst_id,
                            "repo": meta.get('repo'),
                            "version": meta.get('version'),
                            "green_metrics": metrics,
                            "_green_metadata": {
                                "strategy": strat,
                                "model": res_data.get('model', 'unknown'),
                                "valid_tests": len(metrics),
                                "creation_date": datetime.now().isoformat()
                            }
                        }
                        dataset['instances'].append(new_entry)
                        existing_ids.add(unique_key)
                        with open(self.output_file, 'w') as f: json.dump(dataset, f, indent=2)
                        logger.info(f"✅ Saved measurement for {inst_id}")
                    else:
                        logger.warning(f"⚠️ Measurement skipped/failed for {inst_id}")

                except Exception as e:
                    logger.error(f"💥 Failed processing {fpath}: {e}")

# ==============================================================================
#  FUNZIONI ORCHESTRATOR
# ==============================================================================
def load_instances():
    if not INPUT_DATASET.exists(): return []
    with open(INPUT_DATASET, 'r') as f: data = json.load(f)
    return data if isinstance(data, list) else data.get("instances", [])

def is_generated(instance_id, result_dir):
    return os.path.exists(os.path.join(result_dir, f"{instance_id}.json"))

def run_generation(phase_name, script_name, result_dir, instances):
    logger.info(f"\n{'='*60}\n🚀 STARTING GENERATION: {phase_name}\n{'='*60}")
    result_path = PROJECT_ROOT / result_dir
    result_path.mkdir(parents=True, exist_ok=True)
    script_path = PROJECT_ROOT / "scripts" / script_name
    
    for idx, inst in enumerate(instances):
        inst_id = inst['instance_id']
        out_file = result_path / f"{inst_id}.json"
        
        if out_file.exists():
            logger.info(f"⏩ Skipping {inst_id} (Already generated)")
            continue
            
        logger.info(f"▶️  Generating {inst_id} ({idx+1}/{len(instances)})...")
        cmd = ["python", str(script_path), "--instance", inst_id, "--dataset", str(INPUT_DATASET)]
        
        try:
            # Esecuzione script
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_GEN)
            
            # --- FEEDBACK IMMEDIATO (NEW) ---
            if out_file.exists():
                try:
                    with open(out_file, 'r') as f: res = json.load(f)
                    status = res.get("status")
                    if status == "Success":
                        logger.info(f"   ✅ Patch Generated & Applied Successfully")
                    else:
                        logger.warning(f"   ❌ Generation Failed. Status: {status}")
                except Exception:
                    logger.error("   ⚠️ Result file corrupted.")
            else:
                logger.error(f"   ❌ No result file created. (Stderr: {proc.stderr[-200:] if proc.stderr else 'None'})")
            # --------------------------------

        except subprocess.TimeoutExpired:
            logger.error(f"   ⏰ Timeout generating {inst_id}")
        except Exception as e:
            logger.error(f"   💥 Crash generating {inst_id}: {e}")
        
        time.sleep(1)

def main():
    if not os.path.exists(INPUT_DATASET):
        logger.error(f"Dataset not found: {INPUT_DATASET}")
        return

    instances = load_instances()
    logger.info(f"📚 Loaded {len(instances)} target instances.")

    # 1. PHASE 1: GENERATION
    run_generation("ZERO SHOT ORACLE", "run_zero_shot_oracle.py", "results/zs_oracle", instances)
    run_generation("ZERO SHOT REALISTIC", "run_zero_shot_realistic.py", "results/zs_realistic", instances)

    # 2. PHASE 2: MEASUREMENT
    logger.info(f"\n{'='*60}\n📏 STARTING MEASUREMENT & CONSOLIDATION\n{'='*60}")
    engine = GreenMeasurementEngine(INPUT_DATASET, OUTPUT_DATASET)
    engine.update_dataset(
        results_dirs=[PROJECT_ROOT / "results/zs_oracle", PROJECT_ROOT / "results/zs_realistic"],
        strategies=["Oracle", "Realistic"]
    )
    logger.info(f"\n🎉 ALL DONE! Final dataset: {OUTPUT_DATASET}")

if __name__ == "__main__":
    main()