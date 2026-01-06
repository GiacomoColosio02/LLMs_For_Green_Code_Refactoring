"""
BATCH RUNNER ORCHESTRATOR
Executes experiments sequentially for the entire dataset.
Features:
- Resume capability (skips already processed instances).
- Timeout handling (kills hung processes).
- Phase separation (Oracle first, then Realistic).
"""
import json
import os
import subprocess
import time
import logging
from pathlib import Path

# CONFIGURAZIONE
DATASET_PATH = "data/processed/swe_perf_reduced.json"
TIMEOUT_SECONDS = 600  # 10 minuti per istanza max
LOG_FILE = "batch_experiment_run.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BatchRunner")

def load_instances():
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, list) else data["instances"]

def is_processed(instance_id, result_dir):
    """Check if the result file already exists."""
    return os.path.exists(os.path.join(result_dir, f"{instance_id}.json"))

def run_phase(phase_name, script_path, result_dir, instances):
    logger.info(f"\n{'='*50}\n🚀 STARTING PHASE: {phase_name}\n{'='*50}")
    
    os.makedirs(result_dir, exist_ok=True)
    total = len(instances)
    
    for idx, instance in enumerate(instances):
        inst_id = instance['instance_id']
        
        if is_processed(inst_id, result_dir):
            logger.info(f"⏩ Skipping {inst_id} ({idx+1}/{total}) - Already done.")
            continue
            
        logger.info(f"▶️  Running {inst_id} ({idx+1}/{total})...")
        start_time = time.time()
        
        cmd = [
            "python", script_path,
            "--instance", inst_id,
            "--dataset", DATASET_PATH
        ]
        
        try:
            # Eseguiamo come sottoprocesso per isolare la memoria
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=TIMEOUT_SECONDS
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ {inst_id} Finished in {elapsed:.1f}s")
            else:
                logger.error(f"❌ {inst_id} Failed (Exit Code {result.returncode})")
                logger.error(f"   Stderr: {result.stderr[-500:]}") # Log ultimi 500 caratteri di errore
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {inst_id} TIMED OUT after {TIMEOUT_SECONDS}s")
        except Exception as e:
            logger.error(f"💥 {inst_id} CRASHED: {e}")

        # Piccolo respiro tra le istanze per far raffreddare GPU/CPU
        time.sleep(2)

def main():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset not found: {DATASET_PATH}")
        return

    instances = load_instances()
    logger.info(f"Loaded {len(instances)} instances from {DATASET_PATH}")

    # --- PHASE 1: ZERO SHOT ORACLE ---
    run_phase(
        phase_name="ZERO SHOT ORACLE",
        script_path="scripts/run_zero_shot_oracle.py",
        result_dir="results/zs_oracle",
        instances=instances
    )

    # --- PHASE 2: ZERO SHOT REALISTIC ---
    run_phase(
        phase_name="ZERO SHOT REALISTIC",
        script_path="scripts/run_zero_shot_realistic.py",
        result_dir="results/zs_realistic",
        instances=instances
    )

    logger.info("\n🎉 BATCH RUN COMPLETE!")

if __name__ == "__main__":
    main()