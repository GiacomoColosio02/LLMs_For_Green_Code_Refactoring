#!/usr/bin/env python3
"""
Analizza lo stato delle misurazioni SWE-Perf.
Identifica: completate, fallite, mancanti.
"""
import json
from pathlib import Path
from collections import defaultdict

def analyze_measurements(
    dataset_path: str = "data/original/swe_perf_original_20251124.json",
    measurements_dir: str = "data/raw/measurements"
):
    print("=" * 70)
    print("📊 ANALISI STATO MISURAZIONI SWE-PERF")
    print("=" * 70)
    
    # 1. Carica tutte le istanze dal dataset
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    all_instances = {inst['instance_id'] for inst in dataset}
    print(f"\n📋 Istanze totali nel dataset: {len(all_instances)}")
    
    # 2. Trova istanze misurate
    measurements_path = Path(measurements_dir)
    
    if not measurements_path.exists():
        print(f"\n❌ Directory {measurements_dir} non trovata!")
        print("   Nessuna misurazione effettuata.")
        return None
    
    measured_complete = []
    measured_partial = []
    measured_invalid = []
    
    for instance_dir in measurements_path.iterdir():
        if not instance_dir.is_dir():
            continue
            
        instance_id = instance_dir.name
        json_file = instance_dir / "measurements.json"
        
        if not json_file.exists():
            continue
        
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            has_base = bool(data.get('base_measurements', {}).get('tests'))
            has_head = bool(data.get('head_measurements', {}).get('tests'))
            
            if has_base and has_head:
                measured_complete.append(instance_id)
            elif has_base or has_head:
                measured_partial.append({
                    'instance_id': instance_id,
                    'has_base': has_base,
                    'has_head': has_head
                })
            else:
                measured_invalid.append(instance_id)
                
        except json.JSONDecodeError:
            measured_invalid.append(instance_id)
    
    # 3. Trova istanze mancanti
    measured_ids = set(measured_complete) | {p['instance_id'] for p in measured_partial} | set(measured_invalid)
    missing = all_instances - measured_ids
    
    # 4. Analizza log se esiste
    log_file = measurements_path / "measurement_log.txt"
    log_errors = defaultdict(list)
    
    if log_file.exists():
        with open(log_file) as f:
            for line in f:
                if "ERROR" in line or "❌" in line:
                    parts = line.strip().split("|")
                    if len(parts) >= 3:
                        instance_id = parts[1].strip()
                        error = parts[2].strip() if len(parts) > 2 else "Unknown"
                        log_errors[error].append(instance_id)
    
    # 5. Report
    print(f"\n{'='*70}")
    print("📈 RISULTATI")
    print(f"{'='*70}")
    
    print(f"\n✅ Completate (base + head): {len(measured_complete)}")
    print(f"⚠️  Parziali (solo base o head): {len(measured_partial)}")
    print(f"❌ JSON invalidi/vuoti: {len(measured_invalid)}")
    print(f"📭 Mai provate: {len(missing)}")
    
    total_ok = len(measured_complete)
    
    print(f"\n📊 Progresso: {total_ok}/140 ({total_ok/140*100:.1f}%)")
    print(f"📊 Da completare: {140 - total_ok}")
    
    if measured_partial:
        print(f"\n⚠️  ISTANZE PARZIALI ({len(measured_partial)}):")
        for p in measured_partial[:10]:
            status = "base OK" if p['has_base'] else "head OK"
            print(f"   {p['instance_id']}: {status}")
        if len(measured_partial) > 10:
            print(f"   ... e altre {len(measured_partial) - 10}")
    
    if log_errors:
        print(f"\n❌ ERRORI PIÙ COMUNI:")
        for error, instances in sorted(log_errors.items(), key=lambda x: -len(x[1]))[:5]:
            print(f"   [{len(instances)}x] {error[:60]}...")
    
    if missing:
        print(f"\n📭 ISTANZE MAI PROVATE ({len(missing)}):")
        for inst in sorted(missing)[:10]:
            print(f"   {inst}")
        if len(missing) > 10:
            print(f"   ... e altre {len(missing) - 10}")
    
    # 6. Salva lista da ri-misurare
    to_remeasure = list(missing) + [p['instance_id'] for p in measured_partial] + measured_invalid
    
    output_file = measurements_path / "instances_to_remeasure.txt"
    with open(output_file, 'w') as f:
        for inst in sorted(to_remeasure):
            f.write(f"{inst}\n")
    
    print(f"\n💾 Lista istanze da ri-misurare salvata in: {output_file}")
    print(f"   Totale: {len(to_remeasure)} istanze")
    
    print(f"\n{'='*70}")
    
    return {
        'complete': measured_complete,
        'partial': measured_partial,
        'invalid': measured_invalid,
        'missing': list(missing)
    }


if __name__ == "__main__":
    analyze_measurements()
