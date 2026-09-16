#!/usr/bin/env python3
"""Summarize saved experiment process-wall records without calling them GPU kernel time."""

import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
rows=[]
for path in sorted((root/'experiments').glob('**/wall-time.json')):
    payload=json.load(open(path)); key=next((key for key in ('run_wall_seconds','test_wall_seconds') if key in payload),None)
    if key is None: continue
    exit_path=path.with_name('run.exitcode') if path.with_name('run.exitcode').exists() else path.with_name('test.exitcode')
    rows.append({"path":str(path.relative_to(root)),"process_wall_seconds":float(payload[key]),"exit_code":int(exit_path.read_text().strip()) if exit_path.exists() else None,"gpu_snapshot_present":path.with_name('nvidia-before.csv').exists()})
output={"schema_version":1,"definition":"Sum of saved test/run process wall intervals only; includes load/JIT/execution but excludes environment install/build and is not CUDA kernel time or exclusive GPU allocation time.","record_count":len(rows),"process_wall_seconds_total":sum(row['process_wall_seconds'] for row in rows),"process_wall_seconds_with_gpu_snapshots":sum(row['process_wall_seconds'] for row in rows if row['gpu_snapshot_present']),"successful_records":sum(row['exit_code']==0 for row in rows),"failed_records":sum(row['exit_code'] not in (0,None) for row in rows),"unknown_exit_records":sum(row['exit_code'] is None for row in rows),"records":rows}
(root/'results/raw/resource-usage-summary.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n'); print(json.dumps(output,indent=2,sort_keys=True))
