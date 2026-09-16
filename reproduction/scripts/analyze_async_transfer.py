#!/usr/bin/env python3
"""Summarize isolated transfer records without accepting an invalid overlap run."""

import argparse
import json
import statistics
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); a=p.parse_args()
data=json.load(open(a.input)); summary={"schema_version":1,"source_run_passed":bool(data['passed']),"classification":"isolated transfer subset; no overlap or paper-agreement claim","precisions":{}}
for precision in ('W4','FP16'):
    rows=[row for row in data['rows'] if row['precision']==precision]; copies=[row['copy_ms'] for row in rows]; hosts=[row['enqueue_host_ms'] for row in rows]; bandwidth=[row['transfer_bytes']/1e9/(row['copy_ms']/1000) for row in rows]
    summary['precisions'][precision]={"repeats":len(rows),"bytes":rows[0]['transfer_bytes'],"copy_ms":{"min":min(copies),"median":statistics.median(copies),"max":max(copies)},"effective_gbps":{"min":min(bandwidth),"median":statistics.median(bandwidth),"max":max(bandwidth)},"enqueue_host_ms":{"min":min(hosts),"median":statistics.median(hosts),"max":max(hosts)},"incomplete_at_host_return":sum(not row['ready_at_enqueue_return'] for row in rows)}
Path(a.output).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))
