#!/usr/bin/env python3
"""Build deterministic modified-condition manifests from inferred trace windows."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path


def select(rows,factor):
    return [rows[math.floor(k*factor)] for k in range(math.ceil(len(rows)/factor)) if math.floor(k*factor)<len(rows)]


def write_jsonl(path,rows):
    data=''.join(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n' for row in rows).encode()
    Path(path).write_bytes(data); return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--windows',required=True); p.add_argument('--azure-code',required=True); p.add_argument('--burst-v1',required=True); p.add_argument('--output-dir',required=True); a=p.parse_args()
    config=json.load(open(a.config)); windows=json.load(open(a.windows)); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)

    with open(a.azure_code,newline='') as f:
        reader=csv.DictReader(f); first=next(reader); base=datetime.fromisoformat(first['TIMESTAMP'])
        start=base+timedelta(seconds=windows['azure']['relative_start_s']); end=start+timedelta(seconds=windows['duration_s']); window=[]
        for index,row in enumerate(reader,start=1):
            timestamp=datetime.fromisoformat(row['TIMESTAMP'])
            if start<=timestamp<end: window.append((index,row,timestamp))
            elif timestamp>=end: break
    azure=[]
    for kept,(index,row,timestamp) in enumerate(select(window,config['azure_factor'])):
        azure.append({"request_id":f"azure-{kept:04d}","source_row_index":index,"scheduled_s":(timestamp-start).total_seconds(),"source_timestamp":row['TIMESTAMP'],"source_context_tokens":int(row['ContextTokens']),"source_generated_tokens":int(row['GeneratedTokens']),"context_mapping":None})

    start_s=windows['burstgpt']['start_timestamp_s']; end_s=start_s+windows['duration_s']; window=[]
    with open(a.burst_v1,newline='') as f:
        for index,row in enumerate(csv.DictReader(f)):
            timestamp=float(row['Timestamp'])
            if start_s<=timestamp<end_s: window.append((index,row,timestamp))
            elif timestamp>=end_s: break
    burst=[]
    for kept,(index,row,timestamp) in enumerate(select(window,config['burst_factor'])):
        burst.append({"request_id":f"burst-{kept:04d}","source_row_index":index,"scheduled_s":timestamp-start_s,"source_timestamp_s":timestamp,"source_model":row['Model'],"source_request_tokens":int(row['Request tokens']),"source_response_tokens":int(row['Response tokens']),"source_log_type":row['Log Type'],"context_mapping":None})

    azure_path=out/'azure-code-systematic-4.75x.jsonl'; burst_path=out/'burstgpt-v1.1-systematic-1.75x.jsonl'
    hashes={azure_path.name:write_jsonl(azure_path,azure),burst_path.name:write_jsonl(burst_path,burst)}
    summary={"schema_version":1,"classification":config['classification'],"operation":config['operation'],"windows":windows,"azure":{"raw_window_requests":windows['azure']['raw_request_count'],"selected_requests":len(azure),"first_s":azure[0]['scheduled_s'],"last_s":azure[-1]['scheduled_s']},"burstgpt":{"raw_window_requests":windows['burstgpt']['raw_request_count'],"selected_requests":len(burst),"first_s":burst[0]['scheduled_s'],"last_s":burst[-1]['scheduled_s']},"sha256":hashes,"context_mapping":None,"limitations":["Systematic thinning is a frozen reconstruction choice, not recovered author behavior.","No task contexts/prompts have been assigned; these files cannot produce quality metrics."]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__': main()
