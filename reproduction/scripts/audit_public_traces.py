#!/usr/bin/env python3
"""Inventory primary public trace files and deterministic 72-second window statistics."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def max_window(times, width=72.0):
    left=0; best=(0,None,None)
    for right,value in enumerate(times):
        while value-times[left] > width: left+=1
        count=right-left+1
        candidate=(count,times[left],value)
        if count>best[0]: best=candidate
    return {"count":best[0],"start":best[1],"last_arrival":best[2],"width_s":width}


def inspect(path, kind):
    times=[]; prompt=[]; output=[]
    with open(path,newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames
        base=None
        for row in reader:
            if kind=='azure':
                t=datetime.fromisoformat(row['TIMESTAMP']).timestamp()
                if base is None: base=t
                times.append(t-base); prompt.append(int(row['ContextTokens'])); output.append(int(row['GeneratedTokens']))
            else:
                times.append(float(row['Timestamp'])); prompt.append(int(row['Request tokens'])); output.append(int(row['Response tokens']))
    return {
      "path":str(Path(path).resolve()),"bytes":Path(path).stat().st_size,"sha256":sha256(path),"fields":fields,
      "requests":len(times),"start":times[0],"end":times[-1],"duration_s":times[-1]-times[0],
      "first_72s":{"count":sum(t<=times[0]+72 for t in times),"start":times[0]},"max_count_72s":max_window(times),
      "prompt_tokens":{"min":min(prompt),"max":max(prompt)},"output_tokens":{"min":min(output),"max":max(output)},
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument('--azure-code',required=True); p.add_argument('--azure-conv',required=True); p.add_argument('--burst-v1',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    payload={
      "schema_version":1,
      "sources":{
        "azure_repo_commit":"207bed67dd10090b28ad4f745b2cfd41a11aace4",
        "azure_data_introducing_commit":"790921015d50dd6aae7f7e47f39ba0e235ad6b08",
        "burst_release":"v1.1 (2024-06-13; available before MorphServe paper)",
      },
      "files":{
        "azure_code":inspect(a.azure_code,'azure'),"azure_conversation":inspect(a.azure_conv,'azure'),"burstgpt_1_v1.1":inspect(a.burst_v1,'burst')
      },
      "unresolved":{
        "azure_file_choice":"paper says Azure 2023 but not Code vs Conversation",
        "window_offsets":"paper does not identify any 72-second start",
        "downscaling_operation":"4.75x/1.75x do not specify timestamp multiply/divide, thinning, or sampling",
        "context_mapping":"paper does not publish sampled dataset IDs/order/seed",
      },
      "policy":"Max-count and first-window statistics are source characterization only; neither window is selected for a paper-result comparison.",
    }
    Path(a.output).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=='__main__': main()
