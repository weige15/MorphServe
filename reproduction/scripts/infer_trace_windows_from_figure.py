#!/usr/bin/env python3
"""Rank source-trace windows against approximate Figure 1b shapes."""

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def rank(values, plot, minimum_nonzero, limit=10):
    n=len(plot); centered=plot-plot.mean(); plot_ss=float(centered@centered)
    numerator=np.correlate(values,centered,mode='valid')
    cumulative=np.r_[0,np.cumsum(values,dtype=np.float64)]
    cumulative2=np.r_[0,np.cumsum(values*values,dtype=np.float64)]
    sums=cumulative[n:]-cumulative[:-n]; squares=cumulative2[n:]-cumulative2[:-n]
    denominator=np.sqrt(np.maximum((squares-sums*sums/n)*plot_ss,1e-20))
    correlation=numerator/denominator
    nonzero=np.convolve((values>0).astype(np.int16),np.ones(n,dtype=np.int16),mode='valid')
    correlation[nonzero<minimum_nonzero]=-2
    indices=np.argpartition(correlation,-limit)[-limit:]
    indices=indices[np.argsort(correlation[indices])[::-1]]
    return [{"start_s":int(i),"pearson":float(correlation[i]),"nonzero_bins":int(nonzero[i])} for i in indices]


def azure(path,plot,minimum):
    with open(path,newline='') as f:
        rows=list(csv.DictReader(f))
    base=datetime.fromisoformat(rows[0]['TIMESTAMP'])
    seconds=[int((datetime.fromisoformat(row['TIMESTAMP'])-base).total_seconds()) for row in rows]
    count=np.zeros(max(seconds)+1,dtype=np.float32); tokens=np.zeros_like(count)
    for second,row in zip(seconds,rows):
        count[second]+=1; tokens[second]+=int(row['ContextTokens'])/1000
    rankings={"request_count":rank(count,plot,minimum),"source_context_k_tokens":rank(tokens,plot,minimum)}
    start=rankings['source_context_k_tokens'][0]['start_s']; replay_end=start+72
    return {"base_timestamp":base.isoformat(),"rankings":rankings,"plot_alignment_bins":len(plot),"replay_window_s":72,"inferred_start_s":start,"inferred_start_timestamp":(base+timedelta(seconds=start)).isoformat(),"window_request_count":int(count[start:replay_end].sum()),"window_source_context_tokens":int(round(float(tokens[start:replay_end].sum()*1000)))}


def burst(path,plot,minimum):
    records=[]; maximum=0
    with open(path,newline='') as f:
        for row in csv.DictReader(f):
            second=int(float(row['Timestamp'])); maximum=max(maximum,second); records.append((second,int(row['Request tokens'])))
    count=np.zeros(maximum+1,dtype=np.float32); tokens=np.zeros_like(count)
    for second,prompt in records: count[second]+=1; tokens[second]+=prompt/1000
    rankings={"request_count":rank(count,plot,minimum),"source_request_k_tokens":rank(tokens,plot,minimum)}
    start=rankings['source_request_k_tokens'][0]['start_s']; replay_end=start+72
    return {"rankings":rankings,"plot_alignment_bins":len(plot),"replay_window_s":72,"inferred_start_s":start,"window_request_count":int(count[start:replay_end].sum()),"window_source_request_tokens":int(round(float(tokens[start:replay_end].sum()*1000)))}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--azure-code',required=True); p.add_argument('--burst-v1',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    config=json.load(open(a.config)); az=np.asarray(config['azure']['values'],dtype=np.float32); bu=np.asarray(config['burst']['values'],dtype=np.float32)
    result={"schema_version":1,"classification":config['classification'],"method":config['extraction'],"azure_code":azure(a.azure_code,az,config['azure']['minimum_nonzero_source_bins']),"burstgpt_1_v1.1":burst(a.burst_v1,bu,config['burst']['minimum_nonzero_source_bins']),"limitations":["Plot digitization is approximate and partially occluded by legends.","Correlation identifies arrival-shape candidates, not the unpublished downsampling seed/operation or request-to-context mapping.","Candidates are frozen before serving outcomes but are not author-confirmed exact offsets."]}
    Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__': main()
