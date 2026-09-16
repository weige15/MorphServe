#!/usr/bin/env python3
"""Measure full-layer copy/decode overlap with same-precision-history requests."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from morphserve.autoawq_adapter import backup_fp16_layer, load_packed_layer
from morphserve.real_executor import RealMorphingExecutor


def compare(a,b):
    a=a.float(); b=b.float(); diff=a-b
    return {"relative_l2":float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(a)),"max_abs":float(diff.abs().max()),"top1_a":int(a.argmax()),"top1_b":int(b.argmax())}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--fp16-model',required=True); p.add_argument('--w4-model',required=True); p.add_argument('--output',required=True); p.add_argument('--layer',type=int,default=25); p.add_argument('--repeats',type=int,default=3); a=p.parse_args()
    import swiftllm_c
    from swiftllm.engine_config import EngineConfig
    from swiftllm.worker.model import LlamaModel

    config=EngineConfig(org_model_path=a.fp16_model,block_size=16,num_cpu_blocks=0,max_seqs_in_block_table=8,max_blocks_per_seq=64,max_batch_size=8,max_tokens_in_batch=2048)
    model=LlamaModel(config,a.fp16_model); model.load_weights(); model.init_kvcache_and_swap(8)
    backup=backup_fp16_layer(model,a.layer,swiftllm_c); packed=load_packed_layer(a.w4_model,a.layer,swiftllm_c)
    executor=RealMorphingExecutor(model,swiftllm_c,{a.layer:backup},{a.layer:packed})
    tokenizer=AutoTokenizer.from_pretrained(a.fp16_model,local_files_only=True)
    ids=tokenizer('Life blooms like a flower, far away',return_tensors='pt').input_ids[0].tolist()

    # Compile both precision paths and one real cached decode outside timed regions.
    model.forward([ids],[7],[],ignore_kvcache=True,return_logits=True).cpu()
    assert executor.morph_to_w4([a.layer])
    warm_logits=model.forward([ids],[6],[],return_logits=True)[0]
    warm_token=int(warm_logits.argmax().cpu())
    model.forward([[warm_token]],[6],[len(ids)+1],return_logits=True).cpu()
    model.free_seqs_resources([6])
    assert executor.restore_fp16([a.layer]); model.forward([ids],[7],[],ignore_kvcache=True,return_logits=True).cpu()

    first0=int(model.forward([ids],[0],[],return_logits=True)[0].argmax().cpu())
    first1=int(model.forward([ids],[1],[],return_logits=True)[0].argmax().cpu())
    assert first0==first1
    forced=first0; decode_index=0; rows=[]

    def timed_phase(precision,input_token,length):
        origin=torch.cuda.Event(enable_timing=True); copy_start=torch.cuda.Event(enable_timing=True)
        decode_start=torch.cuda.Event(enable_timing=True); decode_end=torch.cuda.Event(enable_timing=True)
        origin.record()
        with torch.cuda.stream(executor.copier.stream):
            executor.copier.stream.wait_event(origin); copy_start.record()
        host_start=time.perf_counter()
        changed=(executor.morph_to_w4 if precision=='W4' else executor.restore_fp16)([a.layer])
        if not changed: raise RuntimeError(f'{precision} transition failed')
        host_ms=(time.perf_counter()-host_start)*1000
        ready=model.layer_transfer_events[a.layer]; ready_at_return=ready.query()
        decode_start.record(); async_logits=model.forward([[input_token]],[0],[length],return_logits=True)[0]; decode_end.record()
        pre_wait=model.layer_transfer_wait_events.pop(a.layer)
        decode_end.synchronize(); ready.synchronize(); async_cpu=async_logits.detach().cpu()
        ref_start=torch.cuda.Event(enable_timing=True); ref_end=torch.cuda.Event(enable_timing=True)
        ref_start.record(); ref_logits=model.forward([[input_token]],[1],[length],return_logits=True)[0]; ref_end.record(); ref_end.synchronize(); ref_cpu=ref_logits.detach().cpu()
        copy_begin=origin.elapsed_time(copy_start); copy_end=origin.elapsed_time(ready); decode_begin=origin.elapsed_time(decode_start); decode_finish=origin.elapsed_time(decode_end)
        pre_wait_time=origin.elapsed_time(pre_wait)
        enclosing_overlap=max(0.0,min(copy_end,decode_finish)-max(copy_begin,decode_begin))
        compute_overlap=max(0.0,min(copy_end,pre_wait_time)-max(copy_begin,decode_begin))
        output_token=int(async_cpu.argmax())
        row={"precision":precision,"forced_input_token":input_token,"forced_output_token":output_token,"decode_length":length,"transfer_bytes":packed['size'] if precision=='W4' else backup['size'],"enqueue_host_ms":host_ms,"ready_at_enqueue_return":ready_at_return,"copy_ms":copy_start.elapsed_time(ready),"decode_concurrent_ms":decode_start.elapsed_time(decode_end),"decode_reference_ms":ref_start.elapsed_time(ref_end),"copy_start_from_origin_ms":copy_begin,"copy_end_from_origin_ms":copy_end,"decode_start_from_origin_ms":decode_begin,"pre_layer_wait_from_origin_ms":pre_wait_time,"decode_end_from_origin_ms":decode_finish,"enclosing_interval_overlap_ms":enclosing_overlap,"pre_layer_compute_overlap_ms":compute_overlap,"transfer_remaining_at_layer_ms":max(0.0,copy_end-pre_wait_time),"exposed_decode_delta_ms":decode_start.elapsed_time(decode_end)-ref_start.elapsed_time(ref_end),"comparison":compare(ref_cpu,async_cpu)}
        return row,output_token

    forced_history=[forced]
    for _ in range(a.repeats):
        length=len(ids)+decode_index+1; row,forced=timed_phase('W4',forced,length); rows.append(row); forced_history.append(forced); decode_index+=1
        length=len(ids)+decode_index+1; row,forced=timed_phase('FP16',forced,length); rows.append(row); forced_history.append(forced); decode_index+=1

    model.free_seqs_resources([0,1]); torch.cuda.synchronize()
    region=swiftllm_c.get_layer_memory_org_gpu(a.layer)
    restored_exact=bool(torch.equal(region[:backup['size']].cpu(),backup['buffer']))
    w4=[row for row in rows if row['precision']=='W4']; fp16=[row for row in rows if row['precision']=='FP16']
    gate={"repeat_count":len(w4)==len(fp16)==a.repeats,"host_enqueue_nonblocking":all(not row['ready_at_enqueue_return'] for row in rows),"same_history_w4_within_envelope":all(row['comparison']['relative_l2']<0.005 and row['comparison']['top1_a']==row['comparison']['top1_b'] for row in w4),"same_history_fp16_within_envelope":all(row['comparison']['relative_l2']<0.005 and row['comparison']['top1_a']==row['comparison']['top1_b'] for row in fp16),"pre_layer_compute_overlap_observed":all(row['pre_layer_compute_overlap_ms']>0 for row in rows),"final_fp16_bytes_exact":restored_exact,"final_state_fp16":executor.active_layers==[]}
    payload={"schema_version":1,"classification":"modified-condition asynchronous full-model overlap","layer":a.layer,"repeats":a.repeats,"request_ids":{"async":0,"same_history_reference":1},"prompt_ids":ids,"forced_history":forced_history,"rows":rows,"final_fp16_bytes_exact":restored_exact,"gate":gate,"passed":all(gate.values())}
    Path(a.output).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); print(json.dumps({"gate":gate,"rows":rows},indent=2)); return 0 if payload['passed'] else 1

if __name__=='__main__': raise SystemExit(main())
