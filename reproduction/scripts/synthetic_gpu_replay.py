#!/usr/bin/env python3
"""Frozen synthetic scheduled-arrival replay against a real FP16 GPU worker."""

import argparse
import asyncio
import concurrent.futures
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from morphserve.replay import RequestSpec, run_replay, summarize, write_artifacts


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--model',required=True); parser.add_argument('--config',required=True); parser.add_argument('--output-dir',required=True); args=parser.parse_args()
    import swiftllm_c  # noqa: F401
    from swiftllm.engine_config import EngineConfig
    from swiftllm.worker.model import LlamaModel

    config_data=json.load(open(args.config)); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    init_start=time.perf_counter()
    engine_config=EngineConfig(org_model_path=args.model,block_size=config_data['block_size'],num_cpu_blocks=0,max_seqs_in_block_table=16,max_blocks_per_seq=128,max_batch_size=16,max_tokens_in_batch=2048)
    model=LlamaModel(engine_config,args.model); model.load_weights(); model.init_kvcache_and_swap(8)
    init_seconds=time.perf_counter()-init_start

    def gpu_call(fn,*values):
        torch.cuda.set_device(0)
        return fn(*values)

    warm_ids=tokenizer("Warm up candidate kernels",return_tensors='pt').input_ids[0].tolist()
    warm_start=time.perf_counter()
    warm_token=model.forward([warm_ids],[15],[])[0]
    model.forward([[warm_token]],[15],[len(warm_ids)+1])
    model.free_seqs_resources([15])
    torch.cuda.synchronize()
    warm_seconds=time.perf_counter()-warm_start
    manager=model.gpu_block_manager
    if isinstance(manager.num_free_blocks,torch.Tensor): manager.num_free_blocks=int(manager.num_free_blocks.item())

    specs=[RequestSpec(**item) for item in config_data['requests']]
    prompt_ids={spec.request_id:tokenizer(spec.prompt,return_tensors='pt').input_ids[0].tolist() for spec in specs}
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def execute():
        lock=asyncio.Lock(); loop=asyncio.get_running_loop()
        async def submit(spec,emit):
            generated=[]; queue_delays=[]; kv_samples=[]
            for step in range(spec.output_len):
                queued=loop.time()
                async with lock:
                    acquired=loop.time(); queue_delays.append(acquired-queued)
                    if step==0:
                        inputs=[prompt_ids[spec.request_id]]; lengths=[]
                    else:
                        inputs=[[generated[-1]]]; lengths=[len(prompt_ids[spec.request_id])+step]
                    token=(await loop.run_in_executor(pool,gpu_call,model.forward,inputs,[int(spec.request_id.split('-')[-1])],lengths))[0]
                    generated.append(int(token))
                    free=int(manager.num_free_blocks.item()) if isinstance(manager.num_free_blocks,torch.Tensor) else int(manager.num_free_blocks)
                    sample={"precision":"FP16","kv_capacity":int(manager.num_blocks),"kv_occupancy":int(manager.num_blocks)-free,"preemptions":None}
                    kv_samples.append(sample); emit(token,sample)
            async with lock:
                await loop.run_in_executor(pool,gpu_call,model.free_seqs_resources,[int(spec.request_id.split('-')[-1])])
                if isinstance(manager.num_free_blocks,torch.Tensor): manager.num_free_blocks=int(manager.num_free_blocks.item())
            return {"queue_delays_s":queue_delays,"queued_time_s":sum(queue_delays),"precision_changes":[],"kv_samples":kv_samples,"preemptions":None,"scheduler_preemptions_measured":False,"generated_text":tokenizer.decode(generated)}
        return await run_replay(specs,submit,config_data['request_timeout_s'])

    try: records=asyncio.run(execute())
    finally: pool.shutdown(wait=True)
    write_artifacts(records,out/'raw.jsonl',out/'summary.json')
    final_free=int(manager.num_free_blocks.item()) if isinstance(manager.num_free_blocks,torch.Tensor) else int(manager.num_free_blocks)
    metadata={"schema_version":1,"classification":config_data['classification'],"init_seconds":init_seconds,"warmup_seconds":warm_seconds,"warmup_excluded":True,"final_kv_capacity":int(manager.num_blocks),"final_kv_free":final_free,"config":config_data,"summary":summarize(records)}
    (out/'run-metadata.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    submit_times=[r['actual_submit_s'] for r in records]
    required_metadata=all(all(key in r['metadata'] for key in ('queue_delays_s','precision_changes','kv_samples','preemptions','scheduler_preemptions_measured','generated_text')) for r in records)
    gate={"submit_span_under_100ms":max(submit_times)-min(submit_times)<0.1,"all_complete":all(r['error'] is None and r['token_count']==3 for r in records),"all_ids_once":len({r['request_id'] for r in records})==len(records)==3,"required_metadata":required_metadata,"final_kv_fully_free":final_free==manager.num_blocks,"preemptions_explicitly_not_measured":all(r['metadata']['preemptions'] is None and not r['metadata']['scheduler_preemptions_measured'] for r in records),"summary_regenerates":json.load(open(out/'summary.json'))==summarize(records)}
    result={"gate":gate,"passed":all(gate.values()),"records":records,"metadata":metadata}; (out/'metrics.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps({"gate":gate,"summary":metadata['summary'],"submit_times":submit_times},indent=2)); return 0 if result['passed'] else 1

if __name__=='__main__': raise SystemExit(main())
