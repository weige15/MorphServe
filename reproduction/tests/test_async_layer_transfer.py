import json
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import swiftllm_c

from morphserve.autoawq_adapter import AsyncLayerCopier
from morphserve.real_executor import RealMorphingExecutor
from swiftllm.worker.model import LlamaModel


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class AsyncLayerTransferTests(unittest.TestCase):
    metrics={}
    def test_pinned_copy_waits_for_prior_use_without_blocking_host(self):
        layer=4101; owner=torch.zeros(4*1024*1024,dtype=torch.uint8,device='cuda')
        source=((torch.arange(owner.numel(),dtype=torch.int64)%251).to(torch.uint8).pin_memory())
        swiftllm_c.register_layer_memory_org_gpu(layer,owner.data_ptr(),owner.numel())
        prior=torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(torch.cuda.default_stream()):
            torch.cuda._sleep(400_000_000)
            owner.fill_(17)
            prior.record()
        model=SimpleNamespace(last_forward_event=prior,layer_transfer_events={})
        copier=AsyncLayerCopier(model,swiftllm_c)
        tensor_map=[{"payload":{"offset":0,"shape":[owner.numel()],"dtype":torch.uint8,"size":owner.numel(),"is_param":True}}]

        started=time.perf_counter(); views,ready=copier.enqueue(layer,source,source.numel(),tensor_map); host_ms=(time.perf_counter()-started)*1000
        prior_unfinished_after_enqueue=not prior.query()
        self.assertTrue(prior_unfinished_after_enqueue)
        self.assertFalse(ready.query())
        self.assertEqual(views[0].data_ptr(),owner.data_ptr())
        copier.wait_current(layer); torch.cuda.synchronize()

        device_ms=prior.elapsed_time(ready)
        self.assertTrue(torch.equal(owner.cpu(),source))
        self.assertLess(host_ms,device_ms)
        self.assertNotIn(layer,model.layer_transfer_events)
        self.metrics['copy']={"bytes":source.numel(),"enqueue_host_ms":host_ms,"prior_unfinished_after_enqueue":prior_unfinished_after_enqueue,"copy_device_ms":device_ms,"same_address":views[0].data_ptr()==owner.data_ptr(),"bytes_exact":True}

    def test_model_waits_immediately_before_affected_layer(self):
        layer_id=4103; owner=torch.full((1024,),17,dtype=torch.uint8,device='cuda')
        source=torch.full((1024,),73,dtype=torch.uint8).pin_memory()
        swiftllm_c.register_layer_memory_org_gpu(layer_id,owner.data_ptr(),owner.numel())
        prior=torch.cuda.Event()
        prior_stream=torch.cuda.Stream()
        with torch.cuda.stream(prior_stream):
            torch.cuda._sleep(400_000_000)
            prior.record()

        model=object.__new__(LlamaModel); torch.nn.Module.__init__(model)
        model.last_forward_event=prior; model.layer_transfer_events={}; model.layer_transfer_wait_events={}; model.layer_quant_list=[]
        model.k_cache=model.v_cache=None; model.k_cache_new=model.v_cache_new=[]
        model.pre_layer=SimpleNamespace(forward=lambda _:torch.zeros((1,1),device='cuda'))
        class Layer:
            def __init__(self,layer_id,read=False): self.layer_id=layer_id; self.read=read
            def forward(self,x,*_): return owner[:1].float().reshape(1,1) if self.read else x
        model.transformer_layers=[Layer(0),Layer(layer_id,True)]
        model.post_layer=SimpleNamespace(forward=lambda x,_state,return_logits=False:x)
        copier=AsyncLayerCopier(model,swiftllm_c)
        tensor_map=[{"x":{"offset":0,"shape":[1024],"dtype":torch.uint8,"size":1024,"is_param":True}}]
        copier.enqueue(layer_id,source,source.numel(),tensor_map)
        state=SimpleNamespace(ignore_kvcache=True)

        result=model._forward(torch.tensor([1],device='cuda'),state)
        torch.cuda.synchronize()

        self.assertEqual(result.item(),73)
        self.assertNotIn(layer_id,model.layer_transfer_events)
        self.assertTrue(model.layer_transfer_wait_events[layer_id].query())
        self.assertIsNot(model.last_forward_event,prior)
        self.assertTrue(model.last_forward_event.query())

    def test_async_model_does_not_accumulate_blocking_restore_events(self):
        model=object.__new__(LlamaModel); torch.nn.Module.__init__(model)
        model.last_forward_event=None; model.layer_transfer_events={}; model.layer_transfer_wait_events={}; model.layer_quant_list=[9]; model.async_layer_transfers=True
        model.k_cache=model.v_cache=None; model.k_cache_new=model.v_cache_new=[]
        model.gpu_block_manager=SimpleNamespace(block_table=None)
        model.pre_layer=SimpleNamespace(forward=lambda _:torch.zeros((1,1),device='cuda'))
        model.transformer_layers=[SimpleNamespace(layer_id=9,forward=lambda x,*_:x)]
        model.post_layer=SimpleNamespace(forward=lambda x,_state,return_logits=False:x)
        with patch.object(swiftllm_c,'record_layer_memory_use',side_effect=AssertionError('redundant C++ event')):
            model._forward(torch.tensor([1],device='cuda'),SimpleNamespace(ignore_kvcache=False))
        torch.cuda.synchronize()
        self.assertTrue(model.last_forward_event.query())

    def test_partial_expansion_failure_publishes_rollback_barrier(self):
        class Extension:
            calls=0
            def acquire_new_kvcache(self,_layer):
                self.calls+=1
                if self.calls==2: raise RuntimeError('second expansion failed')
                k=torch.ones((1,1),device='cuda'); v=torch.ones((1,1),device='cuda')
                k.zero_(); v.zero_()
                return k,v
        manager=SimpleNamespace(num_blocks=2,num_free_blocks=2,is_block_free=torch.ones(2,dtype=torch.bool,device='cuda'),num_blocks_org=2)
        model=SimpleNamespace(gpu_block_manager=manager,k_cache_new=[],v_cache_new=[],kv_cache_new_block_size=0,last_forward_event=None,model_config=SimpleNamespace(num_layers=3),layer_quant_list=[],is_layer_quant_list=[False]*3,explicit_kv_regions=False)
        executor=object.__new__(RealMorphingExecutor); executor.model=model; executor.extension=Extension(); executor.copier=SimpleNamespace(wait_current=lambda _layer:False)
        executor.backups={0:{"base":1024,"size":512},1:{"base":2048,"size":512}}; executor.packed={0:{"size":128},1:{"size":128}}
        executor.fail_expand_calls=set(); executor.expand_calls=0; executor.kv_groups=[]; executor.active_layers=[0,1]; executor.log=[]; executor.layer_stride_bytes=8; executor.poisoned=False

        self.assertFalse(executor.expand_kv([0,1]))

        self.assertIsNotNone(model.last_forward_event)
        model.last_forward_event.synchronize()
        self.assertEqual((manager.num_blocks,manager.num_free_blocks),(2,2)); self.assertEqual(model.k_cache_new,[]); self.assertEqual(executor.kv_groups,[])

    def test_rejects_unpinned_and_oversize_sources(self):
        layer=4102; owner=torch.zeros(1024,dtype=torch.uint8,device='cuda')
        swiftllm_c.register_layer_memory_org_gpu(layer,owner.data_ptr(),owner.numel())
        copier=AsyncLayerCopier(SimpleNamespace(last_forward_event=None,layer_transfer_events={}),swiftllm_c)
        tensor_map=[{"x":{"offset":0,"shape":[1024],"dtype":torch.uint8,"size":1024,"is_param":True}}]
        with self.assertRaisesRegex(ValueError,'pinned'):
            copier.enqueue(layer,torch.zeros(1024,dtype=torch.uint8),1024,tensor_map)
        with self.assertRaisesRegex(ValueError,'exceeds'):
            copier.enqueue(layer,torch.zeros(2048,dtype=torch.uint8).pin_memory(),2048,tensor_map)

    @classmethod
    def tearDownClass(cls):
        output=os.environ.get('MORPHSERVE_TEST_OUTPUT')
        if output: Path(output).write_text(json.dumps(cls.metrics,indent=2,sort_keys=True)+'\n')


if __name__=='__main__': unittest.main()
