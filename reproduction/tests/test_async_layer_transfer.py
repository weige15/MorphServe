import time
import unittest
from types import SimpleNamespace

import torch
import swiftllm_c

from morphserve.autoawq_adapter import AsyncLayerCopier


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class AsyncLayerTransferTests(unittest.TestCase):
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
        self.assertFalse(ready.query())
        self.assertEqual(views[0].data_ptr(),owner.data_ptr())
        copier.wait_current(layer); torch.cuda.synchronize()

        self.assertTrue(torch.equal(owner.cpu(),source))
        self.assertLess(host_ms,prior.elapsed_time(ready))
        self.assertNotIn(layer,model.layer_transfer_events)

    def test_rejects_unpinned_and_oversize_sources(self):
        layer=4102; owner=torch.zeros(1024,dtype=torch.uint8,device='cuda')
        swiftllm_c.register_layer_memory_org_gpu(layer,owner.data_ptr(),owner.numel())
        copier=AsyncLayerCopier(SimpleNamespace(last_forward_event=None,layer_transfer_events={}),swiftllm_c)
        tensor_map=[{"x":{"offset":0,"shape":[1024],"dtype":torch.uint8,"size":1024,"is_param":True}}]
        with self.assertRaisesRegex(ValueError,'pinned'):
            copier.enqueue(layer,torch.zeros(1024,dtype=torch.uint8),1024,tensor_map)
        with self.assertRaisesRegex(ValueError,'exceeds'):
            copier.enqueue(layer,torch.zeros(2048,dtype=torch.uint8).pin_memory(),2048,tensor_map)


if __name__=='__main__': unittest.main()
