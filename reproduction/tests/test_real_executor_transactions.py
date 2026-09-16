import unittest
from types import SimpleNamespace

import torch

from morphserve.real_executor import RealMorphingExecutor


class FakeCopier:
    def __init__(self,fail_source=None,fail_sources=()): self.fail_sources=set(fail_sources); self.fail_sources.add(fail_source) if fail_source is not None else None; self.calls=[]; self.waits=[]
    def enqueue(self,layer,source,size,tensor_map):
        self.calls.append((layer,source))
        if source in self.fail_sources: raise RuntimeError('injected copy failure')
        return [source],object()
    def wait_current(self,layer): self.waits.append(layer); return False


def bare_executor():
    executor=object.__new__(RealMorphingExecutor)
    executor.model=SimpleNamespace(model_config=SimpleNamespace(num_layers=3),transformer_layers=['f0','f1','f2'],layer_quant_list=[],is_layer_quant_list=[False]*3,explicit_kv_regions=False)
    executor.extension=SimpleNamespace(); executor.backups={0:{'buffer':'f0b','size':8,'tensor_map':[]},1:{'buffer':'f1b','size':8,'tensor_map':[]}}
    executor.packed={0:{'buffer':'q0b','size':4,'tensor_map':[]},1:{'buffer':'q1b','size':4,'tensor_map':[]}}
    executor.fp16_objects={0:'f0',1:'f1'}; executor.prepared_quant={0:('q0',[],[]),1:('q1',[],[])}
    executor.active_layers=[]; executor.quant_objects={}; executor.kv_groups=[]; executor.log=[]; executor.layer_stride_bytes=8; executor.poisoned=False
    executor.fail_expand_calls=set(); executor.expand_calls=0
    return executor


class RealExecutorTransactionTests(unittest.TestCase):
    def test_morph_failure_rolls_back_only_committed_layers(self):
        executor=bare_executor(); executor.copier=FakeCopier(fail_source='q1b')

        self.assertFalse(executor.morph_to_w4([0,1]))

        self.assertEqual(executor.active_layers,[])
        self.assertEqual(executor.quant_objects,{})
        self.assertEqual(executor.model.transformer_layers[:2],['f0','f1'])
        self.assertEqual(executor.copier.calls,[(0,'q0b'),(1,'q1b'),(0,'f0b')])

    def test_multi_layer_shrink_validates_all_occupancy_before_mutation(self):
        executor=bare_executor(); manager=SimpleNamespace(num_blocks_org=4,num_blocks=8,num_free_blocks=7,is_block_free=torch.tensor([True,True,True,True,False,True,True,True]))
        executor.model.gpu_block_manager=manager; executor.model.k_cache_new=['k25','k24']; executor.model.v_cache_new=['v25','v24']; executor.model.kv_cache_new_block_size=2
        executor.kv_groups=[{'layer':25,'k':SimpleNamespace(data_ptr=lambda:200),'v':None},{'layer':24,'k':SimpleNamespace(data_ptr=lambda:100),'v':None}]
        before=(list(executor.kv_groups),list(executor.model.k_cache_new),manager.num_blocks,manager.num_free_blocks,manager.is_block_free.clone())

        self.assertFalse(executor.shrink_kv_before_restore([24,25]))

        self.assertEqual(executor.kv_groups,before[0]); self.assertEqual(executor.model.k_cache_new,before[1])
        self.assertEqual((manager.num_blocks,manager.num_free_blocks),before[2:4]); self.assertTrue(torch.equal(manager.is_block_free,before[4]))

    def test_restore_copy_failure_rolls_back_to_original_w4_batch(self):
        executor=bare_executor(); executor.copier=FakeCopier(fail_source='f1b'); executor.active_layers=[0,1]
        executor.quant_objects={0:executor.prepared_quant[0],1:executor.prepared_quant[1]}; executor.model.transformer_layers[:2]=['q0','q1']

        self.assertFalse(executor.restore_fp16([0,1]))

        self.assertEqual(executor.active_layers,[0,1]); self.assertEqual(executor.model.transformer_layers[:2],['q0','q1'])
        self.assertEqual(set(executor.quant_objects),{0,1})
        self.assertEqual(executor.copier.calls,[(0,'f0b'),(1,'f1b'),(0,'q0b')])

    def test_recovery_restores_exact_kv_metadata_if_fp16_restore_fails(self):
        executor=bare_executor(); executor.copier=FakeCopier(fail_source='f0b'); executor.active_layers=[0,1]
        executor.quant_objects={0:executor.prepared_quant[0],1:executor.prepared_quant[1]}; executor.model.transformer_layers[:2]=['q0','q1']
        manager=SimpleNamespace(num_blocks_org=4,num_blocks=8,num_free_blocks=8,is_block_free=torch.ones(8,dtype=torch.bool))
        executor.model.gpu_block_manager=manager; executor.model.k_cache_new=['k0','k1']; executor.model.v_cache_new=['v0','v1']; executor.model.kv_cache_new_block_size=2
        executor.kv_groups=[{'layer':0,'k':SimpleNamespace(data_ptr=lambda:200),'v':'v0'},{'layer':1,'k':SimpleNamespace(data_ptr=lambda:100),'v':'v1'}]
        before=(list(executor.kv_groups),list(executor.model.k_cache_new),list(executor.model.v_cache_new),manager.num_blocks,manager.num_free_blocks,manager.is_block_free.clone(),executor.model.kv_cache_new_block_size)

        self.assertFalse(executor.recover_fp16([1,0]))

        self.assertEqual(executor.kv_groups,before[0]); self.assertEqual(executor.model.k_cache_new,before[1]); self.assertEqual(executor.model.v_cache_new,before[2])
        self.assertEqual((manager.num_blocks,manager.num_free_blocks),(before[3],before[4])); self.assertTrue(torch.equal(manager.is_block_free,before[5])); self.assertEqual(executor.model.kv_cache_new_block_size,before[6])
        self.assertEqual(executor.active_layers,[0,1]); self.assertFalse(executor.poisoned); self.assertEqual(executor.copier.waits,[1,0])

    def test_double_copy_failure_poisoned_executor_blocks_further_actions(self):
        executor=bare_executor(); executor.copier=FakeCopier(fail_sources={'q1b','f0b'})

        self.assertFalse(executor.morph_to_w4([0,1]))

        self.assertTrue(executor.poisoned); self.assertEqual(executor.active_layers,[0]); self.assertEqual(executor.model.transformer_layers[:2],['q0','f1'])
        calls=list(executor.copier.calls); self.assertFalse(executor.restore_fp16([0])); self.assertEqual(executor.copier.calls,calls)

    def test_poisoned_recovery_does_not_reattach_invalid_kv_snapshot(self):
        executor=bare_executor(); executor.copier=FakeCopier(fail_sources={'f0b','q1b'}); executor.active_layers=[0,1]
        executor.quant_objects={0:executor.prepared_quant[0],1:executor.prepared_quant[1]}; executor.model.transformer_layers[:2]=['q0','q1']
        manager=SimpleNamespace(num_blocks_org=4,num_blocks=8,num_free_blocks=8,is_block_free=torch.ones(8,dtype=torch.bool))
        executor.model.gpu_block_manager=manager; executor.model.k_cache_new=['k0','k1']; executor.model.v_cache_new=['v0','v1']; executor.model.kv_cache_new_block_size=2
        executor.kv_groups=[{'layer':0,'k':SimpleNamespace(data_ptr=lambda:200),'v':'v0'},{'layer':1,'k':SimpleNamespace(data_ptr=lambda:100),'v':'v1'}]

        self.assertFalse(executor.recover_fp16([1,0]))

        self.assertTrue(executor.poisoned); self.assertEqual(executor.active_layers,[0]); self.assertEqual(executor.kv_groups,[])
        self.assertEqual(executor.model.k_cache_new,[]); self.assertEqual(executor.model.v_cache_new,[]); self.assertEqual((manager.num_blocks,manager.num_free_blocks),(4,4))
        self.assertIn(("recovery_snapshot_not_reattached", "executor state is uncertain"),executor.log)

    def test_restore_rollback_copy_failure_preserves_truthful_partial_state(self):
        executor=bare_executor(); executor.copier=FakeCopier(fail_sources={'f1b','q0b'}); executor.active_layers=[0,1]
        executor.quant_objects={0:executor.prepared_quant[0],1:executor.prepared_quant[1]}; executor.model.transformer_layers[:2]=['q0','q1']

        self.assertFalse(executor.restore_fp16([0,1]))

        self.assertTrue(executor.poisoned); self.assertEqual(executor.active_layers,[1]); self.assertEqual(executor.model.transformer_layers[:2],['f0','q1']); self.assertEqual(set(executor.quant_objects),{1})

    def test_restore_rejects_invalid_batch_before_first_copy(self):
        executor=bare_executor(); executor.copier=FakeCopier(); executor.active_layers=[0]; executor.quant_objects={0:executor.prepared_quant[0]}; executor.model.transformer_layers[0]='q0'

        self.assertFalse(executor.restore_fp16([0,1]))

        self.assertEqual(executor.active_layers,[0]); self.assertEqual(executor.model.transformer_layers[0],'q0'); self.assertEqual(executor.copier.calls,[])


if __name__=='__main__': unittest.main()
