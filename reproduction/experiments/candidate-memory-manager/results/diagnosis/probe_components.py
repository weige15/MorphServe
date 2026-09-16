import gc, os, sys, torch, swiftllm_c
mode=sys.argv[1]
layer={'quant':4101,'kv':4102,'restore':4103,'full_delete':4104}[mode]
def pinned(n, off=0): return ((torch.arange(n,dtype=torch.int64)+off)%251).to(torch.uint8).pin_memory()
def tmap(name,t): return [{name:{'offset':0,'shape':list(t.shape),'dtype':t.dtype,'size':t.numel()*t.element_size(),'is_param':True}}]
owner=torch.full((4096,),255,dtype=torch.uint8,device='cuda')
quant=pinned(1024,3)
orig=pinned(4096,17)
swiftllm_c.register_layer_memory_org_gpu(layer,owner.data_ptr(),4096)
swiftllm_c.register_layer_memory_org_cpu(layer,orig.data_ptr(),4096)
swiftllm_c.register_layer_memory_quant(layer,quant.data_ptr(),1024)
swiftllm_c.register_layer_memory_tensor_map_org(layer,tmap('orig',orig))
swiftllm_c.register_layer_memory_tensor_map_quant(layer,tmap('quant',quant))
swiftllm_c.register_kv_cache_info(2,1,4,8)
if mode in ('quant','full_delete'):
 r=swiftllm_c.replace_layer_org2quant(layer); torch.cuda.synchronize(); del r
if mode in ('kv','full_delete'):
 k,v=swiftllm_c.acquire_new_kvcache(layer); torch.cuda.synchronize(); del k,v
if mode in ('restore','full_delete'):
 r2=swiftllm_c.replace_layer_quant2org(layer); torch.cuda.synchronize(); del r2
gc.collect(); torch.cuda.synchronize()
del owner, quant, orig
gc.collect(); torch.cuda.synchronize()
print('completed',mode,flush=True)
