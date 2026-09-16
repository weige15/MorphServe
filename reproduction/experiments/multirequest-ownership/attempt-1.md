# Attempt 1 — executor recovery outside inference mode

The real block allocations matched `[0,1,2,3,4]` and `[5]`, and refusal preserved rows/counts/sentinel/FCFS. Recovery could not progress because executor shrink mutated tensors created under `torch.inference_mode` from a normal context; the first recovery raised PyTorch's inference-tensor in-place error. A secondary console JSON print also lacked `default=str`, but `metrics.json` was saved.

Changed retry: decorate real executor action methods with `torch.inference_mode` and make the diagnostic print serializer tolerant. No policy, ownership, or gate changes.
