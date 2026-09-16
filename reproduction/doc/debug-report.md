# Debug Report

## Symptom

The candidate C++ memory-manager tests execute both assertions successfully, then the Python process exits with SIGSEGV (exit 139). This makes the released extension unsafe for a normal serving-process shutdown even though in-process storage behavior is correct.

## Reproduction Command

Working directory: `/nfs/home/s314511048/MorphServe`
Shell: Bash
Runtime: Python 3.12.3; PyTorch 2.4.0+cu121; CUDA runtime 12.1; extension compiled with CUDA toolkit 12.4
Environment: `reproduction/.venv`, created by `uv`; candidate commit `85c4fbf...`
Relevant environment variables:
```text
CUDA_VISIBLE_DEVICES=0
PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=reproduction/results/tmp/candidate-memory-manager/csrc:reproduction/runtime
```

```bash
CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_candidate_memory_manager_test.sh
```

## Expected Behavior

Both GPU tests pass and the interpreter exits normally with status 0, leaving `metrics.json`, logs, and unchanged vendor manifests.

## Actual Behavior

Both tests report `OK`, but interpreter shutdown segfaults and the command exits 139. The retained metrics show correct in-process address bounds, 12 real reclaimed blocks, zero allocator delta, copy/restore, and data preservation. The crash occurs only at process teardown.

## Error Log

```text
test_in_place_copy_reclaimed_views_and_restore ... ok
test_rejects_tail_smaller_than_one_kv_block ... ok
Ran 2 tests in 0.421s
OK
Segmentation fault (core dumped)
```

`cuda-gdb` identifies the crashing destructor:

```text
Thread 1 received SIGSEGV in PyObject_Free
pybind11::handle::dec_ref
pybind11::object::~object
...
std::unordered_map<... pybind11::object ...>::~unordered_map
swiftllm::MemoryManager::layer_memory_tensor_map_org
exit
```

Full evidence: `experiments/candidate-memory-manager/results/diagnosis/cuda-gdb.stdout.log`.

## Failure Layer Classification

Most likely layer:

* Command problem: no
* Permission problem: no
* Shell/script invocation problem: no
* Environment problem: no
* Dependency problem: no (NumPy warning is unrelated)
* Python/package/import problem: no
* GPU/CUDA problem: no
* Distributed/torchrun problem: no
* Filesystem/path problem: no
* Data/checkpoint/model file problem: no
* Code logic problem: yes
* Configuration problem: no
* Resource problem: no
* Concurrency/race problem: no
* Unknown/insufficient evidence: no

Final classification: C++/Python lifetime bug in static extension metadata.

## Hypotheses

### Hypothesis 1: static `py::object` metadata outlives the Python interpreter

Why it could explain the symptom: `MemoryManager` stores nested `py::object` values in process-static C++ unordered maps. Global C++ destructors run during/after Python finalization and attempt `Py_DECREF` on already-invalid Python objects.

Evidence for: `cuda-gdb` shows `PyObject_Free` called through `pybind11::object::~object` while destroying `MemoryManager::layer_memory_tensor_map_org`. The tests finish first; crash is at `exit`. The source declarations in `memory_manager.h/.cpp` use static containers of `py::object`.

Evidence against: small registration-only probes sometimes exit normally, so shutdown order/exact populated maps affect whether the invalid lifetime manifests. This variability is consistent with, not contrary to, a shutdown-order bug.

How to verify: replace persisted Python objects with native C++ metadata and rerun the exact test; exit must become 0.

### Hypothesis 2: unmanaged K/V `c10::Storage` views free or access the owner incorrectly

Why it could explain the symptom: the extension builds K/V tensors over an externally owned CUDA allocation with custom `DataPtr` objects.

Evidence for: the crashing test creates these views.

Evidence against: acquire-only probes with and without explicit view deletion exit 0; the debugger stack contains no `c10::Storage`/CUDA tensor destructor and points directly to static `py::object` metadata.

How to verify: the native-metadata repair should fix shutdown without changing storage ownership; an acquire-only control already passes.

### Hypothesis 3: Torch/CUDA ABI mismatch

Why it could explain the symptom: the extension uses CUDA 12.4 headers with PyTorch's CUDA 12.1 runtime.

Evidence for: mixed toolkit/runtime versions are observable.

Evidence against: build/import succeed, all GPU operations and assertions complete, import-only/range-only/acquire-only processes exit 0, and the backtrace is Python object destruction rather than a CUDA symbol/kernel failure.

How to verify: not the next action; only revisit if the native-metadata repair still crashes.

## Most Likely Root Cause

The released C++ extension stores Python-owned metadata (`pybind11::object`) in static `MemoryManager` containers. At process exit those containers are destroyed after Python object lifetime is no longer valid, and their destructors call `Py_DECREF`, causing the SIGSEGV. This is a deterministic code-lifetime defect in the exercised full registration/replacement path, not a GPU calculation failure.

## Minimal Fix

In the separate reconstruction (never the immutable vendor tree), parse each registered metadata dictionary immediately into a native `TensorInfo` structure containing `offset`, `std::vector<int64_t> shape`, `c10::ScalarType dtype`, and `size`. Store only native values in static maps. Do not retain `py::object` past the binding call. This preserves all runtime behavior while removing Python finalization from global C++ destructors.

A workaround such as `os._exit(0)` or deliberately leaking the static containers would hide the crash and is not acceptable.

## Verification

```bash
CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_candidate_memory_manager_test.sh
```

Expected verification result:

```text
test_in_place_copy_reclaimed_views_and_restore ... ok
test_rejects_tail_smaller_than_one_kv_block ... ok
Ran 2 tests
OK
process exit code: 0
vendor manifest before/after: OK
```
