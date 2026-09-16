# Protocol: native C++ tensor metadata repair

Status: pre-registered after root-cause diagnosis and before repair.

## Hypothesis

Replacing only the process-static nested `py::object` metadata values with native C++ `TensorInfo` records will remove the shutdown segfault without changing copy addresses, reconstructed tensor order/shapes/dtypes, reclaimed K/V capacity, or bytes.

## Scoped change

Create `runtime/candidate-csrc` from the immutable candidate C++ source and change only `memory_manager.h/.cpp`:

- retain the Python registration API;
- parse each input dictionary immediately into `name`, `offset`, `shape`, `c10::ScalarType`, and `size`;
- persist `std::vector<TensorInfo>` in static maps;
- reconstruct tensors from native fields;
- do not change copy streams, synchronization, address arithmetic, allocator ownership, K/V shape calculation, or bindings.

## Frozen verification

Run the identical command and inputs from the failed experiment:

```bash
CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_candidate_memory_manager_test.sh
```

The runner will be pointed to the reconstructed C++ tree but otherwise retain the original tests.

Pass requires:

1. two GPU tests print `OK`;
2. process exit is 0 (not merely assertions passing);
3. address/shape/write/restore metrics match the prior invariants;
4. allocator delta remains zero;
5. candidate vendor manifests remain unchanged;
6. an import-only extension probe still succeeds after Torch preload.

No performance claim is allowed. If exit still crashes, stop and re-diagnose rather than adding teardown workarounds.
