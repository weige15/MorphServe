# Candidate C++ reconstruction provenance

Base: `vendor/author-morphserve/csrc` from `MorphServe/MorphServe@85c4fbf753b6eb47613bdd7ea423c38b48054544`.

Scoped repairs:

1. `src/memory_manager.h` and `.cpp` parse Python registration dictionaries into native `TensorInfo` records before persistence. This fixes the diagnosed interpreter-shutdown crash caused by process-static `py::object` values.
2. `record_layer_memory_use(layer_id)` records an event on the current CUDA stream. FP16 restore waits for all recorded events for that reclaimed region before copying, then destroys them. Call sites must explicitly record every stream that uses a region.

Copy addresses, K/V capacity arithmetic, storage ownership, and the candidate's synchronous copy completion are otherwise unchanged.

Evidence: `doc/debug-report.md`, `experiments/candidate-memory-manager-repair/`, and `experiments/candidate-lifetime-barrier/`.
