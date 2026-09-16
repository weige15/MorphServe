# Candidate C++ reconstruction provenance

Base: `vendor/author-morphserve/csrc` from `MorphServe/MorphServe@85c4fbf753b6eb47613bdd7ea423c38b48054544`.

Scoped repair: `src/memory_manager.h` and `.cpp` parse Python registration dictionaries into native `TensorInfo` records before persistence. This fixes the diagnosed interpreter-shutdown crash caused by process-static `py::object` values. Copy, synchronization, address arithmetic, K/V capacity, storage ownership, and pybind entrypoints are otherwise unchanged.

Evidence: `doc/debug-report.md` and `experiments/candidate-memory-manager-repair/`.
