---
name: cuda
description: CUDA C++ (.cu) kernel reference — TVM-FFI direct-export vs Python-binding entry points, the chevron-launch null-stream pitfall under CUDA-graph capture, the sm_100 fp8→bf16 cvt PTX gap, the load_inline name-cache trap, and the __launch_bounds__ register-spill lever. Also the single-source for generic per-call-overhead / CUDA-graph-capture / PDL theory (Waves-Per-SM decision table) regardless of DSL. Use when writing or debugging hand-written CUDA C++ kernels, or whenever you need generic CUDA-graph / sync-audit / PDL reasoning.
---

# CUDA

Reference for CUDA C++ kernel writers. Detailed guide: `cuda.md`.

## When to consult

- Writing a CUDA `.cu` kernel (TVM-FFI direct export or Python binding).
- Resolving a `RUNTIME_ERROR` traceable to chevron-launch stream binding (legacy null-stream is NOT capture-aware — see `cuda.md` "Kernel-launch stream is NOT optional under CUDA graph capture").
- Reducing per-call overhead: GPU↔CPU sync audit, CUDA-graph capture, or PDL kernel→kernel overlap — generic theory + the Waves-Per-SM decision table live here; per-DSL PDL bindings live in the DSL skills.

## COUPLED references

None directly to runtime. Per-operator CUDA wins / traps live under `docs/prior/` (when your operator has a prior archive).
