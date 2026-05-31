---
name: cute-dsl
description: CuTe DSL reference — NVIDIA's CUTLASS Python DSL. Use whenever writing or debugging a CuTe DSL kernel — covers the `@cute.kernel` + `@cute.jit` + `.launch()` host pattern, `from_dlpack` tensor conversion, API-probing against the installed wheel, the launch-kwarg table, and the well-known graph-capture pitfall (CuTe's TVM-FFI stream binding does NOT pick up CUDA-graph capture mode — flashinfer-bench issue 414).
---

# CuTe DSL

Reference for CuTe DSL kernel writers. Detailed guide: `cute-dsl.md`.

## When to consult

- Writing a kernel via `@cute.kernel.launch()` decorators.
- Selecting layout / tile algebra primitives for matmul or reduction.
- Resolving silent kernel-skip under CUDA graph capture (the second-order gotcha — TVM-FFI stream binding does not capture; see `cute-dsl.md` "Second-order gotcha").

## COUPLED references

None directly to runtime. Per-operator CuTe wins / traps live under `docs/prior/` (when your operator has a prior archive) — notably the `dsa-sparse-attention` archive for graph-capture interactions.
