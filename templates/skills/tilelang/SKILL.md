---
name: tilelang
description: TileLang DSL reference — the `@tilelang.jit` factory + `@T.prim_func` pattern, and the TileLang PDL binding (`T.pdl_trigger()` / `T.pdl_sync()`, with JIT auto-setting the launch attribute so no host-side flag is needed, unlike Triton). Use when writing or tuning a TileLang kernel, or wiring TileLang↔Triton PDL overlap.
---

# TileLang

Reference for TileLang kernel writers. Detailed guide: `tilelang.md`.

## When to consult

- Writing a TileLang kernel with `T.pdl_trigger()` / `T.pdl_sync()` for kernel→kernel overlap.
- Verifying that JIT correctly sets the host launch attribute (no manual host-side flag needed in TileLang).
- Picking pipeline depth / tile shapes for fused operators.

## COUPLED references

None directly to runtime. Per-operator TileLang results / regressions live under `docs/prior/` (when your operator has a prior archive) — note variant `v12` in the `dsa-sparse-attention` archive for the PDL waves-per-SM threshold case.
