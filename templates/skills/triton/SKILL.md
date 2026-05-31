---
name: triton
description: Triton DSL reference for kernel writers — num_warps/num_stages choice, the small-N fp8 MMA-throughput regression, split-K reduce tile form, deterministic tl.join/permute concat (tl.cat is not order-stable), autotune + .triton_cache pitfalls, and the Triton PDL binding. Use whenever writing or tuning a Triton kernel; first stop before guessing tile shapes or pipeline depth.
---

# Triton

Reference for Triton-specific kernel decisions. Detailed guide: `triton.md`.

## When to consult

- Choosing `num_warps` / `num_stages` for a new kernel.
- Diagnosing `TRITON_PRINT_AUTOTUNING=1` config-specific failures.
- Picking MMA tile shape for the GPU SM target.
- Wiring `gdc_launch_dependents()` / `gdc_wait()` for PDL kernel→kernel overlap.

## COUPLED references

None directly to runtime — Triton compilation happens inside the kernel itself; the runtime SKILLs (bench / profiler / sanitizer) consume the result. Per-operator Triton wins live under `docs/prior/` (when your operator has a prior archive) — grep variant headers there for Triton-specific recipes that already worked on this operator.
