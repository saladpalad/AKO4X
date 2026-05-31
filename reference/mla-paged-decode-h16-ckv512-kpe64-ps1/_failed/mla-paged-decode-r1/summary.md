# Failure summary — mla-paged-decode-r1

- **exit_kind**: timeout
- **last_action**: iter-17 A/B vs iter-12 (cap=32 splits for b>=8) — bench launched in background, killed by 10800s timeout before result; kernel left in iter-17 modified state (uncommitted in child)
- **last_stderr_tail**: [benign] no stdin data received in 3s (subprocess.run input plumbing)
- **cited_skills**: bench, flashinfer-bench, profiler-ncu (triton kernel; NCU ran on iter-3 split kernel to confirm 254 regs/thread + 34048 spill reqs)
- **top_frame**: n/a (timeout — not a crash). Sub had committed through iter-12 (Triton split-KV flash-decode with bf16 partial output, AB+10.68% over iter-3, effective ~1.11x); iter-13–16 were dead ends (reverted); iter-17 (cap=32 splits for b≥8) was mid-A/B when killed. Best-confirmed kernel = iter-12 at HEAD (commit 195eda4 in child).
