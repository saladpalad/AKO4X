Take 30 seconds to zoom out:

1. **Trend** — read back the last few iters in `ITERATIONS.md`. Progress, plateau, regression?
2. **Direction** — given the trend, is the current line of experiments still highest-leverage?
3. **Tools** — `scripts/profile.sh` / `scripts/sanitize.sh` / `scripts/diff.sh` / `gpt_pro_*` / `WebSearch` — any of these relevant now?
4. **Prior variants** (`docs/prior/variants/*/kernel.py` headers) — any constraint or dead-end you may have drifted from?
5. **Logging** — last labeled bench has a Summary row in `ITERATIONS.md`? If you wrote an `Expected:` beforehand, capture the actual delta in `## Notes` now while the bench output is fresh.
6. **Tempo** — any >30s tool call you're about to run, or currently blocking on? Background it (`run_in_background: true` for Bash, `ask_async` → other work → `poll` for `gpt_pro_*`). A blocked agent is a slow agent.

Skip bullets that aren't relevant right now.
