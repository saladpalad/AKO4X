#!/bin/bash
# PostToolUse hook: advisory self-review prompt after every N labeled benches.
# Soft periodic priming of under-used tools / resources. No gate, no blocking.
# Configure via config.toml [advisory] frequency (default 3) / enabled (default true).
#
# SMOKE TEST (verifies the hook's stdout JSON actually reaches the model —
# stderr + exit 0 on PostToolUse is invisible to both assistant AND user):
#   echo '{"tool_input":{"command":"bash scripts/bench.sh --label test"}}' \
#     | $0 | python3 -m json.tool
# Expect valid JSON with `hookSpecificOutput.additionalContext` and
# `hookSpecificOutput.hookEventName == "PostToolUse"`. If output is empty,
# the labeled-trajectory count hasn't hit `frequency` — pad and retry:
#   N=$(ls trajectory/ 2>/dev/null | awk -F_ 'NF>=3' | wc -l)
#   PAD=$(( (3 - N % 3) % 3 ))
#   for i in $(seq 1 $PAD); do mkdir -p trajectory/_probe_a_b$i; done
#   # ... re-run the echo | $0 | json.tool pipe ...
#   for i in $(seq 1 $PAD); do rmdir trajectory/_probe_a_b$i; done
#
# Downstream invariants worth noting (do not break these without re-checking):
#   - `--ab-compare LABEL` does NOT contain `--label` substring — correctly skips.
#   - Failed labeled benches (COMPILE_ERROR, INVALID) do not save trajectory dirs
#     (see scripts/bench_utils.py save_trajectory + compute_score paths),
#     so they don't inflate the count.
#   - Counting uses `d.count('_') >= 2` to distinguish labeled from unlabeled
#     trajectory dirs (unlabeled = `YYYYMMDD_HHMMSS` with 1 underscore,
#     labeled = `YYYYMMDD_HHMMSS_<sanitized_label>` with 2+).
#
# Silent-failure design: NOT using `set -e`. Any subcommand failure falls
# through to defaults. Hook errors must never break bench.sh output.
set -u

INPUT="$(cat)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Extract the Bash command from hook JSON input.
COMMAND=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print((d.get('tool_input', {}) or {}).get('command', ''))
except Exception:
    print('')
" <<< "$INPUT" 2>/dev/null || echo "")

# Only trigger on labeled bench. Smoke tests / --ab-compare / profile / etc. skip.
if [[ "$COMMAND" != *"--label"* ]]; then
    exit 0
fi

# Read [advisory] config — frequency + enabled. Defaults: 3, true.
# Uses minimal line-based parse instead of tomllib (which requires Python
# 3.11+; system python on macOS is often 3.9). Zero deps.
CONFIG=$(python3 -c "
freq, enabled = 3, 'true'
try:
    with open('$PROJECT_ROOT/config.toml', 'r') as f:
        lines = f.readlines()
    in_advisory = False
    for line in lines:
        s = line.strip()
        if s.startswith('[') and s.endswith(']'):
            in_advisory = (s == '[advisory]')
            continue
        if not in_advisory or '=' not in s or s.startswith('#'):
            continue
        k, v = [p.strip() for p in s.split('=', 1)]
        # Strip inline comments
        if '#' in v:
            v = v.split('#', 1)[0].strip()
        if k == 'frequency':
            try: freq = int(v)
            except: pass
        elif k == 'enabled':
            enabled = 'true' if v.lower() == 'true' else 'false'
except Exception:
    pass
print(f'{freq}:{enabled}')
" 2>/dev/null || echo "3:true")

FREQUENCY="${CONFIG%:*}"
ENABLED="${CONFIG#*:}"

if [ "$ENABLED" = "false" ]; then
    exit 0
fi

# Count labeled trajectory directories (name has 2+ underscores).
COUNT=$(python3 -c "
import os
try:
    p = '$PROJECT_ROOT/trajectory'
    c = sum(
        1 for d in os.listdir(p)
        if os.path.isdir(os.path.join(p, d)) and d.count('_') >= 2
    )
    print(c)
except Exception:
    print(0)
" 2>/dev/null || echo 0)

# Fire reminder every FREQUENCY labeled benches.
# Emit via stdout JSON (hookSpecificOutput.additionalContext) so the assistant
# actually sees it. Writing to stderr with exit 0 is NOT surfaced to the model
# by Claude Code — it's only logged locally. See:
#   https://code.claude.com/docs/en/hooks.md#postToolUse-hook-output-and-exit-codes
if [ "$COUNT" -gt 0 ] && [ "$((COUNT % FREQUENCY))" -eq 0 ]; then
    REVIEW_FILE="$PROJECT_ROOT/.claude/commands/review.md"
    if [ -f "$REVIEW_FILE" ]; then
        python3 -c "
import json, sys
try:
    with open('$REVIEW_FILE') as f:
        review = f.read()
    msg = (
        f'Self-review checkpoint (every $FREQUENCY labeled benches, now at #$COUNT).\n\n'
        + review
    )
    print(json.dumps({
        'hookSpecificOutput': {
            'hookEventName': 'PostToolUse',
            'additionalContext': msg,
        }
    }))
except Exception as e:
    # Silent failure — never break bench output.
    pass
" 2>/dev/null || true
    fi
fi

exit 0
