"""One-shot: walk reference/<family>/variants/<name>/kernel.py, parse the
"Delta from ..." section of the 5-section header, and write parent.txt
(parent variant name, or "null" for roots).

Run: python scripts/backfill_parent_txt.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference"

# Patterns observed in existing 5-section headers (lessons-convention.md):
#   "# Delta from prior anchor (X)"
#   "# Delta from X"
#   "# Delta vs X"
#   "# ─── Delta from prior anchor (X) ──────────────"
#   "# Delta from scratch"  → root, parent = null
DELTA_LINE_RE = re.compile(
    r"^\s*#\s*[─\-=\s]*Delta\b\s*"
    r"(?:from\s+prior\s+anchor\s*\(([^)]+)\)"          # 1: "(X)"
    r"|from\s+(scratch)\b"                             # 2: literal "scratch"
    r"|from\s+([A-Za-z0-9_][\w\-./]*?)(?:\s|\(|$)"     # 3: bare name after "from"
    r"|vs\s+([A-Za-z0-9_][\w\-./]*?)(?:\s|\(|$))",     # 4: bare name after "vs"
    re.IGNORECASE,
)


def read_header(kernel_path: Path, max_lines: int = 200) -> list[str]:
    """Return the leading comment block (lines starting with # or empty)."""
    lines: list[str] = []
    with kernel_path.open() as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                lines.append(line.rstrip("\n"))
            else:
                break
    return lines


def extract_parent(header_lines: list[str], sibling_names: set[str]) -> tuple[str | None, str | None]:
    """Return (parent_name, reason). parent_name="null" means root.
    parent_name=None means manual review needed; reason explains why.
    """
    for i, line in enumerate(header_lines):
        m = DELTA_LINE_RE.match(line)
        if not m:
            continue
        anchor_paren, scratch, from_bare, vs_bare = m.groups()
        if scratch:
            return "null", "matched 'Delta from scratch'"
        candidate = anchor_paren or from_bare or vs_bare
        if candidate is not None:
            # Strip trailing punctuation / version info that often follows the parent
            # name in headers like "(triton_swap_grid_evict_lsr 1.13×)" or "(X, v12)".
            candidate = candidate.split(",")[0].split()[0].strip()
            if candidate in sibling_names:
                return candidate, f"matched delta header line: {line.strip()!r}"
        # Fallback: bare "Delta from prior anchor" headers list the parent on the
        # next 1-3 lines. Scan for any sibling name token there.
        for j in range(i + 1, min(i + 5, len(header_lines))):
            for tok in re.split(r"[\s(),:;\"`]+", header_lines[j]):
                if tok in sibling_names:
                    return tok, f"matched sibling {tok!r} in line below delta header: {header_lines[j].strip()!r}"
        return (None,
                f"delta line {line.strip()!r} names {candidate!r} which is not a sibling variant; "
                f"no sibling found in next 4 lines either")
    return None, "no Delta section header line matched"


def main() -> int:
    if not REF.is_dir():
        print(f"FATAL: {REF} not found", file=sys.stderr)
        return 1

    backfilled = roots = linked = manual = 0
    manual_review: list[tuple[str, str, str]] = []

    for family in sorted(REF.iterdir()):
        variants_dir = family / "variants"
        if not variants_dir.is_dir():
            continue
        sibling_names = {p.name for p in variants_dir.iterdir() if p.is_dir()}

        for vdir in sorted(variants_dir.iterdir()):
            if not vdir.is_dir():
                continue
            kernel = vdir / "kernel.py"
            if not kernel.is_file():
                continue
            header = read_header(kernel)
            parent, reason = extract_parent(header, sibling_names)

            parent_txt = vdir / "parent.txt"
            # Don't clobber an existing reviewed parent.txt; rerun should be
            # idempotent for already-resolved variants. Only the literal
            # placeholder "TODO_MANUAL_REVIEW" is treated as overwriteable.
            if parent_txt.is_file():
                existing = parent_txt.read_text().strip()
                if existing and existing != "TODO_MANUAL_REVIEW":
                    backfilled += 1
                    if existing == "null":
                        roots += 1
                    elif existing in {p.name for p in vdir.parent.iterdir() if p.is_dir()}:
                        linked += 1
                    continue
            if parent is None:
                manual += 1
                manual_review.append((family.name, vdir.name, reason))
                parent_txt.write_text("TODO_MANUAL_REVIEW\n")
            elif parent == "null":
                roots += 1
                parent_txt.write_text("null\n")
            else:
                linked += 1
                parent_txt.write_text(parent + "\n")
            backfilled += 1

    print(f"Backfilled: {backfilled} variants")
    print(f"  Roots (null):           {roots}")
    print(f"  Linked to parent:       {linked}")
    print(f"  Manual review needed:   {manual}")
    if manual_review:
        print("\nManual review:")
        for family, variant, reason in manual_review:
            print(f"  {family}/{variant}: {reason}")
    return 0 if manual == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
