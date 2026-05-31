"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Triton or CUDA) into a Solution JSON file for evaluation.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import scripts.benchmark_adapter as adapter


def load_config(root: Path = None) -> dict:
    """Load configuration from config.toml.

    root defaults to PROJECT_ROOT (this script's own env). Parent-side
    callers that pack a kernel dir other than their own (the modal
    cheat-check) pass an explicit root.
    """
    config_path = (root or PROJECT_ROOT) / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def build_solution(root: Path = None):
    """Build (without writing) a solution-blob from <root>/config.toml + <root>/solution/.

    Single source of truth for the pack rules — flat solution/ layout,
    target_hardware, destination_passing_style default. Returns
    (blob, meta, language) where blob is the solution.json text and meta is
    {name, definition, author}. pack_solution() wraps this and writes the blob to
    disk; the parent-side modal cheat-check uses the blob directly.
    """
    base = root or PROJECT_ROOT
    config = load_config(base)

    # Validate required config sections and keys
    for section in ("solution", "build"):
        if section not in config:
            raise ValueError(f"config.toml missing required section: [{section}]")
    required_solution_keys = ("name", "definition", "author")
    for key in required_solution_keys:
        if key not in config["solution"]:
            raise ValueError(f"config.toml [solution] missing required key: '{key}'")
    required_build_keys = ("language", "entry_point")
    for key in required_build_keys:
        if key not in config["build"]:
            raise ValueError(f"config.toml [build] missing required key: '{key}'")

    solution_config = config["solution"]
    build_config = config["build"]
    language = build_config["language"]

    # Determine source directory (flat solution/)
    source_dir = base / "solution"
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    build_cfg = {
        "language": language,
        "entry_point": build_config["entry_point"],
        "destination_passing_style": build_config.get("destination_passing_style", False),
    }
    blob = adapter.pack(
        str(source_dir), build_cfg,
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
    )
    meta = {"name": solution_config["name"],
            "definition": solution_config["definition"],
            "author": solution_config["author"]}
    return blob, meta, language


def pack_solution(output_path: Path = None, quiet: bool = False) -> Path:
    """Pack solution files into a solution.json blob; returns the written path."""
    blob, meta, language = build_solution()

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(blob)
    if not quiet:
        print(f"Solution packed: {output_path}")
        print(f"  Name: {meta['name']}")
        print(f"  Definition: {meta['definition']}")
        print(f"  Author: {meta['author']}")
        print(f"  Language: {language}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)"
    )
    args = parser.parse_args()

    try:
        pack_solution(args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
