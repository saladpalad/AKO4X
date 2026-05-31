"""AKO4X closed-loop master agent package.

Re-exports the IO layer from `master.master` so `import master` and
`master.spawn_child(...)` work unchanged when called from the AKO4X
repo root (which is the master CC's cwd).

See MASTER.md for the round-flow protocol.
"""

from master.master import (  # noqa: F401 — re-export for `import master`
    PKG_DIR,
    ROOT,
    SubResult,
    Phase2Result,
    init_campaign,
    read_campaign_mode,
    spawn_child,
    run_sub_phase1,
    send_retrospective_prompt,
    archive_variant,
    archive_failed,
    append_ledger,
)
