"""SQLite-backed campaign telemetry and strict lifecycle transitions."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATES = (
    "CREATED", "PREFLIGHTED", "BASELINED", "PROFILED_BASELINE",
    "OPTIMIZING", "VERIFIED", "BENCHMARKED", "PROFILED_CANDIDATE",
    "PROMOTABLE", "PROMOTED",
)
TERMINAL_STATES = {"FAILED", "PROMOTED"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                project_root TEXT NOT NULL,
                lane TEXT NOT NULL,
                agent TEXT NOT NULL,
                state TEXT NOT NULL,
                status TEXT NOT NULL,
                source_hash TEXT,
                integrity_hash TEXT,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finished_at TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                kind TEXT NOT NULL,
                phase TEXT,
                payload TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );
            CREATE INDEX IF NOT EXISTS events_run_seq ON events(run_id, seq);
            """
        )
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(runs)")}
        if "integrity_hash" not in columns:
            self._db.execute("ALTER TABLE runs ADD COLUMN integrity_hash TEXT")
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def start_run(self, project_root: Path, *, lane: str = "candidate",
                  agent: str = "external", run_id: str | None = None) -> str:
        run_id = run_id or str(uuid.uuid4())
        now = _now()
        with self._lock:
            self._db.execute(
                """INSERT INTO runs
                   (id, project_root, lane, agent, state, status, source_hash,
                    integrity_hash, started_at, updated_at, finished_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, str(project_root.resolve()), lane, agent, "CREATED", "running",
                 None, None, now, now, None),
            )
            self._db.commit()
        self.emit(run_id, "run.started", phase="CREATED", payload={"lane": lane, "agent": agent})
        return run_id

    def emit(self, run_id: str, kind: str, *, phase: str | None = None,
             payload: dict[str, Any] | None = None) -> None:
        now = _now()
        with self._lock:
            self._db.execute(
                "INSERT INTO events(run_id, timestamp, kind, phase, payload) VALUES (?, ?, ?, ?, ?)",
                (run_id, now, kind, phase, json.dumps(payload or {}, sort_keys=True)),
            )
            self._db.execute("UPDATE runs SET updated_at=? WHERE id=?", (now, run_id))
            self._db.commit()

    def heartbeat(self, run_id: str, *, phase: str, detail: str = "") -> None:
        self.emit(run_id, "heartbeat", phase=phase, payload={"detail": detail})

    def transition(self, run_id: str, state: str, *, payload: dict[str, Any] | None = None) -> None:
        row = self.run(run_id)
        current = row["state"]
        if state == "FAILED":
            allowed = current not in TERMINAL_STATES
        elif current in STATES and state in STATES:
            allowed = STATES.index(state) == STATES.index(current) + 1
        else:
            allowed = False
        if not allowed:
            raise ValueError(f"invalid run transition {current} -> {state}")
        now = _now()
        status = "failed" if state == "FAILED" else ("complete" if state == "PROMOTED" else "running")
        finished = now if state in TERMINAL_STATES else None
        with self._lock:
            self._db.execute(
                "UPDATE runs SET state=?, status=?, updated_at=?, finished_at=? WHERE id=?",
                (state, status, now, finished, run_id),
            )
            self._db.commit()
        self.emit(run_id, "state.changed", phase=state, payload=payload)

    def bind_source(self, run_id: str, source_hash: str) -> None:
        with self._lock:
            self._db.execute("UPDATE runs SET source_hash=?, updated_at=? WHERE id=?",
                             (source_hash, _now(), run_id))
            self._db.commit()
        self.emit(run_id, "source.bound", payload={"sha256": source_hash})

    def bind_integrity(self, run_id: str, integrity_hash: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE runs SET integrity_hash=?, updated_at=? WHERE id=?",
                (integrity_hash, _now(), run_id),
            )
            self._db.commit()
        self.emit(run_id, "integrity.bound", payload={"sha256": integrity_hash})

    def finish_checkpoint(self, run_id: str) -> None:
        """Mark a non-promotion command such as doctor as cleanly complete."""
        row = self.run(run_id)
        if row["state"] in TERMINAL_STATES:
            raise ValueError(f"cannot finish terminal run in {row['state']}")
        now = _now()
        with self._lock:
            self._db.execute(
                "UPDATE runs SET status='complete', updated_at=?, finished_at=? WHERE id=?",
                (now, now, run_id),
            )
            self._db.commit()
        self.emit(run_id, "run.completed", phase=row["state"])

    def run(self, run_id: str) -> dict[str, Any]:
        row = self._db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run {run_id}")
        return dict(row)

    def latest_run(self) -> dict[str, Any] | None:
        row = self._db.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def events(self, run_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq", (run_id, after)
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            result.append(item)
        return result
