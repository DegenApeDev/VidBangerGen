from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class Database:
    """Small SQLite repository with atomic persistent job claiming."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self, *, recover_running: bool = True) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'draft',
                    brief_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS concepts (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    hook TEXT NOT NULL,
                    treatment TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    selected INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS shots (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    prompt TEXT NOT NULL,
                    negative_prompt TEXT NOT NULL DEFAULT '',
                    duration_seconds REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'planned',
                    data_json TEXT NOT NULL,
                    selected_candidate_id TEXT,
                    selection_origin TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    shot_id TEXT NOT NULL REFERENCES shots(id) ON DELETE CASCADE,
                    job_id TEXT,
                    status TEXT NOT NULL DEFAULT 'queued',
                    seed INTEGER NOT NULL,
                    draft INTEGER NOT NULL DEFAULT 1,
                    prompt TEXT NOT NULL,
                    settings_json TEXT NOT NULL,
                    artifact_json TEXT,
                    score_json TEXT,
                    total_score REAL,
                    selected INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    lane TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
                    shot_id TEXT REFERENCES shots(id) ON DELETE CASCADE,
                    candidate_id TEXT REFERENCES candidates(id) ON DELETE CASCADE,
                    worker_id TEXT,
                    remote_id TEXT,
                    progress REAL NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 2,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs(lane, status, created_at);
                CREATE INDEX IF NOT EXISTS candidates_shot_idx ON candidates(shot_id, status);
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS exports (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    job_id TEXT,
                    status TEXT NOT NULL DEFAULT 'queued',
                    platform TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    local_path TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    candidate_id TEXT REFERENCES candidates(id) ON DELETE SET NULL,
                    rating INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chains (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'ready',
                    merged_path TEXT,
                    finish_job_id TEXT,
                    finished_path TEXT,
                    finish_metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chain_clips (
                    id TEXT PRIMARY KEY,
                    chain_id TEXT NOT NULL REFERENCES chains(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    prompt TEXT NOT NULL DEFAULT '',
                    strength REAL NOT NULL DEFAULT 0.7,
                    status TEXT NOT NULL DEFAULT 'done',
                    prompt_id TEXT,
                    remote_filename TEXT,
                    local_path TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(chain_id, position)
                );
                CREATE INDEX IF NOT EXISTS chain_clips_chain_idx
                    ON chain_clips(chain_id, position);
                """
            )
            shot_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(shots)").fetchall()
            }
            if "selection_origin" not in shot_columns:
                conn.execute("ALTER TABLE shots ADD COLUMN selection_origin TEXT")
            chain_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(chains)").fetchall()
            }
            if "finished_path" not in chain_columns:
                conn.execute("ALTER TABLE chains ADD COLUMN finished_path TEXT")
            if "finish_job_id" not in chain_columns:
                conn.execute("ALTER TABLE chains ADD COLUMN finish_job_id TEXT")
            if "finish_metadata_json" not in chain_columns:
                conn.execute("ALTER TABLE chains ADD COLUMN finish_metadata_json TEXT")
            # Older builds stored a neutral 73 whenever the vision model could
            # not be reached. That number is not a creative judgment. Preserve
            # the diagnostic JSON while removing it from ratings and analytics.
            conn.execute(
                """UPDATE candidates SET status='unscored',total_score=NULL,updated_at=?
                   WHERE status='scored' AND score_json IS NOT NULL
                   AND json_valid(score_json)
                   AND json_extract(score_json,'$.judge')='technical-fallback-v1'""",
                (utcnow(),),
            )
        if recover_running:
            self.recover_jobs()

    def recover_jobs(self) -> None:
        now = utcnow()
        with self.connect() as conn:
            conn.execute(
                """UPDATE jobs SET status='queued',
                   worker_id=CASE WHEN remote_id IS NULL THEN NULL ELSE worker_id END,
                   error='Recovered after service restart', updated_at=?
                   WHERE status='running' AND cancel_requested=0 AND attempts < max_attempts""",
                (now,),
            )
            conn.execute(
                """UPDATE jobs SET status='failed', error='Retry limit reached before restart',
                   updated_at=?, finished_at=?
                   WHERE status='running' AND attempts >= max_attempts""",
                (now, now),
            )

    def retire_unsafe_studio_prompt_audio_jobs(self) -> int:
        """Cancel unstarted Studio jobs that preserve visual prose as speech.

        Direct/Quick generation is not represented by these durable project
        jobs. Completed media is immutable; only queued or continuity-blocked
        Studio work with the retired raw-prompt audio contract is affected.
        """
        now = utcnow()
        reason = (
            "Retired unsafe Studio raw-prompt audio contract. Regenerate this draft set "
            "using each shot's audio intent, exact Native dialogue, or uploaded voiceover."
        )
        with self._lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """SELECT id,candidate_id FROM jobs
                   WHERE kind IN ('generate','retake')
                   AND project_id IS NOT NULL
                   AND status IN ('queued','blocked')
                   AND json_valid(payload_json)
                   AND json_extract(payload_json,'$.settings.audio_mode')='prompt'"""
            ).fetchall()
            if rows:
                ids = [str(row["id"]) for row in rows]
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"""UPDATE jobs SET status='cancelled',cancel_requested=1,error=?,
                        progress=0,finished_at=?,updated_at=?
                        WHERE id IN ({placeholders})""",
                    (reason, now, now, *ids),
                )
                candidate_ids = [
                    str(row["candidate_id"]) for row in rows if row["candidate_id"]
                ]
                if candidate_ids:
                    candidate_placeholders = ",".join("?" for _ in candidate_ids)
                    conn.execute(
                        f"""UPDATE candidates SET status='cancelled',updated_at=?
                            WHERE id IN ({candidate_placeholders})""",
                        (now, *candidate_ids),
                    )
            conn.execute("COMMIT")
        return len(rows)

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        for key in (
            "brief_json", "data_json", "settings_json", "artifact_json", "score_json",
            "payload_json", "result_json", "options_json", "metadata_json",
            "finish_metadata_json",
        ):
            if key in item:
                item[key.removesuffix("_json")] = _loads(item.pop(key), None)
        for key in ("selected", "draft", "cancel_requested"):
            if key in item:
                item[key] = bool(item[key])
        return item

    def create_project(self, brief: dict[str, Any]) -> dict[str, Any]:
        pid, now = new_id("prj"), utcnow()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO projects(id,title,status,brief_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (pid, brief["title"], "draft", json.dumps(brief), now, now),
            )
        return self.get_project(pid)  # type: ignore[return-value]

    def list_projects(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM projects ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode(row) for row in rows if row is not None]  # type: ignore[misc]

    def get_project(self, project_id: str, expanded: bool = False) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        project = self._decode(row)
        if project and expanded:
            project["concepts"] = self.list_concepts(project_id, include_shots=True)
            project["exports"] = self.list_exports(project_id)
        return project

    def set_project_status(self, project_id: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE projects SET status=?, updated_at=? WHERE id=?",
                (status, utcnow(), project_id),
            )

    def set_project_prompt_mode(self, project_id: str, mode: str) -> None:
        if mode not in ("assisted", "manual"):
            raise ValueError(f"Unsupported prompt mode: {mode}")
        with self._lock, self.connect() as conn:
            row = conn.execute(
                "SELECT brief_json FROM projects WHERE id=?", (project_id,)
            ).fetchone()
            if not row:
                raise KeyError(project_id)
            brief = _loads(row["brief_json"], {}) or {}
            brief["prompt_mode"] = mode
            conn.execute(
                "UPDATE projects SET brief_json=?,updated_at=? WHERE id=?",
                (json.dumps(brief), utcnow(), project_id),
            )

    def refresh_project_status(self, project_id: str) -> str | None:
        """Derive the user-facing lifecycle from durable project state.

        Service methods set optimistic phase labels when work is enqueued. This
        reconciliation closes the phase after terminal jobs, including review
        gates and interrupted batches, so a finished queue can never leave the
        studio claiming that it is still generating.
        """
        with self.connect() as conn:
            project = conn.execute(
                "SELECT status FROM projects WHERE id=?", (project_id,)
            ).fetchone()
            if not project:
                return None
            active = conn.execute(
                """SELECT j.kind,c.draft FROM jobs j
                   LEFT JOIN candidates c ON c.id=j.candidate_id
                   WHERE j.project_id=? AND j.status IN ('queued','running','blocked')""",
                (project_id,),
            ).fetchall()
            if active:
                kinds = {row["kind"] for row in active}
                if "export" in kinds:
                    status = "exporting"
                elif "creative_plan" in kinds:
                    status = "planning"
                elif any(row["draft"] == 0 for row in active if row["kind"] in ("generate", "score")):
                    status = "rendering_winners"
                else:
                    status = "generating"
            elif project["status"] in ("exported", "export_failed"):
                status = str(project["status"])
            else:
                concept = conn.execute(
                    """SELECT id FROM concepts WHERE project_id=?
                       ORDER BY selected DESC,position LIMIT 1""",
                    (project_id,),
                ).fetchone()
                shots = (
                    conn.execute(
                        "SELECT id,status,selected_candidate_id FROM shots WHERE concept_id=? ORDER BY position",
                        (concept["id"],),
                    ).fetchall()
                    if concept else []
                )
                if not concept:
                    failed_plan = conn.execute(
                        """SELECT status FROM jobs WHERE project_id=? AND kind='creative_plan'
                           ORDER BY created_at DESC LIMIT 1""",
                        (project_id,),
                    ).fetchone()
                    status = (
                        "planning_failed"
                        if failed_plan and failed_plan["status"] == "failed" else "draft"
                    )
                elif not shots:
                    status = "planned"
                else:
                    selected = conn.execute(
                        """SELECT c.draft FROM shots s
                           JOIN candidates c ON c.id=s.selected_candidate_id
                           WHERE s.concept_id=?""",
                        (concept["id"],),
                    ).fetchall()
                    has_final = bool(conn.execute(
                        """SELECT 1 FROM candidates c JOIN shots s ON s.id=c.shot_id
                           WHERE s.concept_id=? AND c.draft=0 LIMIT 1""",
                        (concept["id"],),
                    ).fetchone())
                    needs_review = any(row["status"] == "needs_review" for row in shots)
                    if len(selected) == len(shots) and all(row["draft"] == 0 for row in selected):
                        status = "finals_ready"
                    elif has_final and (needs_review or len(selected) < len(shots) or any(row["draft"] for row in selected)):
                        status = "finals_review_required"
                    elif len(selected) == len(shots):
                        status = "candidates_ready"
                    elif needs_review:
                        status = "candidates_review_required"
                    else:
                        interrupted = conn.execute(
                            """SELECT 1 FROM jobs WHERE project_id=? AND kind IN ('generate','score')
                               AND status IN ('failed','cancelled') LIMIT 1""",
                            (project_id,),
                        ).fetchone()
                        status = "generation_interrupted" if interrupted else "planned"
            conn.execute(
                "UPDATE projects SET status=?,updated_at=? WHERE id=?",
                (status, utcnow(), project_id),
            )
        return status

    def reconcile_project_statuses(self) -> None:
        with self.connect() as conn:
            project_ids = [row["id"] for row in conn.execute("SELECT id FROM projects")]
        for project_id in project_ids:
            self.refresh_project_status(project_id)

    def replace_plan(self, project_id: str, concepts: list[dict[str, Any]]) -> None:
        now = utcnow()
        with self._lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM concepts WHERE project_id=?", (project_id,))
            for concept_position, concept in enumerate(concepts):
                cid = concept.get("id") or new_id("con")
                conn.execute(
                    """INSERT INTO concepts(id,project_id,position,title,hook,treatment,data_json,selected,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        cid, project_id, concept_position, concept["title"], concept["hook"],
                        concept["treatment"], json.dumps(concept), int(concept_position == 0), now,
                    ),
                )
                for shot_position, shot in enumerate(concept.get("shots", [])):
                    sid = shot.get("id") or new_id("shot")
                    conn.execute(
                        """INSERT INTO shots(id,project_id,concept_id,position,prompt,negative_prompt,
                           duration_seconds,status,data_json,created_at,updated_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            sid, project_id, cid, shot_position, shot["prompt"],
                            shot.get("negative_prompt", ""), shot.get("duration_seconds", 5.0),
                            "planned", json.dumps(shot), now, now,
                        ),
                    )
            conn.execute(
                "UPDATE projects SET status='planned', updated_at=? WHERE id=?", (now, project_id)
            )
            conn.execute("COMMIT")

    def list_concepts(self, project_id: str, include_shots: bool = False) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM concepts WHERE project_id=? ORDER BY position", (project_id,)
            ).fetchall()
        values = [self._decode(row) for row in rows if row is not None]  # type: ignore[misc]
        if include_shots:
            for value in values:
                value["shots"] = self.list_shots(project_id, value["id"], include_candidates=True)
        return values

    def select_concept(self, project_id: str, concept_id: str) -> None:
        with self._lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE concepts SET selected=0 WHERE project_id=?", (project_id,))
            result = conn.execute(
                "UPDATE concepts SET selected=1 WHERE id=? AND project_id=?", (concept_id, project_id)
            )
            if result.rowcount != 1:
                conn.execute("ROLLBACK")
                raise KeyError(concept_id)
            conn.execute("UPDATE projects SET updated_at=? WHERE id=?", (utcnow(), project_id))
            conn.execute("COMMIT")

    def get_shot(self, shot_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM shots WHERE id=?", (shot_id,)).fetchone()
        return self._decode(row)

    def add_shot(
        self, project_id: str, concept_id: str, prompt: str, negative_prompt: str,
        duration_seconds: float, data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        shot_id, now = new_id("shot"), utcnow()
        with self.connect() as conn:
            position = conn.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM shots WHERE concept_id=?", (concept_id,)
            ).fetchone()[0]
            conn.execute(
                """INSERT INTO shots(id,project_id,concept_id,position,prompt,negative_prompt,
                   duration_seconds,status,data_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    shot_id, project_id, concept_id, position, prompt, negative_prompt,
                    duration_seconds, "planned", json.dumps(data or {}), now, now,
                ),
            )
        return self.get_shot(shot_id)  # type: ignore[return-value]

    def update_shot(self, shot_id: str, values: dict[str, Any]) -> dict[str, Any]:
        editable_data = {
            "title", "purpose", "camera", "audio", "caption", "transition",
            "reference_asset_id", "reference_role", "audio_mode", "dialogue",
            "speaker", "language", "accent", "voiceover_text",
        }
        with self._lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM shots WHERE id=?", (shot_id,)).fetchone()
            if not row:
                conn.execute("ROLLBACK")
                raise KeyError(shot_id)
            candidate_count = conn.execute(
                "SELECT COUNT(*) FROM candidates WHERE shot_id=?", (shot_id,)
            ).fetchone()[0]
            if candidate_count:
                conn.execute("ROLLBACK")
                raise ValueError(
                    "This shot already has candidates; edit the prompt through a retake or create a new shot"
                )
            data = _loads(row["data_json"], {}) or {}
            for key in editable_data:
                if key in values:
                    data[key] = values[key]
            prompt = values.get("prompt", row["prompt"])
            negative = values.get("negative_prompt", row["negative_prompt"])
            duration = values.get("duration_seconds", row["duration_seconds"])
            now = utcnow()
            conn.execute(
                """UPDATE shots SET prompt=?,negative_prompt=?,duration_seconds=?,data_json=?,
                   status='planned',updated_at=? WHERE id=?""",
                (prompt, negative, duration, json.dumps(data), now, shot_id),
            )
            conn.execute(
                "UPDATE projects SET updated_at=? WHERE id=?", (now, row["project_id"])
            )
            conn.execute("COMMIT")
        return self.get_shot(shot_id)  # type: ignore[return-value]

    def list_shots(
        self, project_id: str, concept_id: str | None = None, include_candidates: bool = False
    ) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM shots WHERE project_id=?", [project_id]
        if concept_id:
            sql += " AND concept_id=?"
            args.append(concept_id)
        sql += " ORDER BY position"
        with self.connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        values = [self._decode(row) for row in rows if row is not None]  # type: ignore[misc]
        if include_candidates:
            for value in values:
                value["candidates"] = self.list_candidates(value["id"])
        return values

    def create_candidate(
        self, project_id: str, shot_id: str, prompt: str, seed: int,
        settings: dict[str, Any], draft: bool = True,
    ) -> dict[str, Any]:
        candidate_id, now = new_id("cand"), utcnow()
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO candidates(id,project_id,shot_id,status,seed,draft,prompt,
                   settings_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    candidate_id, project_id, shot_id, "queued", seed, int(draft), prompt,
                    json.dumps(settings), now, now,
                ),
            )
        return self.get_candidate(candidate_id)  # type: ignore[return-value]

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM candidates WHERE id=?", (candidate_id,)).fetchone()
        return self._decode(row)

    def list_candidates(self, shot_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE shot_id=? ORDER BY created_at", (shot_id,)
            ).fetchall()
        return [self._decode(row) for row in rows if row is not None]  # type: ignore[misc]

    def update_candidate(self, candidate_id: str, **values: Any) -> None:
        allowed = {"status", "job_id", "artifact_json", "score_json", "total_score", "selected"}
        assignments, args = [], []
        for key, value in values.items():
            if key not in allowed:
                continue
            assignments.append(f"{key}=?")
            args.append(json.dumps(value) if key.endswith("_json") and value is not None else value)
        if not assignments:
            return
        assignments.append("updated_at=?")
        args.extend([utcnow(), candidate_id])
        with self.connect() as conn:
            conn.execute(f"UPDATE candidates SET {', '.join(assignments)} WHERE id=?", args)

    def select_candidate(
        self, shot_id: str, candidate_id: str, selection_origin: str = "manual"
    ) -> None:
        with self._lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT id FROM candidates WHERE id=? AND shot_id=?", (candidate_id, shot_id)
            ).fetchone()
            if not row:
                conn.execute("ROLLBACK")
                raise KeyError(candidate_id)
            conn.execute("UPDATE candidates SET selected=0 WHERE shot_id=?", (shot_id,))
            conn.execute("UPDATE candidates SET selected=1 WHERE id=?", (candidate_id,))
            conn.execute(
                """UPDATE shots SET selected_candidate_id=?,selection_origin=?,
                   status='selected',updated_at=? WHERE id=?""",
                (candidate_id, selection_origin, utcnow(), shot_id),
            )
            conn.execute("COMMIT")

    def auto_select_candidate(self, shot_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            current = conn.execute(
                "SELECT project_id,selected_candidate_id,selection_origin FROM shots WHERE id=?",
                (shot_id,),
            ).fetchone()
            rows = conn.execute(
                """SELECT * FROM candidates WHERE shot_id=? AND status='scored'
                   ORDER BY total_score DESC, created_at ASC""",
                (shot_id,),
            ).fetchall()
            pending = conn.execute(
                "SELECT COUNT(*) FROM candidates WHERE shot_id=? AND status IN ('queued','generating','generated')",
                (shot_id,),
            ).fetchone()[0]
        if pending:
            return None
        eligible = None
        final_rows = [row for row in rows if not bool(row["draft"])]
        manual_source_id = (
            current["selected_candidate_id"]
            if current and current["selection_origin"] == "manual" else None
        )
        if current and str(current["selection_origin"] or "").startswith("manual"):
            # A human draft lock may be promoted only to its own derived final.
            # All other later score jobs remain unable to replace it.
            if current["selection_origin"] != "manual" or not final_rows:
                return None
            final_rows = [
                row for row in final_rows
                if (_loads(row["settings_json"], {}) or {}).get("hq_of") == manual_source_id
            ]
            if not final_rows:
                return None
        selection_pool = final_rows or rows
        for row in selection_pool:
            score = _loads(row["score_json"], {})
            subject = score.get("subject_check") or {}
            if score.get("judge") == "technical-fallback-v1":
                # Technical validity is useful, but it cannot choose a creative
                # winner without seeing subject identity, action, and hook.
                continue
            if float(row["total_score"] or 0) < 65:
                continue
            if subject.get("present") is False:
                continue
            if subject.get("identity_consistent") is False:
                continue
            eligible = row
            break
        if eligible:
            self.select_candidate(
                shot_id, eligible["id"], "manual-final" if manual_source_id else "auto"
            )
            return self.get_candidate(eligible["id"])
        with self.connect() as conn:
            current = conn.execute(
                "SELECT project_id,selection_origin FROM shots WHERE id=?", (shot_id,)
            ).fetchone()
            if current and not str(current["selection_origin"] or "").startswith("manual"):
                conn.execute("UPDATE candidates SET selected=0 WHERE shot_id=?", (shot_id,))
                conn.execute(
                    """UPDATE shots SET selected_candidate_id=NULL,selection_origin=NULL,
                       status='needs_review',updated_at=? WHERE id=?""",
                    (utcnow(), shot_id),
                )
        if current:
            self.refresh_project_status(current["project_id"])
        return None

    def create_job(
        self, kind: str, lane: str, payload: dict[str, Any], project_id: str | None = None,
        shot_id: str | None = None, candidate_id: str | None = None, max_attempts: int = 2,
        status: str = "queued",
    ) -> dict[str, Any]:
        job_id, now = new_id("job"), utcnow()
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO jobs(id,kind,lane,status,project_id,shot_id,candidate_id,payload_json,
                   max_attempts,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    job_id, kind, lane, status, project_id, shot_id, candidate_id,
                    json.dumps(payload), max_attempts, now, now,
                ),
            )
        if candidate_id:
            self.update_candidate(candidate_id, job_id=job_id)
        return self.get_job(job_id)  # type: ignore[return-value]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._decode(row)

    def list_jobs(self, project_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM jobs", []
        if project_id:
            sql += " WHERE project_id=?"
            args.append(project_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self.connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._decode(row) for row in rows if row is not None]  # type: ignore[misc]

    def claim_job(self, lane: str, worker_id: str) -> dict[str, Any] | None:
        now = utcnow()
        with self._lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT id FROM jobs WHERE lane=? AND status='queued' AND cancel_requested=0
                   AND (worker_id IS NULL OR worker_id=?)
                   ORDER BY created_at LIMIT 1""",
                (lane, worker_id),
            ).fetchone()
            if not row:
                conn.execute("COMMIT")
                return None
            changed = conn.execute(
                """UPDATE jobs SET status='running',worker_id=?,
                   attempts=attempts+CASE WHEN remote_id IS NULL THEN 1 ELSE 0 END,
                   error=NULL,started_at=COALESCE(started_at,?),updated_at=?
                   WHERE id=? AND status='queued'""",
                (worker_id, now, now, row["id"]),
            )
            conn.execute("COMMIT")
        return self.get_job(row["id"]) if changed.rowcount == 1 else None

    def claim_generation_job(
        self, worker_id: str, *, upload_capable: bool, exclusive_capable: bool,
        target_id: str = "primary",
    ) -> dict[str, Any] | None:
        """Atomically drain ordinary GPU work before an exclusive dual-GPU job."""
        now = utcnow()
        with self._lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            exclusive_running = conn.execute(
                """SELECT id FROM jobs WHERE lane='comfy_exclusive' AND status='running'
                   LIMIT 1"""
            ).fetchone()
            if exclusive_running:
                conn.execute("COMMIT")
                return None
            exclusive = conn.execute(
                """SELECT id FROM jobs WHERE lane='comfy_exclusive' AND status='queued'
                   AND cancel_requested=0 AND (worker_id IS NULL OR worker_id=?)
                   AND (COALESCE(json_extract(payload_json,'$.execution_target'),'auto')='auto'
                        OR json_extract(payload_json,'$.execution_target')=?)
                   ORDER BY created_at LIMIT 1""",
                (worker_id, target_id),
            ).fetchone()
            if exclusive:
                regular_running = conn.execute(
                    """SELECT COUNT(*) FROM jobs
                       WHERE lane IN ('comfy','comfy_upload') AND status='running'"""
                ).fetchone()[0]
                if not exclusive_capable or regular_running:
                    conn.execute("COMMIT")
                    return None
                selected_id = exclusive["id"]
            else:
                lanes = ("comfy_upload", "comfy") if upload_capable else ("comfy",)
                placeholders = ",".join("?" for _ in lanes)
                row = conn.execute(
                    f"""SELECT id FROM jobs WHERE lane IN ({placeholders})
                       AND status='queued' AND cancel_requested=0
                       AND (worker_id IS NULL OR worker_id=?)
                       AND (COALESCE(json_extract(payload_json,'$.execution_target'),'auto')='auto'
                            OR json_extract(payload_json,'$.execution_target')=?)
                       ORDER BY CASE lane WHEN 'comfy_upload' THEN 0 ELSE 1 END,
                                created_at LIMIT 1""",
                    (*lanes, worker_id, target_id),
                ).fetchone()
                if not row:
                    conn.execute("COMMIT")
                    return None
                selected_id = row["id"]
            changed = conn.execute(
                """UPDATE jobs SET status='running',worker_id=?,
                   attempts=attempts+CASE WHEN remote_id IS NULL THEN 1 ELSE 0 END,
                   error=NULL,started_at=COALESCE(started_at,?),updated_at=?
                   WHERE id=? AND status='queued'""",
                (worker_id, now, now, selected_id),
            )
            conn.execute("COMMIT")
        return self.get_job(selected_id) if changed.rowcount == 1 else None

    def update_job(self, job_id: str, **values: Any) -> None:
        allowed = {
            "status", "worker_id", "remote_id", "progress", "result_json", "error",
            "cancel_requested", "finished_at",
        }
        assignments, args = [], []
        for key, value in values.items():
            if key not in allowed:
                continue
            assignments.append(f"{key}=?")
            args.append(json.dumps(value) if key == "result_json" and value is not None else value)
        if not assignments:
            return
        assignments.append("updated_at=?")
        args.extend([utcnow(), job_id])
        project_id: str | None = None
        with self.connect() as conn:
            row = conn.execute("SELECT project_id FROM jobs WHERE id=?", (job_id,)).fetchone()
            project_id = str(row["project_id"]) if row and row["project_id"] else None
            conn.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE id=?", args)
        if project_id and values.get("status") in ("succeeded", "failed", "cancelled"):
            self.refresh_project_status(project_id)

    def request_cancel(self, job_id: str) -> bool:
        now = utcnow()
        project_id: str | None = None
        with self.connect() as conn:
            row = conn.execute(
                "SELECT status,candidate_id,project_id FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if not row:
                return False
            project_id = str(row["project_id"]) if row["project_id"] else None
            if row["status"] in ("queued", "blocked"):
                conn.execute(
                    "UPDATE jobs SET status='cancelled',cancel_requested=1,finished_at=?,updated_at=? WHERE id=?",
                    (now, now, job_id),
                )
                if row["candidate_id"]:
                    conn.execute(
                        "UPDATE candidates SET status='cancelled',updated_at=? WHERE id=?",
                        (now, row["candidate_id"]),
                    )
            elif row["status"] == "running":
                conn.execute(
                    "UPDATE jobs SET cancel_requested=1,updated_at=? WHERE id=?", (now, job_id)
                )
        if project_id:
            self.refresh_project_status(project_id)
        return True

    def retry_job(self, job_id: str) -> bool:
        project_id: str | None = None
        with self._lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT j.status,j.kind,j.project_id,j.shot_id,j.candidate_id,j.payload_json,
                          c.draft,s.position,s.concept_id
                   FROM jobs j
                   LEFT JOIN candidates c ON c.id=j.candidate_id
                   LEFT JOIN shots s ON s.id=j.shot_id WHERE j.id=?""",
                (job_id,),
            ).fetchone()
            if not row or row["status"] not in ("failed", "cancelled"):
                conn.execute("COMMIT")
                return False
            payload = _loads(row["payload_json"], {}) or {}
            if (
                row["project_id"]
                and row["kind"] in ("generate", "retake")
                and (payload.get("settings") or {}).get("audio_mode") == "prompt"
            ):
                conn.execute("COMMIT")
                return False
            project_id = str(row["project_id"]) if row["project_id"] else None
            target_status = "queued"
            if (
                row["kind"] == "generate" and row["shot_id"]
                and row["position"] is not None and int(row["position"]) > 0
            ):
                previous = conn.execute(
                    """SELECT c.draft FROM shots s
                       LEFT JOIN candidates c ON c.id=s.selected_candidate_id
                       WHERE s.concept_id=? AND s.position=?""",
                    (row["concept_id"], int(row["position"]) - 1),
                ).fetchone()
                if (
                    not previous or previous["draft"] is None
                    or int(previous["draft"]) != int(row["draft"])
                ):
                    target_status = "blocked"
            now = utcnow()
            result = conn.execute(
                """UPDATE jobs SET status=?,cancel_requested=0,error=NULL,progress=0,
                   worker_id=CASE WHEN remote_id IS NULL THEN NULL ELSE worker_id END,
                   finished_at=NULL,updated_at=?
                   WHERE id=? AND status IN ('failed','cancelled')""",
                (target_status, now, job_id),
            )
            if result.rowcount == 1 and row["candidate_id"]:
                candidate_status = (
                    "generated" if row["kind"] == "score" else "queued"
                )
                conn.execute(
                    "UPDATE candidates SET status=?,updated_at=? WHERE id=?",
                    (candidate_status, now, row["candidate_id"]),
                )
            conn.execute("COMMIT")
        if project_id:
            self.refresh_project_status(project_id)
        return result.rowcount == 1

    def previous_selected_candidate(self, shot_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            shot = conn.execute(
                "SELECT concept_id,position FROM shots WHERE id=?", (shot_id,)
            ).fetchone()
            if not shot or shot["position"] <= 0:
                return None
            row = conn.execute(
                """SELECT c.* FROM shots s JOIN candidates c ON c.id=s.selected_candidate_id
                   WHERE s.concept_id=? AND s.position=?""",
                (shot["concept_id"], shot["position"] - 1),
            ).fetchone()
        return self._decode(row)

    def release_next_shot_jobs(self, shot_id: str) -> int:
        now = utcnow()
        with self._lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                """SELECT s.project_id,s.concept_id,s.position,c.draft AS selected_draft
                   FROM shots s LEFT JOIN candidates c ON c.id=s.selected_candidate_id
                   WHERE s.id=?""",
                (shot_id,),
            ).fetchone()
            if not current:
                conn.execute("COMMIT")
                return 0
            next_shot = conn.execute(
                """SELECT s.id FROM shots s
                   WHERE s.concept_id=? AND s.position>? AND EXISTS (
                       SELECT 1 FROM jobs j JOIN candidates jc ON jc.id=j.candidate_id
                       WHERE j.shot_id=s.id AND j.status='blocked' AND jc.draft=?
                   ) ORDER BY s.position LIMIT 1""",
                (current["concept_id"], current["position"], current["selected_draft"]),
            ).fetchone()
            if not next_shot:
                project_status = (
                    "finals_ready" if current["selected_draft"] == 0 else "candidates_ready"
                )
                conn.execute(
                    "UPDATE projects SET status=?,updated_at=? WHERE id=?",
                    (project_status, now, current["project_id"]),
                )
                conn.execute("COMMIT")
                return 0
            result = conn.execute(
                """UPDATE jobs SET status='queued',updated_at=?
                   WHERE shot_id=? AND status='blocked' AND candidate_id IN (
                       SELECT id FROM candidates WHERE shot_id=? AND draft=?
                   )""",
                (now, next_shot["id"], next_shot["id"], current["selected_draft"]),
            )
            conn.execute(
                "UPDATE shots SET status='generating',updated_at=? WHERE id=?",
                (now, next_shot["id"]),
            )
            conn.execute("COMMIT")
            return result.rowcount

    def create_asset(
        self, project_id: str, kind: str, filename: str, local_path: str,
        mime_type: str, metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        asset_id = new_id("asset")
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO assets VALUES(?,?,?,?,?,?,?,?)",
                (
                    asset_id, project_id, kind, filename, local_path, mime_type,
                    json.dumps(metadata or {}), utcnow(),
                ),
            )
        return self.get_asset(asset_id)  # type: ignore[return-value]

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
        return self._decode(row)

    def list_assets(self, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM assets WHERE project_id=? ORDER BY created_at", (project_id,)
            ).fetchall()
        return [self._decode(row) for row in rows if row is not None]  # type: ignore[misc]

    def create_export(self, project_id: str, platform: str, options: dict[str, Any]) -> dict[str, Any]:
        export_id, now = new_id("export"), utcnow()
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO exports(id,project_id,status,platform,options_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (export_id, project_id, "queued", platform, json.dumps(options), now, now),
            )
        return self.get_export(export_id)  # type: ignore[return-value]

    def get_export(self, export_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM exports WHERE id=?", (export_id,)).fetchone()
        return self._decode(row)

    def update_export(self, export_id: str, **values: Any) -> None:
        allowed = {"job_id", "status", "local_path", "metadata_json"}
        assignments, args = [], []
        for key, value in values.items():
            if key not in allowed:
                continue
            assignments.append(f"{key}=?")
            args.append(json.dumps(value) if key == "metadata_json" and value is not None else value)
        assignments.append("updated_at=?")
        args.extend([utcnow(), export_id])
        with self.connect() as conn:
            conn.execute(f"UPDATE exports SET {', '.join(assignments)} WHERE id=?", args)

    def list_exports(self, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM exports WHERE project_id=? ORDER BY created_at DESC", (project_id,)
            ).fetchall()
        return [self._decode(row) for row in rows if row is not None]  # type: ignore[misc]

    def create_feedback(
        self, project_id: str, candidate_id: str | None, rating: int, label: str, reason: str,
    ) -> dict[str, Any]:
        feedback_id, now = new_id("fb"), utcnow()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO feedback VALUES(?,?,?,?,?,?,?)",
                (feedback_id, project_id, candidate_id, rating, label, reason, now),
            )
            row = conn.execute("SELECT * FROM feedback WHERE id=?", (feedback_id,)).fetchone()
        return dict(row)

    def analytics(self, project_id: str | None = None) -> dict[str, Any]:
        """Aggregate the signals used to tune prompts and candidate fan-out."""
        where = " WHERE project_id=?" if project_id else ""
        args: tuple[Any, ...] = (project_id,) if project_id else ()
        with self.connect() as conn:
            project_count = conn.execute(
                "SELECT COUNT(*) FROM projects" + (" WHERE id=?" if project_id else ""), args
            ).fetchone()[0]
            candidate = conn.execute(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN status='scored' THEN 1 ELSE 0 END) AS scored,
                          SUM(CASE WHEN selected=1 THEN 1 ELSE 0 END) AS selected,
                          AVG(CASE WHEN status='scored' THEN total_score END) AS average_score,
                          MAX(total_score) AS best_score
                   FROM candidates""" + where,
                args,
            ).fetchone()
            jobs = conn.execute(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN status='succeeded' THEN 1 ELSE 0 END) AS succeeded,
                          SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed,
                          AVG(CASE WHEN finished_at IS NOT NULL AND started_at IS NOT NULL
                              THEN (julianday(finished_at)-julianday(started_at))*86400 END)
                              AS average_runtime_seconds
                   FROM jobs""" + where,
                args,
            ).fetchone()
            feedback = conn.execute(
                """SELECT COUNT(*) AS total, AVG(rating) AS average_rating,
                          SUM(CASE WHEN label='reject' THEN 1 ELSE 0 END) AS rejected,
                          SUM(CASE WHEN label='usable' THEN 1 ELSE 0 END) AS usable,
                          SUM(CASE WHEN label='excellent' THEN 1 ELSE 0 END) AS excellent
                   FROM feedback""" + where,
                args,
            ).fetchone()
            score_rows = conn.execute(
                "SELECT score_json, selected FROM candidates" + where
                + " AND score_json IS NOT NULL" if project_id else
                "SELECT score_json, selected FROM candidates WHERE score_json IS NOT NULL",
                args,
            ).fetchall()
            role_rows = conn.execute(
                "SELECT settings_json,status,total_score,selected FROM candidates" + where,
                args,
            ).fetchall()
            role_feedback_rows = conn.execute(
                """SELECT c.settings_json,f.label,f.rating
                   FROM feedback f JOIN candidates c ON c.id=f.candidate_id"""
                + (" WHERE f.project_id=?" if project_id else ""),
                args,
            ).fetchall()

        score_dimensions = {
            "technical_score": [], "prompt_alignment": [], "temporal_coherence": [],
            "aesthetics": [], "hook_strength": [],
        }
        selected_dimensions = {key: [] for key in score_dimensions}
        for row in score_rows:
            score = _loads(row["score_json"], {})
            for key in score_dimensions:
                value = score.get(key)
                if isinstance(value, (int, float)):
                    score_dimensions[key].append(float(value))
                    if row["selected"]:
                        selected_dimensions[key].append(float(value))

        def averages(values: dict[str, list[float]]) -> dict[str, float | None]:
            return {
                key: round(sum(items) / len(items), 2) if items else None
                for key, items in values.items()
            }

        def clean(row: sqlite3.Row) -> dict[str, Any]:
            return {
                key: (round(value, 2) if isinstance(value, float) else (value or 0))
                for key, value in dict(row).items()
            }

        role_stats: dict[str, dict[str, Any]] = {}
        for row in role_rows:
            role = str((_loads(row["settings_json"], {}) or {}).get("take_role") or "")
            if not role:
                continue
            stats = role_stats.setdefault(
                role,
                {
                    "role": role, "renders": 0, "scored": 0, "selected": 0,
                    "score_total": 0.0, "feedback_total": 0, "rating_total": 0.0,
                    "excellent": 0, "rejected": 0,
                },
            )
            stats["renders"] += 1
            if row["status"] == "scored" and row["total_score"] is not None:
                stats["scored"] += 1
                stats["score_total"] += float(row["total_score"])
            if row["selected"]:
                stats["selected"] += 1
        for row in role_feedback_rows:
            role = str((_loads(row["settings_json"], {}) or {}).get("take_role") or "")
            if not role or role not in role_stats:
                continue
            stats = role_stats[role]
            stats["feedback_total"] += 1
            stats["rating_total"] += float(row["rating"])
            stats["excellent"] += int(row["label"] == "excellent")
            stats["rejected"] += int(row["label"] == "reject")
        take_roles = []
        for stats in role_stats.values():
            average_score = (
                stats["score_total"] / stats["scored"] if stats["scored"] else None
            )
            selection_rate = (
                stats["selected"] / stats["scored"] if stats["scored"] else 0.0
            )
            average_rating = (
                stats["rating_total"] / stats["feedback_total"]
                if stats["feedback_total"] else None
            )
            quality_signal = (
                (average_score or 0.0)
                + selection_rate * 8.0
                + ((average_rating - 3.0) * 2.0 if average_rating is not None else 0.0)
            )
            take_roles.append(
                {
                    "role": stats["role"], "renders": stats["renders"],
                    "scored": stats["scored"], "selected": stats["selected"],
                    "average_score": round(average_score, 2) if average_score is not None else None,
                    "selection_rate": round(selection_rate, 3),
                    "feedback_total": stats["feedback_total"],
                    "average_rating": (
                        round(average_rating, 2) if average_rating is not None else None
                    ),
                    "excellent": stats["excellent"], "rejected": stats["rejected"],
                    "quality_signal": round(quality_signal, 2),
                }
            )
        take_roles.sort(
            key=lambda value: (
                value["scored"] > 0, value["quality_signal"], value["scored"]
            ),
            reverse=True,
        )

        return {
            "scope": project_id or "all",
            "projects": project_count,
            "candidates": clean(candidate),
            "jobs": clean(jobs),
            "feedback": clean(feedback),
            "score_dimensions": averages(score_dimensions),
            "selected_score_dimensions": averages(selected_dimensions),
            "take_roles": take_roles,
        }

    # ── generic helpers ────────────────────────────────

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.connect() as conn:
            conn.execute(sql, params)

    def fetchone(self, sql: str, params: tuple = ()) -> tuple | None:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return tuple(row) if row else None

    def fetchall(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self.connect() as conn:
            return [tuple(r) for r in conn.execute(sql, params).fetchall()]

    # ── creative learning ──────────────────────────────

    def creative_learning_context(self, project_id: str, limit: int = 6) -> dict[str, Any]:
        """Return compact, human-labelled examples for the next creative plan.

        Project-specific feedback is ranked first, followed by studio-wide lessons.
        Prompts are deliberately bounded so the planning context cannot grow without
        limit as the studio accumulates renders.
        """
        limit = max(1, min(int(limit), 12))
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT f.project_id, f.label, f.rating, f.reason, f.created_at,
                          c.prompt, c.total_score, c.score_json, p.brief_json
                   FROM feedback f
                   JOIN projects p ON p.id=f.project_id
                   LEFT JOIN candidates c ON c.id=f.candidate_id
                   WHERE f.label IN ('excellent','usable','reject')
                   ORDER BY CASE WHEN f.project_id=? THEN 0 ELSE 1 END,
                            f.created_at DESC
                   LIMIT ?""",
                (project_id, limit * 4),
            ).fetchall()
            winners = conn.execute(
                """SELECT c.project_id, c.prompt, c.total_score, c.score_json, p.brief_json
                   FROM candidates c JOIN projects p ON p.id=c.project_id
                   WHERE c.selected=1 AND c.total_score>=65
                   ORDER BY CASE WHEN c.project_id=? THEN 0 ELSE 1 END,
                            c.total_score DESC LIMIT ?""",
                (project_id, limit),
            ).fetchall()

        def example(row: sqlite3.Row, *, reason: str = "") -> dict[str, Any]:
            brief = _loads(row["brief_json"], {})
            score = _loads(row["score_json"], {}) if "score_json" in row.keys() else {}
            return {
                "topic": str(brief.get("topic", ""))[:300],
                "style": str(brief.get("style", ""))[:160],
                "prompt": str(row["prompt"] or "")[:900],
                "score": round(float(row["total_score"]), 2)
                if row["total_score"] is not None else None,
                "reason": reason[:300],
                "issues": [str(value)[:200] for value in (score.get("issues") or [])[:4]],
            }

        proven: list[dict[str, Any]] = []
        avoid: list[dict[str, Any]] = []
        for row in rows:
            target = avoid if row["label"] == "reject" else proven
            if len(target) < limit:
                item = example(row, reason=str(row["reason"] or ""))
                item["label"] = row["label"]
                item["rating"] = int(row["rating"])
                target.append(item)
        selected = [example(row) for row in winners]
        return {
            "proven_patterns": proven,
            "avoid_patterns": avoid,
            "high_scoring_selected": selected,
            "example_count": len(proven) + len(avoid) + len(selected),
        }

    def create_chain(
        self, remote_filename: str | None = None, prompt: str = "",
        *, chain_id: str | None = None, clip_id: str | None = None,
        local_path: str | None = None, metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        chain_id, now = chain_id or new_id("chain"), utcnow()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO chains(id,status,created_at,updated_at) VALUES(?,?,?,?)",
                (chain_id, "ready", now, now),
            )
            if remote_filename or local_path:
                conn.execute(
                    """INSERT INTO chain_clips(id,chain_id,position,prompt,strength,status,
                       remote_filename,local_path,metadata_json,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        clip_id or new_id("clip"), chain_id, 0, prompt, 0.7, "done",
                        remote_filename, local_path, json.dumps(metadata or {}), now, now,
                    ),
                )
        return self.get_chain(chain_id)  # type: ignore[return-value]

    def add_chain_clip(
        self, chain_id: str, prompt: str, strength: float, *, status: str = "generating",
        prompt_id: str | None = None, remote_filename: str | None = None,
        local_path: str | None = None, metadata: dict[str, Any] | None = None,
        clip_id: str | None = None,
    ) -> dict[str, Any]:
        clip_id, now = clip_id or new_id("clip"), utcnow()
        with self._lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            chain = conn.execute("SELECT id FROM chains WHERE id=?", (chain_id,)).fetchone()
            if not chain:
                conn.execute("ROLLBACK")
                raise KeyError(chain_id)
            position = conn.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM chain_clips WHERE chain_id=?",
                (chain_id,),
            ).fetchone()[0]
            conn.execute(
                """INSERT INTO chain_clips(id,chain_id,position,prompt,strength,status,prompt_id,
                   remote_filename,local_path,metadata_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    clip_id, chain_id, position, prompt, strength, status, prompt_id,
                    remote_filename, local_path, json.dumps(metadata or {}), now, now,
                ),
            )
            conn.execute(
                """UPDATE chains SET status=?,merged_path=NULL,finish_job_id=NULL,
                   finished_path=NULL,finish_metadata_json=NULL,updated_at=? WHERE id=?""",
                ("generating" if status == "generating" else "ready", now, chain_id),
            )
            conn.execute("COMMIT")
        return self.get_chain_clip(clip_id)  # type: ignore[return-value]

    def get_chain_clip(self, clip_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM chain_clips WHERE id=?", (clip_id,)).fetchone()
        return self._decode(row)

    def update_chain_clip(self, clip_id: str, **values: Any) -> None:
        allowed = {"status", "prompt_id", "remote_filename", "local_path", "metadata_json"}
        assignments, args = [], []
        for key, value in values.items():
            if key not in allowed:
                continue
            assignments.append(f"{key}=?")
            args.append(json.dumps(value) if key == "metadata_json" else value)
        if not assignments:
            return
        assignments.append("updated_at=?")
        args.extend([utcnow(), clip_id])
        with self.connect() as conn:
            conn.execute(f"UPDATE chain_clips SET {', '.join(assignments)} WHERE id=?", args)

    def get_chain(self, chain_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM chains WHERE id=?", (chain_id,)).fetchone()
            clips = conn.execute(
                "SELECT * FROM chain_clips WHERE chain_id=? ORDER BY position", (chain_id,)
            ).fetchall()
        chain = self._decode(row)
        if chain:
            chain["clips"] = [self._decode(clip) for clip in clips]
            chain["finish_job"] = (
                self.get_job(str(chain["finish_job_id"]))
                if chain.get("finish_job_id") else None
            )
        return chain

    def list_chains(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id FROM chains ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [chain for row in rows if (chain := self.get_chain(row["id"]))]

    def update_chain(self, chain_id: str, *, status: str, merged_path: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE chains SET status=?,merged_path=?,finish_job_id=NULL,
                   finished_path=NULL,finish_metadata_json=NULL,updated_at=? WHERE id=?""",
                (status, merged_path, utcnow(), chain_id),
            )

    def update_chain_finish(
        self, chain_id: str, *, finished_path: str, metadata: dict[str, Any]
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE chains SET finished_path=?,finish_metadata_json=?,updated_at=?
                   WHERE id=?""",
                (finished_path, json.dumps(metadata), utcnow(), chain_id),
            )

    def update_chain_finish_job(self, chain_id: str, job_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE chains SET finish_job_id=?,finished_path=NULL,
                   finish_metadata_json=NULL,updated_at=? WHERE id=?""",
                (job_id, utcnow(), chain_id),
            )

    def update_chain_finish_failure(self, chain_id: str, message: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE chains SET finished_path=NULL,finish_metadata_json=?,updated_at=?
                   WHERE id=?""",
                (json.dumps({"error": message}), utcnow(), chain_id),
            )

    def import_legacy_chains(self, path: Path) -> int:
        """One-way, idempotent migration from the pre-SQLite `_chains.json` store."""
        if not path.exists():
            return 0
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return 0
        imported = 0
        for chain_id, value in payload.items():
            if self.get_chain(chain_id):
                continue
            clips = value.get("clips") or []
            first = clips[0] if clips else {}
            self.create_chain(
                first.get("video_file"), first.get("prompt", ""), chain_id=chain_id,
                clip_id=first.get("id"),
            )
            for clip in clips[1:]:
                self.add_chain_clip(
                    chain_id, clip.get("prompt", ""), float(clip.get("strength", 0.7)),
                    status=clip.get("status", "done"), prompt_id=clip.get("prompt_id"),
                    remote_filename=clip.get("video_file"), clip_id=clip.get("id"),
                )
            if value.get("merged_file"):
                self.update_chain(chain_id, status=value.get("status", "ready"), merged_path=value["merged_file"])
            imported += 1
        return imported
