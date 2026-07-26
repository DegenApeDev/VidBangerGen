from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from apps.api.database import Database
from apps.api.jobs import ExclusiveInferenceCoordinator, GenerationWorker, LocalWorker
from apps.api.schemas import CreativeBrief
from apps.api.comfy import ComfyClient


class RecordingAdapter:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def build(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.kwargs = kwargs
        raise RuntimeError("stop after workflow arguments are captured")


class UploadOnlyClient:
    worker_id = "test"
    base_url = "http://comfy.test:8188"

    def __init__(self) -> None:
        self.uploaded: list[Path] = []

    async def upload(self, path: Path) -> str:
        self.uploaded.append(path)
        return path.name


class CoordinatedClient:
    def __init__(self, url: str, *, busy: bool = False) -> None:
        self.base_url = url
        self.busy = busy
        self.released = 0

    async def queue_state(self) -> dict[str, list[Any]]:
        return {"running": [["foreign"]] if self.busy else [], "pending": []}

    async def release_managed_models(self) -> None:
        self.released += 1


class CompletedSharedClient:
    worker_id = "shared"
    base_url = "http://comfy.test:8188"

    def __init__(self) -> None:
        self.allow_interrupt: bool | None = None

    async def queue_and_wait(self, _graph, **kwargs):
        self.allow_interrupt = kwargs["allow_interrupt"]
        return {"files": [{"filename": "discarded.mp4", "type": "output"}]}


class SuccessfulClient:
    worker_id = "success"
    base_url = "http://comfy.test:8188"

    async def queue_and_wait(self, _graph, **_kwargs):
        return {"files": [{"filename": "finished.mp4", "type": "output"}]}

    async def download(self, _artifact, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"test video")


class SuccessfulAdapter:
    def build(self, **_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {"profile": "motion-draft-4x3"}


def test_local_upscale_worker_runs_video2x_on_the_configured_gpu(
    test_settings, tmp_path: Path, monkeypatch
):
    db = Database(test_settings.database_path)
    db.initialize()
    source = tmp_path / "source.mp4"
    destination = tmp_path / "upscaled.mp4"
    source.write_bytes(b"video")
    job = db.create_job(
        "upscale", "postprocess",
        {
            "target_id": "local", "scale": 2, "source_path": str(source),
            "destination_path": str(destination),
        },
        max_attempts=1,
    )
    monkeypatch.setattr("apps.api.jobs.shutil.which", lambda _value: "/usr/bin/video2x")
    recorded: list[str] = []

    def run(command, **_kwargs):
        recorded.extend(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"upscaled")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("apps.api.jobs.subprocess.run", run)
    worker = LocalWorker(db, test_settings, None, None)  # type: ignore[arg-type]
    result = worker._upscale(job)

    assert destination.read_bytes() == b"upscaled"
    assert result["target_id"] == "local"
    assert recorded[-2:] == ["-g", "0"]


@pytest.mark.asyncio
async def test_retake_does_not_reattach_reference_audio(test_settings, tmp_path: Path):
    db = Database(test_settings.database_path)
    db.initialize()
    project = db.create_project(
        CreativeBrief(title="Retake", topic="A practical retake test").model_dump(mode="json")
    )
    db.replace_plan(
        project["id"],
        [{
            "title": "Concept", "hook": "Hook", "treatment": "Treatment",
            "shots": [{"prompt": "Original shot", "duration_seconds": 5}],
        }],
    )
    shot = db.list_shots(project["id"])[0]
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"voice")
    audio = db.create_asset(
        project["id"], "reference", "voice.wav", str(audio_path), "audio/wav", {}
    )
    settings = {
        "width": 448, "height": 256, "duration_seconds": 5, "fps": 24,
        "negative_prompt": "", "profile": "motion-draft-4x3",
        "reference_audio_asset_id": audio["id"],
        "retake_start_seconds": 1.0, "retake_end_seconds": 3.0,
    }
    source = db.create_candidate(
        project["id"], shot["id"], "Original shot", 1, settings
    )
    db.update_candidate(
        source["id"], status="generated",
        artifact_json={"local_path": str(source_path)},
    )
    retake = db.create_candidate(
        project["id"], shot["id"], "Improve the middle", 2, settings
    )
    job = db.create_job(
        "retake", "comfy_upload",
        {"settings": settings, "source_candidate_id": source["id"]},
        project_id=project["id"], shot_id=shot["id"], candidate_id=retake["id"],
        max_attempts=1,
    )
    claimed = db.claim_job("comfy_upload", "comfy:test")
    assert claimed and claimed["id"] == job["id"]

    adapter = RecordingAdapter()
    client = UploadOnlyClient()
    worker = GenerationWorker(db, test_settings, adapter, client)  # type: ignore[arg-type]
    await worker._execute(claimed)

    assert adapter.kwargs is not None
    assert adapter.kwargs["retake_video_name"] == "source.mp4"
    assert adapter.kwargs["audio_name"] is None
    assert client.uploaded == [source_path]
    assert db.get_job(job["id"])["status"] == "failed"


@pytest.mark.asyncio
async def test_exclusive_preflight_releases_only_an_idle_managed_peer():
    target = CoordinatedClient("http://rig:8188")
    peer = CoordinatedClient("http://rig:8189")
    coordinator = ExclusiveInferenceCoordinator(  # type: ignore[arg-type]
        [target, peer], (peer.base_url,)
    )
    await coordinator.prepare(target)  # type: ignore[arg-type]
    assert target.released == 0
    assert peer.released == 1

    peer.busy = True
    with pytest.raises(RuntimeError, match="queue to be idle"):
        await coordinator.prepare(target)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_exclusive_preflight_refuses_an_unmanaged_peer():
    target = CoordinatedClient("http://rig:8188")
    peer = CoordinatedClient("http://rig:8189")
    coordinator = ExclusiveInferenceCoordinator([target, peer], ())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="unmanaged peer"):
        await coordinator.prepare(target)  # type: ignore[arg-type]


def test_comfy_queue_ids_are_parsed_without_touching_foreign_entries():
    assert ComfyClient._queued_prompt_ids(
        [[12, "ours", {}, {}], {"prompt_id": "also-ours"}, ["short"]]
    ) == {"ours", "also-ours"}


@pytest.mark.asyncio
async def test_shared_worker_cancellation_discards_output_without_interrupt(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = db.create_project(
        CreativeBrief(title="Cancel", topic="A safe shared cancellation").model_dump(mode="json")
    )
    db.replace_plan(
        project["id"], [{
            "title": "Concept", "hook": "Hook", "treatment": "Treatment",
            "shots": [{"prompt": "One shot", "duration_seconds": 1}],
        }],
    )
    shot = db.list_shots(project["id"])[0]
    settings = {
        "width": 256, "height": 256, "duration_seconds": 1, "fps": 24,
        "negative_prompt": "", "profile": "motion-draft-4x3",
    }
    candidate = db.create_candidate(project["id"], shot["id"], shot["prompt"], 1, settings)
    job = db.create_job(
        "generate", "comfy", {"settings": settings}, project_id=project["id"],
        shot_id=shot["id"], candidate_id=candidate["id"], max_attempts=1,
    )
    claimed = db.claim_job("comfy", "comfy:shared")
    assert claimed and db.request_cancel(job["id"])
    client = CompletedSharedClient()
    worker = GenerationWorker(
        db, test_settings, SuccessfulAdapter(), client  # type: ignore[arg-type]
    )
    await worker._execute(claimed)
    assert client.allow_interrupt is False
    assert db.get_job(job["id"])["status"] == "cancelled"
    assert db.get_candidate(candidate["id"])["status"] == "cancelled"


@pytest.mark.asyncio
async def test_generation_goes_directly_to_manual_review_when_vl_scoring_is_off(
    test_settings, monkeypatch
):
    db = Database(test_settings.database_path)
    db.initialize()
    project = db.create_project(
        CreativeBrief(title="Manual review", topic="A product reveal").model_dump(mode="json")
    )
    db.replace_plan(project["id"], [{
        "title": "Concept", "hook": "Hook", "treatment": "Treatment",
        "shots": [{"prompt": "A bottle rotates in warm light", "duration_seconds": 1}],
    }])
    shot = db.list_shots(project["id"])[0]
    settings = {
        "width": 256, "height": 256, "duration_seconds": 1, "fps": 24,
        "negative_prompt": "", "profile": "motion-draft-4x3",
    }
    candidate = db.create_candidate(project["id"], shot["id"], shot["prompt"], 1, settings)
    job = db.create_job(
        "generate", "comfy", {"settings": settings}, project_id=project["id"],
        shot_id=shot["id"], candidate_id=candidate["id"], max_attempts=1,
    )
    claimed = db.claim_job("comfy", "comfy:success")
    monkeypatch.setattr(
        "apps.api.jobs.extract_continuity_assets",
        lambda *_args: {"last_frame_path": "/tmp/frame.png"},
    )
    async def immediate(function, *args):
        return function(*args)

    monkeypatch.setattr("apps.api.jobs.asyncio.to_thread", immediate)
    worker = GenerationWorker(
        db, test_settings, SuccessfulAdapter(), SuccessfulClient()  # type: ignore[arg-type]
    )

    await worker._execute(claimed)

    saved = db.get_candidate(candidate["id"])
    assert saved["status"] == "unscored"
    assert saved["score"]["judge"] == "manual-review"
    assert db.get_shot(shot["id"])["status"] == "needs_review"
    assert not any(item["kind"] == "score" for item in db.list_jobs(project["id"]))
    assert db.get_job(job["id"])["result"]["scoring"] == "manual-review"
