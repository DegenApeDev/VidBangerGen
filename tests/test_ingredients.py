from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from apps.api.database import Database
from apps.api.ingredients import IngredientsAdapter
from apps.api.jobs import GenerationWorker
from apps.api.schemas import CandidateBatchRequest, CreativeBrief, GenerationSettings
from apps.api.studio import StudioService


def test_ingredients_graph_matches_official_distilled_reference_contract():
    graph, metadata = IngredientsAdapter().build(
        image_name="visual-bible.png",
        reference_description="A red bottle, its exact white roundel, and a clean market aisle.",
        shot_prompt="The bottle slides into a sharp pool of warm light.",
        seed=42,
        filename_prefix="vbg/ingredients/test/take",
    )

    assert graph["6"]["inputs"]["unet_name"] == "ltx-2.3-22b-dev-Q4_K_M.gguf"
    assert graph["7"]["inputs"]["lora_name"] == (
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    )
    assert graph["8"]["inputs"]["lora_name"] == (
        "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
    )
    assert graph["3"]["inputs"]["amount"] == 121
    assert graph["4"]["inputs"] == {
        "width": 768, "height": 448, "length": 121, "batch_size": 1,
    }
    assert graph["13"]["inputs"]["image"] == ["3", 0]
    assert graph["19"]["inputs"]["sampler_name"] == "euler_ancestral_cfg_pp"
    assert graph["20"]["inputs"]["sigmas"].count(",") == 8
    assert "Reference sheet:" in graph["10"]["inputs"]["text"]
    assert "Generated video:" in graph["10"]["inputs"]["text"]
    assert "audio" not in graph["25"]["inputs"]
    assert metadata["workflow"] == "official-distilled-reference-sheet-gguf"
    assert metadata["steps"] == 8


def _ingredients_project(db: Database, tmp_path: Path) -> tuple[dict, dict]:
    project = db.create_project(
        CreativeBrief(
            title="Reference-sheet campaign", topic="A consistent product reveal",
            duration_seconds=5,
        ).model_dump(mode="json")
    )
    db.replace_plan(project["id"], [{
        "title": "One", "hook": "One", "treatment": "One",
        "shots": [{
            "prompt": "The red bottle rotates on a polished counter.",
            "duration_seconds": 5,
        }],
    }])
    sheet_path = tmp_path / "bible.png"
    sheet_path.write_bytes(b"image")
    sheet = db.create_asset(
        project["id"], "reference_sheet", "bible.png", str(sheet_path), "image/png",
        {"notes": "A red bottle, exact white roundel, green cap, and clean store aisle."},
    )
    return project, sheet


def test_studio_queues_ingredients_as_reviewable_final_candidates(test_settings, tmp_path: Path):
    db = Database(test_settings.database_path)
    db.initialize()
    project, sheet = _ingredients_project(db, tmp_path)
    result = StudioService(db).enqueue_candidates(
        project["id"],
        CandidateBatchRequest(
            candidates_per_shot=2,
            settings=GenerationSettings(
                reference_engine="ingredients", reference_image_asset_id=sheet["id"],
                reference_mode="every-shot", audio_mode="silent",
            ),
        ),
    )

    jobs = db.list_jobs(project["id"])
    shot = db.list_shots(project["id"])[0]
    candidates = db.list_candidates(shot["id"])
    assert result["creative_lab_mode"] == "ingredients"
    assert result["render_bucket"] == {"width": 768, "height": 448, "frames": 121, "fps": 24}
    assert len(jobs) == len(candidates) == 2
    assert all(job["kind"] == "ingredients_generate" for job in jobs)
    assert all(job["lane"] == "comfy_upload" and job["status"] == "queued" for job in jobs)
    assert all(candidate["draft"] is False for candidate in candidates)
    assert all(candidate["settings"]["creative_lab_mode"] == "ingredients" for candidate in candidates)


class StubIngredientsAdapter:
    width = 768
    height = 448
    max_duration_seconds = 5.0

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def build(self, **kwargs: Any):
        self.kwargs = kwargs
        return {}, {"engine": "Ingredients test", "seed": kwargs["seed"]}


class IngredientsClient:
    worker_id = "ingredients"
    base_url = "http://comfy.test:8188"

    def __init__(self, candidate_id: str, job_id: str) -> None:
        self.candidate_id = candidate_id
        self.job_id = job_id

    async def upload(self, path: Path) -> str:
        assert path.name.endswith("-reference.png")
        return path.name

    async def queue_and_wait(self, _graph, **kwargs):
        await kwargs["on_queued"]("remote-ingredients")
        await kwargs["progress"](.5, "sampler")
        return {"files": [{
            "filename": f"{self.candidate_id}_00001_.mp4",
            "subfolder": f"vbg/ingredients/{self.job_id}", "type": "output",
        }]}

    async def download(self, _artifact, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"raw ingredients video")


@pytest.mark.asyncio
async def test_ingredients_worker_persists_candidate_and_structural_silence(
    test_settings, tmp_path: Path, monkeypatch,
):
    db = Database(test_settings.database_path)
    db.initialize()
    project, sheet = _ingredients_project(db, tmp_path)
    shot = db.list_shots(project["id"])[0]
    settings = GenerationSettings(
        width=768, height=448, duration_seconds=4, fps=24, draft=False,
        reference_engine="ingredients", reference_image_asset_id=sheet["id"],
        reference_mode="every-shot", reference_sheet_description=(
            "A red bottle, exact white roundel, green cap, and clean store aisle."
        ),
        audio_mode="silent",
    ).model_dump(mode="json")
    settings["workflow_prompt"] = shot["prompt"]
    candidate = db.create_candidate(
        project["id"], shot["id"], shot["prompt"], 42, settings, draft=False,
    )
    job = db.create_job(
        "ingredients_generate", "comfy_upload",
        {
            "settings": settings, "execution_target": "primary",
            "reference_image_path": sheet["local_path"],
            "reference_sheet_description": settings["reference_sheet_description"],
        },
        project_id=project["id"], shot_id=shot["id"], candidate_id=candidate["id"],
        max_attempts=1,
    )
    claimed = db.claim_generation_job(
        "comfy:ingredients", upload_capable=True, exclusive_capable=False,
        target_id="primary",
    )
    assert claimed and claimed["id"] == job["id"]

    def prepare(_source: Path, destination: Path, width: int, height: int):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"prepared sheet")
        return {"path": str(destination), "width": width, "height": height}

    def trim(_source: Path, destination: Path, duration: float):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"trimmed silent video")
        return {"path": str(destination), "duration_seconds": duration, "audio": "omitted"}

    async def immediate(function, *args):
        return function(*args)

    monkeypatch.setattr("apps.api.jobs.prepare_reference_sheet", prepare)
    monkeypatch.setattr("apps.api.jobs.trim_silent_video", trim)
    monkeypatch.setattr(
        "apps.api.jobs.probe_video_info",
        lambda _path: {"width": 768, "height": 448, "frames": 97, "fps": 24.0},
    )
    monkeypatch.setattr(
        "apps.api.jobs.extract_continuity_assets",
        lambda *_args: {"last_frame_path": "/tmp/last.png"},
    )
    monkeypatch.setattr("apps.api.jobs.asyncio.to_thread", immediate)
    adapter = StubIngredientsAdapter()
    worker = GenerationWorker(
        db, test_settings, None, IngredientsClient(candidate["id"], job["id"]),
        ingredients_adapter=adapter,  # type: ignore[arg-type]
    )

    await asyncio.wait_for(worker._execute(claimed), timeout=2)

    saved_job = db.get_job(job["id"])
    saved_candidate = db.get_candidate(candidate["id"])
    assert saved_job and saved_job["status"] == "succeeded"
    assert saved_candidate and saved_candidate["status"] == "unscored"
    assert Path(saved_candidate["artifact"]["local_path"]).read_bytes() == b"trimmed silent video"
    assert saved_candidate["artifact"]["workflow"]["trim"]["audio"] == "omitted"
    assert adapter.kwargs and adapter.kwargs["image_name"].endswith("-reference.png")
