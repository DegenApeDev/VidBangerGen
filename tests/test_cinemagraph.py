from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from apps.api.cinemagraph import CinemagraphAdapter
from apps.api.database import Database
from apps.api.jobs import GenerationWorker
from apps.api.main import create_app
from apps.api.schemas import CinemagraphRequest, CreativeBrief


def test_cinemagraph_graph_uses_dev_gguf_i2v_and_official_motion_contract():
    graph, metadata = CinemagraphAdapter().build(
        image_name="neon-sign.png",
        prompt="only the red neon letters subtly pulse and flicker",
        width=704, height=512, seed=88,
    )

    assert graph["3"]["inputs"] == {
        "width": 704, "height": 512, "length": 25, "batch_size": 1,
    }
    assert graph["5"]["class_type"] == "LTXVImgToVideoConditionOnly"
    assert graph["5"]["inputs"]["strength"] == 1
    assert graph["6"]["inputs"]["unet_name"] == "ltx-2.3-22b-dev-Q4_K_M.gguf"
    assert graph["7"]["inputs"]["lora_name"] == (
        "ltx-2.3-22b-lora-cinemagraph-0.9.safetensors"
    )
    assert graph["8"]["inputs"]["block_indices"] == "29"
    assert graph["10"]["inputs"]["text"].startswith("CINEMAGRAPH_MOTION")
    assert "everything else remains completely frozen" in graph["10"]["inputs"]["text"]
    assert graph["16"]["class_type"] == "STGGuider"
    assert graph["16"]["inputs"]["cfg"] == 4
    assert graph["19"]["inputs"]["steps"] == 30
    assert "audio" not in graph["23"]["inputs"]
    assert metadata["workflow"] == "isolated-image-to-video-dev-stg-gguf"
    assert metadata["audio_mode"] == "silent-disconnected"


def test_cinemagraph_request_validates_strength_and_motion_prompt():
    with pytest.raises(ValidationError):
        CinemagraphRequest(target_id="primary", asset_id="image", prompt="")
    with pytest.raises(ValidationError):
        CinemagraphRequest(
            target_id="primary", asset_id="image", prompt="water moves", strength=3.1,
        )


@pytest.mark.asyncio
async def test_cinemagraph_api_queues_uploaded_image_as_dedicated_job(test_settings, tmp_path):
    app = create_app(test_settings, start_workers=False)
    project = app.state.db.create_project(
        CreativeBrief(title="Loop", topic="A selective motion loop").model_dump(mode="json")
    )
    source = tmp_path / "sign.png"
    source.write_bytes(b"image")
    asset = app.state.db.create_asset(
        project["id"], "image", "sign.png", str(source), "image/png", {},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        response = await client.post(
            "/creative-lab/cinemagraph",
            json={
                "target_id": "primary", "asset_id": asset["id"],
                "prompt": "only the reflected water ripples inside the sunglasses",
                "strength": 1.2,
            },
        )

    assert response.status_code == 202, response.text
    job = response.json()["job"]
    assert job["kind"] == "cinemagraph"
    assert job["lane"] == "comfy_upload"
    assert job["project_id"] == project["id"]
    assert job["payload"]["asset_id"] == asset["id"]
    assert job["payload"]["execution_target"] == "primary"


class StubCinemagraphAdapter:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def build(self, **kwargs: Any):
        self.kwargs = kwargs
        return {}, {"engine": "Cinemagraph test", "seed": kwargs["seed"]}


class CinemagraphClient:
    worker_id = "cinemagraph"
    base_url = "http://comfy.test:8188"

    async def upload(self, path: Path) -> str:
        assert path.name == "prepared-source.png"
        return path.name

    async def queue_and_wait(self, _graph, **kwargs):
        await kwargs["on_queued"]("remote-cinemagraph")
        await kwargs["progress"](.6, "sampler")
        job_id = kwargs["client_id"].removesuffix("-cinemagraph")
        return {"files": [{
            "filename": "cinemagraph_00001_.mp4",
            "subfolder": f"vbg/creative-lab/{job_id}", "type": "output",
        }]}

    async def download(self, _artifact, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"cinemagraph video")


@pytest.mark.asyncio
async def test_cinemagraph_worker_persists_downloadable_silent_output(
    test_settings, tmp_path: Path, monkeypatch,
):
    db = Database(test_settings.database_path)
    db.initialize()
    project = db.create_project(
        CreativeBrief(title="Loop", topic="A selective motion loop").model_dump(mode="json")
    )
    source = tmp_path / "still.png"
    source.write_bytes(b"still")
    destination = tmp_path / "output" / "loop.mp4"
    job = db.create_job(
        "cinemagraph", "comfy_upload",
        {
            "target_id": "primary", "execution_target": "primary",
            "source_path": str(source), "destination_path": str(destination),
            "prompt": "only the steam curls upward", "strength": 1.0, "seed": 9,
        },
        project_id=project["id"], max_attempts=1,
    )
    claimed = db.claim_generation_job(
        "comfy:cinemagraph", upload_capable=True, exclusive_capable=False,
        target_id="primary",
    )
    assert claimed and claimed["id"] == job["id"]

    def prepare(_source: Path, prepared: Path):
        prepared.parent.mkdir(parents=True, exist_ok=True)
        prepared.write_bytes(b"prepared")
        return {"path": str(prepared), "width": 704, "height": 512}

    async def immediate(function, *args):
        return function(*args)

    monkeypatch.setattr("apps.api.jobs.prepare_cinemagraph_image", prepare)
    monkeypatch.setattr(
        "apps.api.jobs.probe_video_info",
        lambda _path: {"width": 704, "height": 512, "frames": 25, "fps": 25.0},
    )
    monkeypatch.setattr("apps.api.jobs.asyncio.to_thread", immediate)
    adapter = StubCinemagraphAdapter()
    worker = GenerationWorker(
        db, test_settings, None, CinemagraphClient(),
        cinemagraph_adapter=adapter,  # type: ignore[arg-type]
    )

    await asyncio.wait_for(worker._execute(claimed), timeout=2)

    saved = db.get_job(job["id"])
    assert saved and saved["status"] == "succeeded"
    assert Path(saved["result"]["local_path"]).read_bytes() == b"cinemagraph video"
    assert saved["result"]["mode"] == "cinemagraph"
    assert adapter.kwargs and adapter.kwargs["image_name"] == "prepared-source.png"
