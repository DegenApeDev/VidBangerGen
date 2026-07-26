from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from apps.api.database import Database
from apps.api.in_outpaint import InOutpaintAdapter
from apps.api.jobs import GenerationWorker
from apps.api.main import create_app
from apps.api.media import prepare_outpaint_inputs, prepare_video_mask, probe_video_info
from apps.api.schemas import CreativeBrief, CreativeTransformRequest


def test_in_outpaint_graph_uses_two_stage_gguf_masked_edit_contract():
    adapter = InOutpaintAdapter()
    graph, metadata = adapter.build(
        video_name="source.mp4", mask_name="mask.mkv",
        prompt="a clean brick wall continues behind the subject",
        strength=1.0, dilation=12, seed=70,
    )

    assert graph["18"]["inputs"]["unet_name"] == "ltx-2.3-22b-dev-Q4_K_M.gguf"
    assert graph["19"]["inputs"] == {
        "model": ["18", 0],
        "lora_name": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "strength_model": 0.5,
    }
    assert graph["20"]["inputs"]["lora_name"] == (
        "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"
    )
    assert graph["25"]["class_type"] == "LTXAddVideoICLoRAGuideAdvanced"
    assert graph["25"]["inputs"]["image"] == ["14", 0]
    assert graph["25"]["inputs"]["attention_strength"] == 1.0
    assert graph["33"]["inputs"]["sigmas"] == adapter.stage_one_sigmas
    assert graph["45"]["inputs"]["sigmas"] == adapter.stage_two_sigmas
    assert graph["38"]["class_type"] == "LTXVLaplacianPyramidBlend"
    assert graph["49"]["class_type"] == "LTXVLaplacianPyramidBlend"
    assert "audio" not in graph["50"]["inputs"]
    assert metadata["workflow"] == "official-two-stage-masked-edit-gguf"
    assert metadata["audio_mode"] == "source-restored-locally"
    assert metadata["stage_one"]["steps"] == 8
    assert metadata["stage_two"]["steps"] == 2


def test_in_outpaint_source_and_request_contracts_are_explicit():
    adapter = InOutpaintAdapter()
    adapter.validate_source({"frames": 121})
    with pytest.raises(ValueError, match="up to 121 frames"):
        adapter.validate_source({"frames": 129})
    with pytest.raises(ValueError, match=r"8n\+1"):
        adapter.validate_source({"frames": 120})
    with pytest.raises(ValidationError, match="uploaded static or animated mask"):
        CreativeTransformRequest(
            mode="in-outpainting", operation="inpaint",
            candidate_id="candidate", target_id="primary",
        )
    with pytest.raises(ValidationError, match="canvas mask automatically"):
        CreativeTransformRequest(
            mode="in-outpainting", operation="outpaint", mask_asset_id="mask",
            candidate_id="candidate", target_id="primary",
        )
    request = CreativeTransformRequest(
        mode="in-outpainting", operation="outpaint", outpaint_direction="right",
        expansion_percent=35, candidate_id="candidate", target_id="primary",
    )
    assert request.mask_asset_id is None
    assert request.outpaint_direction == "right"


def _make_test_video(path: Path) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=64x64:rate=8",
            "-frames:v", "9", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_static_mask_and_outpaint_preparation_follow_source_timeline(tmp_path: Path):
    source = tmp_path / "source.mp4"
    _make_test_video(source)
    mask = tmp_path / "mask.pgm"
    mask.write_bytes(b"P5\n64 64\n255\n" + bytes([255]) * (64 * 64))

    prepared_mask = tmp_path / "mask.mkv"
    metadata = prepare_video_mask(mask, source, prepared_mask)
    assert metadata["kind"] == "static"
    assert metadata["frames"] == 9
    assert probe_video_info(prepared_mask)["frames"] == 9

    outpaint = prepare_outpaint_inputs(
        source, tmp_path / "outpaint", "right", 50, max_side=128,
    )
    assert outpaint["source_dimensions"] == [64, 64]
    assert outpaint["output_dimensions"] == [128, 64]
    assert outpaint["source_rect"] == [0, 0, 64, 64]
    assert probe_video_info(Path(outpaint["source_path"]))["frames"] == 9
    assert probe_video_info(Path(outpaint["mask_path"]))["frames"] == 9


@pytest.mark.asyncio
async def test_inpainting_api_queues_project_mask_and_candidate(test_settings, tmp_path: Path):
    app = create_app(test_settings, start_workers=False)
    project = app.state.db.create_project(
        CreativeBrief(title="Masked edit", topic="Replace an object").model_dump(mode="json")
    )
    app.state.db.replace_plan(project["id"], [{
        "title": "One", "hook": "One", "treatment": "One",
        "shots": [{"prompt": "A person beside a wall", "duration_seconds": 4}],
    }])
    shot = app.state.db.list_shots(project["id"])[0]
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    candidate = app.state.db.create_candidate(
        project["id"], shot["id"], shot["prompt"], 9,
        {"width": 448, "height": 256, "duration_seconds": 4, "fps": 24},
    )
    app.state.db.update_candidate(
        candidate["id"], status="generated", artifact_json={"local_path": str(source)},
    )
    mask_path = tmp_path / "mask.png"
    mask_path.write_bytes(b"mask")
    mask = app.state.db.create_asset(
        project["id"], "image", "mask.png", str(mask_path), "image/png", {},
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        response = await client.post(
            "/creative-lab/transforms",
            json={
                "mode": "in-outpainting", "operation": "inpaint",
                "candidate_id": candidate["id"], "mask_asset_id": mask["id"],
                "target_id": "primary", "prompt": "a plain wall",
                "mask_dilation": 10,
            },
        )

    assert response.status_code == 202, response.text
    job = response.json()["job"]
    assert job["kind"] == "creative_transform"
    assert job["lane"] == "comfy_upload"
    assert job["candidate_id"] == candidate["id"]
    assert job["payload"]["operation"] == "inpaint"
    assert job["payload"]["mask_asset_id"] == mask["id"]
    assert job["payload"]["mask_path"] == str(mask_path)
    assert job["payload"]["mask_dilation"] == 10


class StubInOutpaintAdapter:
    max_side = 1024

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def validate_source(self, info: dict[str, Any]) -> None:
        assert info["frames"] == 121

    def build(self, **kwargs: Any):
        self.kwargs = kwargs
        return {}, {"engine": "In/Outpaint test", "audio_mode": "source-restored-locally"}


class InOutpaintClient:
    worker_id = "in-outpaint"
    base_url = "http://comfy.test:8188"

    def __init__(self) -> None:
        self.uploads: list[str] = []

    async def upload(self, path: Path) -> str:
        self.uploads.append(path.name)
        return path.name

    async def queue_and_wait(self, _graph, **kwargs):
        await kwargs["on_queued"]("remote-in-outpaint")
        await kwargs["progress"](.5, "sampler")
        job_id = kwargs["client_id"].removesuffix("-in-outpaint")
        return {"files": [{
            "filename": "in-outpaint_00001_.mp4",
            "subfolder": f"vbg/creative-lab/{job_id}", "type": "output",
        }]}

    async def download(self, _artifact, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"masked video")


@pytest.mark.asyncio
async def test_in_outpaint_worker_persists_downloadable_masked_output(
    test_settings, tmp_path: Path, monkeypatch,
):
    db = Database(test_settings.database_path)
    db.initialize()
    source = tmp_path / "source.mp4"
    mask = tmp_path / "mask.png"
    destination = tmp_path / "output" / "masked.mp4"
    source.write_bytes(b"source")
    mask.write_bytes(b"mask")
    job = db.create_job(
        "creative_transform", "comfy_upload",
        {
            "mode": "in-outpainting", "operation": "inpaint",
            "target_id": "primary", "execution_target": "primary",
            "source_path": str(source), "mask_path": str(mask),
            "destination_path": str(destination), "prompt": "a clean wall",
            "strength": 1.0, "mask_dilation": 11, "seed": 4,
        },
        max_attempts=1,
    )
    claimed = db.claim_generation_job(
        "comfy:in-outpaint", upload_capable=True, exclusive_capable=False,
        target_id="primary",
    )
    assert claimed and claimed["id"] == job["id"]

    def prepare(_mask: Path, _source: Path, prepared: Path):
        prepared.parent.mkdir(parents=True, exist_ok=True)
        prepared.write_bytes(b"prepared mask")
        return {"path": str(prepared), "kind": "static", "frames": 121}

    async def immediate(function, *args):
        return function(*args)

    monkeypatch.setattr("apps.api.jobs.prepare_video_mask", prepare)
    monkeypatch.setattr(
        "apps.api.jobs.probe_video_info",
        lambda _path: {"width": 768, "height": 448, "frames": 121, "fps": 24.0},
    )
    monkeypatch.setattr("apps.api.jobs.probe_stream_types", lambda _path: {"video"})
    monkeypatch.setattr("apps.api.jobs.asyncio.to_thread", immediate)
    adapter = StubInOutpaintAdapter()
    client = InOutpaintClient()
    worker = GenerationWorker(
        db, test_settings, None, client,  # type: ignore[arg-type]
        in_outpaint_adapter=adapter,  # type: ignore[arg-type]
    )

    await asyncio.wait_for(worker._execute(claimed), timeout=2)

    saved = db.get_job(job["id"])
    assert saved and saved["status"] == "succeeded"
    assert destination.read_bytes() == b"masked video"
    assert saved["result"]["operation"] == "inpaint"
    assert saved["result"]["source_audio_restored"] is False
    assert client.uploads == ["source.mp4", "inpaint-mask.mkv"]
    assert adapter.kwargs and adapter.kwargs["mask_name"] == "inpaint-mask.mkv"
    assert adapter.kwargs["dilation"] == 11
