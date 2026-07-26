from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from apps.api.database import Database
from apps.api.jobs import GenerationWorker
from apps.api.lipdub import LipDubAdapter
from apps.api.main import create_app
from apps.api.media import prepare_lipdub_source, probe_stream_types, probe_video_info
from apps.api.schemas import CreativeBrief, CreativeTransformRequest


def test_lipdub_graph_uses_source_voice_tokens_and_freezes_generated_audio_stage_two():
    adapter = LipDubAdapter()
    graph, metadata = adapter.build(
        video_name="speaker.mp4",
        scene_prompt="A reporter faces the camera on a quiet street",
        dialogue="Nous construisons l'avenir ensemble.", language="French",
        width=768, height=448, frames=121, fps=24, seed=12,
    )

    assert graph["5"]["inputs"]["unet_name"] == "ltx-2.3-22b-dev-Q4_K_M.gguf"
    assert graph["6"]["inputs"]["lora_name"] == (
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    )
    assert graph["6"]["inputs"]["strength_model"] == 0.5
    assert graph["7"]["inputs"]["lora_name"] == (
        "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"
    )
    assert graph["12"]["inputs"]["latent_downscale_factor"] == 1
    assert graph["14"]["class_type"] == "LTXVAudioVAEEncode"
    assert graph["14"]["inputs"]["audio"] == ["2", 1]
    assert graph["15"]["class_type"] == "LTXVSetAudioRefTokens"
    assert graph["18"]["inputs"]["audio_latent"] == ["17", 0]
    assert graph["22"]["inputs"]["sigmas"] == adapter.stage_one_sigmas
    assert graph["29"]["inputs"]["audio_latent"] == ["24", 1]
    assert graph["31"]["inputs"]["audio_latent"] == ["29", 2]
    assert graph["35"]["inputs"]["sigmas"] == adapter.stage_two_sigmas
    assert graph["40"]["inputs"]["samples"] == ["37", 1]
    assert graph["41"]["inputs"]["audio"] == ["40", 0]
    assert 'says in French: "Nous construisons' in graph["9"]["inputs"]["text"]
    assert metadata["workflow"] == "official-two-stage-lipdub-gguf"
    assert metadata["reference_downscale_factor"] == 1
    assert metadata["stage_two"]["audio"] == "frozen-from-stage-one"
    assert metadata["audio_mode"] == "generated-dialogue-authoritative"


def test_lipdub_request_and_prompt_require_exact_operator_dialogue():
    with pytest.raises(ValidationError, match="exact desired dialogue"):
        CreativeTransformRequest(
            mode="lipdub", candidate_id="candidate", target_id="primary",
        )
    request = CreativeTransformRequest(
        mode="lipdub", candidate_id="candidate", target_id="primary",
        dialogue="自由を守ろう。", language="Japanese",
    )
    assert request.dialogue == "自由を守ろう。"
    prompt = LipDubAdapter.compile_prompt("A woman speaks", request.dialogue, request.language)
    assert prompt.endswith('says in Japanese: "自由を守ろう。"')


def _make_speaking_test_video(path: Path, duration: float = 2.1) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=96x64:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000",
            "-t", str(duration), "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_lipdub_preparation_normalizes_audio_video_to_safe_ltx_bucket(tmp_path: Path):
    source = tmp_path / "speaker.mp4"
    prepared = tmp_path / "prepared.mp4"
    _make_speaking_test_video(source)

    metadata = prepare_lipdub_source(source, prepared)

    assert metadata["width"] == 768
    assert metadata["height"] == 448
    assert metadata["frames"] == 49
    assert metadata["fps"] == 24
    assert probe_video_info(prepared)["frames"] == 49
    assert probe_stream_types(prepared) == {"video", "audio"}


@pytest.mark.asyncio
async def test_lipdub_api_preflights_source_speech_and_queues_exact_line(
    test_settings, tmp_path: Path, monkeypatch,
):
    app = create_app(test_settings, start_workers=False)

    async def healthy():
        return {"devices": []}

    async def immediate(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(app.state.comfy_clients[0], "health", healthy)
    monkeypatch.setattr("apps.api.main.asyncio.to_thread", immediate)
    project = app.state.db.create_project(
        CreativeBrief(title="Dub", topic="Translate a presenter").model_dump(mode="json")
    )
    app.state.db.replace_plan(project["id"], [{
        "title": "One", "hook": "One", "treatment": "One",
        "shots": [{"prompt": "A presenter faces the camera", "duration_seconds": 2}],
    }])
    shot = app.state.db.list_shots(project["id"])[0]
    source = tmp_path / "speaker.mp4"
    _make_speaking_test_video(source)
    candidate = app.state.db.create_candidate(
        project["id"], shot["id"], shot["prompt"], 5,
        {"width": 448, "height": 256, "duration_seconds": 2, "fps": 24},
    )
    app.state.db.update_candidate(
        candidate["id"], status="generated", artifact_json={"local_path": str(source)},
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        response = await asyncio.wait_for(
            client.post(
                "/creative-lab/transforms",
                json={
                    "mode": "lipdub", "candidate_id": candidate["id"],
                    "target_id": "primary", "language": "French",
                    "dialogue": "Nous avançons ensemble.",
                    "prompt": "A presenter faces the camera on a city street",
                },
            ),
            timeout=5,
        )

    assert response.status_code == 202, response.text
    job = response.json()["job"]
    assert job["kind"] == "creative_transform"
    assert job["payload"]["mode"] == "lipdub"
    assert job["payload"]["dialogue"] == "Nous avançons ensemble."
    assert job["payload"]["language"] == "French"
    assert job["payload"]["prompt"].startswith("A presenter")


class StubLipDubAdapter:
    max_frames = 121
    base_landscape_size = (768, 448)

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def validate_source(self, info: dict[str, Any]) -> None:
        assert info["frames"] == 49

    def build(self, **kwargs: Any):
        self.kwargs = kwargs
        return {}, {"engine": "LipDub test", "audio_mode": "generated-dialogue-authoritative"}


class LipDubClient:
    worker_id = "lipdub"
    base_url = "http://comfy.test:8188"

    async def upload(self, path: Path) -> str:
        assert path.name == "lipdub-source.mp4"
        return path.name

    async def queue_and_wait(self, _graph, **kwargs):
        await kwargs["on_queued"]("remote-lipdub")
        await kwargs["progress"](.5, "sampler")
        job_id = kwargs["client_id"].removesuffix("-lipdub")
        return {"files": [{
            "filename": "lipdub_00001_.mp4",
            "subfolder": f"vbg/creative-lab/{job_id}", "type": "output",
        }]}

    async def download(self, _artifact, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"dubbed video")


@pytest.mark.asyncio
async def test_lipdub_worker_persists_generated_dialogue_output(
    test_settings, tmp_path: Path, monkeypatch,
):
    db = Database(test_settings.database_path)
    db.initialize()
    source = tmp_path / "source.mp4"
    destination = tmp_path / "output" / "dubbed.mp4"
    source.write_bytes(b"source")
    job = db.create_job(
        "creative_transform", "comfy_upload",
        {
            "mode": "lipdub", "target_id": "primary",
            "execution_target": "primary", "source_path": str(source),
            "destination_path": str(destination), "prompt": "A reporter speaks",
            "dialogue": "We build the future together.", "language": "English",
            "strength": 1.0, "seed": 5,
        },
        max_attempts=1,
    )
    claimed = db.claim_generation_job(
        "comfy:lipdub", upload_capable=True, exclusive_capable=False,
        target_id="primary",
    )
    assert claimed and claimed["id"] == job["id"]

    def prepare(_source: Path, prepared: Path, **_kwargs):
        prepared.parent.mkdir(parents=True, exist_ok=True)
        prepared.write_bytes(b"prepared")
        return {
            "path": str(prepared), "width": 768, "height": 448,
            "frames": 49, "fps": 24, "duration_seconds": 2.042,
        }

    async def immediate(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("apps.api.jobs.prepare_lipdub_source", prepare)
    monkeypatch.setattr(
        "apps.api.jobs.probe_video_info",
        lambda _path: {"width": 1536, "height": 896, "frames": 49, "fps": 24.0},
    )
    monkeypatch.setattr("apps.api.jobs.asyncio.to_thread", immediate)
    adapter = StubLipDubAdapter()
    worker = GenerationWorker(
        db, test_settings, None, LipDubClient(),  # type: ignore[arg-type]
        lipdub_adapter=adapter,  # type: ignore[arg-type]
    )

    await asyncio.wait_for(worker._execute(claimed), timeout=2)

    saved = db.get_job(job["id"])
    assert saved and saved["status"] == "succeeded"
    assert destination.read_bytes() == b"dubbed video"
    assert saved["result"]["mode"] == "lipdub"
    assert saved["result"]["audio_authority"] == "generated-lipdub-dialogue"
    assert adapter.kwargs and adapter.kwargs["dialogue"] == "We build the future together."
    assert adapter.kwargs["video_name"] == "lipdub-source.mp4"
