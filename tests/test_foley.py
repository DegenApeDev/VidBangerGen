from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from apps.api.database import Database
from apps.api.foley import FOLEY_NEGATIVE, FoleyAdapter
from apps.api.jobs import GenerationWorker


def test_foley_graph_freezes_video_and_generates_only_audio():
    graph, metadata = FoleyAdapter().build(
        video_name="source.mp4",
        prompt='A worker says "read the whole prompt" and strikes a steel anvil.',
        seed=42,
        duration_seconds=5,
        filename_prefix="vbg/test/foley",
    )

    assert graph["8"]["inputs"]["unet_name"] == "ltx-2.3-22b-dev-Q4_K_M.gguf"
    assert graph["9"]["inputs"]["lora_name"] == (
        "ltx-2.3-22b-lora-foley-v2a-1.0.safetensors"
    )
    mask = graph["18"]["inputs"]
    assert mask["mask_video"] is False
    assert mask["mask_init_value_video"] == 0
    assert mask["mask_audio"] is True
    assert graph["19"]["inputs"]["modality"] == "AUDIO"
    assert graph["19"]["inputs"]["cfg"] == 6
    assert graph["20"]["inputs"]["skip_blocks"] == "29"
    assert graph["23"]["inputs"]["steps"] == 30
    assert graph["27"]["class_type"] == "SaveAudio"
    assert not any(node["class_type"] in {"CreateVideo", "SaveVideo"} for node in graph.values())
    assert "read the whole prompt" not in graph["11"]["inputs"]["text"]
    assert "No speech is present. No music is present." in graph["11"]["inputs"]["text"]
    assert "speech" in graph["12"]["inputs"]["text"]
    assert graph["12"]["inputs"]["text"] == FOLEY_NEGATIVE
    assert metadata["source_video_policy"] == "bitstream-copied-locally"
    assert metadata["speech_allowed"] is False
    assert metadata["seed"] == 42


def test_foley_rejects_out_of_recipe_strength():
    with pytest.raises(ValueError, match="between 0.8 and 1.0"):
        FoleyAdapter().build(
            video_name="source.mp4", prompt="Shoes cross a wood floor.", strength=1.1,
        )


class FoleyTestAdapter:
    max_frames = 121

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def build(self, **kwargs: Any):
        self.kwargs = kwargs
        return {}, {"engine": "Test Foley", "source_audio_policy": "replace"}


class FoleyClient:
    worker_id = "foley"
    base_url = "http://comfy.test:8188"

    async def upload(self, path: Path) -> str:
        return path.name

    async def queue_and_wait(self, _graph, **kwargs):
        await kwargs["on_queued"]("remote-foley")
        await kwargs["progress"](.5, "audio sampler")
        return {
            "files": [{
                "filename": "foley_00001_.flac",
                "subfolder": f"vbg/creative-lab/{kwargs['client_id'].removesuffix('-foley')}",
                "type": "output",
            }]
        }

    async def download(self, _artifact, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"lossless foley")


@pytest.mark.asyncio
async def test_foley_worker_muxes_audio_without_reencoding_source_video(
    test_settings, tmp_path: Path, monkeypatch,
):
    db = Database(test_settings.database_path)
    db.initialize()
    source = tmp_path / "silent.mp4"
    destination = tmp_path / "with-foley.mp4"
    source.write_bytes(b"source video")
    job = db.create_job(
        "creative_transform", "comfy_upload",
        {
            "mode": "foley-v2a", "target_id": "primary",
            "execution_target": "primary", "source_path": str(source),
            "destination_path": str(destination), "strength": 1.0, "seed": 7,
            "prompt": "Boots land on wet concrete.",
        },
        max_attempts=1,
    )
    claimed = db.claim_generation_job(
        "comfy:foley", upload_capable=True, exclusive_capable=False,
        target_id="primary",
    )
    assert claimed and claimed["id"] == job["id"]
    monkeypatch.setattr(
        "apps.api.jobs.probe_video_info",
        lambda _path: {"width": 768, "height": 448, "frames": 121, "fps": 24.0},
    )
    monkeypatch.setattr("apps.api.jobs.probe_duration", lambda _path: 5.0)

    def fake_mux(_source: Path, audio: Path, output: Path):
        assert audio.read_bytes() == b"lossless foley"
        output.write_bytes(b"source video plus foley")
        return {
            "video_codec": "copy", "video_frames_preserved": True,
            "source_audio_replaced": False,
        }

    monkeypatch.setattr("apps.api.jobs.mux_generated_foley", fake_mux)

    async def immediate(function, *args):
        return function(*args)

    monkeypatch.setattr("apps.api.jobs.asyncio.to_thread", immediate)
    foley = FoleyTestAdapter()
    worker = GenerationWorker(
        db, test_settings, None, FoleyClient(),  # type: ignore[arg-type]
        foley_adapter=foley,  # type: ignore[arg-type]
    )

    await asyncio.wait_for(worker._execute(claimed), timeout=2)

    saved = db.get_job(job["id"])
    assert saved and saved["status"] == "succeeded"
    assert saved["result"]["foley_mux"]["video_codec"] == "copy"
    assert saved["result"]["source_audio_replaced"] is False
    assert destination.read_bytes() == b"source video plus foley"
    assert foley.kwargs and foley.kwargs["prompt"] == "Boots land on wet concrete."
