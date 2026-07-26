from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import httpx
from pydantic import ValidationError

from apps.api.creative_transform import CreativeTransformAdapter
from apps.api.database import Database
from apps.api.jobs import GenerationWorker
from apps.api.main import create_app
from apps.api.model_inventory import query_remote_model_inventory, required_model_files
from apps.api.schemas import CreativeBrief, CreativeTransformRequest


def test_day_to_night_graph_uses_installed_gguf_and_full_video_guide():
    graph, metadata = CreativeTransformAdapter().build(
        mode="day-to-night", video_name="day.mp4", seed=42,
        filename_prefix="vbg/test/night",
    )

    assert graph["8"]["inputs"]["unet_name"] == (
        "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"
    )
    assert graph["9"]["inputs"]["lora_name"] == (
        "ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors"
    )
    assert graph["14"]["class_type"] == "LTXAddVideoICLoRAGuide"
    assert graph["14"]["inputs"]["latent_downscale_factor"] == ["9", 1]
    assert graph["22"]["inputs"]["steps"] == 30
    assert graph["27"]["inputs"].get("audio") is None
    assert "Only the lighting changes from day to night" in graph["11"]["inputs"]["text"]
    assert metadata["audio_mode"] == "source-restored-locally"
    assert metadata["seed"] == 42


@pytest.mark.parametrize(
    ("mode", "lora", "trigger", "audio_policy"),
    [
        ("deblur", "ltx-2.3-22b-ic-lora-deblur-0.9.safetensors", "DEBLUR", "preserve"),
        (
            "decompression", "ltx-2.3-22b-ic-lora-decompression-0.9.safetensors",
            "ENHANCE QUALITY", "preserve",
        ),
        (
            "colorization", "ltx-2.3-22b-ic-lora-colorization-0.9.safetensors",
            "COLORIZE", "preserve",
        ),
        (
            "clean-plate", "ltx-2.3-22b-ic-lora-clean-plate-1.0.safetensors",
            "no people", "discard",
        ),
    ],
)
def test_restoration_graphs_use_their_trained_prompt_contract(
    mode: str, lora: str, trigger: str, audio_policy: str,
):
    graph, metadata = CreativeTransformAdapter().build(
        mode=mode, video_name="source.mp4", prompt="a woman beside a red bicycle",
        seed=8, filename_prefix=f"vbg/test/{mode}",
    )

    assert graph["9"]["inputs"]["lora_name"] == lora
    assert trigger.lower() in graph["11"]["inputs"]["text"].lower()
    assert graph["14"]["inputs"]["image"] == ["4", 0]
    assert metadata["source_audio_policy"] == audio_policy
    assert metadata["workflow"] == "isolated-stage-one-gguf"


def test_water_simulation_uses_distilled_fixed_recipe_and_trained_prompt_contract():
    adapter = CreativeTransformAdapter()
    graph, metadata = adapter.build(
        mode="water-simulation", video_name="dry.mp4",
        prompt="a shallow clear stream surges around the subject's boots with white foam",
        seed=1212, filename_prefix="vbg/test/water",
    )

    assert graph["8"]["inputs"]["unet_name"] == (
        "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"
    )
    assert graph["9"]["inputs"]["lora_name"] == (
        "ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors"
    )
    assert graph["9"]["inputs"]["strength_model"] == 1.2
    assert "ADD WATER" in graph["11"]["inputs"]["text"]
    assert "only water-related elements differ" in graph["11"]["inputs"]["text"]
    assert graph["12"]["inputs"]["text"] == ""
    assert graph["19"]["inputs"]["cfg"] == 1
    assert graph["21"]["inputs"]["sampler_name"] == "euler_ancestral_cfg_pp"
    assert graph["22"]["class_type"] == "ManualSigmas"
    assert graph["22"]["inputs"]["sigmas"] == adapter.distilled_sigmas
    assert metadata["recipe"] == "distilled-fixed"
    assert metadata["steps"] == 8
    assert metadata["source_audio_policy"] == "discard"


def test_water_source_contract_requires_24fps_and_ltx_frame_count():
    adapter = CreativeTransformAdapter()
    adapter.validate_source(
        "water-simulation", {"frames": 121, "fps": 24.0, "width": 768, "height": 448}
    )
    with pytest.raises(ValueError, match="24 fps"):
        adapter.validate_source(
            "water-simulation", {"frames": 121, "fps": 25.0}
        )
    with pytest.raises(ValueError, match=r"8n\+1"):
        adapter.validate_source(
            "water-simulation", {"frames": 120, "fps": 24.0}
        )


def test_instant_shave_uses_dev_gguf_stg_and_trigger():
    graph, metadata = CreativeTransformAdapter().build(
        mode="instant-shave", video_name="bearded.mp4", seed=7,
        filename_prefix="vbg/test/shave",
    )

    assert graph["8"]["inputs"]["unet_name"] == "ltx-2.3-22b-dev-Q4_K_M.gguf"
    assert graph["9"]["inputs"]["lora_name"] == (
        "ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors"
    )
    assert graph["11"]["inputs"]["text"].startswith("REMOVEBEARD ")
    assert "no stubble" in graph["11"]["inputs"]["text"]
    assert "beard, mustache, facial hair, stubble" in graph["12"]["inputs"]["text"]
    assert graph["29"] == {
        "class_type": "LTXVApplySTG",
        "inputs": {"model": ["9", 0], "block_indices": "29"},
    }
    assert graph["19"]["class_type"] == "STGGuider"
    assert graph["19"]["inputs"]["model"] == ["29", 0]
    assert graph["19"]["inputs"]["cfg"] == 4
    assert graph["19"]["inputs"]["stg"] == 1
    assert metadata["recipe"] == "dev-stg"
    assert metadata["stg"]["blocks"] == [29]
    assert metadata["source_audio_policy"] == "preserve"


def test_cross_eyed_uses_dev_gguf_stg_and_identity_safe_prompt():
    graph, metadata = CreativeTransformAdapter().build(
        mode="cross-eyed", video_name="portrait.mp4", seed=23,
        filename_prefix="vbg/test/cross-eyed",
    )

    assert graph["8"]["inputs"]["unet_name"] == "ltx-2.3-22b-dev-Q4_K_M.gguf"
    assert graph["9"]["inputs"]["lora_name"] == (
        "ltx-2.3-22b-ic-lora-cross-eyed-0.9.safetensors"
    )
    assert "severe convergent strabismus" in graph["11"]["inputs"]["text"]
    assert "both eyes continuously turned inward" in graph["11"]["inputs"]["text"]
    assert "normal eyes, straight eyes, corrected eyes" in graph["12"]["inputs"]["text"]
    assert graph["29"] == {
        "class_type": "LTXVApplySTG",
        "inputs": {"model": ["9", 0], "block_indices": "29"},
    }
    assert graph["19"]["class_type"] == "STGGuider"
    assert graph["19"]["inputs"]["cfg"] == 4
    assert graph["19"]["inputs"]["stg"] == 1
    assert metadata["recipe"] == "dev-stg"
    assert metadata["stg"]["blocks"] == [29]
    assert metadata["source_audio_policy"] == "preserve"


def test_water_request_requires_effect_prompt_and_defaults_to_trained_strength():
    with pytest.raises(ValidationError, match="water-motion"):
        CreativeTransformRequest(
            mode="water-simulation", candidate_id="candidate", target_id="primary"
        )
    request = CreativeTransformRequest(
        mode="water-simulation", candidate_id="candidate", target_id="primary",
        prompt="rainwater floods the pavement and splashes around the tires",
    )
    assert request.strength == 1.2


def test_transform_request_accepts_exactly_one_processed_job_source():
    request = CreativeTransformRequest(
        mode="foley-v2a", source_job_id="transform-123", target_id="primary",
    )
    assert request.source_job_id == "transform-123"
    with pytest.raises(ValidationError, match="exactly one"):
        CreativeTransformRequest(
            mode="foley-v2a", source_job_id="transform-123",
            candidate_id="candidate-123", target_id="primary",
        )


@pytest.mark.asyncio
async def test_foley_can_chain_from_a_succeeded_water_job(test_settings, tmp_path: Path):
    app = create_app(test_settings, start_workers=False)
    project = app.state.db.create_project(
        CreativeBrief(title="Water pass", topic="Add water then sound").model_dump(mode="json")
    )
    water_path = tmp_path / "water.mp4"
    water_path.write_bytes(b"water video")
    water = app.state.db.create_job(
        "creative_transform", "comfy_upload",
        {
            "mode": "water-simulation", "target_id": "primary",
            "execution_target": "primary", "prompt": "water surges around boots",
            "source_path": str(tmp_path / "dry.mp4"),
            "destination_path": str(water_path),
        },
        project_id=project["id"], max_attempts=1,
    )
    app.state.db.update_job(
        water["id"], status="succeeded", progress=1.0,
        result_json={"local_path": str(water_path), "mode": "water-simulation"},
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        response = await client.post(
            "/creative-lab/transforms",
            json={
                "mode": "foley-v2a", "source_job_id": water["id"],
                "target_id": "primary",
            },
        )

    assert response.status_code == 202, response.text
    foley = response.json()["job"]
    assert foley["payload"]["source_job_id"] == water["id"]
    assert foley["payload"]["source_path"] == str(water_path)
    assert foley["payload"]["prompt"] == "water surges around boots"
    assert foley["project_id"] == project["id"]


def test_remote_inventory_distinguishes_observed_empty_from_unavailable(monkeypatch):
    monkeypatch.setattr(
        "apps.api.model_inventory.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="night.safetensors|123\nfoley.safetensors|456\n",
            stderr="",
        ),
    )
    assert query_remote_model_inventory("user@rig", "/models/loras") == {
        "night.safetensors": 123, "foley.safetensors": 456,
    }
    assert query_remote_model_inventory("user@rig", None) is None
    assert required_model_files({"model_files": {"2x": "two", "4x": "four"}}) == [
        "two", "four",
    ]


class TransformAdapter:
    max_frames = 121

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def build(self, **kwargs: Any):
        self.kwargs = kwargs
        return {}, {"engine": "Test day to night", "source_audio_policy": "preserve"}

    def limits(self, _mode: str):
        return {"max_frames": 121, "max_side": 768}


class TransformClient:
    worker_id = "transform"
    base_url = "http://comfy.test:8188"

    async def upload(self, path: Path) -> str:
        return path.name

    async def queue_and_wait(self, _graph, **kwargs):
        await kwargs["on_queued"]("remote-transform")
        await kwargs["progress"](.5, "sampler")
        return {
            "files": [{
                "filename": "transform_00001_.mp4",
                "subfolder": f"vbg/creative-lab/{kwargs['client_id'].removesuffix('-creative')}",
                "type": "output",
            }]
        }

    async def download(self, _artifact, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"night video")


@pytest.mark.asyncio
async def test_creative_transform_worker_persists_downloadable_output(
    test_settings, tmp_path: Path, monkeypatch
):
    db = Database(test_settings.database_path)
    db.initialize()
    source = tmp_path / "day.mp4"
    destination = tmp_path / "night.mp4"
    source.write_bytes(b"day video")
    job = db.create_job(
        "creative_transform", "comfy_upload",
        {
            "mode": "day-to-night", "target_id": "primary",
            "execution_target": "primary", "source_path": str(source),
            "destination_path": str(destination), "strength": 1.0, "seed": 7,
        },
        max_attempts=1,
    )
    claimed = db.claim_generation_job(
        "comfy:transform", upload_capable=True, exclusive_capable=False,
        target_id="primary",
    )
    assert claimed and claimed["id"] == job["id"]
    monkeypatch.setattr(
        "apps.api.jobs.probe_video_info",
        lambda _path: {"width": 768, "height": 448, "frames": 97, "fps": 24.0},
    )
    monkeypatch.setattr("apps.api.jobs.probe_stream_types", lambda _path: ["video"])
    # This is a worker state-machine test, not an executor integration test.
    # Running the already-stubbed probes inline also keeps pytest-asyncio's
    # strict loop teardown from waiting on a disposable default executor.
    async def immediate(function, *args):
        return function(*args)

    monkeypatch.setattr("apps.api.jobs.asyncio.to_thread", immediate)
    transform = TransformAdapter()
    worker = GenerationWorker(
        db, test_settings, None, TransformClient(),  # type: ignore[arg-type]
        transform_adapter=transform,  # type: ignore[arg-type]
    )

    await asyncio.wait_for(worker._execute(claimed), timeout=2)

    saved = db.get_job(job["id"])
    assert saved and saved["status"] == "succeeded"
    assert saved["result"]["local_path"] == str(destination)
    assert saved["result"]["source_audio_restored"] is False
    assert destination.read_bytes() == b"night video"
    assert transform.kwargs and transform.kwargs["mode"] == "day-to-night"
