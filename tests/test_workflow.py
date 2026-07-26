from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest

from apps.api.config import PROJECT_ROOT
from apps.api.comfy import (
    ComfyClient, ComfyError, ComfyOrphanedPromptError, parse_history_entry,
)
from apps.api.workflow import LTXWorkflowAdapter, WorkflowError, WorkflowStore


def test_default_workflow_is_a_local_versioned_gguf_graph(test_settings):
    manifest_path = PROJECT_ROOT / "apps/api/workflows/ltx23.json"
    store = WorkflowStore(replace(test_settings, workflow_manifest=manifest_path))

    graph = store.load_graph()

    assert store.manifest["source"] == {
        "kind": "file", "path": "ltx23_base_graph.json",
    }
    assert graph["3940"] == {
        "class_type": "UnetLoaderGGUF",
        "inputs": {"unet_name": "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"},
    }
    assert graph["4980"]["inputs"]["api_key"] == ""
    assert graph["4981"]["inputs"]["api_key"] == ""


def test_relative_workflow_path_resolves_from_manifest_directory(
    test_settings, tmp_path: Path,
):
    workflow_dir = tmp_path / "versioned-workflow"
    workflow_dir.mkdir()
    graph = json.loads(
        (PROJECT_ROOT / "apps/api/workflows/ltx23_base_graph.json").read_text()
    )
    (workflow_dir / "base.json").write_text(json.dumps(graph))
    manifest = json.loads(
        (PROJECT_ROOT / "apps/api/workflows/ltx23.json").read_text()
    )
    manifest["source"] = {"kind": "file", "path": "base.json"}
    manifest_path = workflow_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    store = WorkflowStore(replace(test_settings, workflow_manifest=manifest_path))

    assert store.load_graph()["3940"]["class_type"] == "UnetLoaderGGUF"


def test_remote_workflow_sources_are_rejected_without_network_access(test_settings):
    manifest = json.loads(test_settings.workflow_manifest.read_text())
    manifest["source"] = {
        "kind": "ssh", "remote_path": "/srv/comfyui/workflow.json",
    }
    test_settings.workflow_manifest.write_text(json.dumps(manifest))

    with pytest.raises(WorkflowError, match="inference worker"):
        WorkflowStore(test_settings).load_graph()


@pytest.mark.asyncio
async def test_upload_retries_transient_connection_failures(monkeypatch, tmp_path: Path):
    attempts = 0

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"name": "clip.mp4", "subfolder": ""}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise httpx.ConnectError(
                    "All connection attempts failed", request=httpx.Request("POST", url)
                )
            return Response()

    async def no_sleep(_seconds):
        return None

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr("apps.api.comfy.httpx.AsyncClient", Client)
    monkeypatch.setattr("apps.api.comfy.asyncio.sleep", no_sleep)

    uploaded = await ComfyClient("http://comfy.invalid").upload(source)

    assert uploaded == "clip.mp4"
    assert attempts == 3


@pytest.mark.asyncio
async def test_prompt_queue_retries_transient_connection_failures(monkeypatch):
    attempts = 0

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"prompt_id": "queued-after-retry"}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise httpx.ConnectError(
                    "All connection attempts failed", request=httpx.Request("POST", url)
                )
            return Response()

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("apps.api.comfy.httpx.AsyncClient", Client)
    monkeypatch.setattr("apps.api.comfy.asyncio.sleep", no_sleep)

    prompt_id = await ComfyClient("http://comfy.invalid").queue({"1": {}})

    assert prompt_id == "queued-after-retry"
    assert attempts == 3


def test_workflow_applies_all_controls_and_seed(test_settings):
    adapter = LTXWorkflowAdapter(WorkflowStore(test_settings))
    graph, metadata = adapter.build(
        prompt="A chrome hummingbird takes flight",
        negative_prompt="flicker",
        width=360,
        height=640,
        duration_seconds=5,
        fps=24,
        seed=1234,
        filename_prefix="vbg/test/candidate",
    )
    assert graph["3059"]["inputs"] == {"width": 360, "height": 640}
    assert graph["4988"]["inputs"]["value"] == 121
    assert graph["4980"]["inputs"]["prompt"] == "A chrome hummingbird takes flight"
    assert graph["2483"]["inputs"]["text"] == "A chrome hummingbird takes flight"
    assert graph["4981"]["inputs"]["prompt"] == "flicker"
    assert graph["2612"]["inputs"]["text"] == "flicker"
    assert graph["4832"]["inputs"]["noise_seed"] == 1234
    assert graph["4967"]["inputs"]["noise_seed"] == 1235
    assert len(graph["4984"]["inputs"]["sigmas"].split(",")) == 5
    assert len(graph["4985"]["inputs"]["sigmas"].split(",")) == 4
    assert "4987" not in graph
    assert "2004" not in graph
    assert graph["4528"]["inputs"]["video_latent"] == ["3059", 0]
    assert graph["4969"]["inputs"]["video_latent"] == ["4975", 0]
    assert graph["4852"]["inputs"]["filename_prefix"] == "vbg/test/candidate"
    assert metadata["output_width"] == 720
    assert metadata["output_height"] == 1280
    assert metadata["stages"]["motion_steps"] == 4
    assert metadata["stages"]["latent_upscale_steps"] == 3


def test_silent_workflow_structurally_removes_the_audio_track(test_settings):
    adapter = LTXWorkflowAdapter(WorkflowStore(test_settings))
    graph, metadata = adapter.build(
        prompt="A quiet landscape", negative_prompt="speech", width=448, height=256,
        duration_seconds=5, fps=24, seed=4, filename_prefix="vbg/silent",
        include_audio=False,
    )

    assert "audio" not in graph["4849"]["inputs"]
    assert metadata["audio_output"] == "none"


def test_workflow_adds_ic_lora_for_image_conditioning(test_settings):
    adapter = LTXWorkflowAdapter(WorkflowStore(test_settings))
    graph, metadata = adapter.build(
        prompt="Subject continues walking",
        negative_prompt="",
        width=448,
        height=256,
        duration_seconds=3,
        fps=24,
        seed=8,
        filename_prefix="vbg/i2v",
        image_name="input/reference.png",
        ic_lora_strength=0.6,
        image_condition_strength=0.75,
    )
    lora_nodes = [
        (node_id, node) for node_id, node in graph.items()
        if node["class_type"] == "LTXICLoRALoaderModelOnly"
    ]
    assert len(lora_nodes) == 1
    lora_id, lora = lora_nodes[0]
    assert lora["inputs"]["strength_model"] == 0.6
    assert graph["4828"]["inputs"]["model"] == [lora_id, 0]
    assert graph["4964"]["inputs"]["model"] == [lora_id, 0]
    assert graph["4987"]["inputs"]["value"] is False
    assert graph["3159"]["inputs"]["strength"] == 0.75
    assert metadata["mode"] == "i2v"


def test_workflow_preserves_audio_seed_and_generates_remainder(test_settings):
    adapter = LTXWorkflowAdapter(WorkflowStore(test_settings))
    graph, metadata = adapter.build(
        prompt="Speaker continues the sentence", negative_prompt="", width=256, height=448,
        duration_seconds=5, fps=24, seed=10, filename_prefix="vbg/audio",
        audio_name="voice_seed.wav", audio_seed_seconds=3.25,
    )
    mask_nodes = [
        (node_id, node) for node_id, node in graph.items()
        if node["class_type"] == "LTXVSetAudioVideoMaskByTime"
    ]
    assert len(mask_nodes) == 1
    mask_id, mask = mask_nodes[0]
    assert mask["inputs"]["start_time"] == 3.25
    assert mask["inputs"]["mask_video"] is False
    assert mask["inputs"]["mask_init_value_video"] == 1.0
    assert mask["inputs"]["mask_audio"] is True
    assert mask["inputs"]["mask_init_value_audio"] == 0.0
    assert graph["4828"]["inputs"]["positive"] == [mask_id, 0]
    assert graph["4829"]["inputs"]["latent_image"] == [mask_id, 2]
    assert metadata["audio_seed"]["preserved_seconds"] == 3.25


def test_workflow_uses_full_motion_tail_before_last_frame_fallback(test_settings):
    adapter = LTXWorkflowAdapter(WorkflowStore(test_settings))
    graph, metadata = adapter.build(
        prompt="Continue the motion", negative_prompt="", width=448, height=256,
        duration_seconds=3, fps=24, seed=11, filename_prefix="vbg/motion-tail",
        video_name="tail17.mp4", motion_guide_strength=0.8,
    )
    guide_nodes = [
        (node_id, node) for node_id, node in graph.items()
        if node["class_type"] == "LTXVAddGuide"
    ]
    assert len(guide_nodes) == 1
    guide_id, guide = guide_nodes[0]
    assert guide["inputs"]["frame_idx"] == 0
    assert guide["inputs"]["strength"] == 0.8
    assert graph["4528"]["inputs"]["video_latent"] == [guide_id, 2]
    assert graph["4828"]["inputs"]["positive"] == [guide_id, 0]
    crop_nodes = [
        (node_id, node) for node_id, node in graph.items()
        if node["class_type"] == "LTXVCropGuides"
    ]
    assert len(crop_nodes) == 1
    crop_id, crop = crop_nodes[0]
    assert crop["inputs"]["latent"] == ["4845", 0]
    assert graph["4845"]["inputs"]["av_latent"] == ["4829", 0]
    assert graph["4975"]["inputs"]["samples"] == [crop_id, 2]
    assert graph["4964"]["inputs"]["positive"] == [crop_id, 0]
    assert graph["4964"]["inputs"]["negative"] == [crop_id, 1]
    assert graph["4995"]["inputs"]["latents"] == ["4973", 0]
    assert "4987" not in graph
    assert metadata["mode"] == "v2v"
    assert metadata["motion_guide"]["frames"] == 17
    assert metadata["motion_guide"]["crop_stage"] == "post_av_separation_pre_upscale"


def test_workflow_retake_masks_only_requested_av_interval(test_settings):
    adapter = LTXWorkflowAdapter(WorkflowStore(test_settings))
    graph, metadata = adapter.build(
        prompt="Replace the middle action", negative_prompt="", width=256, height=448,
        duration_seconds=5, fps=24, seed=12, filename_prefix="vbg/retake",
        retake_video_name="source.mp4", retake_start_seconds=1.25,
        retake_end_seconds=3.5,
    )
    mask_nodes = [
        (node_id, node) for node_id, node in graph.items()
        if node["class_type"] == "LTXVSetAudioVideoMaskByTime"
    ]
    assert len(mask_nodes) == 1
    mask_id, mask = mask_nodes[0]
    assert mask["inputs"]["start_time"] == 1.25
    assert mask["inputs"]["end_time"] == 3.5
    assert mask["inputs"]["mask_video"] is True
    assert mask["inputs"]["mask_audio"] is True
    assert mask["inputs"]["mask_init_value_video"] == 0.0
    assert mask["inputs"]["mask_init_value_audio"] == 0.0
    assert graph["4829"]["inputs"]["latent_image"] == [mask_id, 2]
    assert metadata["mode"] == "retake"
    assert metadata["retake"]["preserved_outside_interval"] is True


def test_history_parser_understands_comfy_prompt_list():
    entry = {
        "prompt": [
            7,
            "pid",
            {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "actual prompt"}}},
            {"create_time": 1000},
            ["output"],
        ],
        "outputs": {
            "2": {"images": [{"filename": "clip.mp4", "subfolder": "vbg/a", "type": "output"}]}
        },
        "status": {
            "completed": True,
            "messages": [
                ["execution_start", {"timestamp": 1500}],
                ["execution_success", {"timestamp": 3500}],
            ],
        },
    }
    value = parse_history_entry("pid", entry)
    assert value["prompt"] == "actual prompt"
    assert value["created_at_ms"] == 1000
    assert value["elapsed_seconds"] == 2.0
    assert value["files"][0]["subfolder"] == "vbg/a"


def test_history_parser_surfaces_execution_error_messages():
    entry = {
        "prompt": [1, "pid", {}, {"create_time": 1000}, ["output"]],
        "outputs": {},
        "status": {
            "status_str": "error",
            "completed": False,
            "messages": [
                ["execution_start", {"timestamp": 1500}],
                ["execution_error", {
                    "timestamp": 2500,
                    "exception_type": "AttributeError",
                    "exception_message": "NestedTensor has no clone",
                }],
            ],
        },
    }
    value = parse_history_entry("pid", entry)
    assert value["status"] == "error"
    assert value["error"] == "NestedTensor has no clone"
    assert value["elapsed_seconds"] == 1.0


@pytest.mark.asyncio
async def test_history_poll_retries_transient_http_timeout(monkeypatch):
    client = ComfyClient("http://comfy.invalid", timeout_seconds=5)
    calls = 0

    async def transient_history(_prompt_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("busy encoding")
        return {"status": "done", "files": [{"filename": "clip.mp4"}]}

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(client, "history", transient_history)
    monkeypatch.setattr("apps.api.comfy.asyncio.sleep", no_sleep)
    result = await client.wait_for_completion("prompt-id")
    assert calls == 2
    assert result["files"][0]["filename"] == "clip.mp4"


@pytest.mark.asyncio
async def test_execution_error_is_not_treated_as_a_retryable_socket_drop(monkeypatch):
    client = ComfyClient("http://comfy.invalid", timeout_seconds=5)

    class Socket:
        async def recv(self):
            return '{"type":"execution_error","data":{"prompt_id":"pid","exception_message":"bad latent"}}'

    class Connection:
        async def __aenter__(self):
            return Socket()

        async def __aexit__(self, *_args):
            return False

    async def queue(_graph, client_id=None):
        return "pid"

    monkeypatch.setattr("apps.api.comfy.connect", lambda *_args, **_kwargs: Connection())
    monkeypatch.setattr(client, "queue", queue)
    with pytest.raises(ComfyError, match="bad latent"):
        await client.queue_and_wait({}, "client")


@pytest.mark.asyncio
async def test_history_poll_detects_prompt_lost_after_worker_restart(monkeypatch):
    client = ComfyClient(
        "http://comfy.invalid", timeout_seconds=5, orphan_grace_seconds=30
    )

    async def missing_history(_prompt_id):
        return None

    async def empty_queue():
        return {"running": [], "pending": []}

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(client, "history", missing_history)
    monkeypatch.setattr(client, "queue_state", empty_queue)
    monkeypatch.setattr("apps.api.comfy.asyncio.sleep", no_sleep)
    with pytest.raises(ComfyOrphanedPromptError, match="disappeared"):
        await client.wait_for_completion("lost-prompt")


@pytest.mark.asyncio
async def test_history_poll_allows_queue_to_history_commit_grace(monkeypatch):
    client = ComfyClient(
        "http://comfy.invalid", timeout_seconds=5, orphan_grace_seconds=120
    )
    calls = 0

    async def delayed_history(_prompt_id):
        nonlocal calls
        calls += 1
        if calls <= 12:
            return None
        return {"status": "done", "files": [{"filename": "late.mp4"}]}

    async def empty_queue():
        return {"running": [], "pending": []}

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(client, "history", delayed_history)
    monkeypatch.setattr(client, "queue_state", empty_queue)
    monkeypatch.setattr("apps.api.comfy.asyncio.sleep", no_sleep)

    result = await client.wait_for_completion("late-history-prompt")
    assert calls == 13
    assert result["files"][0]["filename"] == "late.mp4"


@pytest.mark.asyncio
async def test_prompt_in_queue_detects_running_and_pending_entries(monkeypatch):
    client = ComfyClient("http://comfy.invalid", timeout_seconds=5)

    async def queue_state():
        return {
            "running": [[4, "running-prompt", {}, {}]],
            "pending": [{"prompt_id": "pending-prompt"}],
        }

    monkeypatch.setattr(client, "queue_state", queue_state)
    assert await client.prompt_in_queue("running-prompt")
    assert await client.prompt_in_queue("pending-prompt")
    assert not await client.prompt_in_queue("missing-prompt")
