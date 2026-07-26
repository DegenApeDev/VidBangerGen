from __future__ import annotations

from dataclasses import replace
import json

import pytest

from apps.api.workflow import LTXWorkflowAdapter, WorkflowError, WorkflowStore


def test_production_profile_fails_loudly_until_rig_capabilities_exist(test_settings):
    adapter = LTXWorkflowAdapter(WorkflowStore(test_settings))
    with pytest.raises(WorkflowError, match="rtx_vssr"):
        adapter.build(
            prompt="production shot", negative_prompt="", width=640, height=360,
            duration_seconds=5, fps=24, seed=1, filename_prefix="vbg/production",
            profile="production-4x3x1-vssr",
        )


def test_gguf_final_profile_is_ready_and_keeps_the_q4_loader(test_settings):
    store = WorkflowStore(test_settings)
    assert store.profile_status("quality-final-gguf-4x3")["ready"] is True
    graph, metadata = LTXWorkflowAdapter(store).build(
        prompt="stable quality final", negative_prompt="", width=640, height=360,
        duration_seconds=5, fps=24, seed=1, filename_prefix="vbg/gguf-final",
        profile="quality-final-gguf-4x3",
    )
    assert graph["3940"]["class_type"] == "UnetLoaderGGUF"
    assert metadata["profile"] == "quality-final-gguf-4x3"
    assert metadata["model_loader"]["kind"] == "workflow-default-gguf-q4-k-m"
    assert metadata["stages"]["motion_steps"] == 4
    assert metadata["stages"]["latent_upscale_steps"] == 3


def test_fp8_profile_is_gated_then_swaps_only_the_in_memory_loader(test_settings):
    store = WorkflowStore(test_settings)
    adapter = LTXWorkflowAdapter(store)
    with pytest.raises(WorkflowError, match="fp8_exclusive_inference"):
        adapter.build(
            prompt="quality final", negative_prompt="", width=640, height=360,
            duration_seconds=5, fps=24, seed=1, filename_prefix="vbg/fp8",
            profile="quality-final-fp8-4x3",
        )

    manifest = json.loads(test_settings.workflow_manifest.read_text())
    manifest["available_capabilities"].append("fp8_exclusive_inference")
    test_settings.workflow_manifest.write_text(json.dumps(manifest))
    configured = replace(
        test_settings,
        comfyui_urls=("http://comfy.test:8188", "http://comfy.test:8189"),
        exclusive_comfy_url="http://comfy.test:8188",
        managed_comfy_urls=("http://comfy.test:8189",),
    )
    enabled = LTXWorkflowAdapter(WorkflowStore(configured))
    graph, metadata = enabled.build(
        prompt="quality final", negative_prompt="", width=640, height=360,
        duration_seconds=5, fps=24, seed=1, filename_prefix="vbg/fp8",
        profile="quality-final-fp8-4x3",
    )
    assert graph["3940"]["class_type"] == "CheckpointLoaderSimpleDisTorch2MultiGPU"
    assert graph["3940"]["inputs"] == {
        "ckpt_name": "ltx-2.3-22b-distilled-fp8.safetensors",
        "compute_device": "cuda:0", "virtual_vram_gb": 8.0,
        "donor_device": "cuda:1", "expert_mode_allocations": "",
        "eject_models": True,
    }
    assert metadata["model_loader"]["exclusive_dual_gpu"] is True
