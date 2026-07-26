from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.api.config import PROJECT_ROOT, Settings


def minimal_graph() -> dict:
    return {
        "3059": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 384, "height": 256}},
        "4988": {"class_type": "PrimitiveInt", "inputs": {"value": 121}},
        "4989": {"class_type": "PrimitiveFloat", "inputs": {"value": 24.0}},
        "4980": {"class_type": "GemmaAPITextEncode", "inputs": {"prompt": "old"}},
        "2483": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
        "4981": {"class_type": "GemmaAPITextEncode", "inputs": {"prompt": "old negative"}},
        "2612": {"class_type": "CLIPTextEncode", "inputs": {"text": "old negative"}},
        "4832": {"class_type": "RandomNoise", "inputs": {"noise_seed": 43}},
        "4967": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "4984": {"class_type": "ManualSigmas", "inputs": {"sigmas": "old"}},
        "4985": {"class_type": "ManualSigmas", "inputs": {"sigmas": "old"}},
        "2004": {"class_type": "LoadImage", "inputs": {"image": "example.png"}},
        "4990": {"class_type": "ResizeImageMaskNode", "inputs": {}},
        "3336": {"class_type": "LTXVPreprocess", "inputs": {}},
        "4987": {"class_type": "PrimitiveBoolean", "inputs": {"value": True}},
        "3159": {"class_type": "LTXVImgToVideoConditionOnly", "inputs": {"strength": 0.7}},
        "4970": {"class_type": "LTXVImgToVideoConditionOnly", "inputs": {"strength": 1.0}},
        "3940": {"class_type": "UnetLoaderGGUF", "inputs": {}},
        "4828": {"class_type": "CFGGuider", "inputs": {"model": ["3940", 0]}},
        "4964": {"class_type": "CFGGuider", "inputs": {"model": ["3940", 0]}},
        "1241": {"class_type": "LTXVConditioning", "inputs": {}},
        "4528": {"class_type": "LTXVConcatAVLatent", "inputs": {"audio_latent": ["3980", 0]}},
        "4969": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["4970", 0]}},
        "4975": {"class_type": "LTXVLatentUpsampler", "inputs": {}},
        "4829": {"class_type": "SamplerCustomAdvanced", "inputs": {"latent_image": ["4528", 0]}},
        "4845": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["4829", 0]}},
        "4973": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["4971", 0]}},
        "4995": {"class_type": "LTXVTiledVAEDecode", "inputs": {"latents": ["4973", 0]}},
        "4010": {"class_type": "LTXVAudioVAELoader", "inputs": {}},
        "90100": {"class_type": "VAELoader", "inputs": {}},
        "4849": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["4995", 0], "fps": ["4989", 0], "audio": ["4848", 0]},
        },
        "4852": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "output"}},
    }


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(minimal_graph()))
    manifest = json.loads((PROJECT_ROOT / "apps/api/workflows/ltx23.json").read_text())
    manifest["source"] = {"kind": "file", "path": str(graph_path)}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    settings = Settings(
        data_dir=tmp_path,
        database_path=tmp_path / "test.sqlite3",
        artifact_dir=tmp_path / "output",
        upload_dir=tmp_path / "uploads",
        workflow_manifest=manifest_path,
        comfyui_urls=("http://comfy.test:8188",),
        comfyui_remote_workflow="/unused.json",
        ssh_host="nobody@invalid",
        ollama_url="http://127.0.0.1:1",
        director_model="test-director",
        vision_model="test-vision",
        max_upload_bytes=1024 * 1024,
        generation_timeout_seconds=30,
        cors_origins=("http://localhost:5173",),
    )
    settings.ensure_directories()
    return settings
