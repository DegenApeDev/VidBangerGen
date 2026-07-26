from __future__ import annotations

import asyncio
import json

import httpx

from apps.api.config import SETTINGS
from apps.api.workflow import WorkflowStore


REQUIRED_BASE_NODES = {
    "EmptyLTXVLatentVideo",
    "LTXVLatentUpsampler",
    "LTXVTiledVAEDecode",
    "LTXVSetAudioVideoMaskByTime",
    "LTXVAudioVAEEncode",
    "LoadVideo",
    "GetVideoComponents",
    "LTXVAddGuide",
    "LTXVCropGuides",
}
PRODUCTION_NODES = {
    "RTXVideoSuperResolution": "rtx_vssr",
    "VHS_VideoCombine": "vhs_video_combine",
}


async def inspect(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        stats, nodes = await asyncio.gather(
            client.get(f"{url}/system_stats", timeout=15),
            client.get(f"{url}/object_info", timeout=60),
        )
    stats.raise_for_status()
    nodes.raise_for_status()
    registry = nodes.json()
    missing_base = sorted(REQUIRED_BASE_NODES - registry.keys())
    return {
        "url": url,
        "healthy": True,
        "devices": [value.get("name") for value in stats.json().get("devices", [])],
        "missing_base_nodes": missing_base,
        "production_capabilities": {
            capability: node_name in registry
            for node_name, capability in PRODUCTION_NODES.items()
        },
        "audio_time_mask": "LTXVSetAudioVideoMaskByTime" in registry,
        "motion_segment_guide": all(
            name in registry for name in ("LoadVideo", "GetVideoComponents", "LTXVAddGuide")
        ),
        "cloud_flash_vsr_present": "WavespeedFlashVSRNode" in registry,
        "local_rtx_vssr_node_present": "RTXVideoSuperResolution" in registry,
        # Runtime and dependency-path acceptance are recorded separately; node
        # registration alone must never enable the production profile.
        "local_rtx_vssr_verified": False,
    }


async def main() -> int:
    results = []
    for url in SETTINGS.comfyui_urls:
        try:
            results.append(await inspect(url))
        except Exception as exc:
            results.append({"url": url, "healthy": False, "error": str(exc)})
    profiles = {}
    store = WorkflowStore(SETTINGS)
    for name in store.manifest.get("profiles", {}):
        profiles[name] = store.profile_status(name)
    print(json.dumps({
        "workers": results,
        "configured_worker_count": len(SETTINGS.comfyui_urls),
        "dual_worker_ready": len(results) >= 2 and all(value.get("healthy") for value in results),
        "profiles": profiles,
    }, indent=2))
    return 0 if results and all(value.get("healthy") for value in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
