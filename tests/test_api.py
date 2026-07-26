from __future__ import annotations

import base64
import asyncio
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from apps.api.creative import CreativeDirector
from apps.api.database import Database
from apps.api.main import _continuation_safe_size, create_app
from apps.api.schemas import CandidateBatchRequest, CreativeBrief


def test_continuation_canvas_caps_motion_guide_memory_without_upscaling_small_inputs():
    assert _continuation_safe_size(640, 360) == (448, 256)
    assert _continuation_safe_size(360, 640) == (256, 448)
    assert _continuation_safe_size(512, 512) == (320, 320)
    assert _continuation_safe_size(384, 256) == (384, 256)


def test_app_construction_does_not_recover_live_jobs(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = db.create_project(
        CreativeBrief(title="Live owner", topic="A live job must stay running").model_dump(
            mode="json"
        )
    )
    job = db.create_job("creative_plan", "local", {}, project_id=project["id"])
    assert db.claim_job("local", "already-running-worker")["id"] == job["id"]

    create_app(test_settings, start_workers=False)

    assert db.get_job(job["id"])["status"] == "running"


@pytest.mark.asyncio
async def test_project_api_and_job_controls(test_settings):
    app = create_app(test_settings, start_workers=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post(
            "/projects",
            json={
                "brief": {
                    "title": "Neon bakery",
                    "topic": "A croissant unfolds into a glowing miniature city",
                    "platform": "reels",
                    "duration_seconds": 15,
                }
            },
        )
        assert created.status_code == 201, created.text
        project = created.json()
        assert project["brief"]["aspect_ratio"] == "9:16"

        plan = await client.post(f"/projects/{project['id']}/plan", json={"concept_count": 3})
        assert plan.status_code == 202
        job_id = plan.json()["id"]
        assert (await client.get(f"/jobs/{job_id}")).json()["status"] == "queued"
        assert (await client.post(f"/jobs/{job_id}/cancel")).status_code == 202
        assert (await client.get(f"/jobs/{job_id}")).json()["status"] == "cancelled"
        assert (await client.post(f"/jobs/{job_id}/retry")).status_code == 202
        assert (await client.get(f"/jobs/{job_id}")).json()["status"] == "queued"

        expanded = await client.get(f"/projects/{project['id']}")
        assert expanded.status_code == 200
        assert expanded.json()["title"] == "Neon bakery"

        lab = await client.get("/creative-lab")
        assert lab.status_code == 200
        ingredients = next(
            mode for mode in lab.json()["modes"] if mode["id"] == "ingredients"
        )
        assert ingredients["ready"] is True
        assert ingredients["missing_capabilities"] == []
        assert lab.json()["active_pipeline"]["upscale_factor"] == 2
        assert lab.json()["active_pipeline"]["upscale_steps"] == 3
        assert lab.json()["vision_scoring_enabled"] is False
        assert lab.json()["recommended_next_mode"] == "ingredients"
        night = next(
            mode for mode in lab.json()["modes"] if mode["id"] == "day-to-night"
        )
        assert night["ready"] is True
        assert night["readiness_state"] == "ready"
        foley = next(
            mode for mode in lab.json()["modes"] if mode["id"] == "foley-v2a"
        )
        assert foley["ready"] is True
        assert foley["readiness_state"] == "ready"
        water = next(
            mode for mode in lab.json()["modes"] if mode["id"] == "water-simulation"
        )
        shave = next(
            mode for mode in lab.json()["modes"] if mode["id"] == "instant-shave"
        )
        cross_eyed = next(
            mode for mode in lab.json()["modes"] if mode["id"] == "cross-eyed"
        )
        assert water["ready"] is True
        assert shave["ready"] is True
        assert cross_eyed["ready"] is True
        cinemagraph = next(
            mode for mode in lab.json()["modes"] if mode["id"] == "cinemagraph"
        )
        assert cinemagraph["ready"] is True
        assert lab.json()["model_inventory"]["read_only"] is True
        assert {
            mode["id"] for mode in lab.json()["modes"] if mode["collection_member"]
        } == {
            "ingredients", "in-outpainting", "pixel-spatial-upscaler",
            "water-simulation", "deblur", "decompression", "cross-eyed",
            "instant-shave", "colorization", "day-to-night", "cinemagraph",
            "foley-v2a", "clean-plate",
        }
        assert lab.json()["companion_modes"][0]["id"] == "lipdub"
        assert lab.json()["companion_modes"][0]["ready"] is True
        assert lab.json()["companion_modes"][0]["readiness_state"] == "ready"

        # Storyboard editing validates concept ownership and persists new shots.
        app.state.db.replace_plan(
            project["id"],
            [{
                "title": "Only concept", "hook": "Hook", "treatment": "Treatment",
                "shots": [{"prompt": "Opening", "duration_seconds": 1}],
            }],
        )
        concept_id = app.state.db.list_concepts(project["id"])[0]["id"]
        estimate = await client.get(
            f"/projects/{project['id']}/generation-estimate?candidates_per_shot=4"
        )
        assert estimate.status_code == 200, estimate.text
        assert estimate.json()["render_count"] == 4
        assert estimate.json()["configured_workers"] == 1
        assert estimate.json()["estimated_wall_minutes"] > 0
        added = await client.post(
            f"/projects/{project['id']}/concepts/{concept_id}/shots",
            json={"prompt": "A second beat", "duration_seconds": 1, "purpose": "payoff"},
        )
        assert added.status_code == 201, added.text
        assert added.json()["position"] == 1

        edited = await client.patch(
            f"/shots/{added.json()['id']}",
            json={
                "title": "Bigger payoff", "prompt": "The city erupts into warm light",
                "duration_seconds": 2.5, "camera": "slow crane reveal",
            },
        )
        assert edited.status_code == 200, edited.text
        assert edited.json()["prompt"] == "The city erupts into warm light"
        assert edited.json()["duration_seconds"] == 2.5
        assert edited.json()["data"]["title"] == "Bigger payoff"
        assert edited.json()["data"]["camera"] == "slow crane reveal"

        app.state.db.create_candidate(
            project["id"], added.json()["id"], "Already rendered", 1, {}
        )
        locked = await client.patch(
            f"/shots/{added.json()['id']}", json={"prompt": "Silent mismatch"}
        )
        assert locked.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["json", "form"])
async def test_direct_t2v_accepts_current_and_legacy_request_formats(
    test_settings, monkeypatch, encoding
):
    app = create_app(test_settings, start_workers=True)
    queued: list[dict] = []

    async def healthy():
        return {"devices": []}

    async def queue(graph):
        queued.append(graph)
        return f"prompt-{encoding}"

    monkeypatch.setattr(app.state.comfy_clients[0], "health", healthy)
    monkeypatch.setattr(app.state.comfy_clients[0], "queue", queue)
    payload = {
        "prompt": "A silver bird takes flight through warm morning haze",
        "negative_prompt": "flicker, text",
        "width": 448,
        "height": 256,
        "duration_seconds": 5,
        "profile": "motion-draft-4x3",
    }
    kwargs = {"json": payload} if encoding == "json" else {"data": payload}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/generate/t2v", **kwargs)

    assert response.status_code == 200, response.text
    assert response.json()["prompt_id"] == f"prompt-{encoding}"
    compiled = queued[0]["4980"]["inputs"]["prompt"]
    assert compiled.startswith(payload["prompt"])
    assert "natural environmental ambience and synchronized scene Foley" in compiled
    assert "zero human voices" in compiled
    assert queued[0]["2483"]["inputs"]["text"] == compiled
    assert "spoken words" in queued[0]["2612"]["inputs"]["text"]
    assert "audio" in queued[0]["4849"]["inputs"]
    assert "2004" not in queued[0]


@pytest.mark.asyncio
async def test_direct_t2v_keeps_native_dialogue_out_of_visual_prose(
    test_settings, monkeypatch
):
    app = create_app(test_settings, start_workers=True)
    queued: list[dict] = []

    async def healthy():
        return {"devices": []}

    async def queue(graph):
        queued.append(graph)
        return "prompt-native-dialogue"

    monkeypatch.setattr(app.state.comfy_clients[0], "health", healthy)
    monkeypatch.setattr(app.state.comfy_clients[0], "queue", queue)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/generate/t2v",
            json={
                "prompt": 'A host faces the lens and says "old inline words." '
                "The camera slowly pushes in.",
                "dialogue": "The future starts here.",
                "audio_mode": "native-dialogue",
                "duration_seconds": 5,
                "width": 448,
                "height": 256,
            },
        )

    assert response.status_code == 200, response.text
    compiled = queued[0]["2483"]["inputs"]["text"]
    assert compiled.count("The future starts here.") == 1
    assert "old inline words" not in compiled
    assert "camera slowly pushes in" in compiled
    assert queued[0]["4849"]["inputs"].get("audio")


@pytest.mark.asyncio
async def test_direct_t2v_blocks_explicit_dialogue_in_ambient_mode_before_queueing(
    test_settings, monkeypatch
):
    app = create_app(test_settings, start_workers=True)
    queued: list[dict] = []

    async def healthy():
        return {"devices": []}

    async def queue(graph):
        queued.append(graph)
        return "must-not-queue"

    monkeypatch.setattr(app.state.comfy_clients[0], "health", healthy)
    monkeypatch.setattr(app.state.comfy_clients[0], "queue", queue)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/generate/t2v",
            json={
                "prompt": (
                    "A hippie points at an AI video on his monitor. "
                    'Speaks only "Far Out Dude" as the room glows warmly.'
                ),
                "audio_mode": "ambient",
                "duration_seconds": 5,
                "width": 448,
                "height": 256,
            },
        )

    assert response.status_code == 422
    assert "Choose Native quoted speech" in response.json()["detail"]
    assert queued == []


@pytest.mark.asyncio
async def test_manual_plan_preserves_exact_prompt_without_ollama_job(test_settings):
    app = create_app(test_settings, start_workers=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post(
            "/projects",
            json={"brief": {
                "title": "Human cut", "topic": "A precise human-authored shot",
                "duration_seconds": 5, "prompt_mode": "manual",
            }},
        )
        project_id = created.json()["id"]
        exact_prompt = "  Chrome bird holds perfectly still.\nThen it takes flight on the beat.  "
        saved = await client.post(
            f"/projects/{project_id}/manual-plan",
            json={
                "shots": [{
                    "title": "Exact opening", "purpose": "hook",
                    "duration_seconds": 5, "prompt": exact_prompt,
                    "negative_prompt": "flicker, text",
                }]
            },
        )
        assert saved.status_code == 201, saved.text
        assert saved.json()["provider"] == "human"
        assert saved.json()["prompt_integrity"] == "exact"
        expanded = (await client.get(f"/projects/{project_id}")).json()
        assert expanded["brief"]["prompt_mode"] == "manual"
        assert expanded["concepts"][0]["shots"][0]["prompt"] == exact_prompt
        assert not any(job["kind"] == "creative_plan" for job in expanded["jobs"])

        bad_duration = await client.post(
            f"/projects/{project_id}/manual-plan",
            json={
                "regenerate": True,
                "shots": [{"duration_seconds": 4, "prompt": "Too short"}],
            },
        )
        assert bad_duration.status_code == 422
        assert "total 4s" in bad_duration.json()["detail"]


@pytest.mark.asyncio
async def test_script_project_queues_one_faithful_storyboard_breakdown(test_settings):
    app = create_app(test_settings, start_workers=False)
    script = """INT. LAB - NIGHT

NIA watches a glass seed unfold into a glowing tree.

NIA
We finally made something alive."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post(
            "/projects",
            json={"brief": {
                "title": "Glass Seed", "topic": "Nia creates a living glass tree",
                "duration_seconds": 15, "source_kind": "script", "script": script,
                "prompt_mode": "manual",
            }},
        )
        assert created.status_code == 201, created.text
        project = created.json()
        assert project["brief"]["source_kind"] == "script"
        assert project["brief"]["prompt_mode"] == "assisted"
        assert project["brief"]["script"] == script

        planned = await client.post(
            f"/projects/{project['id']}/plan", json={"concept_count": 5}
        )
        assert planned.status_code == 202, planned.text
        assert planned.json()["payload"]["concept_count"] == 1


@pytest.mark.asyncio
async def test_generation_preflight_creates_nothing_when_workers_are_offline(
    test_settings, monkeypatch
):
    app = create_app(test_settings, start_workers=True)

    async def offline():
        raise OSError("isolated network")

    for comfy in app.state.comfy_clients:
        monkeypatch.setattr(comfy, "health", offline)
    project = app.state.db.create_project(
        CreativeBrief(title="Offline", topic="Do not create broken work").model_dump(
            mode="json"
        )
    )
    app.state.db.replace_plan(project["id"], [{
        "title": "Exact", "hook": "Exact", "treatment": "Exact",
        "shots": [{"prompt": "No mutation before preflight", "duration_seconds": 5}],
    }])
    shot = app.state.db.list_shots(project["id"])[0]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/projects/{project['id']}/generate", json={"candidates_per_shot": 2}
        )
    assert response.status_code == 503
    assert "No generation work was queued" in response.json()["detail"]
    assert app.state.db.list_candidates(shot["id"]) == []
    assert app.state.db.list_jobs(project["id"]) == []


@pytest.mark.asyncio
async def test_upload_preflight_requires_the_healthy_upload_worker(
    test_settings, monkeypatch
):
    settings = replace(
        test_settings,
        comfyui_urls=("http://shared.test:8188", "http://upload.test:8189"),
        upload_capable_urls=("http://upload.test:8189",),
    )
    app = create_app(settings, start_workers=True)

    async def healthy():
        return {"devices": []}

    async def offline():
        raise OSError("upload worker offline")

    monkeypatch.setattr(app.state.comfy_clients[0], "health", healthy)
    monkeypatch.setattr(app.state.comfy_clients[1], "health", offline)
    project = app.state.db.create_project(
        CreativeBrief(title="Upload gate", topic="Conditioned generation").model_dump(
            mode="json"
        )
    )
    app.state.db.replace_plan(project["id"], [{
        "title": "Exact", "hook": "Exact", "treatment": "Exact",
        "shots": [{"prompt": "Conditioned shot", "duration_seconds": 5}],
    }])
    shot = app.state.db.list_shots(project["id"])[0]
    asset = app.state.db.create_asset(
        project["id"], "image", "anchor.png", "/tmp/anchor.png", "image/png"
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/projects/{project['id']}/generate",
            json={
                "candidates_per_shot": 2,
                "settings": {"reference_image_asset_id": asset["id"]},
            },
        )
    assert response.status_code == 503
    assert "upload-capable ComfyUI worker" in response.json()["detail"]
    assert app.state.db.list_candidates(shot["id"]) == []


@pytest.mark.asyncio
async def test_execution_targets_are_capability_gated_before_jobs_are_created(
    test_settings, monkeypatch
):
    settings = replace(test_settings, local_upscaler_binary="definitely-not-video2x")
    app = create_app(settings, start_workers=False)

    async def healthy():
        return {"devices": []}

    monkeypatch.setattr(app.state.comfy_clients[0], "health", healthy)
    project = app.state.db.create_project(
        CreativeBrief(title="Routing", topic="Route this render safely").model_dump(
            mode="json"
        )
    )
    app.state.db.replace_plan(project["id"], [{
        "title": "One", "hook": "One", "treatment": "One",
        "shots": [{"prompt": "A clean camera move", "duration_seconds": 5}],
    }])
    shot = app.state.db.list_shots(project["id"])[0]
    before = len(app.state.db.list_jobs(project["id"]))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        targets = await client.get("/execution-targets")
        assert targets.status_code == 200, targets.text
        local = next(value for value in targets.json()["targets"] if value["id"] == "local")
        assert local["available"] is False
        assert local["capabilities"] == ["post-upscale"]

        rejected = await client.post(
            f"/projects/{project['id']}/generate",
            json={"settings": {"execution_target": "local"}},
        )
        assert rejected.status_code == 422
        assert "cannot run ltx generation" in rejected.json()["detail"]
        unavailable_upscale = await client.post(
            "/upscales",
            json={"remote_filename": "vbg/example.mp4", "target_id": "local", "scale": 2},
        )
        assert unavailable_upscale.status_code == 409
        assert "not installed" in unavailable_upscale.json()["detail"]
    assert len(app.state.db.list_jobs(project["id"])) == before
    assert app.state.db.list_candidates(shot["id"]) == []


@pytest.mark.asyncio
async def test_project_generation_queues_official_ingredients_candidates(
    test_settings, tmp_path, monkeypatch,
):
    app = create_app(test_settings, start_workers=False)

    async def healthy():
        return {"devices": []}

    monkeypatch.setattr(app.state.comfy_clients[0], "health", healthy)
    project = app.state.db.create_project(
        CreativeBrief(
            title="Ingredients", topic="Keep one product consistent", duration_seconds=5,
        ).model_dump(mode="json")
    )
    app.state.db.replace_plan(project["id"], [{
        "title": "One", "hook": "One", "treatment": "One",
        "shots": [{"prompt": "A red bottle rotates on a counter", "duration_seconds": 5}],
    }])
    shot = app.state.db.list_shots(project["id"])[0]
    sheet_path = tmp_path / "bible.png"
    sheet_path.write_bytes(b"image")
    sheet = app.state.db.create_asset(
        project["id"], "reference_sheet", "bible.png", str(sheet_path), "image/png",
        {"notes": "A red bottle, exact white roundel, green cap, and clean store aisle."},
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/projects/{project['id']}/generate",
            json={
                "shot_ids": [shot["id"]], "candidates_per_shot": 1,
                "settings": {
                    "reference_engine": "ingredients",
                    "reference_image_asset_id": sheet["id"],
                    "reference_mode": "every-shot", "audio_mode": "silent",
                    "execution_target": "primary",
                },
            },
        )

    assert response.status_code == 202, response.text
    job = app.state.db.get_job(response.json()["job_ids"][0])
    candidate = app.state.db.get_candidate(job["candidate_id"])
    assert job["kind"] == "ingredients_generate"
    assert job["lane"] == "comfy_upload"
    assert job["payload"]["reference_image_path"] == str(sheet_path)
    assert candidate["draft"] is False
    assert candidate["settings"]["render_bucket"]["frames"] == 121


@pytest.mark.asyncio
async def test_primary_pixel_upscale_is_queued_for_the_upload_worker(
    test_settings, tmp_path
):
    app = create_app(test_settings, start_workers=False)
    project = app.state.db.create_project(
        CreativeBrief(title="Pixel pass", topic="Enhance an approved clip").model_dump(
            mode="json"
        )
    )
    app.state.db.replace_plan(project["id"], [{
        "title": "One", "hook": "One", "treatment": "One",
        "shots": [{"prompt": "A polished cinematic robot shot", "duration_seconds": 5}],
    }])
    shot = app.state.db.list_shots(project["id"])[0]
    source = tmp_path / "approved.mp4"
    source.write_bytes(b"video")
    candidate = app.state.db.create_candidate(
        project["id"], shot["id"], shot["prompt"], 42,
        {"width": 384, "height": 256, "duration_seconds": 5, "fps": 24},
    )
    app.state.db.update_candidate(
        candidate["id"], status="generated", artifact_json={"local_path": str(source)}
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/upscales",
            json={"candidate_id": candidate["id"], "target_id": "primary", "scale": 4},
        )

    assert response.status_code == 202, response.text
    job = response.json()["job"]
    assert job["lane"] == "comfy_upload"
    assert job["payload"]["execution_target"] == "primary"
    assert job["payload"]["prompt"] == shot["prompt"]
    assert job["payload"]["scale"] == 4


@pytest.mark.asyncio
async def test_merged_chain_can_be_queued_for_one_final_pixel_upscale(
    test_settings, tmp_path
):
    app = create_app(test_settings, start_workers=False)
    opening = tmp_path / "opening.mp4"
    continuation = tmp_path / "continuation.mp4"
    merged = tmp_path / "merged.mp4"
    opening.write_bytes(b"opening")
    continuation.write_bytes(b"continuation")
    merged.write_bytes(b"merged-video")
    chain = app.state.db.create_chain(
        prompt="Opening camera move", local_path=str(opening)
    )
    app.state.db.add_chain_clip(
        chain["id"], "Continuation camera move", 0.7,
        status="done", local_path=str(continuation),
    )
    app.state.db.update_chain(
        chain["id"], status="merged", merged_path=str(merged)
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/upscales",
            json={"chain_id": chain["id"], "target_id": "primary", "scale": 2},
        )

    assert response.status_code == 202, response.text
    job = response.json()["job"]
    assert job["lane"] == "comfy_upload"
    assert job["payload"]["source_path"] == str(merged)
    assert job["payload"]["chain_id"] == chain["id"]
    assert job["payload"]["scale"] == 2
    assert job["payload"]["prompt"] == (
        "Opening camera move Continuation camera move"
    )


@pytest.mark.asyncio
async def test_timed_out_single_pass_upscale_can_recover_video_and_audio(
    test_settings, tmp_path, monkeypatch
):
    app = create_app(test_settings, start_workers=False)
    source = tmp_path / "source.mp4"
    destination = tmp_path / "recovered.mp4"
    source.write_bytes(b"source-with-audio")
    job = app.state.db.create_job(
        "upscale", "comfy_upload",
        {
            "target_id": "primary", "scale": 2,
            "source_path": str(source), "destination_path": str(destination),
        },
        max_attempts=1,
    )
    app.state.db.update_job(
        job["id"], status="failed", error="Generation timed out after 1200 seconds"
    )

    async def download(artifact, output):
        assert artifact["filename"] == "pass-000_00001_.mp4"
        assert artifact["subfolder"] == f"vbg/upscales/{job['id']}"
        output.write_bytes(b"remote-video-only")

    def probe(path):
        if path == source:
            return {"frames": 121, "width": 1280, "height": 704, "fps": 24.0}
        assert path == destination
        return {"frames": 121, "width": 1536, "height": 832, "fps": 24.0}

    def restore(video_only, original, output):
        assert original == source
        assert video_only.read_bytes() == b"remote-video-only"
        output.write_bytes(b"video-with-restored-audio")
        video_only.unlink()
        return {"source_audio_restored": True, "audio_codec": "aac"}

    async def immediate(func, *args):
        return func(*args)

    app.state.comfy_clients[0].download = download
    monkeypatch.setattr("apps.api.main.probe_video_info", probe)
    monkeypatch.setattr("apps.api.main.restore_source_audio", restore)
    monkeypatch.setattr("apps.api.main.asyncio.to_thread", immediate)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(f"/jobs/{job['id']}/recover-upscale")

    assert response.status_code == 200, response.text
    recovered = response.json()
    assert recovered["status"] == "succeeded"
    assert recovered["progress"] == 1.0
    assert recovered["result"]["source_audio_restored"] is True
    assert destination.read_bytes() == b"video-with-restored-audio"


@pytest.mark.asyncio
async def test_day_to_night_is_queued_as_isolated_creative_lab_job(
    test_settings, tmp_path, monkeypatch
):
    app = create_app(test_settings, start_workers=False)

    async def healthy():
        return {"devices": []}

    monkeypatch.setattr(app.state.comfy_clients[0], "health", healthy)
    project = app.state.db.create_project(
        CreativeBrief(title="Night pass", topic="Relight an approved clip").model_dump(
            mode="json"
        )
    )
    app.state.db.replace_plan(project["id"], [{
        "title": "One", "hook": "One", "treatment": "One",
        "shots": [{"prompt": "A cafe during the afternoon", "duration_seconds": 4}],
    }])
    shot = app.state.db.list_shots(project["id"])[0]
    source = tmp_path / "day.mp4"
    source.write_bytes(b"video")
    candidate = app.state.db.create_candidate(
        project["id"], shot["id"], shot["prompt"], 9,
        {"width": 448, "height": 256, "duration_seconds": 4, "fps": 24},
    )
    app.state.db.update_candidate(
        candidate["id"], status="generated", artifact_json={"local_path": str(source)}
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/creative-lab/transforms",
            json={
                "mode": "day-to-night", "candidate_id": candidate["id"],
                "target_id": "primary", "strength": .9,
            },
        )

    assert response.status_code == 202, response.text
    job = response.json()["job"]
    assert job["kind"] == "creative_transform"
    assert job["lane"] == "comfy_upload"
    assert job["candidate_id"] == candidate["id"]
    assert job["payload"]["mode"] == "day-to-night"
    assert job["payload"]["prompt"] == ""
    assert job["payload"]["execution_target"] == "primary"


@pytest.mark.asyncio
async def test_foley_job_defaults_to_candidate_visual_action(
    test_settings, tmp_path, monkeypatch,
):
    app = create_app(test_settings, start_workers=False)

    async def healthy():
        return {"devices": []}

    monkeypatch.setattr(app.state.comfy_clients[0], "health", healthy)
    project = app.state.db.create_project(
        CreativeBrief(title="Foley pass", topic="Sound an approved clip").model_dump(
            mode="json"
        )
    )
    app.state.db.replace_plan(project["id"], [{
        "title": "One", "hook": "One", "treatment": "One",
        "shots": [{"prompt": "Heavy boots splash through a shallow puddle", "duration_seconds": 4}],
    }])
    shot = app.state.db.list_shots(project["id"])[0]
    source = tmp_path / "silent.mp4"
    source.write_bytes(b"video")
    candidate = app.state.db.create_candidate(
        project["id"], shot["id"], shot["prompt"], 9,
        {"width": 448, "height": 256, "duration_seconds": 4, "fps": 24},
    )
    app.state.db.update_candidate(
        candidate["id"], status="generated", artifact_json={"local_path": str(source)}
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/creative-lab/transforms",
            json={
                "mode": "foley-v2a", "candidate_id": candidate["id"],
                "target_id": "primary", "strength": 1.0,
            },
        )

    assert response.status_code == 202, response.text
    job = response.json()["job"]
    assert job["kind"] == "creative_transform"
    assert job["payload"]["mode"] == "foley-v2a"
    assert job["payload"]["prompt"] == shot["prompt"]


@pytest.mark.asyncio
async def test_failed_pixel_retry_checks_upload_worker_before_requeue(
    test_settings, monkeypatch
):
    app = create_app(test_settings, start_workers=True)
    failed = app.state.db.create_job(
        "upscale", "comfy_upload",
        {
            "target_id": "primary", "execution_target": "primary", "scale": 2,
            "source_path": "/tmp/source.mp4", "destination_path": "/tmp/output.mp4",
        },
        max_attempts=1, status="failed",
    )

    async def offline():
        raise httpx.ConnectError("worker offline")

    monkeypatch.setattr(app.state.comfy_clients[0], "health", offline)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(f"/jobs/{failed['id']}/retry")

    assert response.status_code == 503
    assert app.state.db.get_job(failed["id"])["status"] == "failed"


@pytest.mark.asyncio
async def test_reference_sheet_assets_require_an_image(test_settings):
    app = create_app(test_settings, start_workers=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        project = (await client.post(
            "/projects",
            json={"brief": {"title": "Bible", "topic": "A recurring mascot"}},
        )).json()
        rejected = await client.post(
            f"/projects/{project['id']}/assets",
            data={"kind": "reference_sheet"},
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )
        assert rejected.status_code == 422

        accepted = await client.post(
            f"/projects/{project['id']}/assets",
            data={"kind": "reference_sheet"},
            files={"file": (
                "visual-bible.png",
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                ),
                "image/png",
            )},
        )
        assert accepted.status_code == 201, accepted.text
        assert accepted.json()["kind"] == "reference_sheet"


@pytest.mark.asyncio
async def test_brand_assets_store_reusable_identity_metadata(test_settings):
    app = create_app(test_settings, start_workers=False)
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        project = (await client.post(
            "/projects",
            json={"brief": {"title": "Product reel", "topic": "A branded bottle reveal"}},
        )).json()
        missing_role = await client.post(
            f"/projects/{project['id']}/assets",
            data={"kind": "brand"},
            files={"file": ("logo.png", png, "image/png")},
        )
        assert missing_role.status_code == 422

        accepted = await client.post(
            f"/projects/{project['id']}/assets",
            data={
                "kind": "brand", "brand_role": "logo", "label": "North mark",
                "notes": "Keep the white leaf and green ring",
            },
            files={"file": ("logo.png", png, "image/png")},
        )
        assert accepted.status_code == 201, accepted.text
        asset = accepted.json()
        assert asset["kind"] == "brand"
        assert asset["metadata"]["brand_role"] == "logo"
        assert asset["metadata"]["label"] == "North mark"

        app.state.db.replace_plan(project["id"], [{
            "title": "Brand", "hook": "Brand", "treatment": "Brand",
            "shots": [{"prompt": "A model raises the branded sign", "duration_seconds": 5}],
        }])
        shot = app.state.db.list_shots(project["id"])[0]
        attached = await client.patch(
            f"/shots/{shot['id']}",
            json={"reference_asset_id": asset["id"], "reference_role": "sign"},
        )
        assert attached.status_code == 200, attached.text
        assert attached.json()["data"]["reference_asset_id"] == asset["id"]
        assert attached.json()["data"]["reference_role"] == "sign"


@pytest.mark.asyncio
async def test_manual_winner_releases_once_and_remains_authoritative(test_settings):
    app = create_app(test_settings, start_workers=False)
    brief = CreativeBrief(
        title="Manual lock", topic="A chrome drone crosses a red laser grid",
        duration_seconds=15,
    ).model_dump(mode="json")
    project = app.state.db.create_project(brief)
    app.state.db.replace_plan(
        project["id"], CreativeDirector(test_settings)._fallback(brief, 1)
    )
    app.state.studio.enqueue_candidates(
        project["id"], CandidateBatchRequest(candidates_per_shot=2)
    )
    shots = app.state.db.list_shots(project["id"])
    candidates = app.state.db.list_candidates(shots[0]["id"])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = await client.post(
            f"/shots/{shots[0]['id']}/select", json={"candidate_id": candidates[0]["id"]}
        )
        assert first.status_code == 200
        assert first.json()["data"]["released_next_jobs"] == 2
        second = await client.post(
            f"/shots/{shots[0]['id']}/select", json={"candidate_id": candidates[1]["id"]}
        )
        assert second.status_code == 200
        assert second.json()["data"]["released_next_jobs"] == 0

    blocked = [
        value for value in app.state.db.list_jobs(project["id"])
        if value["shot_id"] == shots[2]["id"] and value["status"] == "blocked"
    ]
    assert len(blocked) == 2
    for index, candidate in enumerate(candidates):
        app.state.db.update_candidate(
            candidate["id"], status="scored", total_score=80 + index,
            score_json={"subject_check": {"present": True, "identity_consistent": True}},
        )
    assert app.state.db.auto_select_candidate(shots[0]["id"]) is None
    assert app.state.db.get_shot(shots[0]["id"])["selected_candidate_id"] == candidates[1]["id"]


@pytest.mark.asyncio
async def test_validation_rejects_invalid_brief(test_settings):
    app = create_app(test_settings, start_workers=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/projects",
            json={"brief": {"title": "", "topic": "x", "duration_seconds": 2}},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_built_studio_is_served_by_the_api(test_settings):
    app = create_app(test_settings, start_workers=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/studio/")
    assert response.status_code == 200
    assert "VidBangerGen" in response.text


@pytest.mark.asyncio
async def test_analytics_and_legacy_chain_registration(test_settings):
    app = create_app(test_settings, start_workers=False)

    async def no_remote_history():
        return []

    app.state.comfy_clients[0].all_history = no_remote_history
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        chain = await client.post(
            "/chain", json={"video_file": "vbg/legacy/source.mp4", "prompt": "opening"}
        )
        assert chain.status_code == 201, chain.text
        chain_id = chain.json()["id"]
        restored = await client.get(f"/chain/{chain_id}")
        assert restored.status_code == 200
        assert restored.json()["clips"][0]["remote_filename"] == "vbg/legacy/source.mp4"

        history = await client.get("/history?limit=12")
        assert history.status_code == 200
        assert history.json()["generations"] == []
        assert history.json()["chains"][0]["id"] == chain_id
        assert history.json()["chains"][0]["clips"][0]["prompt"] == "opening"

        analytics = await client.get("/analytics")
        assert analytics.status_code == 200
        assert analytics.json()["projects"] == 0


@pytest.mark.asyncio
async def test_chain_clip_review_and_reject_latest_continuation(test_settings, tmp_path):
    app = create_app(test_settings, start_workers=False)
    opening_path = tmp_path / "opening.mp4"
    continuation_path = tmp_path / "continuation.mp4"
    opening_path.write_bytes(b"opening-video")
    continuation_path.write_bytes(b"continuation-video")
    chain = app.state.db.create_chain(
        prompt="Opening", local_path=str(opening_path)
    )
    continuation = app.state.db.add_chain_clip(
        chain["id"], "Next story beat", 0.7, status="done",
        local_path=str(continuation_path),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        preview = await client.get(
            f"/chain/{chain['id']}/clips/{continuation['id']}/output"
        )
        assert preview.status_code == 200
        assert preview.content == b"continuation-video"

        rejected = await client.delete(
            f"/chain/{chain['id']}/clips/{continuation['id']}"
        )
        assert rejected.status_code == 200, rejected.text
        assert rejected.json()["clips"][1]["status"] == "rejected"
        assert rejected.json()["status"] == "ready"

        opening_rejected = await client.delete(
            f"/chain/{chain['id']}/clips/{chain['clips'][0]['id']}"
        )
        assert opening_rejected.status_code == 409


@pytest.mark.asyncio
async def test_chain_get_recovers_completed_continuation_after_local_restart(
    test_settings, tmp_path
):
    app = create_app(test_settings, start_workers=False)
    opening_path = tmp_path / "opening.mp4"
    opening_path.write_bytes(b"opening-video")
    chain = app.state.db.create_chain(prompt="Opening", local_path=str(opening_path))
    continuation = app.state.db.add_chain_clip(
        chain["id"], "Recovered story beat", 0.7,
        status="generating", prompt_id="remote-prompt",
    )

    async def completed_history(prompt_id):
        assert prompt_id == "remote-prompt"
        return {
            "prompt_id": prompt_id,
            "status": "done",
            "files": [
                {"filename": "remote.mp4", "subfolder": "vbg/recovered", "type": "output"}
            ],
        }

    async def download(_artifact, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"recovered-video")

    app.state.comfy_clients[0].history = completed_history
    app.state.comfy_clients[0].download = download

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        recovered = await client.get(f"/chain/{chain['id']}")

    assert recovered.status_code == 200, recovered.text
    clip = recovered.json()["clips"][1]
    assert clip["id"] == continuation["id"]
    assert clip["status"] == "done"
    assert clip["remote_filename"] == "vbg/recovered/remote.mp4"
    assert Path(clip["local_path"]).read_bytes() == b"recovered-video"
    assert clip["metadata"]["remote_recovery"]["mode"] == "history-after-local-restart"


@pytest.mark.asyncio
async def test_quick_chain_can_create_and_download_a_social_finish(
    test_settings, tmp_path, monkeypatch
):
    app = create_app(test_settings, start_workers=False)
    opening_path = tmp_path / "opening.mp4"
    opening_path.write_bytes(b"opening-video")
    chain = app.state.db.create_chain(prompt="Opening", local_path=str(opening_path))

    def finish(clips, destination, options, music, voiceover, logo):
        assert clips == [opening_path]
        assert options["platform"] == "custom"
        assert options["width"] == 720
        assert options["height"] == 1280
        assert options["captions"] == "HOOK\nPAYOFF"
        assert music is voiceover is logo is None
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"finished-social-video")
        return {
            "path": str(destination), "width": 720, "height": 1280,
            "captions_burned": True,
        }

    monkeypatch.setattr("apps.api.jobs.render_timeline", finish)
    async def immediate(func, *args):
        return func(*args)
    monkeypatch.setattr("apps.api.jobs.asyncio.to_thread", immediate)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        rendered = await asyncio.wait_for(
            client.post(
                f"/chain/{chain['id']}/finish",
                files={
                    "platform": (None, "custom"), "width": (None, "720"),
                    "height": (None, "1280"),
                    "captions": (None, "HOOK\nPAYOFF"),
                },
            ),
            timeout=3,
        )
        assert rendered.status_code == 202, rendered.text
        queued = rendered.json()["chain"]
        assert queued["finish_job"]["status"] == "queued"
        assert queued["finished_path"] is None

    job = app.state.db.claim_job("postprocess", "test-finisher")
    assert job and job["id"] == queued["finish_job_id"]
    await app.state.pool.local_worker._execute(job)
    finished = app.state.db.get_chain(chain["id"])
    assert finished["finish_job"]["status"] == "succeeded"
    assert finished["finish_metadata"]["captions_burned"] is True

    assert Path(finished["finished_path"]).read_bytes() == b"finished-social-video"
    download_route = next(
        route for route in app.routes
        if getattr(route, "path", "") == "/chain/{chain_id}/finished"
    )
    download = await download_route.endpoint(chain["id"])
    assert Path(download.path).read_bytes() == b"finished-social-video"

    app.state.db.add_chain_clip(chain["id"], "New beat", 0.7, status="generating")
    assert app.state.db.get_chain(chain["id"])["finished_path"] is None
    assert app.state.db.get_chain(chain["id"])["finish_metadata"] is None
