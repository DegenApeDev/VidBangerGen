from __future__ import annotations

import pytest

from apps.api.creative import CreativeDirector
from apps.api.database import Database
from apps.api.schemas import (
    CandidateBatchRequest, CreativeBrief, ExportRequest, GenerationSettings,
)
from apps.api.studio import StudioService
from apps.api.workflow import WorkflowStore


def make_project(db: Database, test_settings):
    brief = CreativeBrief(
        title="Coffee robot",
        topic="A tiny robot discovers espresso and moves at impossible speed",
        duration_seconds=15,
    ).model_dump(mode="json")
    project = db.create_project(brief)
    concepts = CreativeDirector(test_settings)._fallback(brief, 3)
    db.replace_plan(project["id"], concepts)
    return project


def test_candidate_batch_is_sequential_and_persistent(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    result = StudioService(db).enqueue_candidates(
        project["id"], CandidateBatchRequest(candidates_per_shot=3)
    )
    jobs = db.list_jobs(project["id"])
    assert result["candidate_count"] == 9
    assert len([job for job in jobs if job["status"] == "queued"]) == 3
    assert len([job for job in jobs if job["status"] == "blocked"]) == 6

    claimed = [db.claim_job("comfy", "gpu0") for _ in range(3)]
    assert all(claimed)
    assert db.claim_job("comfy", "gpu1") is None

    first_shot = next(
        shot for shot in db.list_shots(project["id"])
        if db.list_candidates(shot["id"])
    )
    takes = db.list_candidates(first_shot["id"])
    assert [take["settings"]["take_role"] for take in takes] == [
        "clean-baseline", "motion-push", "hook-clarity"
    ]
    assert len({take["prompt"] for take in takes}) == 3
    assert all(first_shot["prompt"] in take["prompt"] for take in takes)

    for index, take in enumerate(takes):
        db.update_candidate(
            take["id"], status="scored", total_score=70 + index * 5,
            score_json={"total": 70 + index * 5},
        )
    db.select_candidate(first_shot["id"], takes[-1]["id"])
    db.create_feedback(project["id"], takes[-1]["id"], 5, "excellent", "best hook")
    roles = db.analytics(project["id"])["take_roles"]
    assert roles[0]["role"] == "hook-clarity"
    assert roles[0]["average_score"] == 80
    assert roles[0]["excellent"] == 1


def test_quick_finish_job_and_chain_link_survive_local_api_restart(
    test_settings, tmp_path
):
    db = Database(test_settings.database_path)
    db.initialize()
    opening = tmp_path / "opening.mp4"
    opening.write_bytes(b"video")
    chain = db.create_chain(prompt="Opening", local_path=str(opening))
    job = db.create_job(
        "chain_finish", "postprocess",
        {
            "chain_id": chain["id"], "clip_paths": [str(opening)],
            "destination_path": str(tmp_path / "finished.mp4"), "options": {},
        },
    )
    db.update_chain_finish_job(chain["id"], job["id"])
    assert db.claim_job("postprocess", "old-api")["id"] == job["id"]

    Database(test_settings.database_path).initialize(recover_running=True)

    recovered = db.get_chain(chain["id"])
    assert recovered["finish_job_id"] == job["id"]
    assert recovered["finish_job"]["status"] == "queued"
    assert recovered["finished_path"] is None


def test_fixed_base_seed_produces_reproducible_distinct_takes(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    StudioService(db).enqueue_candidates(
        project["id"], CandidateBatchRequest(
            candidates_per_shot=3, settings=GenerationSettings(seed=100)
        ),
    )
    first_shot = db.list_shots(project["id"])[0]
    assert [value["seed"] for value in db.list_candidates(first_shot["id"])] == [100, 101, 102]


def test_generation_workers_claim_only_their_explicit_target(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    pinned = db.create_job(
        "generate", "comfy", {"execution_target": "renderbox", "settings": {}}
    )
    automatic = db.create_job(
        "generate", "comfy", {"execution_target": "auto", "settings": {}}
    )

    primary_job = db.claim_generation_job(
        "comfy:primary", upload_capable=True, exclusive_capable=False,
        target_id="primary",
    )
    assert primary_job and primary_job["id"] == automatic["id"]
    renderbox_job = db.claim_generation_job(
        "comfy:renderbox", upload_capable=True, exclusive_capable=False,
        target_id="renderbox",
    )
    assert renderbox_job and renderbox_job["id"] == pinned["id"]


def test_manual_mode_varies_seed_without_rewriting_prompt(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    brief = CreativeBrief(
        title="Exact operator prompt", topic="A human-controlled generation",
        duration_seconds=5, prompt_mode="manual",
    ).model_dump(mode="json")
    project = db.create_project(brief)
    exact_prompt = "A cobalt sphere drops once, splashes, and remains visible."
    db.replace_plan(project["id"], [{
        "title": "Manual", "hook": "Manual", "treatment": "Manual",
        "shots": [{"prompt": exact_prompt, "duration_seconds": 5}],
    }])
    product = db.create_asset(
        project["id"], "brand", "sphere.png", "/tmp/sphere.png", "image/png",
        {"brand_role": "product", "notes": "Keep the cobalt finish"},
    )
    StudioService(db).enqueue_candidates(
        project["id"], CandidateBatchRequest(
            candidates_per_shot=3,
            settings=GenerationSettings(
                seed=700, reference_image_asset_id=product["id"]
            ),
        ),
    )
    shot = db.list_shots(project["id"])[0]
    candidates = db.list_candidates(shot["id"])
    assert [value["seed"] for value in candidates] == [700, 701, 702]
    assert [value["prompt"] for value in candidates] == [exact_prompt] * 3
    assert {value["settings"]["take_role"] for value in candidates} == {
        "manual-seed-variation"
    }


def test_native_dialogue_mode_uses_only_enabled_shot_dialogue(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    brief = CreativeBrief(
        title="Dialogue contract", topic="A reporter delivers one short line",
        duration_seconds=5,
    ).model_dump(mode="json")
    project = db.create_project(brief)
    db.replace_plan(project["id"], [{
        "title": "One speaker", "hook": "Line", "treatment": "Direct",
        "shots": [{
            "prompt": "A reporter looks into the camera on a quiet street.",
            "duration_seconds": 5, "audio_mode": "native-dialogue",
            "dialogue": "We can build this together.", "speaker": "The reporter",
            "language": "English", "accent": "Canadian",
            "audio": "quiet street ambience",
        }],
    }])

    StudioService(db).enqueue_candidates(
        project["id"], CandidateBatchRequest(
            candidates_per_shot=1,
            settings=GenerationSettings(audio_mode="shot"),
        ),
    )

    shot = db.list_shots(project["id"])[0]
    candidate = db.list_candidates(shot["id"])[0]
    assert (
        'speaks one short English line with a Canadian accent: '
        '"We can build this together."'
    ) in candidate["prompt"]
    assert "There is no narrator and no off-screen speaker" in candidate["prompt"]
    assert candidate["settings"]["audio_contract"]["dialogue_words"] == 5
    assert candidate["settings"]["audio_mode"] == "native-dialogue"


def test_studio_rejects_raw_prompt_audio_before_creating_jobs(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)

    with pytest.raises(ValueError, match="gibberish"):
        StudioService(db).enqueue_candidates(
            project["id"], CandidateBatchRequest(
                candidates_per_shot=1,
                settings=GenerationSettings(audio_mode="prompt"),
            ),
        )

    assert db.list_jobs(project["id"]) == []
    assert all(
        db.list_candidates(shot["id"]) == []
        for shot in db.list_shots(project["id"])
    )


def test_unstarted_legacy_studio_prompt_audio_jobs_are_retired(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    shot = db.list_shots(project["id"])[0]
    candidate = db.create_candidate(
        project["id"], shot["id"], "A presenter talking about a product.", 10,
        {"audio_mode": "prompt"}, draft=True,
    )
    job = db.create_job(
        "generate", "comfy",
        {"settings": {"audio_mode": "prompt"}, "execution_target": "primary"},
        project_id=project["id"], shot_id=shot["id"], candidate_id=candidate["id"],
        status="blocked",
    )

    assert db.retire_unsafe_studio_prompt_audio_jobs() == 1
    assert db.retire_unsafe_studio_prompt_audio_jobs() == 0
    retired = db.get_job(job["id"])
    assert retired and retired["status"] == "cancelled"
    assert "Regenerate this draft set" in retired["error"]
    assert db.get_candidate(candidate["id"])["status"] == "cancelled"
    assert db.retry_job(job["id"]) is False


def test_studio_default_resolves_each_shot_audio_intent(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    brief = CreativeBrief(
        title="Mixed audio", topic="A quiet visual story", duration_seconds=10,
    ).model_dump(mode="json")
    project = db.create_project(brief)
    db.replace_plan(project["id"], [{
        "title": "Mixed", "hook": "Mixed", "treatment": "Mixed",
        "shots": [
            {
                "prompt": "A mascot talking about a glowing computer.",
                "duration_seconds": 5, "audio_mode": "ambient",
                "audio": "soft room tone",
            },
            {
                "prompt": "A reporter looks into the lens.",
                "duration_seconds": 5, "audio_mode": "native-dialogue",
                "dialogue": "Local tools belong to everyone.",
                "speaker": "The reporter", "language": "English",
            },
        ],
    }])

    StudioService(db).enqueue_candidates(
        project["id"], CandidateBatchRequest(candidates_per_shot=1),
    )

    shots = db.list_shots(project["id"])
    first = db.list_candidates(shots[0]["id"])[0]
    second = db.list_candidates(shots[1]["id"])[0]
    assert first["settings"]["audio_mode"] == "ambient"
    assert first["settings"]["audio_contract"]["speech_allowed"] is False
    assert "talking about" not in first["settings"]["workflow_prompt"].lower()
    assert second["settings"]["audio_mode"] == "native-dialogue"
    assert second["settings"]["audio_contract"]["dialogue"] == (
        "Local tools belong to everyone."
    )


def test_project_status_reconciles_terminal_and_retried_generation(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    result = StudioService(db).enqueue_candidates(
        project["id"], CandidateBatchRequest(candidates_per_shot=1)
    )
    jobs = [db.get_job(job_id) for job_id in result["job_ids"]]
    for job in jobs:
        assert job and db.request_cancel(job["id"])
    assert db.get_project(project["id"])["status"] == "generation_interrupted"

    assert jobs[0] and db.retry_job(jobs[0]["id"])
    assert db.get_project(project["id"])["status"] == "generating"


def test_project_reference_image_routes_first_shot_to_upload_worker(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    image = db.create_asset(
        project["id"], "image", "anchor.png", "/tmp/anchor.png", "image/png"
    )
    result = StudioService(db).enqueue_candidates(
        project["id"], CandidateBatchRequest(
            candidates_per_shot=2,
            settings=GenerationSettings(reference_image_asset_id=image["id"]),
        ),
    )
    first_jobs = [db.get_job(value) for value in result["first_shot_job_ids"]]
    assert all(value and value["lane"] == "comfy_upload" for value in first_jobs)
    assert all(value and value["payload"]["image_path"] == "/tmp/anchor.png" for value in first_jobs)


def test_brand_reference_can_anchor_every_storyboard_shot(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    brand = db.create_asset(
        project["id"], "brand", "bottle.png", "/tmp/bottle.png", "image/png",
        {"brand_role": "product", "label": "Hero bottle"},
    )
    result = StudioService(db).enqueue_candidates(
        project["id"], CandidateBatchRequest(
            candidates_per_shot=1,
            settings=GenerationSettings(
                reference_image_asset_id=brand["id"], reference_mode="every-shot"
            ),
        ),
    )
    jobs = [db.get_job(job_id) for job_id in result["job_ids"]]
    assert len(jobs) == 3
    assert all(job and job["lane"] == "comfy_upload" for job in jobs)
    assert all(job and job["payload"]["image_path"] == "/tmp/bottle.png" for job in jobs)
    assert jobs[0] and "reference_overrides_continuity" not in jobs[0]["payload"]
    assert all(
        job and job["payload"].get("reference_overrides_continuity") is True
        for job in jobs[1:]
    )
    candidates = [db.get_candidate(job["candidate_id"]) for job in jobs if job]
    assert all(
        candidate and candidate["settings"]["reference_image_asset_id"] == brand["id"]
        for candidate in candidates
    )


def test_project_reference_audio_routes_first_shot_to_upload_worker(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    audio = db.create_asset(
        project["id"], "reference", "voice.wav", "/tmp/voice.wav", "audio/wav"
    )
    result = StudioService(db).enqueue_candidates(
        project["id"], CandidateBatchRequest(
            candidates_per_shot=2,
            settings=GenerationSettings(reference_audio_asset_id=audio["id"]),
        ),
    )
    first_jobs = [db.get_job(value) for value in result["first_shot_job_ids"]]
    assert all(value and value["lane"] == "comfy_upload" for value in first_jobs)


def test_candidate_references_are_validated_before_jobs_are_created(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    wrong_kind = db.create_asset(
        project["id"], "image", "not-audio.png", "/tmp/not-audio.png", "image/png"
    )
    with pytest.raises(ValueError, match="Reference audio asset has the wrong media role"):
        StudioService(db).enqueue_candidates(
            project["id"], CandidateBatchRequest(
                settings=GenerationSettings(reference_audio_asset_id=wrong_kind["id"]),
            ),
        )
    assert db.list_jobs(project["id"]) == []


def test_auto_selection_releases_next_shot(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    StudioService(db).enqueue_candidates(
        project["id"], CandidateBatchRequest(candidates_per_shot=2)
    )
    concept = next(value for value in db.list_concepts(project["id"]) if value["selected"])
    first_shot = db.list_shots(project["id"], concept["id"])[0]
    candidates = db.list_candidates(first_shot["id"])
    for index, candidate in enumerate(candidates):
        db.update_candidate(
            candidate["id"], status="scored", score_json={"total": 70 + index},
            total_score=70 + index,
        )
    selected = db.auto_select_candidate(first_shot["id"])
    assert selected["id"] == candidates[-1]["id"]
    assert db.release_next_shot_jobs(first_shot["id"]) == 2
    jobs = db.list_jobs(project["id"])
    assert len([job for job in jobs if job["status"] == "queued"]) == 4


def test_manual_selection_is_never_overwritten_by_later_scoring(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    StudioService(db).enqueue_candidates(
        project["id"], CandidateBatchRequest(candidates_per_shot=2)
    )
    shot = db.list_shots(project["id"])[0]
    candidates = db.list_candidates(shot["id"])
    db.select_candidate(shot["id"], candidates[0]["id"], "manual")
    for index, candidate in enumerate(candidates):
        db.update_candidate(
            candidate["id"], status="scored", total_score=70 + index * 20,
            score_json={"subject_check": {"present": True, "identity_consistent": True}},
        )
    assert db.auto_select_candidate(shot["id"]) is None
    assert db.get_shot(shot["id"])["selected_candidate_id"] == candidates[0]["id"]


def test_manual_draft_lock_promotes_only_its_verified_final(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    shot = db.list_shots(project["id"])[0]
    chosen = db.create_candidate(project["id"], shot["id"], shot["prompt"], 4, {})
    other = db.create_candidate(project["id"], shot["id"], shot["prompt"], 5, {})
    db.update_candidate(chosen["id"], status="scored", total_score=80)
    db.update_candidate(other["id"], status="scored", total_score=90)
    db.select_candidate(shot["id"], chosen["id"], "manual")
    final = db.create_candidate(
        project["id"], shot["id"], shot["prompt"], 4,
        {"hq_of": chosen["id"]}, draft=False,
    )
    db.update_candidate(
        final["id"], status="scored", total_score=78,
        score_json={
            "judge": "qwen3-vl:8b",
            "subject_check": {"present": True, "identity_consistent": True},
        },
    )
    selected = db.auto_select_candidate(shot["id"])
    assert selected and selected["id"] == final["id"]
    assert db.get_shot(shot["id"])["selection_origin"] == "manual-final"
    assert db.release_next_shot_jobs(shot["id"]) == 0
    assert db.get_project(project["id"])["status"] == "finals_ready"


def test_technical_fallback_requires_human_winner_review(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    shot = db.list_shots(project["id"])[0]
    candidate = db.create_candidate(project["id"], shot["id"], shot["prompt"], 7, {})
    db.update_candidate(
        candidate["id"], status="scored", total_score=73,
        score_json={"judge": "technical-fallback-v1", "total": 73},
    )
    assert db.auto_select_candidate(shot["id"]) is None
    assert db.get_shot(shot["id"])["status"] == "needs_review"
    assert db.get_project(project["id"])["status"] == "candidates_review_required"


def test_startup_migrates_legacy_neutral_73_to_unscored(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    shot = db.list_shots(project["id"])[0]
    candidate = db.create_candidate(project["id"], shot["id"], shot["prompt"], 73, {})
    db.update_candidate(
        candidate["id"], status="scored", total_score=73,
        score_json={"judge": "technical-fallback-v1", "total": 73, "technical_score": 30},
    )

    db.initialize(recover_running=False)
    migrated = db.get_candidate(candidate["id"])
    assert migrated["status"] == "unscored"
    assert migrated["total_score"] is None
    assert migrated["score"]["technical_score"] == 30


def test_winner_renders_fall_back_to_gguf_and_are_idempotent(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    concept = next(value for value in db.list_concepts(project["id"]) if value["selected"])
    for shot in db.list_shots(project["id"], concept["id"]):
        candidate = db.create_candidate(
            project["id"], shot["id"], shot["prompt"], 123,
            GenerationSettings(width=256, height=448).model_dump(mode="json"),
        )
        db.update_candidate(candidate["id"], status="scored", total_score=80)
        db.select_candidate(shot["id"], candidate["id"], "manual")

    service = StudioService(db, WorkflowStore(test_settings))
    result = service.enqueue_winner_renders(project["id"])
    assert result["profile"] == "quality-final-gguf-4x3"
    assert "Q4_K_M GGUF" in result["fallback_reason"]
    assert result["render_count"] == 3
    final_candidates = [
        value for shot in db.list_shots(project["id"], concept["id"])
        for value in db.list_candidates(shot["id"]) if not value["draft"]
    ]
    assert {value["settings"]["profile"] for value in final_candidates} == {
        "quality-final-gguf-4x3"
    }
    with pytest.raises(ValueError, match="already has a final render"):
        service.enqueue_winner_renders(project["id"])


def test_failed_later_final_can_resume_after_selected_previous_final(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    concept = next(value for value in db.list_concepts(project["id"]) if value["selected"])
    shots = db.list_shots(project["id"], concept["id"])
    for shot in shots:
        draft = db.create_candidate(
            project["id"], shot["id"], shot["prompt"], 20,
            GenerationSettings(width=256, height=448).model_dump(mode="json"),
        )
        db.update_candidate(draft["id"], status="scored", total_score=80)
        db.select_candidate(shot["id"], draft["id"], "manual")
    service = StudioService(db, WorkflowStore(test_settings))
    service.enqueue_winner_renders(project["id"])

    first_final = next(value for value in db.list_candidates(shots[0]["id"]) if not value["draft"])
    db.update_candidate(first_final["id"], status="scored", total_score=80)
    db.select_candidate(shots[0]["id"], first_final["id"], "manual")
    second_final = next(value for value in db.list_candidates(shots[1]["id"]) if not value["draft"])
    second_job = db.get_job(second_final["job_id"])
    assert second_job and db.request_cancel(second_job["id"])

    resumed = service.enqueue_winner_renders(project["id"])
    assert resumed["render_count"] == 1
    replacement = db.get_job(resumed["job_ids"][0])
    assert replacement and replacement["shot_id"] == shots[1]["id"]
    assert replacement["status"] == "queued"


def test_bad_or_wrong_subject_candidate_requires_review(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    concept = next(value for value in db.list_concepts(project["id"]) if value["selected"])
    shot = db.list_shots(project["id"], concept["id"])[0]
    candidate = db.create_candidate(project["id"], shot["id"], shot["prompt"], 9, {})
    db.update_candidate(
        candidate["id"], status="scored", total_score=90,
        score_json={"total": 90, "subject_check": {"present": False}},
    )
    assert db.auto_select_candidate(shot["id"]) is None
    refreshed = db.get_shot(shot["id"])
    assert refreshed["status"] == "needs_review"
    assert refreshed["selected_candidate_id"] is None
    assert db.get_project(project["id"])["status"] == "candidates_review_required"


def test_identity_mismatch_is_not_auto_selected(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    concept = next(value for value in db.list_concepts(project["id"]) if value["selected"])
    shot = db.list_shots(project["id"], concept["id"])[0]
    candidate = db.create_candidate(project["id"], shot["id"], shot["prompt"], 10, {})
    db.update_candidate(
        candidate["id"], status="scored", total_score=88,
        score_json={
            "total": 88,
            "subject_check": {"present": True, "identity_consistent": False},
        },
    )
    assert db.auto_select_candidate(shot["id"]) is None
    assert db.get_shot(shot["id"])["status"] == "needs_review"


def test_production_candidate_replaces_higher_scoring_draft(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    concept = next(value for value in db.list_concepts(project["id"]) if value["selected"])
    shot = db.list_shots(project["id"], concept["id"])[0]
    draft = db.create_candidate(project["id"], shot["id"], shot["prompt"], 1, {}, draft=True)
    final = db.create_candidate(project["id"], shot["id"], shot["prompt"], 1, {}, draft=False)
    db.update_candidate(
        draft["id"], status="scored", total_score=95,
        score_json={"subject_check": {"present": True}},
    )
    db.update_candidate(
        final["id"], status="scored", total_score=75,
        score_json={"subject_check": {"present": True}},
    )
    selected = db.auto_select_candidate(shot["id"])
    assert selected["id"] == final["id"]


def test_selected_shot_subset_skips_unqueued_storyboard_gaps(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    concept = next(value for value in db.list_concepts(project["id"]) if value["selected"])
    shots = db.list_shots(project["id"], concept["id"])
    StudioService(db).enqueue_candidates(
        project["id"], CandidateBatchRequest(
            candidates_per_shot=1, shot_ids=[shots[0]["id"], shots[2]["id"]]
        ),
    )
    first = db.list_candidates(shots[0]["id"])[0]
    db.update_candidate(
        first["id"], status="scored", total_score=80,
        score_json={"subject_check": {"present": True}},
    )
    assert db.auto_select_candidate(shots[0]["id"])
    assert db.release_next_shot_jobs(shots[0]["id"]) == 1
    third_job = next(value for value in db.list_jobs(project["id"]) if value["shot_id"] == shots[2]["id"])
    assert third_job["status"] == "queued"


def test_blocked_job_can_be_cancelled(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    result = StudioService(db).enqueue_candidates(
        project["id"], CandidateBatchRequest(candidates_per_shot=1)
    )
    job = next(value for value in db.list_jobs(project["id"]) if value["status"] == "blocked")
    assert db.request_cancel(job["id"])
    assert db.get_job(job["id"])["status"] == "cancelled"
    assert db.get_candidate(job["candidate_id"])["status"] == "cancelled"
    assert db.retry_job(job["id"])
    assert db.get_job(job["id"])["status"] == "blocked"
    assert db.get_candidate(job["candidate_id"])["status"] == "queued"


def test_export_rejects_foreign_or_wrong_role_assets(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    other = make_project(db, test_settings)
    foreign = db.create_asset(
        other["id"], "music", "bed.wav", "/tmp/bed.wav", "audio/wav"
    )
    wrong_role = db.create_asset(
        project["id"], "reference_sheet", "bible.png", "/tmp/bible.png", "image/png"
    )
    service = StudioService(db)
    with pytest.raises(ValueError, match="does not belong"):
        service.enqueue_export(
            project["id"], ExportRequest(platform="reels", music_asset_id=foreign["id"])
        )
    with pytest.raises(ValueError, match="wrong media role"):
        service.enqueue_export(
            project["id"], ExportRequest(platform="reels", music_asset_id=wrong_role["id"])
        )


def test_running_jobs_recover_after_restart(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    job = db.create_job("creative_plan", "local", {}, project_id=project["id"])
    claimed = db.claim_job("local", "worker")
    assert claimed["id"] == job["id"]
    Database(test_settings.database_path).initialize()
    assert db.get_job(job["id"])["status"] == "queued"


def test_remote_retry_keeps_worker_affinity_for_output_recovery(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    job = db.create_job("generate", "comfy", {}, project_id=project["id"])
    claimed = db.claim_job("comfy", "gpu1")
    db.update_job(
        claimed["id"], status="failed", remote_id="remote-finished",
        error="status connection timed out",
    )
    assert db.retry_job(job["id"])
    assert db.claim_job("comfy", "gpu0") is None
    recovered = db.claim_job("comfy", "gpu1")
    assert recovered["remote_id"] == "remote-finished"
    assert recovered["worker_id"] == "gpu1"
    assert recovered["attempts"] == claimed["attempts"]
    assert recovered["error"] is None


def test_restart_recovery_preserves_remote_prompt_and_worker(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    job = db.create_job("generate", "comfy", {}, project_id=project["id"])
    claimed = db.claim_job("comfy", "gpu1")
    db.update_job(claimed["id"], remote_id="remote-running")
    Database(test_settings.database_path).initialize()
    recovered = db.get_job(job["id"])
    assert recovered["status"] == "queued"
    assert recovered["worker_id"] == "gpu1"
    assert recovered["remote_id"] == "remote-running"


def test_exclusive_generation_drains_regular_jobs_atomically(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    regular = db.create_job("generate", "comfy", {})
    running = db.claim_generation_job(
        "gpu0", upload_capable=False, exclusive_capable=True
    )
    assert running and running["id"] == regular["id"]
    exclusive = db.create_job("generate", "comfy_exclusive", {})
    assert db.claim_generation_job(
        "gpu1", upload_capable=True, exclusive_capable=False
    ) is None
    assert db.claim_generation_job(
        "gpu0", upload_capable=False, exclusive_capable=True
    ) is None
    db.update_job(running["id"], status="succeeded")
    claimed = db.claim_generation_job(
        "gpu0", upload_capable=False, exclusive_capable=True
    )
    assert claimed and claimed["id"] == exclusive["id"]


def test_feedback_analytics_and_chains_are_persistent(test_settings):
    db = Database(test_settings.database_path)
    db.initialize()
    project = make_project(db, test_settings)
    concept = next(value for value in db.list_concepts(project["id"]) if value["selected"])
    shot = db.list_shots(project["id"], concept["id"])[0]
    candidate = db.create_candidate(
        project["id"], shot["id"], shot["prompt"], 42,
        {"width": 256, "height": 448},
    )
    db.update_candidate(
        candidate["id"], status="scored", selected=True, total_score=84,
        score_json={
            "technical_score": 28, "prompt_alignment": 21,
            "temporal_coherence": 17, "aesthetics": 11, "hook_strength": 7,
        },
    )
    db.create_feedback(project["id"], candidate["id"], 5, "excellent", "strong hook")

    analytics = db.analytics(project["id"])
    assert analytics["candidates"]["average_score"] == 84
    assert analytics["feedback"]["excellent"] == 1
    assert analytics["selected_score_dimensions"]["hook_strength"] == 7
    learning = db.creative_learning_context(project["id"])
    assert learning["example_count"] >= 1
    assert learning["proven_patterns"][0]["reason"] == "strong hook"
    assert learning["high_scoring_selected"][0]["prompt"] == candidate["prompt"]

    chain = db.create_chain("folder/source.mp4", "opening")
    db.add_chain_clip(chain["id"], "continuation", 0.65, remote_filename="next.mp4", status="done")
    restored = Database(test_settings.database_path).get_chain(chain["id"])
    assert [clip["prompt"] for clip in restored["clips"]] == ["opening", "continuation"]
