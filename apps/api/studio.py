from __future__ import annotations

import secrets
from typing import Any

from .audio_prompt import compile_audio_prompt, visual_only_prompt
from .database import Database
from .ingredients import IngredientsAdapter
from .schemas import (
    AspectRatio, CandidateBatchRequest, ExportRequest, GenerationProfile, GenerationSettings,
)
from .workflow import WorkflowStore


ASPECT_DRAFT_SIZES = {
    AspectRatio.VERTICAL.value: (256, 448),
    AspectRatio.LANDSCAPE.value: (448, 256),
    AspectRatio.SQUARE.value: (320, 320),
}
ASPECT_FINAL_SIZES = {
    AspectRatio.VERTICAL.value: (360, 640),
    AspectRatio.LANDSCAPE.value: (640, 360),
    AspectRatio.SQUARE.value: (512, 512),
}
VISUAL_REFERENCE_KINDS = {"image", "reference_sheet", "brand"}

TAKE_DIRECTIONS = (
    (
        "clean-baseline",
        "Execute the described shot cleanly and literally. Prioritize exact subject identity, "
        "readable action, stable anatomy, and a coherent beginning-to-end motion arc.",
    ),
    (
        "motion-push",
        "Increase the energy of the same action through continuous physical motion, foreground "
        "parallax, and a decisive camera move. Do not change the subject, location, or outcome.",
    ),
    (
        "hook-clarity",
        "Make the first half-second an unmistakable stop-scroll image while preserving the same "
        "action and payoff. Establish the required subject immediately and keep it readable.",
    ),
    (
        "camera-alternative",
        "Use a bolder but physically plausible camera interpretation with strong depth separation "
        "and subject lock. Preserve every required identity, prop, action, and story fact.",
    ),
    (
        "premium-detail",
        "Favor premium commercial lighting, tactile material detail, controlled highlights, and "
        "clean silhouettes without reducing motion or altering the required content.",
    ),
    (
        "rhythmic-action",
        "Stage the same action with a clear anticipation, impact, and reaction beat that can cut "
        "rhythmically. Keep motion continuous and all required elements unchanged.",
    ),
    (
        "composition-read",
        "Strengthen visual hierarchy and mobile-screen readability: one dominant subject, clear "
        "foreground/background layers, and no distracting additions or identity drift.",
    ),
    (
        "payoff-emphasis",
        "Build the same action toward the clearest possible final visual change, then hold a stable "
        "hero composition long enough to register. Do not invent a different ending.",
    ),
)


def candidate_take_prompt(
    base_prompt: str, take_index: int,
    directions: tuple[tuple[str, str], ...] = TAKE_DIRECTIONS,
) -> tuple[str, str]:
    role, direction = directions[take_index % len(directions)]
    return f"{base_prompt.rstrip()} Take direction: {direction}", role


class StudioService:
    def __init__(self, db: Database, workflow_store: WorkflowStore | None = None):
        self.db = db
        self.workflow_store = workflow_store

    def _take_directions(self, project_id: str) -> tuple[tuple[str, str], ...]:
        """Keep a safe control take, then prioritize strategies that win locally."""
        project_roles = self.db.analytics(project_id).get("take_roles", [])
        learned = [value for value in project_roles if value.get("scored", 0) > 0]
        if not learned:
            learned = [
                value for value in self.db.analytics().get("take_roles", [])
                if value.get("scored", 0) > 0
            ]
        by_role = {role: direction for role, direction in TAKE_DIRECTIONS}
        ordered_roles = ["clean-baseline"]
        ordered_roles.extend(
            value["role"] for value in learned
            if value["role"] in by_role and value["role"] != "clean-baseline"
        )
        ordered_roles.extend(role for role in by_role if role not in ordered_roles)
        return tuple((role, by_role[role]) for role in ordered_roles)

    def _enqueue_ingredients_candidates(
        self,
        project: dict[str, Any],
        concept: dict[str, Any],
        shots: list[dict[str, Any]],
        base: GenerationSettings,
        reference_sheet: dict[str, Any] | None,
        candidates_per_shot: int,
    ) -> dict[str, Any]:
        if not reference_sheet or reference_sheet.get("kind") != "reference_sheet":
            raise ValueError(
                "Ingredients requires a Visual bible asset, not an ordinary image or brand anchor"
            )
        if base.reference_audio_asset_id or base.audio_mode != "silent":
            raise ValueError(
                "Ingredients candidates are structurally silent; add exact voiceover or music at export"
            )
        too_long = [
            shot for shot in shots
            if float(shot["duration_seconds"]) > IngredientsAdapter.max_duration_seconds
        ]
        if too_long:
            positions = ", ".join(str(int(shot["position"]) + 1) for shot in too_long)
            raise ValueError(
                "Ingredients currently supports story beats up to 5 seconds. "
                f"Split or shorten shot(s) {positions}; longer stories can chain those beats."
            )
        metadata = reference_sheet.get("metadata") or {}
        reference_description = (
            base.reference_sheet_description.strip()
            or str(metadata.get("notes") or "").strip()
        )
        if len(reference_description) < 10:
            raise ValueError(
                "Describe the visual-bible panels so Ingredients knows which characters, "
                "products, props, wardrobe, logos, and location it should preserve"
            )

        jobs: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        directions = self._take_directions(project["id"])
        manual = project["brief"].get("prompt_mode") == "manual"
        for shot in shots:
            visual_prompt = visual_only_prompt(shot["prompt"])
            _, guarded_negative, audio_contract = compile_audio_prompt(
                shot["prompt"], base.negative_prompt or shot["negative_prompt"],
                mode="silent", duration_seconds=float(shot["duration_seconds"]),
            )
            for take_index in range(candidates_per_shot):
                seed = (
                    (base.seed + take_index) % (2**63 - 1)
                    if base.seed >= 0 else secrets.randbelow(2**63 - 2)
                )
                if manual:
                    action_prompt, take_role = visual_prompt, "manual-seed-variation"
                    candidate_prompt = shot["prompt"]
                else:
                    action_prompt, take_role = candidate_take_prompt(
                        visual_prompt, take_index, directions
                    )
                    candidate_prompt = action_prompt
                settings = base.model_copy(
                    update={
                        "width": IngredientsAdapter.width,
                        "height": IngredientsAdapter.height,
                        "duration_seconds": float(shot["duration_seconds"]),
                        "fps": float(IngredientsAdapter.fps),
                        "negative_prompt": guarded_negative,
                        "draft": False,
                        "reference_mode": "every-shot",
                        "audio_mode": "silent",
                    }
                ).model_dump(mode="json")
                settings.update({
                    "take_index": take_index, "take_role": take_role,
                    "base_prompt": shot["prompt"], "workflow_prompt": action_prompt,
                    "audio_contract": audio_contract,
                    "creative_lab_mode": "ingredients",
                    "reference_sheet_filename": reference_sheet["filename"],
                    "reference_sheet_description": reference_description,
                    "render_bucket": {
                        "width": IngredientsAdapter.width,
                        "height": IngredientsAdapter.height,
                        "frames": IngredientsAdapter.frames,
                        "fps": IngredientsAdapter.fps,
                    },
                })
                candidate = self.db.create_candidate(
                    project["id"], shot["id"], candidate_prompt, seed, settings,
                    draft=False,
                )
                job = self.db.create_job(
                    "ingredients_generate", "comfy_upload",
                    {
                        "settings": settings,
                        "execution_target": base.execution_target,
                        "reference_image_path": reference_sheet["local_path"],
                        "reference_sheet_description": reference_description,
                    },
                    project_id=project["id"], shot_id=shot["id"],
                    candidate_id=candidate["id"], status="queued",
                )
                jobs.append(job)
                candidates.append(candidate)
        self.db.set_project_status(project["id"], "generating")
        return {
            "concept_id": concept["id"], "candidate_count": len(candidates),
            "job_ids": [job["id"] for job in jobs],
            "first_shot_job_ids": [job["id"] for job in jobs],
            "creative_lab_mode": "ingredients",
            "render_bucket": {
                "width": IngredientsAdapter.width, "height": IngredientsAdapter.height,
                "frames": IngredientsAdapter.frames, "fps": IngredientsAdapter.fps,
            },
        }

    def enqueue_plan(self, project_id: str, concept_count: int) -> dict[str, Any]:
        self.db.set_project_status(project_id, "planning")
        return self.db.create_job(
            "creative_plan", "local", {"concept_count": concept_count},
            project_id=project_id, max_attempts=1,
        )

    def enqueue_candidates(
        self, project_id: str, request: CandidateBatchRequest
    ) -> dict[str, Any]:
        project = self.db.get_project(project_id)
        if not project:
            raise KeyError(project_id)
        concepts = self.db.list_concepts(project_id)
        if not concepts:
            raise ValueError("Create a creative plan before generating candidates")
        concept = (
            next((value for value in concepts if value["id"] == request.concept_id), None)
            if request.concept_id
            else next((value for value in concepts if value["selected"]), concepts[0])
        )
        if not concept:
            raise ValueError("Requested concept does not belong to this project")
        self.db.select_concept(project_id, concept["id"])
        shots = self.db.list_shots(project_id, concept["id"])
        if request.shot_ids:
            requested = set(request.shot_ids)
            shots = [shot for shot in shots if shot["id"] in requested]
        if not shots:
            raise ValueError("No shots selected for generation")

        aspect = project["brief"]["aspect_ratio"]
        default_width, default_height = ASPECT_DRAFT_SIZES[aspect]
        base = request.settings or GenerationSettings(
            width=default_width, height=default_height, audio_mode="shot"
        )
        if base.audio_mode == "prompt":
            raise ValueError(
                "Raw prompt-controlled speech is not available in Studio because descriptive "
                "visual prose can become gibberish. Use each shot's audio intent, Native "
                "dialogue with one exact short line, or an uploaded voiceover."
            )
        reference_image_asset: dict[str, Any] | None = None
        if base.reference_image_asset_id:
            asset = self.db.get_asset(base.reference_image_asset_id)
            if not asset or asset["project_id"] != project_id:
                raise ValueError("Reference image does not belong to this project")
            if (
                asset["kind"] not in VISUAL_REFERENCE_KINDS
                or not str(asset.get("mime_type", "")).startswith("image/")
            ):
                raise ValueError("Reference image asset has the wrong media role")
            reference_image_asset = asset
        if base.reference_engine == "ingredients":
            return self._enqueue_ingredients_candidates(
                project, concept, shots, base, reference_image_asset,
                request.candidates_per_shot,
            )
        if base.reference_audio_asset_id:
            asset = self.db.get_asset(base.reference_audio_asset_id)
            if not asset or asset["project_id"] != project_id:
                raise ValueError("Reference audio does not belong to this project")
            if asset["kind"] not in ("voiceover", "reference"):
                raise ValueError("Reference audio asset has the wrong media role")
        jobs, candidates = [], []
        take_directions = self._take_directions(project_id)
        first_position = min(shot["position"] for shot in shots)
        shot_references: dict[str, dict[str, Any] | None] = {}
        for shot in shots:
            shot_data = shot.get("data") or {}
            shot_reference_id = str(shot_data.get("reference_asset_id") or "").strip()
            use_global = bool(
                reference_image_asset
                and (
                    base.reference_mode == "every-shot"
                    or shot["position"] == first_position
                )
            )
            asset = (
                self.db.get_asset(shot_reference_id)
                if shot_reference_id else (reference_image_asset if use_global else None)
            )
            if asset and (
                asset["project_id"] != project_id
                or asset["kind"] not in VISUAL_REFERENCE_KINDS
                or not str(asset.get("mime_type", "")).startswith("image/")
            ):
                raise ValueError(f"Shot {shot['position'] + 1} has an invalid visual reference")
            shot_references[shot["id"]] = asset
        # Compile every shot before creating any candidates/jobs so an invalid
        # dialogue budget cannot leave a partially queued production behind.
        shot_audio: dict[str, tuple[str, str, dict[str, Any]]] = {}
        for shot in shots:
            shot_data = shot.get("data") or {}
            selected_audio_mode = (
                str(shot_data.get("audio_mode") or "ambient")
                if base.audio_mode == "shot"
                else str(base.audio_mode)
            )
            if selected_audio_mode == "native-dialogue" and (
                base.audio_mode != "shot"
                and (
                    shot_data.get("audio_mode") != "native-dialogue"
                    or not str(shot_data.get("dialogue") or "").strip()
                )
            ):
                # "Only enabled shot dialogue" is intentionally mixed: shots
                # without a deliberate spoken line remain structurally silent.
                selected_audio_mode = "silent"
            try:
                shot_audio[shot["id"]] = compile_audio_prompt(
                    shot["prompt"], base.negative_prompt or shot["negative_prompt"],
                    mode=selected_audio_mode,
                    duration_seconds=float(shot["duration_seconds"]),
                    dialogue=str(shot_data.get("dialogue") or ""),
                    speaker=str(shot_data.get("speaker") or ""),
                    language=str(shot_data.get("language") or "English"),
                    accent=str(shot_data.get("accent") or ""),
                    ambience=str(shot_data.get("audio") or ""),
                )
            except ValueError as exc:
                raise ValueError(f"Shot {int(shot['position']) + 1} audio: {exc}") from exc
        for shot in shots:
            shot_data = shot.get("data") or {}
            compiled_prompt, compiled_negative, audio_contract = shot_audio[shot["id"]]
            visual_reference = shot_references[shot["id"]]
            reference_direction = ""
            if visual_reference and project["brief"].get("prompt_mode") != "manual":
                metadata = visual_reference.get("metadata") or {}
                role = str(
                    shot_data.get("reference_role")
                    or metadata.get("brand_role")
                    or "subject"
                ).replace("_", " ")
                label = str(metadata.get("label") or visual_reference["filename"])
                notes = str(metadata.get("notes") or "").strip()
                reference_direction = (
                    f" Visual reference requirement: use the supplied image as the exact "
                    f"visual identity for the {role} ({label}). Preserve its silhouette, "
                    "colors, markings, proportions, and material design across motion; do not "
                    "redesign, replace, or duplicate it."
                    + (f" Non-negotiable reference details: {notes}." if notes else "")
                )
            for take_index in range(request.candidates_per_shot):
                seed = (
                    (base.seed + take_index) % (2**63 - 1)
                    if base.seed >= 0 else secrets.randbelow(2**63 - 2)
                )
                manual_prompt = project["brief"].get("prompt_mode") == "manual"
                if manual_prompt:
                    # Human-in-the-loop mode is authoritative. Best-of-N varies
                    # only the seed. Persist and display the exact authored
                    # visual prompt; the separately recorded workflow prompt
                    # may append only the operator-selected audio contract.
                    candidate_prompt, take_role = shot["prompt"], "manual-seed-variation"
                else:
                    candidate_prompt, take_role = candidate_take_prompt(
                        f"{compiled_prompt.rstrip()}{reference_direction}",
                        take_index, take_directions,
                    )
                settings = base.model_copy(
                    update={
                        "duration_seconds": float(shot["duration_seconds"]),
                        "negative_prompt": compiled_negative,
                        "draft": True,
                        # Persist the resolved contract, never the Studio-only
                        # "shot" policy or a global mode that was downgraded.
                        "audio_mode": audio_contract["mode"],
                        "reference_image_asset_id": (
                            visual_reference["id"] if visual_reference else None
                        ),
                    }
                ).model_dump(mode="json")
                settings.update(
                    {
                        "take_index": take_index,
                        "take_role": take_role,
                        "base_prompt": shot["prompt"],
                        "workflow_prompt": (
                            compiled_prompt if manual_prompt else candidate_prompt
                        ),
                        "audio_contract": audio_contract,
                        **(
                            {"reference_direction": reference_direction.strip()}
                            if reference_direction else {}
                        ),
                    }
                )
                candidate = self.db.create_candidate(
                    project_id, shot["id"], candidate_prompt, seed, settings, draft=True
                )
                status = "queued" if shot["position"] == first_position else "blocked"
                image_path = visual_reference["local_path"] if visual_reference else None
                reference_overrides_continuity = bool(image_path and shot["position"] > 0)
                job = self.db.create_job(
                    "generate",
                    (
                        "comfy_upload"
                        if shot["position"] > 0 or image_path or base.reference_audio_asset_id
                        else "comfy"
                    ),
                    {
                        "settings": settings,
                        "execution_target": base.execution_target,
                        **({"image_path": image_path} if image_path else {}),
                        **(
                            {"reference_overrides_continuity": True}
                            if reference_overrides_continuity else {}
                        ),
                    },
                    project_id=project_id, shot_id=shot["id"], candidate_id=candidate["id"],
                    status=status,
                )
                jobs.append(job)
                candidates.append(candidate)
        self.db.set_project_status(project_id, "generating")
        return {
            "concept_id": concept["id"], "candidate_count": len(candidates),
            "job_ids": [job["id"] for job in jobs],
            "first_shot_job_ids": [job["id"] for job in jobs if job["status"] == "queued"],
        }

    def enqueue_winner_renders(
        self, project_id: str, execution_target: str = "auto"
    ) -> dict[str, Any]:
        profile = GenerationProfile.PRODUCTION.value
        fallback_reason: str | None = None
        if self.workflow_store:
            status = self.workflow_store.profile_status(profile)
            missing = [
                value for value in status["missing_capabilities"]
                if value != "audio_video_time_mask"
            ]
            if missing:
                # The stable production fallback deliberately remains on the
                # working Q4_K_M GGUF graph. FP8 is never selected implicitly.
                profile = GenerationProfile.GGUF_FINAL.value
                gguf_status = self.workflow_store.profile_status(profile)
                gguf_missing = [
                    value for value in gguf_status["missing_capabilities"]
                    if value != "audio_video_time_mask"
                ]
                if gguf_missing:
                    raise ValueError(
                        "Production profile is waiting for worker workflow capabilities: "
                        + ", ".join(missing)
                        + "; GGUF final fallback is waiting for: " + ", ".join(gguf_missing)
                    )
                fallback_reason = (
                    "Owner-gated production stages unavailable; using the stable "
                    "LTX 2.3 Q4_K_M GGUF 4+3 final path"
                )
        project = self.db.get_project(project_id)
        if not project:
            raise KeyError(project_id)
        concepts = self.db.list_concepts(project_id)
        concept = next((value for value in concepts if value["selected"]), None)
        if not concept:
            raise ValueError("Select a concept first")
        shots = self.db.list_shots(project_id, concept["id"])
        width, height = ASPECT_FINAL_SIZES[project["brief"]["aspect_ratio"]]
        jobs: list[dict[str, Any]] = []
        profile_config = (
            self.workflow_store.manifest.get("profiles", {}).get(profile, {})
            if self.workflow_store else {}
        )
        exclusive = bool(
            profile_config.get("model_loader", {}).get("exclusive_dual_gpu", False)
        )
        prior_final_selected = True
        for shot in shots:
            selected_id = shot.get("selected_candidate_id")
            selected = self.db.get_candidate(selected_id) if selected_id else None
            if not selected:
                raise ValueError(f"Shot {shot['position'] + 1} has no selected draft")
            existing_finals = [
                value for value in self.db.list_candidates(shot["id"])
                if not value["draft"] and value["status"] not in ("failed", "cancelled")
            ]
            if existing_finals:
                prior_final_selected = not bool(selected["draft"])
                continue
            settings = {
                **selected["settings"], "width": width, "height": height,
                "draft": False, "hq_of": selected["id"],
                "profile": profile, "execution_target": execution_target,
            }
            reference_image = None
            reference_overrides_continuity = False
            if settings.get("reference_image_asset_id"):
                asset = self.db.get_asset(str(settings["reference_image_asset_id"]))
                if (
                    not asset or asset["project_id"] != project_id
                    or asset["kind"] not in VISUAL_REFERENCE_KINDS
                    or not str(asset.get("mime_type", "")).startswith("image/")
                ):
                    raise ValueError("Selected draft's reference image is unavailable")
                reference_image = asset["local_path"]
                reference_overrides_continuity = bool(shot["position"] > 0)
            reference_audio_id = settings.get("reference_audio_asset_id")
            if reference_audio_id:
                asset = self.db.get_asset(str(reference_audio_id))
                if (
                    not asset or asset["project_id"] != project_id
                    or asset["kind"] not in ("voiceover", "reference")
                ):
                    raise ValueError("Selected draft's reference audio is unavailable")
            candidate = self.db.create_candidate(
                project_id, shot["id"], selected["prompt"], selected["seed"], settings, draft=False
            )
            job = self.db.create_job(
                "generate",
                "comfy_exclusive" if exclusive
                else (
                    "comfy_upload"
                    if shot["position"] > 0 or reference_image or reference_audio_id
                    else "comfy"
                ),
                {
                    "settings": settings, "hq_of": selected["id"],
                    "execution_target": execution_target,
                    **({"image_path": reference_image} if reference_image else {}),
                    **(
                        {"reference_overrides_continuity": True}
                        if reference_overrides_continuity else {}
                    ),
                },
                project_id=project_id, shot_id=shot["id"], candidate_id=candidate["id"],
                status="queued" if prior_final_selected else "blocked",
            )
            jobs.append(job)
            # Later finals must condition on this one after it is selected.
            prior_final_selected = False
        if not jobs:
            raise ValueError(
                "Every selected shot already has a final render. Retry a failed job or "
                "select a different draft before requesting another final pass."
            )
        self.db.set_project_status(project_id, "rendering_winners")
        return {
            "job_ids": [job["id"] for job in jobs], "render_count": len(jobs),
            "profile": profile, "fallback_reason": fallback_reason,
        }

    def enqueue_export(self, project_id: str, request: ExportRequest) -> dict[str, Any]:
        project = self.db.get_project(project_id)
        if not project:
            raise KeyError(project_id)
        asset_rules = (
            (request.music_asset_id, {"music"}, "Music"),
            (request.voiceover_asset_id, {"voiceover"}, "Voiceover"),
            (request.logo_asset_id, VISUAL_REFERENCE_KINDS, "Logo"),
        )
        for asset_id, allowed_kinds, label in asset_rules:
            if not asset_id:
                continue
            asset = self.db.get_asset(asset_id)
            if not asset or asset["project_id"] != project_id:
                raise ValueError(f"{label} asset does not belong to this project")
            if asset["kind"] not in allowed_kinds:
                raise ValueError(f"{label} asset has the wrong media role")
            if label == "Logo" and not str(asset.get("mime_type", "")).startswith("image/"):
                raise ValueError("Logo asset must be an image")
        options = request.model_dump(mode="json")
        export = self.db.create_export(project_id, request.platform.value, options)
        job = self.db.create_job(
            "export", "local", {"export_id": export["id"]}, project_id=project_id,
            max_attempts=1,
        )
        self.db.update_export(export["id"], job_id=job["id"])
        self.db.set_project_status(project_id, "exporting")
        return self.db.get_export(export["id"])  # type: ignore[return-value]
