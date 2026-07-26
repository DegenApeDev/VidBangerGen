from __future__ import annotations

import asyncio
import hashlib
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any

import httpx

from .comfy import ComfyClient, ComfyError, ComfyOrphanedPromptError
from .config import Settings
from .cinemagraph import CinemagraphAdapter
from .creative import CreativeDirector
from .creative_transform import CreativeTransformAdapter
from .database import Database, utcnow
from .foley import FoleyAdapter
from .ingredients import IngredientsAdapter
from .in_outpaint import InOutpaintAdapter
from .lipdub import LipDubAdapter
from .media import (
    extract_continuity_assets, merge_upscale_chunks, mux_generated_foley,
    prepare_cinemagraph_image, prepare_lipdub_source, prepare_outpaint_inputs,
    prepare_video_mask,
    prepare_reference_audio,
    prepare_reference_sheet, prepare_upscale_chunks, probe_duration, probe_stream_types,
    probe_video_info, render_timeline, restore_source_audio, trim_silent_video,
)
from .pixel_upscale import PixelSpatialUpscaleAdapter
from .scoring import MediaScorer
from .workflow import LTXWorkflowAdapter, WorkflowError


VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aac"}


class ExclusiveInferenceCoordinator:
    def __init__(self, clients: list[ComfyClient], managed_urls: tuple[str, ...]):
        self.clients = clients
        self.managed_urls = set(managed_urls)

    async def prepare(self, target: ComfyClient) -> None:
        states = await asyncio.gather(*(client.queue_state() for client in self.clients))
        busy = [
            client.base_url for client, state in zip(self.clients, states)
            if state["running"] or state["pending"]
        ]
        if busy:
            raise RuntimeError(
                "FP8 final requires every remote inference queue to be idle: " + ", ".join(busy)
            )
        peers = [client for client in self.clients if client.base_url != target.base_url]
        unmanaged = [
            client.base_url for client in peers if client.base_url not in self.managed_urls
        ]
        if unmanaged:
            raise RuntimeError(
                "FP8 final will not release an unmanaged peer worker: " + ", ".join(unmanaged)
            )
        for client in peers:
            await client.release_managed_models()
        verified = await asyncio.gather(*(client.queue_state() for client in self.clients))
        if any(state["running"] or state["pending"] for state in verified):
            raise RuntimeError("A remote inference job arrived during FP8 preflight")


class GenerationWorker:
    def __init__(
        self, db: Database, settings: Settings, adapter: LTXWorkflowAdapter, client: ComfyClient,
        coordinator: ExclusiveInferenceCoordinator | None = None,
        upscale_adapter: PixelSpatialUpscaleAdapter | None = None,
        transform_adapter: CreativeTransformAdapter | None = None,
        ingredients_adapter: IngredientsAdapter | None = None,
        foley_adapter: FoleyAdapter | None = None,
        cinemagraph_adapter: CinemagraphAdapter | None = None,
        in_outpaint_adapter: InOutpaintAdapter | None = None,
        lipdub_adapter: LipDubAdapter | None = None,
    ):
        self.db = db
        self.settings = settings
        self.adapter = adapter
        self.client = client
        self.worker_id = f"comfy:{client.worker_id}"
        target = settings.target_for_url(client.base_url)
        self.target_id = target.id if target else "primary"
        self.upload_capable = (
            not settings.upload_capable_urls or client.base_url in settings.upload_capable_urls
        )
        self.exclusive_capable = client.base_url == settings.exclusive_comfy_url
        self.interrupt_capable = client.base_url in settings.interrupt_capable_urls
        self.coordinator = coordinator
        self.upscale_adapter = upscale_adapter or PixelSpatialUpscaleAdapter()
        self.transform_adapter = transform_adapter or CreativeTransformAdapter()
        self.ingredients_adapter = ingredients_adapter or IngredientsAdapter()
        self.foley_adapter = foley_adapter or FoleyAdapter()
        self.cinemagraph_adapter = cinemagraph_adapter or CinemagraphAdapter()
        self.in_outpaint_adapter = in_outpaint_adapter or InOutpaintAdapter()
        self.lipdub_adapter = lipdub_adapter or LipDubAdapter()
        self.running = False
        self.active_job_id: str | None = None

    async def run(self) -> None:
        self.running = True
        while self.running:
            job = self.db.claim_generation_job(
                self.worker_id, upload_capable=self.upload_capable,
                exclusive_capable=self.exclusive_capable, target_id=self.target_id,
            )
            if not job:
                await asyncio.sleep(0.75)
                continue
            self.active_job_id = job["id"]
            await self._execute(job)
            self.active_job_id = None

    async def stop(self) -> None:
        self.running = False

    async def _execute(self, job: dict[str, Any]) -> None:
        candidate_id = job.get("candidate_id")
        try:
            if job["kind"] == "upscale":
                await self._execute_pixel_upscale(job)
                return
            if job["kind"] == "creative_transform":
                await self._execute_creative_transform(job)
                return
            if job["kind"] == "cinemagraph":
                await self._execute_cinemagraph(job)
                return
            if job["kind"] == "ingredients_generate":
                await self._execute_ingredients(job)
                return
            if job["kind"] not in ("generate", "retake"):
                raise RuntimeError(f"Unsupported Comfy job kind: {job['kind']}")
            if job["lane"] == "comfy_exclusive":
                if not self.exclusive_capable or not self.coordinator:
                    raise RuntimeError("No authorized exclusive inference coordinator is configured")
                await self.coordinator.prepare(self.client)
            candidate = self.db.get_candidate(candidate_id) if candidate_id else None
            if not candidate:
                raise RuntimeError("Generation job has no candidate")
            self.db.update_candidate(candidate_id, status="generating")
            payload = job["payload"]
            settings = payload["settings"]
            image_name = None
            video_name = None
            audio_name = None
            retake_video_name = None
            audio_seed: dict[str, Any] | None = None
            continuity: dict[str, Any] = {"mode": "none"}

            source_image_path = payload.get("image_path")
            retake_source_id = payload.get("source_candidate_id")
            if retake_source_id:
                retake_source = self.db.get_candidate(str(retake_source_id))
                if not retake_source or not retake_source.get("artifact"):
                    raise RuntimeError("Retake source candidate is unavailable")
                retake_path = Path(retake_source["artifact"]["local_path"])
                if not retake_path.exists():
                    raise RuntimeError("Retake source media is missing")
                retake_video_name = await self.client.upload(retake_path)
                continuity = {
                    "mode": "time-masked-av-retake", "source_candidate_id": retake_source_id,
                    "start_seconds": settings.get("retake_start_seconds"),
                    "end_seconds": settings.get("retake_end_seconds"),
                }
            reference_overrides_continuity = bool(
                source_image_path and payload.get("reference_overrides_continuity")
            )
            previous = (
                None if retake_source_id or reference_overrides_continuity
                else self.db.previous_selected_candidate(job["shot_id"])
            )
            if reference_overrides_continuity:
                continuity = {
                    "mode": "visual-reference",
                    "reference_image_asset_id": settings.get("reference_image_asset_id"),
                }
            if previous and previous.get("artifact"):
                previous_artifact = previous["artifact"]
                motion_tail_path = previous_artifact.get("motion_tail_path")
                source_image_path = previous_artifact.get("last_frame_path")
                continuity = {
                    "mode": "motion-tail",
                    "source_candidate_id": previous["id"],
                    "motion_tail_path": motion_tail_path,
                    "last_frame_path": source_image_path,
                }
                if motion_tail_path and Path(motion_tail_path).exists():
                    video_name = await self.client.upload(Path(motion_tail_path))
                    source_image_path = None
                else:
                    continuity["mode"] = "last-frame-fallback"
            if source_image_path and not video_name:
                source_path = Path(source_image_path)
                if not source_path.exists():
                    raise RuntimeError(f"Conditioning asset is missing: {source_path}")
                image_name = await self.client.upload(source_path)

            # A retake already uses the source clip as an AV guide. Re-attaching
            # the project's reference-audio seed would create two competing
            # time masks, which the workflow intentionally rejects. The source
            # clip supplies the preserved audio context for retakes instead.
            reference_audio_id = (
                None if retake_source_id else settings.get("reference_audio_asset_id")
            )
            if reference_audio_id:
                asset = self.db.get_asset(str(reference_audio_id))
                if not asset or asset["project_id"] != candidate["project_id"]:
                    raise RuntimeError("Reference audio asset does not belong to this project")
                if asset["kind"] not in ("voiceover", "reference"):
                    raise RuntimeError("Reference audio asset must be voiceover or reference audio")
                remote_input_name = (asset.get("metadata") or {}).get("comfy_input_name")
                source_duration = await asyncio.to_thread(
                    probe_duration, Path(asset["local_path"])
                )
                generation_duration = float(settings["duration_seconds"])
                requested_seed = float(settings.get("audio_seed_seconds", 4.0))
                if remote_input_name and source_duration >= generation_duration:
                    audio_name = str(remote_input_name)
                    audio_seed = {
                        "path": asset["local_path"], "transport": "existing-comfy-input",
                        "source_duration_seconds": round(source_duration, 3),
                        "duration_seconds": round(generation_duration, 3),
                        "seed_seconds": round(
                            min(requested_seed, generation_duration, source_duration), 3
                        ),
                    }
                else:
                    prepared_path = (
                        self.settings.artifact_dir / candidate["project_id"] / "audio_seeds"
                        / f"{candidate_id}.wav"
                    )
                    audio_seed = await asyncio.to_thread(
                        prepare_reference_audio, Path(asset["local_path"]), prepared_path,
                        generation_duration, requested_seed,
                    )
                    audio_seed["transport"] = "comfy-upload"
                    audio_name = await self.client.upload(prepared_path)

            graph, workflow_metadata = self.adapter.build(
                prompt=str(settings.get("workflow_prompt") or candidate["prompt"]),
                negative_prompt=settings.get("negative_prompt", ""),
                width=int(settings["width"]),
                height=int(settings["height"]),
                duration_seconds=float(settings["duration_seconds"]),
                fps=float(settings["fps"]),
                seed=int(candidate["seed"]),
                filename_prefix=f"vbg/{candidate['project_id']}/{candidate_id}",
                image_name=image_name,
                ic_lora_strength=float(settings.get("ic_lora_strength", 0.5)),
                image_condition_strength=float(settings.get("image_condition_strength", 0.9)),
                profile=str(settings.get("profile", "motion-draft-4x3")),
                video_name=video_name,
                motion_guide_strength=float(settings.get("motion_guide_strength", 0.85)),
                retake_video_name=retake_video_name,
                retake_start_seconds=float(settings.get("retake_start_seconds", 0.0)),
                retake_end_seconds=(
                    float(settings["retake_end_seconds"])
                    if settings.get("retake_end_seconds") is not None else None
                ),
                audio_name=audio_name,
                audio_seed_seconds=float((audio_seed or {}).get("seed_seconds", 4.0)),
                include_audio=(settings.get("audio_contract") or {}).get("mode") != "silent",
            )
            async def update_progress(value: float, _node: str | None) -> None:
                self.db.update_job(job["id"], progress=value)

            async def queued(prompt_id: str) -> None:
                self.db.update_job(job["id"], remote_id=prompt_id, progress=0.01)

            def cancelled() -> bool:
                current = self.db.get_job(job["id"])
                return bool(current and current.get("cancel_requested"))

            result: dict[str, Any] | None = None
            existing_remote_id = str(job.get("remote_id") or "")
            if existing_remote_id:
                # A previous attempt may have lost its HTTP/status connection after
                # ComfyUI finished.  Recover that exact output on the same worker
                # before spending GPU time on a duplicate render.
                existing = await self.client.history(existing_remote_id)
                remote_still_queued = (
                    existing is None
                    and await self.client.prompt_in_queue(existing_remote_id)
                )
                if remote_still_queued or (existing and existing["status"] == "running"):
                    existing = await self.client.wait_for_completion(
                        existing_remote_id, progress=update_progress, cancelled=cancelled,
                        allow_interrupt=self.interrupt_capable,
                    )
                if existing and existing["status"] == "done" and existing.get("files"):
                    result = existing
                    await update_progress(1.0, "recovered remote output")
            if result is None:
                result = await self.client.queue_and_wait(
                    graph, client_id=job["id"], on_queued=queued,
                    progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                )
            # Shared workers are deliberately non-interruptible. If the user
            # cancels there, let our prompt finish and discard the result,
            # avoiding a global interrupt that could hit another account's job.
            if cancelled():
                raise asyncio.CancelledError
            prompt_id = str((self.db.get_job(job["id"]) or {}).get("remote_id") or "")
            artifact = next(
                (
                    item for item in result.get("files", [])
                    if Path(item.get("filename", "")).suffix.lower() in VIDEO_EXTENSIONS
                ),
                None,
            )
            if not artifact:
                raise RuntimeError("ComfyUI completed without a video output")
            extension = Path(artifact["filename"]).suffix.lower() or ".mp4"
            local_path = (
                self.settings.artifact_dir / candidate["project_id"] / candidate["shot_id"]
                / f"{candidate_id}{extension}"
            )
            await self.client.download(artifact, local_path)
            continuity_assets = await asyncio.to_thread(
                extract_continuity_assets, local_path, local_path.parent / "continuity"
            )
            saved_artifact = {
                **artifact,
                "local_path": str(local_path),
                "worker_url": self.client.base_url,
                "remote_prompt_id": prompt_id,
                "workflow": workflow_metadata,
                "continuity": continuity,
                "reference_audio": audio_seed,
                **continuity_assets,
            }
            self.db.update_candidate(candidate_id, status="generated", artifact_json=saved_artifact)
            score_job = None
            if self.settings.vision_scoring_enabled:
                score_job = self.db.create_job(
                    "score", "local", {"candidate_id": candidate_id},
                    project_id=candidate["project_id"], shot_id=candidate["shot_id"],
                    candidate_id=candidate_id, max_attempts=1,
                )
            else:
                self.db.update_candidate(
                    candidate_id,
                    status="unscored",
                    score_json={
                        "available": False,
                        "judge": "manual-review",
                        "summary": "Vision scoring is disabled; choose the keeper by eye.",
                        "issues": [],
                    },
                    total_score=None,
                )
                # Once the final candidate for this shot finishes, move the
                # storyboard to its explicit human-review gate.
                self.db.auto_select_candidate(candidate["shot_id"])
            self.db.update_job(
                job["id"], status="succeeded", progress=1.0,
                result_json={
                    "artifact": saved_artifact,
                    "scoring": "queued" if score_job else "manual-review",
                    **({"score_job_id": score_job["id"]} if score_job else {}),
                },
                finished_at=utcnow(),
            )
        except asyncio.CancelledError:
            current = self.db.get_job(job["id"]) or job
            user_requested = bool(current.get("cancel_requested"))
            if candidate_id and job["kind"] in ("generate", "retake", "ingredients_generate"):
                self.db.update_candidate(
                    candidate_id, status="cancelled" if user_requested else "queued"
                )
            if user_requested:
                self.db.update_job(
                    job["id"], status="cancelled", error="Cancelled by user", progress=0,
                    finished_at=utcnow(),
                )
            else:
                self.db.update_job(
                    job["id"], status="queued",
                    error="Worker stopped; remote output will be recovered on restart",
                    progress=0,
                )
            # Task cancellation is how the pool performs a clean service shutdown.
            # Do not swallow it; a user-requested remote interrupt is raised without
            # setting the task's cancelling flag and the worker loop may continue.
            task = asyncio.current_task()
            if task and task.cancelling():
                raise
        except (
            ComfyError, WorkflowError, OSError, RuntimeError, ValueError, httpx.HTTPError
        ) as exc:
            fresh = self.db.get_job(job["id"]) or job
            retry = int(fresh.get("attempts", 1)) < int(fresh.get("max_attempts", 1))
            self.db.update_job(
                job["id"], status="queued" if retry else "failed", error=str(exc),
                worker_id=None, remote_id=None, progress=0,
                **({} if retry else {"finished_at": utcnow()}),
            )
            if candidate_id and job["kind"] in ("generate", "retake", "ingredients_generate"):
                self.db.update_candidate(candidate_id, status="queued" if retry else "failed")
        except Exception as exc:  # keep workers alive while preserving diagnostics
            self.db.update_job(
                job["id"], status="failed",
                error=f"Unexpected worker failure: {exc}\n{traceback.format_exc(limit=5)}",
                finished_at=utcnow(),
            )
            if candidate_id and job["kind"] in ("generate", "retake", "ingredients_generate"):
                self.db.update_candidate(candidate_id, status="failed")

    async def _execute_pixel_upscale(self, job: dict[str, Any]) -> None:
        payload = job["payload"]
        target = self.settings.execution_target(str(payload.get("target_id") or ""))
        if not target or target.kind != "comfyui" or "post-upscale" not in target.capabilities:
            raise RuntimeError("The selected target cannot run LTX Pixel Spatial upscaling")
        source = Path(payload["source_path"])
        destination = Path(payload["destination_path"])
        if not source.exists():
            raise RuntimeError("The source video is no longer available")
        info = await asyncio.to_thread(probe_video_info, source)
        streams = await asyncio.to_thread(probe_stream_types, source)
        has_source_audio = "audio" in streams
        scale = int(payload.get("scale", 2))
        raw_seed = int(payload.get("seed", -1))
        seed = raw_seed if raw_seed >= 0 else int.from_bytes(
            hashlib.blake2b(job["id"].encode(), digest_size=8).digest(), "big"
        ) % (2**63 - 2)
        prompt = str(payload.get("prompt") or (
            "Preserve the original scene, motion, subjects, lighting, and camera "
            "movement with clean cinematic detail."
        ))
        negative_prompt = str(payload.get("negative_prompt") or (
            "flicker, unstable identity, warped anatomy, changed composition, "
            "text artifacts, oversharpening"
        ))
        work_dir = destination.parent / ".pixel-passes"
        frame_normalization: dict[str, Any] | None = None
        if (
            int(info["frames"]) > self.upscale_adapter.max_frames
            or int(info["frames"]) % 8 != 1
        ):
            prepared = await asyncio.to_thread(
                prepare_upscale_chunks, source, work_dir / "source",
                self.upscale_adapter.max_frames, 9,
            )
            pass_sources = [Path(value["path"]) for value in prepared["chunks"]]
            pass_frames = [int(value["frames"]) for value in prepared["chunks"]]
            overlaps = [
                float(value["overlap_frames"]) / float(prepared["fps"])
                for value in prepared["chunks"]
            ]
            fps = float(prepared["fps"])
            frame_normalization = prepared.get("frame_normalization")
        else:
            pass_sources = [source]
            pass_frames = [int(info["frames"])]
            overlaps = [0.0]
            fps = float(info["fps"] or 24.0)

        pass_outputs: list[Path] = []
        pass_metadata: list[dict[str, Any]] = []
        for index, (pass_source, expected_frames) in enumerate(zip(pass_sources, pass_frames)):
            if self._job_cancelled(job["id"]):
                raise asyncio.CancelledError
            pass_output = (
                destination if len(pass_sources) == 1 and not has_source_audio
                else work_dir / "rendered" / f"pass-{index:03d}.mp4"
            )
            checkpoint_ready = False
            if pass_output.exists():
                try:
                    checkpoint_ready = (
                        await asyncio.to_thread(probe_video_info, pass_output)
                    )["frames"] == expected_frames
                except (OSError, RuntimeError, ValueError):
                    checkpoint_ready = False
            if checkpoint_ready:
                metadata = {"recovery": "local-pass-checkpoint", "pass": index + 1}
            else:
                pass_output.unlink(missing_ok=True)
                metadata = await self._queue_pixel_upscale_pass(
                    job=job, source=pass_source, destination=pass_output,
                    source_width=int(info["width"]), source_height=int(info["height"]),
                    # Pixel Spatial is a visual finishing pass. Keeping its AV
                    # latent silent prevents it from rewriting approved speech;
                    # the untouched source track is restored locally below.
                    has_audio=False, prompt=prompt,
                    negative_prompt=negative_prompt, scale=scale, seed=seed,
                    pass_index=index, pass_count=len(pass_sources),
                )
            pass_outputs.append(pass_output)
            pass_metadata.append(metadata)

        merge_metadata: dict[str, Any] | None = None
        if len(pass_outputs) > 1:
            self.db.update_job(job["id"], progress=0.96)
            merge_metadata = await asyncio.to_thread(
                merge_upscale_chunks, pass_outputs, source, destination, overlaps, fps,
            )
            shutil.rmtree(work_dir, ignore_errors=True)
        elif has_source_audio:
            self.db.update_job(job["id"], progress=0.96)
            merge_metadata = await asyncio.to_thread(
                restore_source_audio, pass_outputs[0], source, destination,
            )
            shutil.rmtree(work_dir, ignore_errors=True)
        output_info = await asyncio.to_thread(probe_video_info, destination)
        self.db.update_job(
            job["id"], status="succeeded", progress=1.0,
            result_json={
                "local_path": str(destination), "target_id": target.id,
                "target_label": target.label, "scale": scale,
                "engine": "LTX 2.3 Pixel Spatial Upscaler", "source_path": str(source),
                "worker_url": self.client.base_url,
                "workflow": {
                    "passes": pass_metadata, "pass_count": len(pass_sources),
                    "source_frames": int(info["frames"]), "max_frames_per_pass": 121,
                    "overlap_frames": 9 if len(pass_sources) > 1 else 0,
                    "frame_normalization": frame_normalization,
                    "merge": merge_metadata,
                },
                "output_dimensions": [output_info["width"], output_info["height"]],
            },
            finished_at=utcnow(),
        )

    async def _execute_cinemagraph(self, job: dict[str, Any]) -> None:
        payload = job["payload"]
        target = self.settings.execution_target(str(payload.get("target_id") or ""))
        if not target or target.kind != "comfyui" or "post-upscale" not in target.capabilities:
            raise RuntimeError("The selected target cannot run Cinemagraph generation")
        source = Path(payload["source_path"])
        destination = Path(payload["destination_path"])
        if not source.exists():
            raise RuntimeError("The Cinemagraph source image is no longer available")
        prepared = destination.parent / "prepared-source.png"
        preparation = await asyncio.to_thread(
            prepare_cinemagraph_image, source, prepared,
        )
        remote_name = await self.client.upload(prepared)
        basename = "cinemagraph"
        subfolder = f"vbg/creative-lab/{job['id']}"
        graph, metadata = self.cinemagraph_adapter.build(
            image_name=remote_name, prompt=str(payload.get("prompt") or ""),
            negative_prompt=str(payload.get("negative_prompt") or ""),
            strength=float(payload.get("strength", 1.0)),
            seed=int(payload.get("seed", -1)),
            width=int(preparation["width"]), height=int(preparation["height"]),
            filename_prefix=f"{subfolder}/{basename}",
        )

        async def update_progress(value: float, _node: str | None) -> None:
            self.db.update_job(job["id"], progress=min(0.95, 0.02 + 0.93 * value))

        async def queued(prompt_id: str) -> None:
            self.db.update_job(job["id"], remote_id=prompt_id)

        def cancelled() -> bool:
            return self._job_cancelled(job["id"])

        expected = {
            "filename": f"{basename}_00001_.mp4", "subfolder": subfolder,
            "type": "output",
        }

        def expected_result(value: dict[str, Any] | None) -> bool:
            return bool(value and any(
                item.get("filename") == expected["filename"]
                and item.get("subfolder", "") == expected["subfolder"]
                for item in value.get("files", [])
            ))

        result: dict[str, Any] | None = None
        already_downloaded = False
        existing_remote_id = str((self.db.get_job(job["id"]) or {}).get("remote_id") or "")
        if existing_remote_id:
            existing = await self.client.history(existing_remote_id)
            still_queued = existing is None and await self.client.prompt_in_queue(existing_remote_id)
            if still_queued or (existing and existing["status"] == "running"):
                existing = await self.client.wait_for_completion(
                    existing_remote_id, progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                )
            if expected_result(existing):
                result = existing
        if result is None:
            try:
                result = await self.client.queue_and_wait(
                    graph, client_id=f"{job['id']}-cinemagraph", on_queued=queued,
                    progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                )
            except ComfyOrphanedPromptError:
                try:
                    await self.client.download(expected, destination)
                    result = {"status": "done", "files": [expected]}
                    already_downloaded = True
                    metadata["remote_recovery"] = "deterministic-output"
                except (ComfyError, httpx.HTTPError, OSError):
                    await asyncio.sleep(3)
                    result = await self.client.queue_and_wait(
                        graph, client_id=f"{job['id']}-cinemagraph-retry",
                        on_queued=queued, progress=update_progress, cancelled=cancelled,
                        allow_interrupt=self.interrupt_capable,
                    )
                    metadata["remote_recovery"] = "single-automatic-retry"
        if cancelled():
            raise asyncio.CancelledError
        artifact = next(
            (
                item for item in (result or {}).get("files", [])
                if item.get("filename") == expected["filename"]
                and Path(item.get("filename", "")).suffix.lower() in VIDEO_EXTENSIONS
            ),
            None,
        )
        if not artifact:
            raise RuntimeError("Cinemagraph completed without a video output")
        if not already_downloaded:
            await self.client.download(artifact, destination)
        prepared.unlink(missing_ok=True)
        output_info = await asyncio.to_thread(probe_video_info, destination)
        self.db.update_job(
            job["id"], status="succeeded", progress=1.0,
            result_json={
                "local_path": str(destination), "source_path": str(source),
                "target_id": target.id, "target_label": target.label,
                "worker_url": self.client.base_url, "mode": "cinemagraph",
                "engine": metadata["engine"], "workflow": metadata,
                "image_preparation": preparation,
                "output_dimensions": [output_info["width"], output_info["height"]],
                "frames": output_info["frames"], "fps": output_info["fps"],
            },
            finished_at=utcnow(),
        )

    async def _execute_creative_transform(self, job: dict[str, Any]) -> None:
        payload = job["payload"]
        target = self.settings.execution_target(str(payload.get("target_id") or ""))
        if not target or target.kind != "comfyui" or "post-upscale" not in target.capabilities:
            raise RuntimeError("The selected target cannot run Creative Lab transforms")
        source = Path(payload["source_path"])
        destination = Path(payload["destination_path"])
        if not source.exists():
            raise RuntimeError("The Creative Lab source video is no longer available")
        info = await asyncio.to_thread(probe_video_info, source)
        mode = str(payload["mode"])
        if mode == "foley-v2a":
            await self._execute_foley_transform(job, target, source, destination, info)
            return
        if mode == "in-outpainting":
            await self._execute_in_outpaint_transform(
                job, target, source, destination, info,
            )
            return
        if mode == "lipdub":
            await self._execute_lipdub_transform(job, target, source, destination)
            return
        limits = self.transform_adapter.limits(mode)
        validate_source = getattr(self.transform_adapter, "validate_source", None)
        if validate_source:
            validate_source(mode, info)
        if int(info["frames"]) > limits["max_frames"]:
            raise RuntimeError(
                f"{mode.replace('-', ' ').title()} currently supports up to "
                f"{limits['max_frames']} frames per isolated pass; "
                f"this clip has {int(info['frames'])} frames"
            )
        streams = await asyncio.to_thread(probe_stream_types, source)
        has_source_audio = "audio" in streams
        remote_name = await self.client.upload(source)
        basename = "transform"
        subfolder = f"vbg/creative-lab/{job['id']}"
        graph, metadata = self.transform_adapter.build(
            mode=mode, video_name=remote_name,
            prompt=str(payload.get("prompt") or ""),
            negative_prompt=str(payload.get("negative_prompt") or ""),
            strength=float(payload.get("strength", 1.0)),
            seed=int(payload.get("seed", -1)),
            filename_prefix=f"{subfolder}/{basename}",
        )

        async def update_progress(value: float, _node: str | None) -> None:
            self.db.update_job(job["id"], progress=min(0.94, 0.02 + 0.92 * value))

        async def queued(prompt_id: str) -> None:
            self.db.update_job(job["id"], remote_id=prompt_id)

        def cancelled() -> bool:
            return self._job_cancelled(job["id"])

        expected = {
            "filename": f"{basename}_00001_.mp4", "subfolder": subfolder,
            "type": "output",
        }

        def expected_result(value: dict[str, Any] | None) -> bool:
            return bool(value and any(
                item.get("filename") == expected["filename"]
                and item.get("subfolder", "") == expected["subfolder"]
                for item in value.get("files", [])
            ))

        result: dict[str, Any] | None = None
        preserve_source_audio = bool(
            has_source_audio and metadata.get("source_audio_policy") == "preserve"
        )
        visual_path = (
            destination.with_name(destination.stem + "-visual.mp4")
            if preserve_source_audio else destination
        )
        already_downloaded = False
        existing_remote_id = str((self.db.get_job(job["id"]) or {}).get("remote_id") or "")
        if existing_remote_id:
            existing = await self.client.history(existing_remote_id)
            still_queued = existing is None and await self.client.prompt_in_queue(existing_remote_id)
            if still_queued or (existing and existing["status"] == "running"):
                existing = await self.client.wait_for_completion(
                    existing_remote_id, progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                )
            if expected_result(existing):
                result = existing
        if result is None:
            try:
                result = await self.client.queue_and_wait(
                    graph, client_id=f"{job['id']}-creative", on_queued=queued,
                    progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                )
            except ComfyOrphanedPromptError:
                try:
                    await self.client.download(expected, visual_path)
                    result = {"status": "done", "files": [expected]}
                    already_downloaded = True
                    metadata["remote_recovery"] = "deterministic-output"
                except (ComfyError, httpx.HTTPError, OSError):
                    await asyncio.sleep(3)
                    result = await self.client.queue_and_wait(
                        graph, client_id=f"{job['id']}-creative-retry", on_queued=queued,
                        progress=update_progress, cancelled=cancelled,
                        allow_interrupt=self.interrupt_capable,
                    )
                    metadata["remote_recovery"] = "single-automatic-retry"
        if cancelled():
            raise asyncio.CancelledError
        artifact = next(
            (
                item for item in (result or {}).get("files", [])
                if item.get("filename") == expected["filename"]
                and Path(item.get("filename", "")).suffix.lower() in VIDEO_EXTENSIONS
            ),
            None,
        )
        if not artifact:
            raise RuntimeError("Creative Lab transform completed without a video output")
        if not already_downloaded:
            await self.client.download(artifact, visual_path)
        audio_restore: dict[str, Any] | None = None
        if preserve_source_audio:
            self.db.update_job(job["id"], progress=0.97)
            audio_restore = await asyncio.to_thread(
                restore_source_audio, visual_path, source, destination,
            )
            visual_path.unlink(missing_ok=True)
        output_info = await asyncio.to_thread(probe_video_info, destination)
        self.db.update_job(
            job["id"], status="succeeded", progress=1.0,
            result_json={
                "local_path": str(destination), "source_path": str(source),
                "target_id": target.id, "target_label": target.label,
                "worker_url": self.client.base_url, "mode": payload["mode"],
                "engine": metadata["engine"], "workflow": metadata,
                "source_audio_restored": preserve_source_audio,
                "audio_restore": audio_restore,
                "output_dimensions": [output_info["width"], output_info["height"]],
                "frames": output_info["frames"],
            },
            finished_at=utcnow(),
        )

    async def _execute_in_outpaint_transform(
        self,
        job: dict[str, Any],
        target: Any,
        source: Path,
        destination: Path,
        info: dict[str, Any],
    ) -> None:
        payload = job["payload"]
        self.in_outpaint_adapter.validate_source(info)
        operation = str(payload.get("operation") or "inpaint")
        work_dir = destination.parent / "prepared"
        if operation == "inpaint":
            mask_source = Path(str(payload.get("mask_path") or ""))
            if not mask_source.exists():
                raise RuntimeError("The inpainting mask is no longer available")
            prepared_source = source
            prepared_mask = work_dir / "inpaint-mask.mkv"
            mask_metadata = await asyncio.to_thread(
                prepare_video_mask, mask_source, source, prepared_mask,
            )
            preparation: dict[str, Any] = {
                "operation": "inpaint", "mask": mask_metadata,
                "source_dimensions": [info["width"], info["height"]],
            }
        elif operation == "outpaint":
            preparation = await asyncio.to_thread(
                prepare_outpaint_inputs, source, work_dir,
                str(payload.get("outpaint_direction") or "all"),
                int(payload.get("expansion_percent") or 25),
                self.in_outpaint_adapter.max_side,
            )
            prepared_source = Path(preparation["source_path"])
            prepared_mask = Path(preparation["mask_path"])
        else:
            raise RuntimeError(f"Unsupported In/Outpainting operation: {operation}")
        source_name, mask_name = await asyncio.gather(
            self.client.upload(prepared_source), self.client.upload(prepared_mask),
        )
        basename = "in-outpaint"
        subfolder = f"vbg/creative-lab/{job['id']}"
        graph, metadata = self.in_outpaint_adapter.build(
            video_name=source_name, mask_name=mask_name,
            prompt=str(payload.get("prompt") or ""),
            negative_prompt=str(payload.get("negative_prompt") or ""),
            strength=float(payload.get("strength", 1.0)),
            dilation=int(payload.get("mask_dilation", 15)),
            seed=int(payload.get("seed", -1)),
            filename_prefix=f"{subfolder}/{basename}",
        )
        metadata["operation"] = operation
        metadata["preparation"] = preparation

        async def update_progress(value: float, _node: str | None) -> None:
            self.db.update_job(job["id"], progress=min(0.94, 0.02 + 0.92 * value))

        async def queued(prompt_id: str) -> None:
            self.db.update_job(job["id"], remote_id=prompt_id)

        def cancelled() -> bool:
            return self._job_cancelled(job["id"])

        expected = {
            "filename": f"{basename}_00001_.mp4", "subfolder": subfolder,
            "type": "output",
        }

        def expected_result(value: dict[str, Any] | None) -> bool:
            return bool(value and any(
                item.get("filename") == expected["filename"]
                and item.get("subfolder", "") == expected["subfolder"]
                for item in value.get("files", [])
            ))

        streams = await asyncio.to_thread(probe_stream_types, source)
        preserve_source_audio = "audio" in streams
        visual_path = (
            destination.with_name(destination.stem + "-visual.mp4")
            if preserve_source_audio else destination
        )
        result: dict[str, Any] | None = None
        already_downloaded = False
        existing_remote_id = str((self.db.get_job(job["id"]) or {}).get("remote_id") or "")
        if existing_remote_id:
            existing = await self.client.history(existing_remote_id)
            still_queued = existing is None and await self.client.prompt_in_queue(existing_remote_id)
            if still_queued or (existing and existing["status"] == "running"):
                existing = await self.client.wait_for_completion(
                    existing_remote_id, progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                )
            if expected_result(existing):
                result = existing
        if result is None:
            try:
                result = await self.client.queue_and_wait(
                    graph, client_id=f"{job['id']}-in-outpaint", on_queued=queued,
                    progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                )
            except ComfyOrphanedPromptError:
                try:
                    await self.client.download(expected, visual_path)
                    result = {"status": "done", "files": [expected]}
                    already_downloaded = True
                    metadata["remote_recovery"] = "deterministic-output"
                except (ComfyError, httpx.HTTPError, OSError):
                    await asyncio.sleep(3)
                    result = await self.client.queue_and_wait(
                        graph, client_id=f"{job['id']}-in-outpaint-retry",
                        on_queued=queued, progress=update_progress, cancelled=cancelled,
                        allow_interrupt=self.interrupt_capable,
                    )
                    metadata["remote_recovery"] = "single-automatic-retry"
        if cancelled():
            raise asyncio.CancelledError
        artifact = next(
            (
                item for item in (result or {}).get("files", [])
                if item.get("filename") == expected["filename"]
                and Path(item.get("filename", "")).suffix.lower() in VIDEO_EXTENSIONS
            ),
            None,
        )
        if not artifact:
            raise RuntimeError("In/Outpainting completed without a video output")
        if not already_downloaded:
            await self.client.download(artifact, visual_path)
        audio_restore: dict[str, Any] | None = None
        if preserve_source_audio:
            self.db.update_job(job["id"], progress=0.97)
            audio_restore = await asyncio.to_thread(
                restore_source_audio, visual_path, source, destination,
            )
            visual_path.unlink(missing_ok=True)
        output_info = await asyncio.to_thread(probe_video_info, destination)
        for prepared_path in (prepared_mask,):
            prepared_path.unlink(missing_ok=True)
        if prepared_source != source:
            prepared_source.unlink(missing_ok=True)
        self.db.update_job(
            job["id"], status="succeeded", progress=1.0,
            result_json={
                "local_path": str(destination), "source_path": str(source),
                "target_id": target.id, "target_label": target.label,
                "worker_url": self.client.base_url, "mode": "in-outpainting",
                "operation": operation, "engine": metadata["engine"],
                "workflow": metadata, "source_audio_restored": preserve_source_audio,
                "audio_restore": audio_restore,
                "output_dimensions": [output_info["width"], output_info["height"]],
                "frames": output_info["frames"], "fps": output_info["fps"],
            },
            finished_at=utcnow(),
        )

    async def _execute_lipdub_transform(
        self,
        job: dict[str, Any],
        target: Any,
        source: Path,
        destination: Path,
    ) -> None:
        payload = job["payload"]
        prepared = destination.parent / "prepared" / "lipdub-source.mp4"
        preparation = await asyncio.to_thread(
            prepare_lipdub_source, source, prepared,
            fps=24, max_frames=self.lipdub_adapter.max_frames,
            landscape_size=self.lipdub_adapter.base_landscape_size,
        )
        prepared_info = await asyncio.to_thread(probe_video_info, prepared)
        self.lipdub_adapter.validate_source(prepared_info)
        remote_name = await self.client.upload(prepared)
        basename = "lipdub"
        subfolder = f"vbg/creative-lab/{job['id']}"
        graph, metadata = self.lipdub_adapter.build(
            video_name=remote_name,
            scene_prompt=str(payload.get("prompt") or ""),
            dialogue=str(payload.get("dialogue") or ""),
            language=str(payload.get("language") or "English"),
            width=int(preparation["width"]), height=int(preparation["height"]),
            frames=int(preparation["frames"]), fps=int(preparation["fps"]),
            negative_prompt=str(payload.get("negative_prompt") or ""),
            strength=float(payload.get("strength", 1.0)),
            seed=int(payload.get("seed", -1)),
            filename_prefix=f"{subfolder}/{basename}",
        )
        metadata["preparation"] = preparation

        async def update_progress(value: float, _node: str | None) -> None:
            self.db.update_job(job["id"], progress=min(0.96, 0.03 + 0.93 * value))

        async def queued(prompt_id: str) -> None:
            self.db.update_job(job["id"], remote_id=prompt_id)

        def cancelled() -> bool:
            return self._job_cancelled(job["id"])

        expected = {
            "filename": f"{basename}_00001_.mp4", "subfolder": subfolder,
            "type": "output",
        }

        def expected_result(value: dict[str, Any] | None) -> bool:
            return bool(value and any(
                item.get("filename") == expected["filename"]
                and item.get("subfolder", "") == expected["subfolder"]
                for item in value.get("files", [])
            ))

        result: dict[str, Any] | None = None
        already_downloaded = False
        existing_remote_id = str((self.db.get_job(job["id"]) or {}).get("remote_id") or "")
        if existing_remote_id:
            existing = await self.client.history(existing_remote_id)
            still_queued = existing is None and await self.client.prompt_in_queue(existing_remote_id)
            if still_queued or (existing and existing["status"] == "running"):
                existing = await self.client.wait_for_completion(
                    existing_remote_id, progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                )
            if expected_result(existing):
                result = existing
        if result is None:
            try:
                result = await self.client.queue_and_wait(
                    graph, client_id=f"{job['id']}-lipdub", on_queued=queued,
                    progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                )
            except ComfyOrphanedPromptError:
                try:
                    await self.client.download(expected, destination)
                    result = {"status": "done", "files": [expected]}
                    already_downloaded = True
                    metadata["remote_recovery"] = "deterministic-output"
                except (ComfyError, httpx.HTTPError, OSError):
                    await asyncio.sleep(3)
                    result = await self.client.queue_and_wait(
                        graph, client_id=f"{job['id']}-lipdub-retry",
                        on_queued=queued, progress=update_progress, cancelled=cancelled,
                        allow_interrupt=self.interrupt_capable,
                    )
                    metadata["remote_recovery"] = "single-automatic-retry"
        if cancelled():
            raise asyncio.CancelledError
        artifact = next(
            (
                item for item in (result or {}).get("files", [])
                if item.get("filename") == expected["filename"]
                and Path(item.get("filename", "")).suffix.lower() in VIDEO_EXTENSIONS
            ),
            None,
        )
        if not artifact:
            raise RuntimeError("LipDub completed without an audio-video output")
        if not already_downloaded:
            await self.client.download(artifact, destination)
        prepared.unlink(missing_ok=True)
        output_info = await asyncio.to_thread(probe_video_info, destination)
        self.db.update_job(
            job["id"], status="succeeded", progress=1.0,
            result_json={
                "local_path": str(destination), "source_path": str(source),
                "target_id": target.id, "target_label": target.label,
                "worker_url": self.client.base_url, "mode": "lipdub",
                "engine": metadata["engine"], "workflow": metadata,
                "dialogue": str(payload.get("dialogue") or ""),
                "language": str(payload.get("language") or "English"),
                "audio_authority": "generated-lipdub-dialogue",
                "output_dimensions": [output_info["width"], output_info["height"]],
                "frames": output_info["frames"], "fps": output_info["fps"],
            },
            finished_at=utcnow(),
        )

    async def _execute_foley_transform(
        self,
        job: dict[str, Any],
        target: Any,
        source: Path,
        destination: Path,
        info: dict[str, Any],
    ) -> None:
        """Generate only Foley audio, then mux it onto untouched source frames."""
        payload = job["payload"]
        frames = int(info["frames"])
        if frames <= 0:
            raise RuntimeError("Foley requires a source with a countable video frame stream")
        if frames > self.foley_adapter.max_frames:
            raise RuntimeError(
                f"Foley V2A currently supports up to {self.foley_adapter.max_frames} frames "
                f"per 24 GB pass; this clip has {frames} frames"
            )
        fps = float(info.get("fps") or 24.0)
        duration = await asyncio.to_thread(probe_duration, source)
        remote_name = await self.client.upload(source)
        basename = "foley"
        subfolder = f"vbg/creative-lab/{job['id']}"
        graph, metadata = self.foley_adapter.build(
            video_name=remote_name,
            prompt=str(payload.get("prompt") or "A natural cinematic scene"),
            negative_prompt=str(payload.get("negative_prompt") or ""),
            strength=float(payload.get("strength", 1.0)),
            seed=int(payload.get("seed", -1)),
            duration_seconds=duration,
            filename_prefix=f"{subfolder}/{basename}",
        )

        async def update_progress(value: float, _node: str | None) -> None:
            self.db.update_job(job["id"], progress=min(0.94, 0.02 + 0.92 * value))

        async def queued(prompt_id: str) -> None:
            self.db.update_job(job["id"], remote_id=prompt_id)

        def cancelled() -> bool:
            return self._job_cancelled(job["id"])

        expected = {
            "filename": f"{basename}_00001_.flac", "subfolder": subfolder,
            "type": "output",
        }

        def audio_artifact(value: dict[str, Any] | None) -> dict[str, Any] | None:
            return next(
                (
                    item for item in (value or {}).get("files", [])
                    if Path(str(item.get("filename") or "")).suffix.lower()
                    in AUDIO_EXTENSIONS
                ),
                None,
            )

        result: dict[str, Any] | None = None
        existing_remote_id = str((self.db.get_job(job["id"]) or {}).get("remote_id") or "")
        if existing_remote_id:
            existing = await self.client.history(existing_remote_id)
            still_queued = existing is None and await self.client.prompt_in_queue(existing_remote_id)
            if still_queued or (existing and existing["status"] == "running"):
                existing = await self.client.wait_for_completion(
                    existing_remote_id, progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                )
            if audio_artifact(existing):
                result = existing

        generated_audio = destination.with_suffix(".foley.flac")
        already_downloaded = False
        if result is None:
            try:
                result = await self.client.queue_and_wait(
                    graph, client_id=f"{job['id']}-foley", on_queued=queued,
                    progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                )
            except ComfyOrphanedPromptError:
                try:
                    await self.client.download(expected, generated_audio)
                    result = {"status": "done", "files": [expected]}
                    already_downloaded = True
                    metadata["remote_recovery"] = "deterministic-audio-output"
                except (ComfyError, httpx.HTTPError, OSError):
                    await asyncio.sleep(3)
                    result = await self.client.queue_and_wait(
                        graph, client_id=f"{job['id']}-foley-retry", on_queued=queued,
                        progress=update_progress, cancelled=cancelled,
                        allow_interrupt=self.interrupt_capable,
                    )
                    metadata["remote_recovery"] = "single-automatic-retry"
        if cancelled():
            raise asyncio.CancelledError
        artifact = audio_artifact(result)
        if not artifact:
            raise RuntimeError("Foley V2A completed without an audio output")
        if not already_downloaded:
            await self.client.download(artifact, generated_audio)
        self.db.update_job(job["id"], progress=0.97)
        mux = await asyncio.to_thread(
            mux_generated_foley, source, generated_audio, destination,
        )
        generated_audio.unlink(missing_ok=True)
        output_info = await asyncio.to_thread(probe_video_info, destination)
        self.db.update_job(
            job["id"], status="succeeded", progress=1.0,
            result_json={
                "local_path": str(destination), "source_path": str(source),
                "target_id": target.id, "target_label": target.label,
                "worker_url": self.client.base_url, "mode": "foley-v2a",
                "engine": metadata["engine"], "workflow": metadata,
                "foley_mux": mux, "source_audio_restored": False,
                "source_audio_replaced": mux["source_audio_replaced"],
                "output_dimensions": [output_info["width"], output_info["height"]],
                "frames": output_info["frames"], "fps": fps,
            },
            finished_at=utcnow(),
        )

    async def _execute_ingredients(self, job: dict[str, Any]) -> None:
        payload = job["payload"]
        candidate_id = str(job.get("candidate_id") or "")
        candidate = self.db.get_candidate(candidate_id) if candidate_id else None
        if not candidate:
            raise RuntimeError("Ingredients job has no candidate")
        target = self.settings.execution_target(self.target_id)
        if not target or target.kind != "comfyui" or "ltx-generation" not in target.capabilities:
            raise RuntimeError("The selected target cannot run Ingredients generation")
        source = Path(str(payload.get("reference_image_path") or ""))
        if not source.exists():
            raise RuntimeError("The Ingredients visual bible is no longer available")
        settings = payload["settings"]
        requested_duration = float(settings["duration_seconds"])
        if requested_duration > self.ingredients_adapter.max_duration_seconds:
            raise RuntimeError("Ingredients story beats must be 5 seconds or shorter")
        self.db.update_candidate(candidate_id, status="generating")

        work_dir = (
            self.settings.artifact_dir / candidate["project_id"] / candidate["shot_id"]
            / ".ingredients"
        )
        prepared_sheet = work_dir / f"{job['id']}-reference.png"
        preparation = await asyncio.to_thread(
            prepare_reference_sheet, source, prepared_sheet,
            self.ingredients_adapter.width, self.ingredients_adapter.height,
        )
        remote_name = await self.client.upload(prepared_sheet)
        basename = candidate_id
        subfolder = f"vbg/ingredients/{job['id']}"
        graph, workflow_metadata = self.ingredients_adapter.build(
            image_name=remote_name,
            reference_description=str(payload["reference_sheet_description"]),
            shot_prompt=str(settings.get("workflow_prompt") or candidate["prompt"]),
            negative_prompt=str(settings.get("negative_prompt") or ""),
            strength=float(settings.get("ingredients_strength", 1.0)),
            seed=int(candidate["seed"]),
            filename_prefix=f"{subfolder}/{basename}",
        )
        workflow_metadata["reference_preparation"] = preparation

        async def update_progress(value: float, _node: str | None) -> None:
            self.db.update_job(job["id"], progress=min(0.94, 0.02 + 0.92 * value))

        async def queued(prompt_id: str) -> None:
            self.db.update_job(job["id"], remote_id=prompt_id)

        def cancelled() -> bool:
            return self._job_cancelled(job["id"])

        expected = {
            "filename": f"{basename}_00001_.mp4", "subfolder": subfolder,
            "type": "output",
        }

        def expected_result(value: dict[str, Any] | None) -> bool:
            return bool(value and any(
                item.get("filename") == expected["filename"]
                and item.get("subfolder", "") == expected["subfolder"]
                for item in value.get("files", [])
            ))

        result: dict[str, Any] | None = None
        raw_path = work_dir / f"{job['id']}-raw.mp4"
        already_downloaded = False
        existing_remote_id = str((self.db.get_job(job["id"]) or {}).get("remote_id") or "")
        if existing_remote_id:
            existing = await self.client.history(existing_remote_id)
            still_queued = existing is None and await self.client.prompt_in_queue(
                existing_remote_id
            )
            if still_queued or (existing and existing["status"] == "running"):
                existing = await self.client.wait_for_completion(
                    existing_remote_id, progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                )
            if expected_result(existing):
                result = existing
        if result is None:
            try:
                result = await self.client.queue_and_wait(
                    graph, client_id=f"{job['id']}-ingredients", on_queued=queued,
                    progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                )
            except ComfyOrphanedPromptError:
                try:
                    await self.client.download(expected, raw_path)
                    result = {"status": "done", "files": [expected]}
                    already_downloaded = True
                    workflow_metadata["remote_recovery"] = "deterministic-output"
                except (ComfyError, httpx.HTTPError, OSError):
                    await asyncio.sleep(3)
                    result = await self.client.queue_and_wait(
                        graph, client_id=f"{job['id']}-ingredients-retry",
                        on_queued=queued, progress=update_progress, cancelled=cancelled,
                        allow_interrupt=self.interrupt_capable,
                    )
                    workflow_metadata["remote_recovery"] = "single-automatic-retry"
        if cancelled():
            raise asyncio.CancelledError
        artifact = next(
            (
                item for item in (result or {}).get("files", [])
                if item.get("filename") == expected["filename"]
                and Path(item.get("filename", "")).suffix.lower() in VIDEO_EXTENSIONS
            ),
            None,
        )
        if not artifact:
            raise RuntimeError("Ingredients completed without a video output")
        if not already_downloaded:
            await self.client.download(artifact, raw_path)

        local_path = (
            self.settings.artifact_dir / candidate["project_id"] / candidate["shot_id"]
            / f"{candidate_id}.mp4"
        )
        self.db.update_job(job["id"], progress=0.96)
        trim_metadata = await asyncio.to_thread(
            trim_silent_video, raw_path, local_path, requested_duration,
        )
        output_info = await asyncio.to_thread(probe_video_info, local_path)
        continuity_assets = await asyncio.to_thread(
            extract_continuity_assets, local_path, local_path.parent / "continuity"
        )
        raw_path.unlink(missing_ok=True)
        prepared_sheet.unlink(missing_ok=True)
        saved_artifact = {
            **artifact, "local_path": str(local_path),
            "worker_url": self.client.base_url,
            "remote_prompt_id": str(
                (self.db.get_job(job["id"]) or {}).get("remote_id") or ""
            ),
            "workflow": {**workflow_metadata, "trim": trim_metadata},
            "continuity": {
                "mode": "ingredients-reference-sheet",
                "reference_image_asset_id": settings.get("reference_image_asset_id"),
            },
            "reference_audio": None,
            **continuity_assets,
        }
        self.db.update_candidate(
            candidate_id, status="generated", artifact_json=saved_artifact
        )
        score_job = None
        if self.settings.vision_scoring_enabled:
            score_job = self.db.create_job(
                "score", "local", {"candidate_id": candidate_id},
                project_id=candidate["project_id"], shot_id=candidate["shot_id"],
                candidate_id=candidate_id, max_attempts=1,
            )
        else:
            self.db.update_candidate(
                candidate_id, status="unscored",
                score_json={
                    "available": False, "judge": "manual-review",
                    "summary": (
                        "Vision scoring is disabled; review reference-sheet identity by eye."
                    ),
                    "issues": [],
                },
                total_score=None,
            )
            self.db.auto_select_candidate(candidate["shot_id"])
        self.db.update_job(
            job["id"], status="succeeded", progress=1.0,
            result_json={
                "artifact": saved_artifact,
                "scoring": "queued" if score_job else "manual-review",
                "output_dimensions": [output_info["width"], output_info["height"]],
                "frames": output_info["frames"],
                **({"score_job_id": score_job["id"]} if score_job else {}),
            },
            finished_at=utcnow(),
        )

    def _job_cancelled(self, job_id: str) -> bool:
        current = self.db.get_job(job_id)
        return bool(current and current.get("cancel_requested"))

    async def _queue_pixel_upscale_pass(
        self, *, job: dict[str, Any], source: Path, destination: Path,
        source_width: int, source_height: int, has_audio: bool,
        prompt: str, negative_prompt: str, scale: int, seed: int,
        pass_index: int, pass_count: int,
    ) -> dict[str, Any]:
        remote_name = await self.client.upload(source)
        basename = f"pass-{pass_index:03d}"
        subfolder = f"vbg/upscales/{job['id']}"
        graph, metadata = self.upscale_adapter.build(
            video_name=remote_name, prompt=prompt, negative_prompt=negative_prompt,
            scale=scale, source_width=source_width, source_height=source_height,
            has_audio=has_audio, seed=seed,
            filename_prefix=f"{subfolder}/{basename}",
        )

        async def update_progress(value: float, _node: str | None) -> None:
            overall = 0.02 + 0.92 * ((pass_index + min(1.0, value)) / pass_count)
            self.db.update_job(job["id"], progress=min(0.94, overall))

        async def queued(prompt_id: str) -> None:
            self.db.update_job(job["id"], remote_id=prompt_id)

        def cancelled() -> bool:
            return self._job_cancelled(job["id"])

        expected = {
            "filename": f"{basename}_00001_.mp4", "subfolder": subfolder, "type": "output",
        }

        def expected_result(value: dict[str, Any] | None) -> bool:
            return bool(value and any(
                item.get("filename") == expected["filename"]
                and item.get("subfolder", "") == expected["subfolder"]
                for item in value.get("files", [])
            ))

        result: dict[str, Any] | None = None
        already_downloaded = False
        existing_remote_id = str((self.db.get_job(job["id"]) or {}).get("remote_id") or "")
        if existing_remote_id:
            existing = await self.client.history(existing_remote_id)
            still_queued = existing is None and await self.client.prompt_in_queue(existing_remote_id)
            if still_queued or (existing and existing["status"] == "running"):
                existing = await self.client.wait_for_completion(
                    existing_remote_id, progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                    timeout_seconds=max(7200, self.client.timeout_seconds),
                )
            if expected_result(existing):
                result = existing

        if result is None:
            try:
                result = await self.client.queue_and_wait(
                    graph, client_id=f"{job['id']}-{basename}", on_queued=queued,
                    progress=update_progress, cancelled=cancelled,
                    allow_interrupt=self.interrupt_capable,
                    timeout_seconds=max(7200, self.client.timeout_seconds),
                )
            except ComfyOrphanedPromptError:
                try:
                    await self.client.download(expected, destination)
                    result = {"status": "done", "files": [expected]}
                    already_downloaded = True
                    metadata["remote_recovery"] = "deterministic-output"
                except (ComfyError, httpx.HTTPError, OSError):
                    await asyncio.sleep(3)
                    result = await self.client.queue_and_wait(
                        graph, client_id=f"{job['id']}-{basename}-retry",
                        on_queued=queued, progress=update_progress, cancelled=cancelled,
                        allow_interrupt=self.interrupt_capable,
                        timeout_seconds=max(7200, self.client.timeout_seconds),
                    )
                    metadata["remote_recovery"] = "single-automatic-retry"

        if cancelled():
            raise asyncio.CancelledError
        artifact = next(
            (
                item for item in (result or {}).get("files", [])
                if Path(item.get("filename", "")).suffix.lower() in VIDEO_EXTENSIONS
                and item.get("filename") == expected["filename"]
            ),
            None,
        )
        if not artifact:
            raise RuntimeError(
                f"Pixel Spatial pass {pass_index + 1}/{pass_count} completed without video"
            )
        if not already_downloaded:
            await self.client.download(artifact, destination)
        metadata["pass"] = pass_index + 1
        metadata["pass_count"] = pass_count
        return metadata


class LocalWorker:
    def __init__(
        self, db: Database, settings: Settings, director: CreativeDirector,
        scorer: MediaScorer,
    ):
        self.db = db
        self.settings = settings
        self.director = director
        self.scorer = scorer
        self.worker_id = "local:orchestrator"
        self.running = False
        self.active_job_id: str | None = None

    async def run(self) -> None:
        self.running = True
        while self.running:
            job = self.db.claim_job("local", self.worker_id)
            if not job:
                job = self.db.claim_job("postprocess", self.worker_id)
            if not job:
                await asyncio.sleep(0.5)
                continue
            self.active_job_id = job["id"]
            await self._execute(job)
            self.active_job_id = None

    async def stop(self) -> None:
        self.running = False

    async def _execute(self, job: dict[str, Any]) -> None:
        try:
            if job["kind"] == "creative_plan":
                result = await self._plan(job)
            elif job["kind"] == "score":
                result = await self._score(job)
            elif job["kind"] == "export":
                result = await asyncio.to_thread(self._export, job)
            elif job["kind"] == "chain_finish":
                result = await asyncio.to_thread(self._chain_finish, job)
            elif job["kind"] == "upscale":
                result = await asyncio.to_thread(self._upscale, job)
            else:
                raise RuntimeError(f"Unsupported local job kind: {job['kind']}")
            self.db.update_job(
                job["id"], status="succeeded", progress=1.0, result_json=result,
                finished_at=utcnow(),
            )
        except Exception as exc:
            if job["kind"] == "export":
                export_id = job.get("payload", {}).get("export_id")
                if export_id:
                    self.db.update_export(export_id, status="failed", metadata_json={"error": str(exc)})
                if job.get("project_id"):
                    self.db.set_project_status(job["project_id"], "export_failed")
            elif job["kind"] == "chain_finish":
                chain_id = str(job.get("payload", {}).get("chain_id") or "")
                chain = self.db.get_chain(chain_id) if chain_id else None
                if chain and chain.get("finish_job_id") == job["id"]:
                    self.db.update_chain_finish_failure(chain_id, str(exc))
            self.db.update_job(
                job["id"], status="failed",
                error=f"{exc}\n{traceback.format_exc(limit=5)}", finished_at=utcnow(),
            )

    def _upscale(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job["payload"]
        target_id = str(payload["target_id"])
        target = self.settings.execution_target(target_id)
        if not target or "post-upscale" not in target.capabilities:
            raise RuntimeError(f"Upscale target is not configured: {target_id}")
        source = Path(payload["source_path"])
        destination = Path(payload["destination_path"])
        if not source.exists():
            raise RuntimeError("The source video is no longer available")
        destination.parent.mkdir(parents=True, exist_ok=True)
        scale = int(payload.get("scale", 2))
        self.db.update_job(job["id"], progress=0.08)
        if target.kind == "local-video2x":
            executable = shutil.which(self.settings.local_upscaler_binary)
            if not executable:
                raise RuntimeError(
                    f"{target.label} is configured but {self.settings.local_upscaler_binary!r} "
                    "is not installed or not visible to the API process"
                )
            command = [
                executable, "-i", str(source), "-o", str(destination),
                "-p", "realesrgan", "-s", str(scale),
                "--realesrgan-model", "realesrgan-plus", "-g", str(target.gpu_index),
            ]
            result = subprocess.run(
                command, capture_output=True, text=True,
                timeout=max(1800, self.settings.generation_timeout_seconds * 2),
            )
            if result.returncode != 0:
                message = (result.stderr or result.stdout or "Video2X failed").strip()
                raise RuntimeError(message[-1_000:])
        elif target.kind == "ssh-video2x":
            # Paths are generated locally from the job ID and contain no user
            # input. SSH hosts are trusted environment configuration, never a
            # browser-supplied value.
            safe_job = "".join(char for char in job["id"] if char.isalnum() or char in "_-")
            remote_source = f"/tmp/vbg-{safe_job}-source{source.suffix or '.mp4'}"
            remote_output = f"/tmp/vbg-{safe_job}-upscaled.mp4"
            ssh = ["ssh", "-F", "/dev/null", "-o", "BatchMode=yes", target.ssh_host or ""]
            try:
                with source.open("rb") as handle:
                    uploaded = subprocess.run(
                        [*ssh, f"cat > {remote_source}"], stdin=handle,
                        capture_output=True, timeout=600,
                    )
                if uploaded.returncode != 0:
                    raise RuntimeError(uploaded.stderr.decode(errors="replace")[-1_000:])
                rendered = subprocess.run(
                    [
                        *ssh,
                        f"video2x -i {remote_source} -o {remote_output} -p realesrgan "
                        f"-s {scale} --realesrgan-model realesrgan-plus -g {target.gpu_index}",
                    ],
                    capture_output=True, text=True,
                    timeout=max(1800, self.settings.generation_timeout_seconds * 2),
                )
                if rendered.returncode != 0:
                    raise RuntimeError((rendered.stderr or rendered.stdout)[-1_000:])
                with destination.open("wb") as handle:
                    downloaded = subprocess.run(
                        [*ssh, "cat", remote_output], stdout=handle, stderr=subprocess.PIPE,
                        timeout=900,
                    )
                if downloaded.returncode != 0:
                    destination.unlink(missing_ok=True)
                    raise RuntimeError(downloaded.stderr.decode(errors="replace")[-1_000:])
            finally:
                subprocess.run(
                    [*ssh, "rm", "-f", remote_source, remote_output],
                    capture_output=True, timeout=30,
                )
        else:
            raise RuntimeError(f"Unsupported upscale worker kind: {target.kind}")
        if not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError("The upscaler finished without creating an output video")
        return {
            "local_path": str(destination), "target_id": target.id,
            "target_label": target.label, "scale": scale,
            "engine": "Video2X / Real-ESRGAN", "source_path": str(source),
        }

    async def _plan(self, job: dict[str, Any]) -> dict[str, Any]:
        project = self.db.get_project(job["project_id"])
        if not project:
            raise RuntimeError("Project no longer exists")
        self.db.update_job(job["id"], progress=0.1)
        learning_context = self.db.creative_learning_context(project["id"])
        plan = await self.director.plan(
            project["brief"], int(job["payload"].get("concept_count", 3)),
            learning_context,
        )
        self.db.replace_plan(project["id"], plan["concepts"])
        return {
            "provider": plan["provider"], "model": plan["model"],
            "concept_count": len(plan["concepts"]),
            "shot_count": sum(len(value.get("shots", [])) for value in plan["concepts"]),
            "learning_examples": int(plan.get("learning_examples", 0)),
        }

    async def _score(self, job: dict[str, Any]) -> dict[str, Any]:
        candidate = self.db.get_candidate(job["candidate_id"])
        if not candidate or not candidate.get("artifact"):
            raise RuntimeError("Candidate output is unavailable for scoring")
        project = self.db.get_project(candidate["project_id"])
        self.db.update_job(job["id"], progress=0.2)
        score = await self.scorer.score(
            Path(candidate["artifact"]["local_path"]), candidate["prompt"],
            project["brief"] if project else {},
        )
        if score["total"] is None:
            # A technical probe is not a creative opinion. Preserve the media
            # for manual review and make it explicitly rescoreable instead of
            # publishing the old neutral 73 for every clip.
            self.db.update_candidate(
                candidate["id"], status="unscored", score_json=score, total_score=None
            )
            self.db.auto_select_candidate(candidate["shot_id"])
            return {
                "score": score,
                "auto_selected_candidate_id": None,
                "rescore_required": True,
            }
        self.db.update_candidate(
            candidate["id"], status="scored", score_json=score, total_score=score["total"]
        )
        selected = self.db.auto_select_candidate(candidate["shot_id"])
        if selected:
            self.db.release_next_shot_jobs(candidate["shot_id"])
        return {"score": score, "auto_selected_candidate_id": selected["id"] if selected else None}

    def _export(self, job: dict[str, Any]) -> dict[str, Any]:
        export_id = job["payload"]["export_id"]
        export = self.db.get_export(export_id)
        if not export:
            raise RuntimeError("Export no longer exists")
        concepts = self.db.list_concepts(export["project_id"])
        concept = next((value for value in concepts if value["selected"]), concepts[0] if concepts else None)
        if not concept:
            raise RuntimeError("Project has no selected concept")
        shots = self.db.list_shots(export["project_id"], concept["id"])
        clips: list[Path] = []
        for shot in shots:
            candidate_id = shot.get("selected_candidate_id")
            candidate = self.db.get_candidate(candidate_id) if candidate_id else None
            if not candidate or not candidate.get("artifact"):
                raise RuntimeError(f"Shot {shot['position'] + 1} has no selected output")
            clips.append(Path(candidate["artifact"]["local_path"]))
        options = export["options"]
        music = self.db.get_asset(options["music_asset_id"]) if options.get("music_asset_id") else None
        voice = self.db.get_asset(options["voiceover_asset_id"]) if options.get("voiceover_asset_id") else None
        logo = self.db.get_asset(options["logo_asset_id"]) if options.get("logo_asset_id") else None
        destination = self.settings.artifact_dir / export["project_id"] / "exports" / f"{export_id}.mp4"
        self.db.update_export(export_id, status="rendering")
        result = render_timeline(
            clips, destination, options,
            Path(music["local_path"]) if music else None,
            Path(voice["local_path"]) if voice else None,
            Path(logo["local_path"]) if logo else None,
        )
        self.db.update_export(export_id, status="ready", local_path=str(destination), metadata_json=result)
        self.db.set_project_status(export["project_id"], "exported")
        return result

    def _chain_finish(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job["payload"]
        chain_id = str(payload["chain_id"])
        chain = self.db.get_chain(chain_id)
        if not chain:
            raise RuntimeError("Quick story no longer exists")
        if chain.get("finish_job_id") != job["id"]:
            raise RuntimeError("This social finish was superseded by a newer edit")
        clips = [Path(value) for value in payload.get("clip_paths", [])]
        if not clips or any(not path.exists() for path in clips):
            raise RuntimeError("One or more kept story clips are no longer available")
        self.db.update_job(job["id"], progress=0.12)

        def optional_path(name: str) -> Path | None:
            value = str(payload.get(name) or "")
            if not value:
                return None
            path = Path(value)
            if not path.exists():
                raise RuntimeError(f"The uploaded {name.removesuffix('_path')} is missing")
            return path

        destination = Path(payload["destination_path"])
        self.db.update_job(job["id"], progress=0.25)
        result = render_timeline(
            clips, destination, payload["options"],
            optional_path("music_path"), optional_path("voiceover_path"),
            optional_path("logo_path"),
        )
        self.db.update_chain_finish(
            chain_id, finished_path=str(destination), metadata=result
        )
        return {**result, "chain_id": chain_id}


class WorkerPool:
    def __init__(self, generation_workers: list[GenerationWorker], local_worker: LocalWorker):
        self.generation_workers = generation_workers
        self.local_worker = local_worker
        self.tasks: list[asyncio.Task[None]] = []

    def start(self) -> None:
        if self.tasks:
            return
        self.tasks = [
            asyncio.create_task(worker.run(), name=worker.worker_id)
            for worker in self.generation_workers
        ]
        self.tasks.append(asyncio.create_task(self.local_worker.run(), name=self.local_worker.worker_id))

    async def stop(self) -> None:
        for worker in self.generation_workers:
            await worker.stop()
        await self.local_worker.stop()
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

    def status(self) -> list[dict[str, Any]]:
        return [
            {
                "id": worker.worker_id,
                "url": worker.client.base_url,
                "lane": "comfy",
                "target_id": worker.target_id,
                "upload_capable": worker.upload_capable,
                "interrupt_capable": worker.interrupt_capable,
                "exclusive_capable": worker.exclusive_capable,
                "active_job_id": worker.active_job_id,
            }
            for worker in self.generation_workers
        ] + [
            {
                "id": self.local_worker.worker_id,
                "lane": "local",
                "target_id": "local",
                "active_job_id": self.local_worker.active_job_id,
            }
        ]
