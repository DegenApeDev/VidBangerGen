from __future__ import annotations

import asyncio
import json
import math
import mimetypes
import secrets
import shutil
import subprocess
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator
from urllib.parse import urlencode

import httpx
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from .comfy import ComfyClient, ComfyError, ComfyOrphanedPromptError
from .audio_prompt import compile_audio_prompt, preview_audio_prompt
from .config import SETTINGS, Settings
from .creative import CreativeDirector
from .model_inventory import query_remote_model_inventory, required_model_files
from .database import Database, new_id, utcnow
from .gpu import query_gpu_status
from .jobs import ExclusiveInferenceCoordinator, GenerationWorker, LocalWorker, WorkerPool
from .media import (
    extract_continuity_assets,
    merge_chain_clips,
    probe_stream_types,
    probe_video_info,
    restore_source_audio,
    safe_filename,
)
from .schemas import (
    ApiMessage,
    CandidateBatchRequest,
    CinemagraphRequest,
    CreativeTransformRequest,
    ExecutionTargetRequest,
    ExportRequest,
    FeedbackRequest,
    LegacyT2VRequest,
    ManualPlanRequest,
    PlanRequest,
    ProjectCreate,
    PromptPreviewRequest,
    QueueResponse,
    RetakeRequest,
    SelectCandidateRequest,
    ShotCreate,
    ShotUpdate,
    UpscaleRequest,
)
from .scoring import MediaScorer
from .studio import ASPECT_DRAFT_SIZES, StudioService
from .workflow import LTXWorkflowAdapter, WorkflowError, WorkflowStore


CREATIVE_LAB_MANIFEST = Path(__file__).parent / "workflows" / "creative_lab.json"
WEB_DIST = Path(__file__).parents[1] / "web" / "dist"
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
VISUAL_REFERENCE_KINDS = {"image", "reference_sheet", "brand"}


def _estimate_seconds(width: int, height: int, frames: int) -> int:
    return max(20, round(184 * (width * height / (384 * 256)) * (frames / 121)))


def _continuation_safe_size(width: int, height: int) -> tuple[int, int]:
    """Bound motion-tail conditioning to the proven single-3090 draft canvas.

    A continuation carries 17 guide frames plus the IC LoRA, so its peak memory
    is higher than an ordinary T2V/I2V render at the same requested size. The
    overlap-aware merger scales the result back to the opening clip's canvas.
    """
    if width * height <= 448 * 256:
        return width, height
    ratio = width / height
    if ratio >= 1.2:
        return 448, 256
    if ratio <= 1 / 1.2:
        return 256, 448
    return 320, 320


def create_app(settings: Settings = SETTINGS, *, start_workers: bool = True) -> FastAPI:
    settings.ensure_directories()
    db = Database(settings.database_path)
    # Constructing a test app must not recover jobs owned by a live process.
    db.initialize(recover_running=False)
    if settings.legacy_chains_path:
        db.import_legacy_chains(settings.legacy_chains_path)

    store = WorkflowStore(settings)
    adapter = LTXWorkflowAdapter(store)
    clients = [
        ComfyClient(url, timeout_seconds=settings.generation_timeout_seconds)
        for url in settings.comfyui_urls
    ]
    # Chain continuations predate the durable job worker. Track the ones owned
    # by this API process so read-side recovery never races their normal
    # completion/download path. The set is intentionally empty after restart,
    # allowing persisted remote prompt IDs to be reconciled.
    active_chain_clips: set[str] = set()
    director = CreativeDirector(settings)
    scorer = MediaScorer(settings)
    studio = StudioService(db, store)
    coordinator = ExclusiveInferenceCoordinator(clients, settings.managed_comfy_urls)
    generation_workers = [
        GenerationWorker(db, settings, adapter, client, coordinator) for client in clients
    ]
    local_worker = LocalWorker(db, settings, director, scorer)
    pool = WorkerPool(generation_workers, local_worker)

    def resolve_target(target_id: str, capability: str):
        if target_id == "auto":
            return None
        target = settings.execution_target(target_id)
        if not target:
            raise HTTPException(422, f"Unknown execution target: {target_id}")
        if capability not in target.capabilities:
            raise HTTPException(
                422, f"{target.label} cannot run {capability.replace('-', ' ')} jobs"
            )
        return target

    def eligible_clients(target_id: str, lane: str) -> list[ComfyClient]:
        target = resolve_target(
            target_id, "continuation" if lane == "comfy_upload" else "ltx-generation"
        )
        allowed_urls = set(target.urls) if target else None
        values = [
            client for client in clients
            if allowed_urls is None or client.base_url in allowed_urls
        ]
        if lane == "comfy_upload":
            values = [
                client for client in values
                if not settings.upload_capable_urls
                or client.base_url in settings.upload_capable_urls
            ]
        elif lane == "comfy_exclusive":
            values = [
                client for client in values
                if client.base_url == settings.exclusive_comfy_url
            ]
        return values

    async def require_inference_capacity(
        lane: str = "comfy", target_id: str = "auto"
    ) -> None:
        """Reject before durable mutations if the required worker class is offline."""
        if not start_workers:
            resolve_target(
                target_id, "continuation" if lane == "comfy_upload" else "ltx-generation"
            )
            return
        eligible = eligible_clients(target_id, lane)
        if not eligible:
            target = settings.execution_target(target_id) if target_id != "auto" else None
            label = target.label if target else "the automatic target pool"
            raise HTTPException(503, f"No configured {lane} worker exists on {label}")
        checks = await asyncio.gather(
            *(client.health() for client in eligible), return_exceptions=True
        )
        healthy = {
            client.base_url for client, result in zip(eligible, checks)
            if not isinstance(result, Exception)
        }
        if lane == "comfy_upload":
            ready = healthy
            selected_target = (
                settings.execution_target(target_id) if target_id != "auto" else None
            )
            requirement = (
                f"upload-capable {selected_target.label} worker"
                if selected_target else "upload-capable ComfyUI worker"
            )
        elif lane == "comfy_exclusive":
            ready = (
                healthy
                if settings.exclusive_comfy_url in healthy and len(healthy) == len(clients)
                else set()
            )
            requirement = "exclusive inference worker and every scheduler peer"
        else:
            ready = healthy
            requirement = "LTX inference worker"
        if not ready:
            configured = ", ".join(client.base_url for client in eligible)
            raise HTTPException(
                503,
                f"No healthy {requirement} is available. No generation work was queued. "
                f"Configured workers: {configured}",
            )

    async def choose_client(target_id: str, lane: str) -> ComfyClient:
        values = eligible_clients(target_id, lane)
        checks = await asyncio.gather(
            *(client.health() for client in values), return_exceptions=True
        )
        healthy = [
            client for client, result in zip(values, checks)
            if not isinstance(result, Exception)
        ]
        if not healthy:
            await require_inference_capacity(lane, target_id)
            raise HTTPException(503, "No healthy target was found")
        # Preserve the stable shared-worker behavior for raw T2V; conditioned
        # work has already been restricted to upload-capable workers above.
        if lane == "comfy":
            return next(
                (
                    client for client in healthy
                    if client.base_url not in settings.upload_capable_urls
                ),
                healthy[0],
            )
        return healthy[0]

    async def require_upscale_capacity(target_id: str):
        target = resolve_target(target_id, "post-upscale")
        if target is None:
            raise HTTPException(422, "Choose a specific post-upscale target")
        if target.kind == "local-video2x":
            if not shutil.which(settings.local_upscaler_binary):
                raise HTTPException(
                    409,
                    f"{target.label} is configured, but {settings.local_upscaler_binary} "
                    "is not installed for the API user yet",
                )
        elif target.kind == "ssh-video2x":
            def probe() -> bool:
                value = subprocess.run(
                    [
                        "ssh", "-F", "/dev/null", "-o", "BatchMode=yes",
                        "-o", "ConnectTimeout=4", target.ssh_host or "",
                        "command -v video2x",
                    ],
                    capture_output=True, timeout=7,
                )
                return value.returncode == 0

            try:
                ready = await asyncio.to_thread(probe)
            except (OSError, subprocess.SubprocessError):
                ready = False
            if not ready:
                raise HTTPException(503, f"{target.label} cannot currently run Video2X")
        elif target.kind == "comfyui":
            await require_inference_capacity("comfy_upload", target.id)
        return target

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if start_workers:
            db.retire_unsafe_studio_prompt_audio_jobs()
            db.recover_jobs()
            db.reconcile_project_statuses()
            pool.start()
        yield
        if start_workers:
            await pool.stop()

    app = FastAPI(
        title="VidBangerGen",
        version="1.0.0",
        description="Local-first creative planning and LTX 2.3 production studio.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.db = db
    app.state.workflow_store = store
    app.state.adapter = adapter
    app.state.comfy_clients = clients
    app.state.studio = studio
    app.state.pool = pool
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "VidBangerGen",
            "version": app.version,
            "workflow_version": store.version,
            "worker_count": len(clients),
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        checks = await asyncio.gather(
            *(client.health() for client in clients), return_exceptions=True
        )
        workers = []
        for client, result in zip(clients, checks):
            if isinstance(result, Exception):
                workers.append({
                    "url": client.base_url, "healthy": False,
                    "error": f"{type(result).__name__}: {str(result)[:240]}",
                })
            else:
                devices = result.get("devices", [])
                workers.append({
                    "url": client.base_url,
                    "healthy": True,
                    "devices": [device.get("name") for device in devices],
                    "comfyui_version": result.get("system", {}).get("comfyui_version"),
                })
        return {
            "status": "ok" if any(worker["healthy"] for worker in workers) else "degraded",
            "database": "ok", "workers": workers, "pool": pool.status(),
        }

    @app.get("/workers")
    async def workers() -> dict[str, Any]:
        return {"workers": pool.status(), "configured_comfy_urls": list(settings.comfyui_urls)}

    @app.get("/execution-targets")
    async def execution_targets() -> dict[str, Any]:
        health_results = await asyncio.gather(
            *(client.health() for client in clients), return_exceptions=True
        )
        comfy_health = {
            client.base_url: not isinstance(result, Exception)
            for client, result in zip(clients, health_results)
        }
        configured_targets = settings.resolved_execution_targets()
        values: list[dict[str, Any]] = []
        for target in configured_targets:
            if target.kind == "comfyui":
                workers = [
                    {"url": url, "healthy": comfy_health.get(url, False)}
                    for url in target.urls
                ]
                available = any(worker["healthy"] for worker in workers)
                reason = None if available else "No ComfyUI worker in this target is reachable"
            elif target.kind == "local-video2x":
                executable = shutil.which(settings.local_upscaler_binary)
                available = bool(executable)
                workers = []
                reason = (
                    None if available else
                    f"Install {settings.local_upscaler_binary} on the VidBangerGen machine"
                )
            else:
                def probe_ssh() -> bool:
                    result = subprocess.run(
                        [
                            "ssh", "-F", "/dev/null", "-o", "BatchMode=yes",
                            "-o", "ConnectTimeout=4", target.ssh_host or "",
                            "command -v video2x",
                        ],
                        capture_output=True, timeout=7,
                    )
                    return result.returncode == 0

                try:
                    available = await asyncio.to_thread(probe_ssh)
                except (OSError, subprocess.SubprocessError):
                    available = False
                workers = []
                reason = None if available else "SSH or Video2X is unavailable on this worker"
            values.append({
                "id": target.id, "label": target.label, "kind": target.kind,
                "capabilities": list(target.capabilities), "description": target.description,
                "available": available, "unavailable_reason": reason, "workers": workers,
                "gpu_index": target.gpu_index,
            })
        default_post_upscale = next(
            (
                target.id
                for target in configured_targets
                if "post-upscale" in target.capabilities
            ),
            "",
        )
        return {
            "targets": values,
            "defaults": {
                "generation": "auto",
                "continuation": "auto",
                "post_upscale": default_post_upscale,
            },
            "policy": (
                "Generation includes the LTX latent upscale on its inference target. "
                "Post-upscale is an isolated LTX Pixel Spatial GGUF job on the "
                "selected capable target."
            ),
        }

    @app.get("/gpu-status")
    async def gpu_status() -> dict[str, Any]:
        if not settings.ssh_host:
            raise HTTPException(
                404,
                "Remote GPU monitoring is not configured. Run `vbg setup --force` "
                "and provide an SSH target.",
            )
        try:
            return await asyncio.to_thread(query_gpu_status, settings.ssh_host)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            raise HTTPException(
                503, f"Remote GPU telemetry is unavailable: {str(exc)[:300]}"
            ) from exc

    @app.get("/analytics")
    async def analytics() -> dict[str, Any]:
        return db.analytics()

    @app.get("/creative-lab")
    async def creative_lab() -> dict[str, Any]:
        catalog = json.loads(CREATIVE_LAB_MANIFEST.read_text())
        available = store.available_capabilities()
        inventory = (
            await asyncio.to_thread(
                query_remote_model_inventory, settings.ssh_host, settings.remote_lora_dir
            )
            if settings.remote_lora_dir else None
        )
        inventory_required = bool(settings.remote_lora_dir)
        def readiness(mode: dict[str, Any]) -> dict[str, Any]:
            required = mode.get("required_capabilities", [])
            missing = [capability for capability in required if capability not in available]
            files = required_model_files(mode)
            missing_files = (
                [name for name in files if name not in inventory]
                if inventory is not None else []
            )
            installed = None if inventory is None else not missing_files
            inventory_ready = installed is True or not inventory_required
            ready = not missing and inventory_ready
            if ready:
                readiness_state = "ready"
            elif installed is True and missing:
                readiness_state = "installed_workflow_pending"
            elif missing_files:
                readiness_state = "model_missing"
            elif inventory_required and inventory is None:
                readiness_state = "inventory_unavailable"
            else:
                readiness_state = "workflow_pending"
            return {
                **mode, "ready": ready, "missing_capabilities": missing,
                "required_model_files": files, "installed": installed,
                "missing_model_files": missing_files,
                "readiness_state": readiness_state,
            }

        modes = [readiness(mode) for mode in catalog.get("modes", [])]
        companions = [
            readiness(mode) for mode in catalog.get("companion_modes", [])
        ]
        return {
            **catalog,
            "active_pipeline": {
                "name": "Motion-first latent upscale",
                "ready": True,
                "model": "LTX 2.3 distilled Q4_K_M GGUF",
                "generation_steps": 4,
                "upscale_steps": 3,
                "upscale_factor": 2,
                "kind": "latent",
            },
            "vision_scoring_enabled": settings.vision_scoring_enabled,
            "recommended_next_mode": "ingredients",
            "model_inventory": {
                "observed": inventory is not None,
                "required_directory": settings.remote_lora_dir,
                "file_count": len(inventory or {}),
                "read_only": True,
            },
            "modes": modes,
            "companion_modes": companions,
        }

    # Project workflow -------------------------------------------------

    @app.post("/projects", status_code=201)
    async def create_project(payload: ProjectCreate) -> dict[str, Any]:
        return db.create_project(payload.brief.model_dump(mode="json"))

    @app.get("/projects")
    async def list_projects(limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
        return {"projects": db.list_projects(limit)}

    @app.get("/projects/{project_id}")
    async def get_project(project_id: str) -> dict[str, Any]:
        project = db.get_project(project_id, expanded=True)
        if not project:
            raise HTTPException(404, "Project not found")
        project["jobs"] = db.list_jobs(project_id)
        project["assets"] = db.list_assets(project_id)
        return project

    @app.get("/projects/{project_id}/analytics")
    async def project_analytics(project_id: str) -> dict[str, Any]:
        if not db.get_project(project_id):
            raise HTTPException(404, "Project not found")
        return db.analytics(project_id)

    @app.get("/projects/{project_id}/generation-estimate")
    async def generation_estimate(
        project_id: str,
        candidates_per_shot: int = Query(4, ge=1, le=8),
        execution_target: str = Query("auto", pattern=r"^(auto|[a-z0-9][a-z0-9-]{0,47})$"),
        reference_engine: str = Query("union", pattern="^(union|ingredients)$"),
    ) -> dict[str, Any]:
        project = db.get_project(project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        concepts = db.list_concepts(project_id)
        concept = next((value for value in concepts if value["selected"]), None)
        if not concept:
            raise HTTPException(409, "Select a concept before estimating generation")
        shots = db.list_shots(project_id, concept["id"])
        ingredients = reference_engine == "ingredients"
        width, height = (768, 448) if ingredients else ASPECT_DRAFT_SIZES[
            project["brief"]["aspect_ratio"]
        ]
        ordinary_clients = eligible_clients(execution_target, "comfy")
        upload_clients = eligible_clients(execution_target, "comfy_upload")
        upload_workers = len(upload_clients)
        rows, total_wall, total_gpu = [], 0, 0
        for index, shot in enumerate(shots):
            frames = 121 if ingredients else adapter.frame_count(
                float(shot["duration_seconds"]), 24.0
            )
            one_render = _estimate_seconds(width, height, frames)
            eligible_workers = (
                max(1, upload_workers) if ingredients
                else (len(ordinary_clients) if index == 0 else max(1, upload_workers))
            )
            batches = math.ceil(candidates_per_shot / max(1, eligible_workers))
            wall = one_render * batches
            total_wall += wall
            total_gpu += one_render * candidates_per_shot
            rows.append({
                "shot_id": shot["id"], "position": index,
                "duration_seconds": shot["duration_seconds"], "frames": frames,
                "render_seconds_each": one_render, "eligible_workers": eligible_workers,
                "batches": batches, "estimated_wall_seconds": wall,
            })
        return {
            "candidates_per_shot": candidates_per_shot,
            "shot_count": len(shots), "render_count": len(shots) * candidates_per_shot,
            "width": width, "height": height, "fps": 24,
            "configured_workers": (
                upload_workers if ingredients else len(ordinary_clients)
            ), "upload_workers": upload_workers,
            "execution_target": execution_target,
            "reference_engine": reference_engine,
            "estimated_wall_seconds": total_wall,
            "estimated_wall_minutes": round(total_wall / 60, 1),
            "estimated_range_minutes": [
                round(total_wall * 0.8 / 60, 1), round(total_wall * 1.35 / 60, 1)
            ],
            "gpu_render_seconds": total_gpu, "shots": rows,
            "basis": (
                "Conservative 768×448 Ingredients estimate; excludes queue contention and review"
                if ingredients else
                "Configured-worker benchmark calibration; excludes queue contention "
                "and manual review"
            ),
        }

    @app.post("/projects/{project_id}/plan", status_code=202)
    async def plan_project(project_id: str, payload: PlanRequest) -> dict[str, Any]:
        project = db.get_project(project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        existing = db.list_concepts(project_id)
        if existing and not payload.regenerate:
            raise HTTPException(409, "Project already has a plan; set regenerate=true to replace it")
        db.set_project_prompt_mode(project_id, "assisted")
        concept_count = 1 if project["brief"].get("source_kind") == "script" else payload.concept_count
        return studio.enqueue_plan(project_id, concept_count)

    @app.post("/projects/{project_id}/manual-plan", status_code=201)
    async def save_manual_plan(
        project_id: str, payload: ManualPlanRequest
    ) -> dict[str, Any]:
        project = db.get_project(project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        existing = db.list_concepts(project_id)
        if existing and not payload.regenerate:
            raise HTTPException(409, "Project already has a plan; set regenerate=true to replace it")
        duration = round(sum(shot.duration_seconds for shot in payload.shots), 3)
        target = float(project["brief"]["duration_seconds"])
        if abs(duration - target) > 0.05:
            raise HTTPException(
                422,
                f"Manual shot durations total {duration:g}s; the project duration is {target:g}s",
            )
        concepts = [{
            "title": payload.title,
            "hook": payload.hook,
            "treatment": payload.treatment,
            "retention_reason": "Human-authored prompt direction",
            "shots": [shot.model_dump(mode="json") for shot in payload.shots],
        }]
        db.set_project_prompt_mode(project_id, "manual")
        db.replace_plan(project_id, concepts)
        return {
            "mode": "manual", "provider": "human", "model": None,
            "concept_count": 1, "shot_count": len(payload.shots),
            "prompt_integrity": "exact",
        }

    @app.post("/projects/{project_id}/concepts/{concept_id}/select")
    async def select_concept(project_id: str, concept_id: str) -> ApiMessage:
        try:
            db.select_concept(project_id, concept_id)
        except KeyError:
            raise HTTPException(404, "Concept not found") from None
        return ApiMessage(message="Concept selected", data={"concept_id": concept_id})

    @app.post("/projects/{project_id}/generate", status_code=202)
    async def generate_candidates(
        project_id: str, payload: CandidateBatchRequest
    ) -> dict[str, Any]:
        project = db.get_project(project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        concepts = db.list_concepts(project_id)
        concept = (
            next((value for value in concepts if value["id"] == payload.concept_id), None)
            if payload.concept_id
            else next(
                (value for value in concepts if value["selected"]),
                concepts[0] if concepts else None,
            )
        )
        if concept:
            shots = db.list_shots(project_id, concept["id"])
            if payload.shot_ids:
                requested = set(payload.shot_ids)
                shots = [shot for shot in shots if shot["id"] in requested]
            generation = payload.settings
            execution_target = generation.execution_target if generation else "auto"
            if generation and generation.reference_engine == "ingredients":
                if "creative_lab_ingredients" not in store.available_capabilities():
                    raise HTTPException(409, "The Ingredients workflow is not enabled")
                if not generation.reference_image_asset_id:
                    raise HTTPException(422, "Ingredients requires a Visual bible asset")
            upload_required = any(
                shot["position"] > 0
                or bool((shot.get("data") or {}).get("reference_asset_id"))
                for shot in shots
            ) or bool(
                generation and (
                    generation.reference_image_asset_id
                    or generation.reference_audio_asset_id
                )
            )
            if shots:
                await require_inference_capacity(
                    "comfy_upload" if upload_required else "comfy", execution_target
                )
        try:
            return studio.enqueue_candidates(project_id, payload)
        except KeyError:
            raise HTTPException(404, "Project not found") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/projects/{project_id}/concepts/{concept_id}/shots", status_code=201)
    async def create_shot(
        project_id: str, concept_id: str, payload: ShotCreate
    ) -> dict[str, Any]:
        if not db.get_project(project_id):
            raise HTTPException(404, "Project not found")
        if not any(value["id"] == concept_id for value in db.list_concepts(project_id)):
            raise HTTPException(404, "Concept not found in this project")
        return db.add_shot(
            project_id, concept_id, payload.prompt, payload.negative_prompt,
            payload.duration_seconds, payload.model_dump(mode="json"),
        )

    @app.patch("/shots/{shot_id}")
    async def update_shot(shot_id: str, payload: ShotUpdate) -> dict[str, Any]:
        shot = db.get_shot(shot_id)
        if not shot:
            raise HTTPException(404, "Shot not found")
        if payload.reference_asset_id:
            asset = db.get_asset(payload.reference_asset_id)
            if (
                not asset
                or asset["project_id"] != shot["project_id"]
                or asset["kind"] not in VISUAL_REFERENCE_KINDS
                or not str(asset.get("mime_type", "")).startswith("image/")
            ):
                raise HTTPException(422, "Shot reference must be an image from this project")
        try:
            return db.update_shot(shot_id, payload.model_dump(exclude_none=True))
        except KeyError:
            raise HTTPException(404, "Shot not found") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/projects/{project_id}/render-winners", status_code=202)
    async def render_winners(
        project_id: str, payload: ExecutionTargetRequest | None = Body(None)
    ) -> dict[str, Any]:
        target_id = payload.execution_target if payload else "auto"
        project = db.get_project(project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        concepts = db.list_concepts(project_id)
        concept = next((value for value in concepts if value["selected"]), None)
        shots = db.list_shots(project_id, concept["id"]) if concept else []
        upload_required = any(shot["position"] > 0 for shot in shots)
        for shot in shots:
            selected = (
                db.get_candidate(shot["selected_candidate_id"])
                if shot.get("selected_candidate_id") else None
            )
            candidate_settings = (selected or {}).get("settings") or {}
            upload_required = upload_required or bool(
                candidate_settings.get("reference_image_asset_id")
                or candidate_settings.get("reference_audio_asset_id")
            )
        if shots:
            await require_inference_capacity(
                "comfy_upload" if upload_required else "comfy", target_id
            )
        try:
            return studio.enqueue_winner_renders(project_id, target_id)
        except KeyError:
            raise HTTPException(404, "Project not found") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/projects/{project_id}/exports", status_code=202)
    async def export_project(project_id: str, payload: ExportRequest) -> dict[str, Any]:
        if not db.get_project(project_id):
            raise HTTPException(404, "Project not found")
        try:
            return studio.enqueue_export(project_id, payload)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/exports/{export_id}")
    async def get_export(export_id: str) -> dict[str, Any]:
        value = db.get_export(export_id)
        if not value:
            raise HTTPException(404, "Export not found")
        return value

    @app.get("/exports/{export_id}/download")
    async def download_export(export_id: str) -> FileResponse:
        value = db.get_export(export_id)
        if not value or value["status"] != "ready" or not value.get("local_path"):
            raise HTTPException(404, "Export is not ready")
        path = Path(value["local_path"])
        if not path.exists():
            raise HTTPException(410, "Export file is no longer present")
        return FileResponse(path, media_type="video/mp4", filename=f"vidbangergen-{export_id}.mp4")

    # Assets and review -----------------------------------------------

    @app.post("/projects/{project_id}/assets", status_code=201)
    async def upload_asset(
        project_id: str,
        kind: str = Form(
            ..., pattern="^(image|video|music|voiceover|reference|reference_sheet|brand)$"
        ),
        comfy_input_name: str | None = Form(None, max_length=255),
        label: str | None = Form(None, max_length=120),
        brand_role: str | None = Form(
            None,
            pattern="^(logo|product|character|wardrobe|sign|location|style_reference)$",
        ),
        notes: str | None = Form(None, max_length=1_000),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        if not db.get_project(project_id):
            raise HTTPException(404, "Project not found")
        if kind in ("reference_sheet", "brand") and not (
            file.content_type or ""
        ).startswith("image/"):
            label_name = "A brand asset" if kind == "brand" else "A visual bible/reference sheet"
            raise HTTPException(422, f"{label_name} must be an image")
        if kind == "brand" and not brand_role:
            raise HTTPException(422, "Brand assets require a role")
        filename = safe_filename(file.filename or "upload.bin")
        asset_id = new_id("assetfile")
        destination = settings.upload_dir / project_id / f"{asset_id}_{filename}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        try:
            with destination.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, "Uploaded asset is too large")
                    handle.write(chunk)
        except Exception:
            if size > settings.max_upload_bytes:
                destination.unlink(missing_ok=True)
            raise
        if comfy_input_name and PurePosixPath(comfy_input_name).name != comfy_input_name:
            destination.unlink(missing_ok=True)
            raise HTTPException(422, "ComfyUI input name must be a plain filename")
        return db.create_asset(
            project_id, kind, filename, str(destination),
            file.content_type or "application/octet-stream", {
                "size_bytes": size,
                **({"comfy_input_name": comfy_input_name} if comfy_input_name else {}),
                **({"label": label.strip()} if label and label.strip() else {}),
                **({"brand_role": brand_role} if brand_role else {}),
                **({"notes": notes.strip()} if notes and notes.strip() else {}),
            },
        )

    @app.get("/assets/{asset_id}")
    async def download_asset(asset_id: str) -> FileResponse:
        asset = db.get_asset(asset_id)
        if not asset:
            raise HTTPException(404, "Asset not found")
        path = Path(asset["local_path"])
        if not path.exists():
            raise HTTPException(410, "Asset file is no longer present")
        return FileResponse(path, media_type=asset["mime_type"], filename=asset["filename"])

    @app.post("/projects/{project_id}/feedback", status_code=201)
    async def add_feedback(
        project_id: str, payload: FeedbackRequest, candidate_id: str | None = None
    ) -> dict[str, Any]:
        if not db.get_project(project_id):
            raise HTTPException(404, "Project not found")
        if candidate_id:
            candidate = db.get_candidate(candidate_id)
            if not candidate or candidate["project_id"] != project_id:
                raise HTTPException(404, "Candidate not found in this project")
        return db.create_feedback(
            project_id, candidate_id, payload.rating, payload.label, payload.reason
        )

    @app.post("/shots/{shot_id}/select")
    async def select_candidate(
        shot_id: str, payload: SelectCandidateRequest
    ) -> ApiMessage:
        shot = db.get_shot(shot_id)
        if not shot:
            raise HTTPException(404, "Shot not found")
        previous = (
            db.get_candidate(shot["selected_candidate_id"])
            if shot.get("selected_candidate_id") else None
        )
        requested = db.get_candidate(payload.candidate_id)
        try:
            db.select_candidate(shot_id, payload.candidate_id)
        except KeyError:
            raise HTTPException(404, "Candidate does not belong to this shot") from None
        phase_changed = (
            previous is None
            or (requested is not None and bool(previous["draft"]) != bool(requested["draft"]))
        )
        released = db.release_next_shot_jobs(shot_id) if phase_changed else 0
        return ApiMessage(
            message="Candidate selected",
            data={"candidate_id": payload.candidate_id, "released_next_jobs": released},
        )

    @app.post("/candidates/{candidate_id}/retake", status_code=202)
    async def retake_candidate(
        candidate_id: str, payload: RetakeRequest
    ) -> dict[str, Any]:
        source = db.get_candidate(candidate_id)
        if not source:
            raise HTTPException(404, "Candidate not found")
        execution_target = str(source.get("settings", {}).get("execution_target") or "auto")
        await require_inference_capacity("comfy_upload", execution_target)
        seed = payload.seed if payload.seed >= 0 else secrets.randbelow(2**63 - 2)
        settings_payload = {
            **source["settings"],
            "retake_start_seconds": payload.start_seconds,
            "retake_end_seconds": payload.end_seconds,
        }
        candidate = db.create_candidate(
            source["project_id"], source["shot_id"], payload.prompt, seed,
            settings_payload, draft=source["draft"],
        )
        job = db.create_job(
            "retake", "comfy_upload",
            {
                "settings": settings_payload, "source_candidate_id": candidate_id,
                "execution_target": execution_target,
            },
            project_id=source["project_id"], shot_id=source["shot_id"],
            candidate_id=candidate["id"],
        )
        return {"candidate": candidate, "job": job, "mode": "time-masked-av-retake"}

    @app.post("/candidates/{candidate_id}/rescore", status_code=202)
    async def rescore_candidate(candidate_id: str) -> dict[str, Any]:
        candidate = db.get_candidate(candidate_id)
        if not candidate or not candidate.get("artifact"):
            raise HTTPException(404, "Candidate media is unavailable")
        active = next(
            (
                job for job in db.list_jobs(candidate["project_id"], 500)
                if job.get("candidate_id") == candidate_id
                and job["kind"] == "score"
                and job["status"] in ("queued", "running")
            ),
            None,
        )
        if active:
            raise HTTPException(409, "Candidate already has a scoring job in progress")
        db.update_candidate(candidate_id, status="generated")
        return db.create_job(
            "score", "local", {"candidate_id": candidate_id},
            project_id=candidate["project_id"], shot_id=candidate["shot_id"],
            candidate_id=candidate_id, max_attempts=1,
        )

    @app.get("/candidates/{candidate_id}/media")
    async def candidate_media(candidate_id: str) -> FileResponse:
        candidate = db.get_candidate(candidate_id)
        if not candidate or not candidate.get("artifact"):
            raise HTTPException(404, "Candidate media is unavailable")
        path = Path(candidate["artifact"]["local_path"])
        if not path.exists():
            raise HTTPException(410, "Candidate media is no longer present")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.post("/upscales", status_code=202)
    async def create_upscale(payload: UpscaleRequest) -> dict[str, Any]:
        target = await require_upscale_capacity(payload.target_id)
        project_id: str | None = None
        candidate_id: str | None = None
        if payload.source_job_id:
            source_job = db.get_job(payload.source_job_id)
            if (
                not source_job or source_job["status"] != "succeeded"
                or source_job["kind"] not in ("creative_transform", "upscale", "cinemagraph")
                or not (source_job.get("result") or {}).get("local_path")
            ):
                raise HTTPException(404, "Processed Creative Lab source is unavailable")
            source = Path((source_job.get("result") or {})["local_path"])
            project_id = source_job.get("project_id")
            candidate_id = source_job.get("candidate_id")
            source_prompt = str(
                payload.prompt
                or source_job.get("payload", {}).get("prompt")
                or (source_job.get("result") or {}).get("workflow", {}).get("prompt")
                or ""
            )
        elif payload.candidate_id:
            candidate = db.get_candidate(payload.candidate_id)
            if not candidate or not candidate.get("artifact"):
                raise HTTPException(404, "Candidate media is unavailable")
            source = Path(candidate["artifact"]["local_path"])
            project_id = candidate["project_id"]
            candidate_id = candidate["id"]
            source_prompt = str(payload.prompt or candidate.get("prompt") or "")
        elif payload.chain_clip_id:
            clip = db.get_chain_clip(payload.chain_clip_id)
            if not clip or clip["status"] != "done":
                raise HTTPException(404, "Chain clip is unavailable")
            source = await ensure_chain_clip_local(clip["chain_id"], clip)
            source_prompt = str(payload.prompt or clip.get("prompt") or "")
        elif payload.chain_id:
            chain = db.get_chain(payload.chain_id)
            if not chain or not chain.get("merged_path"):
                raise HTTPException(404, "Merged chain output is unavailable")
            source = Path(chain["merged_path"])
            source_prompt = str(
                payload.prompt
                or " ".join(
                    str(clip.get("prompt") or "")
                    for clip in chain.get("clips", [])
                    if clip.get("status") == "done"
                )
            )[:5_000]
        else:
            artifact = remote_artifact(payload.remote_filename or "")
            source_id = new_id("source")
            source = settings.artifact_dir / "upscales" / "sources" / f"{source_id}.mp4"
            last_error: Exception | None = None
            downloaded = False
            for client in clients:
                try:
                    await client.download(artifact, source)
                    downloaded = True
                    break
                except (ComfyError, httpx.HTTPError, OSError) as exc:
                    last_error = exc
                    source.unlink(missing_ok=True)
            if not downloaded:
                raise HTTPException(502, str(last_error or "Source render could not be downloaded"))
            source_prompt = str(payload.prompt or "")
        if not source.exists():
            raise HTTPException(410, "Source media is no longer present")
        upscale_id = new_id("upscale")
        destination = (
            settings.artifact_dir / "upscales" / upscale_id
            / f"{source.stem}-{payload.scale}x.mp4"
        )
        job = db.create_job(
            "upscale", "comfy_upload" if target.kind == "comfyui" else "postprocess",
            {
                "target_id": target.id, "scale": payload.scale,
                "source_path": str(source), "destination_path": str(destination),
                "execution_target": target.id, "prompt": source_prompt,
                **({"source_job_id": payload.source_job_id} if payload.source_job_id else {}),
                **({"chain_id": payload.chain_id} if payload.chain_id else {}),
            },
            project_id=project_id, candidate_id=candidate_id, max_attempts=1,
        )
        return {
            "job": job, "target": {"id": target.id, "label": target.label},
            "scale": payload.scale,
        }

    @app.post("/creative-lab/transforms", status_code=202)
    async def create_creative_transform(
        payload: CreativeTransformRequest,
    ) -> dict[str, Any]:
        target = await require_upscale_capacity(payload.target_id)
        if target.kind != "comfyui":
            raise HTTPException(422, "Creative Lab transforms require a ComfyUI target")
        project_id: str | None = None
        candidate_id: str | None = None
        if payload.source_job_id:
            source_job = db.get_job(payload.source_job_id)
            if (
                not source_job or source_job["status"] != "succeeded"
                or source_job["kind"] not in ("creative_transform", "upscale", "cinemagraph")
                or not (source_job.get("result") or {}).get("local_path")
            ):
                raise HTTPException(404, "Processed Creative Lab source is unavailable")
            source = Path((source_job.get("result") or {})["local_path"])
            project_id = source_job.get("project_id")
            candidate_id = source_job.get("candidate_id")
            source_prompt = str(
                payload.prompt
                or source_job.get("payload", {}).get("prompt")
                or (source_job.get("result") or {}).get("workflow", {}).get("prompt")
                or ""
            )
        elif payload.candidate_id:
            candidate = db.get_candidate(payload.candidate_id)
            if not candidate or not candidate.get("artifact"):
                raise HTTPException(404, "Candidate media is unavailable")
            source = Path(candidate["artifact"]["local_path"])
            project_id = candidate["project_id"]
            candidate_id = candidate["id"]
            source_prompt = str(
                payload.prompt
                or (candidate.get("prompt") if payload.mode == "foley-v2a" else "")
                or ""
            )
        elif payload.chain_clip_id:
            clip = db.get_chain_clip(payload.chain_clip_id)
            if not clip or clip["status"] != "done":
                raise HTTPException(404, "Chain clip is unavailable")
            source = await ensure_chain_clip_local(clip["chain_id"], clip)
            source_prompt = str(
                payload.prompt
                or (clip.get("prompt") if payload.mode == "foley-v2a" else "")
                or ""
            )
        else:
            artifact = remote_artifact(payload.remote_filename or "")
            source_id = new_id("source")
            source = (
                settings.artifact_dir / "creative-lab" / "sources" / f"{source_id}.mp4"
            )
            last_error: Exception | None = None
            downloaded = False
            for client in clients:
                try:
                    await client.download(artifact, source)
                    downloaded = True
                    break
                except (ComfyError, httpx.HTTPError, OSError) as exc:
                    last_error = exc
                    source.unlink(missing_ok=True)
            if not downloaded:
                raise HTTPException(502, str(last_error or "Source render could not be downloaded"))
            source_prompt = str(payload.prompt or "")
        if payload.mode == "foley-v2a" and not source_prompt.strip():
            raise HTTPException(
                422,
                "Foley needs a short description of the visible action, surfaces, and materials",
            )
        mask_path: str | None = None
        if payload.mode == "in-outpainting" and payload.operation == "inpaint":
            mask_asset = db.get_asset(str(payload.mask_asset_id or ""))
            if not mask_asset:
                raise HTTPException(404, "Inpainting mask asset is unavailable")
            if not str(mask_asset.get("mime_type") or "").startswith(("image/", "video/")):
                raise HTTPException(422, "Inpainting mask must be an image or video asset")
            if project_id and mask_asset["project_id"] != project_id:
                raise HTTPException(422, "Inpainting mask must belong to the source project")
            project_id = project_id or mask_asset["project_id"]
            mask = Path(mask_asset["local_path"])
            if not mask.exists():
                raise HTTPException(410, "Inpainting mask file is no longer present")
            mask_path = str(mask)
        if not source.exists():
            raise HTTPException(410, "Source media is no longer present")
        if payload.mode == "lipdub":
            if "creative_lab_lipdub" not in store.available_capabilities():
                raise HTTPException(409, "The LipDub workflow is not enabled")
            try:
                streams = await asyncio.to_thread(probe_stream_types, source)
            except RuntimeError as exc:
                raise HTTPException(422, str(exc)) from exc
            if "audio" not in streams:
                raise HTTPException(
                    422,
                    "LipDub needs a source video with one speaker's reference speech audio",
                )
        transform_id = new_id("transform")
        destination = (
            settings.artifact_dir / "creative-lab" / transform_id
            / f"{source.stem}-{payload.mode}.mp4"
        )
        job = db.create_job(
            "creative_transform", "comfy_upload",
            {
                "mode": payload.mode, "target_id": target.id,
                "source_path": str(source), "destination_path": str(destination),
                "execution_target": target.id, "prompt": source_prompt,
                "negative_prompt": payload.negative_prompt,
                "strength": payload.strength, "seed": payload.seed,
                **({"source_job_id": payload.source_job_id} if payload.source_job_id else {}),
                **(
                    {
                        "operation": payload.operation,
                        "mask_asset_id": payload.mask_asset_id,
                        "mask_path": mask_path,
                        "outpaint_direction": payload.outpaint_direction,
                        "expansion_percent": payload.expansion_percent,
                        "mask_dilation": payload.mask_dilation,
                    }
                    if payload.mode == "in-outpainting" else {}
                ),
                **(
                    {"dialogue": payload.dialogue, "language": payload.language}
                    if payload.mode == "lipdub" else {}
                ),
            },
            project_id=project_id, candidate_id=candidate_id, max_attempts=1,
        )
        return {
            "job": job, "mode": payload.mode,
            "target": {"id": target.id, "label": target.label},
        }

    @app.post("/creative-lab/cinemagraph", status_code=202)
    async def create_cinemagraph(payload: CinemagraphRequest) -> dict[str, Any]:
        target = await require_upscale_capacity(payload.target_id)
        if target.kind != "comfyui":
            raise HTTPException(422, "Cinemagraph requires a ComfyUI target")
        asset = db.get_asset(payload.asset_id)
        if not asset:
            raise HTTPException(404, "Cinemagraph source image not found")
        if not str(asset.get("mime_type") or "").startswith("image/"):
            raise HTTPException(422, "Cinemagraph requires an uploaded image asset")
        source = Path(asset["local_path"])
        if not source.exists():
            raise HTTPException(410, "Cinemagraph source image is no longer present")
        job_id = new_id("cinemagraph")
        destination = (
            settings.artifact_dir / "creative-lab" / job_id / "cinemagraph.mp4"
        )
        job = db.create_job(
            "cinemagraph", "comfy_upload",
            {
                "target_id": target.id, "execution_target": target.id,
                "asset_id": asset["id"], "source_path": str(source),
                "destination_path": str(destination), "prompt": payload.prompt.strip(),
                "negative_prompt": payload.negative_prompt,
                "strength": payload.strength, "seed": payload.seed,
            },
            project_id=asset["project_id"], max_attempts=1,
        )
        return {
            "job": job, "mode": "cinemagraph",
            "target": {"id": target.id, "label": target.label},
        }

    @app.get("/jobs/{job_id}/output")
    async def job_output(job_id: str) -> FileResponse:
        job = db.get_job(job_id)
        if not job or job["kind"] not in (
            "upscale", "creative_transform", "cinemagraph",
        ):
            raise HTTPException(404, "Processed output not found")
        if job["status"] != "succeeded" or not job.get("result"):
            raise HTTPException(409, "Processed output is not ready")
        path = Path(job["result"].get("local_path") or "")
        if not path.exists():
            raise HTTPException(410, "Upscale output is no longer present")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    # Jobs and progress ------------------------------------------------

    @app.get("/jobs")
    async def list_jobs(
        project_id: str | None = None, limit: int = Query(100, ge=1, le=500)
    ) -> dict[str, Any]:
        return {"jobs": db.list_jobs(project_id, limit)}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @app.post("/jobs/{job_id}/cancel", status_code=202)
    async def cancel_job(job_id: str) -> ApiMessage:
        if not db.request_cancel(job_id):
            raise HTTPException(404, "Job not found")
        return ApiMessage(message="Cancellation requested", data={"job_id": job_id})

    @app.post("/jobs/{job_id}/retry", status_code=202)
    async def retry_job(job_id: str) -> ApiMessage:
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] not in ("failed", "cancelled"):
            raise HTTPException(409, "Only failed or cancelled jobs can be retried")
        if job["kind"] in ("generate", "retake", "ingredients_generate"):
            await require_inference_capacity(
                job["lane"], str(job.get("payload", {}).get("execution_target") or "auto")
            )
        elif job["kind"] in ("upscale", "creative_transform", "cinemagraph"):
            await require_upscale_capacity(
                str(job.get("payload", {}).get("target_id") or "")
            )
        if not db.retry_job(job_id):
            raise HTTPException(409, "Only failed or cancelled jobs can be retried")
        return ApiMessage(message="Job queued for retry", data={"job_id": job_id})

    @app.post("/jobs/{job_id}/recover-upscale")
    async def recover_upscale(job_id: str) -> dict[str, Any]:
        job = db.get_job(job_id)
        if not job or job["kind"] != "upscale":
            raise HTTPException(404, "Upscale job not found")
        if job["status"] == "succeeded":
            return job
        if job["status"] not in ("failed", "cancelled"):
            raise HTTPException(409, "Upscale is still running")
        payload = job.get("payload") or {}
        source = Path(str(payload.get("source_path") or ""))
        destination = Path(str(payload.get("destination_path") or ""))
        if not source.exists() or not destination.name:
            raise HTTPException(410, "Original upscale source is unavailable")
        info = await asyncio.to_thread(probe_video_info, source)
        if int(info["frames"]) > 121:
            raise HTTPException(
                409,
                "This multi-pass upscale needs worker recovery; do not queue a duplicate",
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        video_only = destination.with_name(f".{destination.stem}.video-only.mp4")
        artifact = {
            "filename": "pass-000_00001_.mp4",
            "subfolder": f"vbg/upscales/{job_id}",
            "type": "output",
        }
        last_error: Exception | None = None
        for client in clients:
            try:
                await client.download(artifact, video_only)
                break
            except (ComfyError, httpx.HTTPError, OSError) as exc:
                last_error = exc
                video_only.unlink(missing_ok=True)
        else:
            raise HTTPException(
                404, str(last_error or "Completed remote upscale was not found")
            )
        try:
            audio = await asyncio.to_thread(
                restore_source_audio, video_only, source, destination
            )
            output = await asyncio.to_thread(probe_video_info, destination)
        except (OSError, RuntimeError, ValueError) as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(500, str(exc)) from exc
        db.update_job(
            job_id, status="succeeded", progress=1.0, error=None,
            result_json={
                "local_path": str(destination),
                "target_id": payload.get("target_id"),
                "scale": int(payload.get("scale") or 2),
                "engine": "LTX 2.3 Pixel Spatial Upscaler",
                "source_path": str(source),
                "output_dimensions": [output["width"], output["height"]],
                "recovery": "deterministic-output-after-local-timeout",
                **audio,
            },
            worker_id=None, remote_id=None, finished_at=utcnow(),
        )
        return db.get_job(job_id) or job

    @app.get("/events")
    async def events(
        request: Request, project_id: str | None = None
    ) -> StreamingResponse:
        async def stream() -> AsyncIterator[str]:
            previous = ""
            while not await request.is_disconnected():
                snapshot = json.dumps(db.list_jobs(project_id, 200), separators=(",", ":"))
                if snapshot != previous:
                    yield f"event: jobs\ndata: {snapshot}\n\n"
                    previous = snapshot
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(1)
        return StreamingResponse(stream(), media_type="text/event-stream")

    # Durable legacy chains -------------------------------------------

    def remote_artifact(filename: str) -> dict[str, str]:
        path = PurePosixPath(filename)
        if not path.name or ".." in path.parts:
            raise HTTPException(422, "Invalid output filename")
        return {
            "filename": path.name,
            "subfolder": str(path.parent) if str(path.parent) != "." else "",
            "type": "output",
        }

    async def ensure_chain_clip_local(chain_id: str, clip: dict[str, Any]) -> Path:
        if clip.get("local_path"):
            local = Path(clip["local_path"])
            if local.exists():
                return local
        remote = clip.get("remote_filename")
        if not remote:
            raise HTTPException(410, f"Chain clip {clip['id']} has no available media")
        suffix = Path(remote).suffix.lower() or ".mp4"
        local = settings.artifact_dir / "legacy_chains" / chain_id / f"{clip['id']}{suffix}"
        last_error: Exception | None = None
        for client in clients:
            try:
                await client.download(remote_artifact(remote), local)
                db.update_chain_clip(clip["id"], local_path=str(local))
                return local
            except (ComfyError, httpx.HTTPError, OSError) as exc:
                last_error = exc
        raise HTTPException(502, str(last_error or "Remote clip is unavailable"))

    async def save_chain_finish_upload(
        chain_id: str, role: str, upload: UploadFile | None, required_stream: str,
    ) -> Path | None:
        if upload is None:
            return None
        filename = safe_filename(upload.filename or f"{role}.bin")
        destination = (
            settings.artifact_dir / "legacy_chains" / chain_id / "finish-inputs"
            / f"{role}-{filename}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        try:
            with destination.open("wb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, f"{role.title()} upload is too large")
                    handle.write(chunk)
            if size == 0:
                raise HTTPException(422, f"{role.title()} upload is empty")
            streams = await asyncio.to_thread(probe_stream_types, destination)
            if required_stream not in streams:
                raise HTTPException(
                    422, f"{role.title()} must contain a valid {required_stream} stream"
                )
            if role == "logo" and not (upload.content_type or "").startswith("image/"):
                raise HTTPException(422, "Logo must be a PNG, JPG, or WebP image")
            return destination
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def chain_clip_age_seconds(clip: dict[str, Any]) -> float:
        try:
            updated = datetime.fromisoformat(str(clip.get("updated_at") or ""))
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            return max(0.0, (datetime.now(UTC) - updated).total_seconds())
        except ValueError:
            return 0.0

    async def reconcile_chain(chain: dict[str, Any]) -> dict[str, Any]:
        """Recover continuations orphaned only by a local API restart.

        ComfyUI remains authoritative and is queried read-only. A prompt is not
        declared lost unless every configured worker answered both history and
        queue checks and the existing 120-second queue/history grace elapsed.
        """
        generating = [
            clip for clip in chain.get("clips", [])
            if clip.get("status") == "generating" and clip.get("id") not in active_chain_clips
        ]
        for clip in generating:
            prompt_id = str(clip.get("prompt_id") or "")
            if not prompt_id:
                if chain_clip_age_seconds(clip) > 120:
                    message = (
                        "Continuation stopped before ComfyUI returned a prompt ID. "
                        "The previous kept clip is safe; add the continuation again."
                    )
                    db.update_chain_clip(
                        clip["id"], status="failed",
                        metadata_json={**(clip.get("metadata") or {}), "error": message},
                    )
                    db.update_chain(chain["id"], status="failed")
                continue

            async def probe(client: ComfyClient) -> dict[str, Any]:
                try:
                    history = await client.history(prompt_id)
                    queued = False if history else await client.prompt_in_queue(prompt_id)
                    return {
                        "client": client, "history": history, "queued": queued,
                        "reachable": True,
                    }
                except (ComfyError, httpx.HTTPError, OSError) as exc:
                    return {"client": client, "reachable": False, "error": str(exc)}

            probes = await asyncio.gather(*(probe(client) for client in clients))
            completed = next(
                (
                    value for value in probes
                    if value.get("history") and value["history"].get("status") == "done"
                ),
                None,
            )
            failed = next(
                (
                    value for value in probes
                    if value.get("history") and value["history"].get("status") == "error"
                ),
                None,
            )
            if completed:
                history = completed["history"]
                artifact = next(
                    (
                        item for item in history.get("files", [])
                        if Path(item.get("filename", "")).suffix.lower() in VIDEO_EXTENSIONS
                    ),
                    None,
                )
                if artifact:
                    suffix = Path(artifact["filename"]).suffix.lower() or ".mp4"
                    destination = (
                        settings.artifact_dir / "legacy_chains" / chain["id"]
                        / f"{clip['id']}{suffix}"
                    )
                    try:
                        await completed["client"].download(artifact, destination)
                    except (ComfyError, httpx.HTTPError, OSError):
                        # A transient read failure must not turn a successfully
                        # rendered remote clip into a failed story beat. The
                        # next chain poll will attempt the recovery again.
                        destination.unlink(missing_ok=True)
                        continue
                    remote_name = str(
                        PurePosixPath(artifact.get("subfolder", "")) / artifact["filename"]
                    ).lstrip("/")
                    db.update_chain_clip(
                        clip["id"], status="done", remote_filename=remote_name,
                        local_path=str(destination), metadata_json={
                            **(clip.get("metadata") or {}),
                            "remote_recovery": {
                                "mode": "history-after-local-restart",
                                "prompt_id": prompt_id,
                            },
                        },
                    )
                    db.update_chain(chain["id"], status="ready")
                else:
                    db.update_chain_clip(
                        clip["id"], status="failed", metadata_json={
                            **(clip.get("metadata") or {}),
                            "error": "Recovered ComfyUI prompt completed without a video output.",
                        },
                    )
                    db.update_chain(chain["id"], status="failed")
                continue
            if failed:
                message = str(failed["history"].get("error") or "ComfyUI continuation failed")
                db.update_chain_clip(
                    clip["id"], status="failed",
                    metadata_json={**(clip.get("metadata") or {}), "error": message},
                )
                db.update_chain(chain["id"], status="failed")
                continue
            if any(
                value.get("history")
                and value["history"].get("status") in ("queued", "running")
                for value in probes
            ):
                continue
            if any(value.get("queued") for value in probes):
                continue
            if not all(value.get("reachable") for value in probes):
                continue
            if chain_clip_age_seconds(clip) <= 120:
                continue

            expected = {
                "filename": f"{clip['id']}_00001_.mp4",
                "subfolder": f"vbg/chains/{chain['id']}", "type": "output",
            }
            recovered = False
            for client in clients:
                destination = (
                    settings.artifact_dir / "legacy_chains" / chain["id"]
                    / f"{clip['id']}.mp4"
                )
                try:
                    await client.download(expected, destination)
                except (ComfyError, httpx.HTTPError, OSError):
                    destination.unlink(missing_ok=True)
                    continue
                db.update_chain_clip(
                    clip["id"], status="done",
                    remote_filename=f"vbg/chains/{chain['id']}/{expected['filename']}",
                    local_path=str(destination), metadata_json={
                        **(clip.get("metadata") or {}),
                        "remote_recovery": {
                            "mode": "deterministic-output-after-local-restart",
                            "prompt_id": prompt_id,
                        },
                    },
                )
                db.update_chain(chain["id"], status="ready")
                recovered = True
                break
            if not recovered:
                message = (
                    f"Remote prompt {prompt_id} is no longer queued and has no history or "
                    "recoverable output. The previous kept clip is safe; add this beat again."
                )
                db.update_chain_clip(
                    clip["id"], status="failed",
                    metadata_json={**(clip.get("metadata") or {}), "error": message},
                )
                db.update_chain(chain["id"], status="failed")
        return db.get_chain(chain["id"]) or chain

    @app.post("/chain", status_code=201)
    async def create_chain(payload: dict[str, Any] | None = Body(None)) -> dict[str, Any]:
        value = payload or {}
        filename = value.get("video_file") or value.get("filename")
        local_path = None
        if value.get("local_path"):
            candidate = Path(str(value["local_path"]))
            if candidate.exists():
                local_path = str(candidate)
        return db.create_chain(
            str(filename) if filename else None,
            str(value.get("prompt") or ""),
            local_path=local_path,
            metadata={"execution_target": str(value.get("execution_target") or "auto")},
        )

    @app.get("/chain/{chain_id}")
    async def get_chain(chain_id: str) -> dict[str, Any]:
        chain = db.get_chain(chain_id)
        if not chain:
            raise HTTPException(404, "Chain not found")
        return await reconcile_chain(chain)

    @app.get("/chain/{chain_id}/clips/{clip_id}/output")
    async def chain_clip_output(chain_id: str, clip_id: str) -> Response:
        chain = db.get_chain(chain_id)
        clip = db.get_chain_clip(clip_id)
        if not chain or not clip or clip["chain_id"] != chain_id:
            raise HTTPException(404, "Chain clip not found")
        if clip["status"] != "done":
            raise HTTPException(409, "Chain clip is not available for review")
        path = await ensure_chain_clip_local(chain_id, clip)
        return Response(
            path.read_bytes(), media_type="video/mp4",
            headers={"Content-Disposition": f'inline; filename="{path.name}"'},
        )

    @app.delete("/chain/{chain_id}/clips/{clip_id}")
    async def reject_chain_clip(chain_id: str, clip_id: str) -> dict[str, Any]:
        chain = db.get_chain(chain_id)
        clip = db.get_chain_clip(clip_id)
        if not chain or not clip or clip["chain_id"] != chain_id:
            raise HTTPException(404, "Chain clip not found")
        if clip["position"] == 0:
            raise HTTPException(409, "The opening clip cannot be rejected")
        active_after = [
            value for value in chain["clips"]
            if value["position"] > clip["position"]
            and value["status"] in ("done", "generating")
        ]
        if active_after:
            raise HTTPException(
                409, "Only the latest continuation can be rejected without breaking continuity"
            )
        if clip["status"] != "done":
            raise HTTPException(409, "Only a completed continuation can be rejected")
        db.update_chain_clip(
            clip_id, status="rejected",
            metadata_json={
                **(clip.get("metadata") or {}),
                "review": {"decision": "rejected", "at": utcnow()},
            },
        )
        db.update_chain(chain_id, status="ready")
        return db.get_chain(chain_id)  # type: ignore[return-value]

    @app.post("/chain/{chain_id}/continue")
    async def continue_chain(
        chain_id: str,
        prompt: str = Form(..., min_length=1, max_length=5_000),
        strength: float = Form(0.7, ge=0, le=1),
        width: int = Form(640, ge=256, le=1024, multiple_of=8),
        height: int = Form(360, ge=256, le=1024, multiple_of=8),
        negative_prompt: str = Form("", max_length=2_000),
        duration_seconds: float = Form(5.0, ge=1.0, le=20.0),
        fps: float = Form(24.0, ge=8.0, le=60.0),
        seed: int = Form(-1, ge=-1),
        audio_mode: str = Form(
            "ambient", pattern="^(prompt|ambient|silent|native-dialogue)$"
        ),
        dialogue: str = Form("", max_length=1_000),
        speaker: str = Form("", max_length=120),
        language: str = Form("English", max_length=80),
        accent: str = Form("", max_length=120),
        execution_target: str = Form(
            "auto", pattern=r"^(auto|[a-z0-9][a-z0-9-]{0,47})$"
        ),
    ) -> dict[str, Any]:
        chain = db.get_chain(chain_id)
        if not chain:
            raise HTTPException(404, "Chain not found")
        if not chain["clips"]:
            raise HTTPException(409, "Chain has no source clip")
        if any(clip["status"] == "generating" for clip in chain["clips"]):
            raise HTTPException(409, "Chain already has a generation in progress")
        if chain.get("finish_job") and chain["finish_job"]["status"] in ("queued", "running"):
            raise HTTPException(409, "Wait for the active social finish before extending")
        previous_clip = next(
            (clip for clip in reversed(chain["clips"]) if clip["status"] == "done"), None
        )
        if not previous_clip:
            raise HTTPException(409, "Chain has no successful source clip")
        await require_inference_capacity("comfy_upload", execution_target)
        continuation_client = await choose_client(execution_target, "comfy_upload")
        previous_path = await ensure_chain_clip_local(chain_id, previous_clip)
        clip: dict[str, Any] | None = None
        try:
            continuity = await asyncio.to_thread(
                extract_continuity_assets,
                previous_path,
                settings.artifact_dir / "legacy_chains" / chain_id / "continuity",
            )
            # Five-second motion-tail conditioning repeatedly restarted the
            # dedicated 24 GB worker near VAE decode. The final-frame IC-LoRA
            # path uses the same proven I2V graph as ordinary image generation,
            # retaining the exact boundary composition without the extra 17
            # guide frames in the sampling latent.
            image_name = await continuation_client.upload(Path(continuity["last_frame_path"]))
            clip = db.add_chain_clip(
                chain_id, prompt, strength, status="generating",
                metadata={"execution_target": execution_target},
            )
            active_chain_clips.add(clip["id"])
            continuation_width, continuation_height = _continuation_safe_size(width, height)
            compiled_prompt, compiled_negative, audio_contract = compile_audio_prompt(
                prompt, negative_prompt, mode=audio_mode,
                duration_seconds=duration_seconds,
                dialogue=dialogue, speaker=speaker, language=language, accent=accent,
            )
            graph, metadata = adapter.build(
                prompt=compiled_prompt, negative_prompt=compiled_negative,
                width=continuation_width, height=continuation_height,
                duration_seconds=duration_seconds, fps=fps, seed=seed,
                filename_prefix=f"vbg/chains/{chain_id}/{clip['id']}",
                image_name=image_name, ic_lora_strength=0.5,
                image_condition_strength=max(0.65, min(1.0, strength + 0.2)),
                profile="motion-draft-4x3",
                include_audio=audio_mode != "silent",
            )
            metadata["audio_contract"] = audio_contract
            metadata["continuation_canvas"] = {
                "requested_input": [width, height],
                "applied_input": [continuation_width, continuation_height],
                "merged_delivery_canvas": "opening clip",
                "reason": "single-worker motion-guide memory safety",
            }
            metadata["continuation_mode"] = {
                "kind": "last-frame-ic-lora",
                "source": image_name,
                "strength": 0.5,
                "image_condition_strength": max(0.65, min(1.0, strength + 0.2)),
                "motion_tail_fallback_available": continuity["motion_tail_path"],
            }

            async def queued(prompt_id: str) -> None:
                db.update_chain_clip(clip["id"], prompt_id=prompt_id)

            destination = (
                settings.artifact_dir / "legacy_chains" / chain_id / f"{clip['id']}.mp4"
            )
            remote_recovery: dict[str, Any] | None = None
            try:
                result = await continuation_client.queue_and_wait(
                    graph, client_id=f"chain-{clip['id']}", on_queued=queued
                )
            except ComfyOrphanedPromptError as orphaned:
                # SaveVideo names are deterministic because every chain clip
                # has a unique prefix. If another user cleared Comfy history
                # after completion, recover the already-rendered file before
                # considering another GPU run.
                first_prompt_id = str(
                    (db.get_chain_clip(clip["id"]) or {}).get("prompt_id") or ""
                )
                expected_artifact = {
                    "filename": f"{clip['id']}_00001_.mp4",
                    "subfolder": f"vbg/chains/{chain_id}",
                    "type": "output",
                }
                try:
                    await continuation_client.download(expected_artifact, destination)
                    result = {
                        "status": "done", "files": [expected_artifact],
                        "prompt_id": first_prompt_id,
                    }
                    remote_recovery = {
                        "mode": "deterministic-output-recovery",
                        "orphaned_prompt_id": first_prompt_id,
                    }
                except (ComfyError, httpx.HTTPError, OSError):
                    # The disposable 8189 worker may have restarted under host
                    # memory pressure. Its writable /tmp input survives that
                    # service restart, so retry this exact graph once rather
                    # than forcing the operator to rebuild the chain manually.
                    await asyncio.sleep(3)
                    result = await continuation_client.queue_and_wait(
                        graph, client_id=f"chain-{clip['id']}-retry",
                        on_queued=queued,
                    )
                    remote_recovery = {
                        "mode": "single-automatic-retry",
                        "orphaned_prompt_id": first_prompt_id,
                        "reason": str(orphaned),
                    }
            artifact = next(
                (
                    item for item in result.get("files", [])
                    if Path(item.get("filename", "")).suffix.lower() in VIDEO_EXTENSIONS
                ),
                None,
            )
            if not artifact:
                raise RuntimeError("ComfyUI completed without a video output")
            suffix = Path(artifact["filename"]).suffix.lower() or ".mp4"
            final_destination = (
                settings.artifact_dir / "legacy_chains" / chain_id / f"{clip['id']}{suffix}"
            )
            if not destination.exists() or destination != final_destination:
                await continuation_client.download(artifact, final_destination)
            destination = final_destination
            remote_name = str(
                PurePosixPath(artifact.get("subfolder", "")) / artifact["filename"]
            ).lstrip("/")
            db.update_chain_clip(
                clip["id"], status="done", remote_filename=remote_name,
                local_path=str(destination), metadata_json={
                    "workflow": metadata,
                    "execution_target": execution_target,
                    "continuity_source": {
                        **continuity,
                        "mode": "last-frame-ic-lora",
                        "overlap_seconds": 0.0,
                    },
                    **({"remote_recovery": remote_recovery} if remote_recovery else {}),
                },
            )
            db.update_chain(chain_id, status="ready")
            return db.get_chain(chain_id)  # type: ignore[return-value]
        except (ComfyError, WorkflowError, OSError, RuntimeError, ValueError, httpx.HTTPError) as exc:
            if clip:
                db.update_chain_clip(clip["id"], status="failed", metadata_json={"error": str(exc)})
            db.update_chain(chain_id, status="failed")
            raise HTTPException(502, str(exc)) from exc
        finally:
            if clip:
                active_chain_clips.discard(clip["id"])

    @app.post("/chain/{chain_id}/merge")
    async def merge_chain(chain_id: str) -> dict[str, Any]:
        chain = db.get_chain(chain_id)
        if not chain:
            raise HTTPException(404, "Chain not found")
        if any(clip["status"] == "generating" for clip in chain["clips"]):
            raise HTTPException(409, "A chain clip is still generating")
        if chain.get("finish_job") and chain["finish_job"]["status"] in ("queued", "running"):
            raise HTTPException(409, "A social finish is still rendering")
        completed = [clip for clip in chain["clips"] if clip["status"] == "done"]
        if not completed:
            raise HTTPException(409, "Chain has no successful clips to merge")
        clips = [await ensure_chain_clip_local(chain_id, clip) for clip in completed]
        overlaps = []
        for clip in completed:
            metadata = clip.get("metadata") or {}
            continuity = metadata.get("continuity_source") or {}
            overlap = continuity.get("overlap_seconds")
            if overlap is None:
                workflow = metadata.get("workflow") or {}
                guide = workflow.get("motion_guide") or {}
                if guide:
                    overlap = float(guide.get("frames", 0)) / float(workflow.get("fps", 24.0))
            overlaps.append(float(overlap or 0.0))
        destination = settings.artifact_dir / "legacy_chains" / chain_id / "merged.mp4"
        try:
            result = await asyncio.to_thread(merge_chain_clips, clips, destination, overlaps)
        except RuntimeError as exc:
            db.update_chain(chain_id, status="failed")
            raise HTTPException(500, str(exc)) from exc
        db.update_chain(chain_id, status="merged", merged_path=str(destination))
        return {"chain": db.get_chain(chain_id), "output": result}

    @app.get("/chain/{chain_id}/output")
    async def chain_output(chain_id: str) -> FileResponse:
        chain = db.get_chain(chain_id)
        if not chain or not chain.get("merged_path"):
            raise HTTPException(404, "Merged chain output is unavailable")
        path = Path(chain["merged_path"])
        if not path.exists():
            raise HTTPException(410, "Merged chain output is no longer present")
        return FileResponse(path, media_type="video/mp4", filename=f"chain-{chain_id}.mp4")

    @app.post("/chain/{chain_id}/finish", status_code=202)
    async def finish_chain(
        chain_id: str,
        platform: str = Form(
            "reels", pattern="^(tiktok|reels|shorts|youtube|x|custom)$"
        ),
        width: int | None = Form(None, ge=256, le=2160),
        height: int | None = Form(None, ge=256, le=2160),
        transition_seconds: float = Form(0.15, ge=0.0, le=1.0),
        captions: str = Form("", max_length=20_000),
        original_audio_volume: float = Form(0.0, ge=0.0, le=2.0),
        music_volume: float = Form(0.18, ge=0.0, le=2.0),
        voiceover_volume: float = Form(1.0, ge=0.0, le=2.0),
        logo_position: str = Form(
            "bottom-right", pattern="^(top-left|top-right|bottom-left|bottom-right)$"
        ),
        logo_width_percent: float = Form(14.0, ge=3.0, le=40.0),
        logo_opacity: float = Form(1.0, ge=0.1, le=1.0),
        music: UploadFile | None = File(None),
        voiceover: UploadFile | None = File(None),
        logo: UploadFile | None = File(None),
    ) -> dict[str, Any]:
        """Create a social-ready local delivery from a Quick continuity chain."""
        chain = db.get_chain(chain_id)
        if not chain:
            raise HTTPException(404, "Chain not found")
        if any(clip["status"] == "generating" for clip in chain["clips"]):
            raise HTTPException(409, "Wait for the active continuation before finishing")
        if chain.get("finish_job") and chain["finish_job"]["status"] in ("queued", "running"):
            raise HTTPException(409, "This story already has a social finish in progress")
        completed = [clip for clip in chain["clips"] if clip["status"] == "done"]
        if not completed:
            raise HTTPException(409, "Chain has no kept clips to finish")
        if (width is None) != (height is None):
            raise HTTPException(422, "Custom delivery requires both width and height")
        clips = [await ensure_chain_clip_local(chain_id, clip) for clip in completed]
        music_path = await save_chain_finish_upload(chain_id, "music", music, "audio")
        voice_path = await save_chain_finish_upload(
            chain_id, "voiceover", voiceover, "audio"
        )
        logo_path = await save_chain_finish_upload(chain_id, "logo", logo, "video")
        options: dict[str, Any] = {
            "platform": platform,
            "transition_seconds": transition_seconds,
            "captions": captions,
            "burn_captions": bool(captions.strip()),
            "original_audio_volume": original_audio_volume,
            "music_volume": music_volume,
            "voiceover_volume": voiceover_volume,
            "logo_position": logo_position,
            "logo_width_percent": logo_width_percent,
            "logo_opacity": logo_opacity,
        }
        if width is not None and height is not None:
            options.update({"width": width, "height": height})
        destination = (
            settings.artifact_dir / "legacy_chains" / chain_id / "finished-social.mp4"
        )
        job = db.create_job(
            "chain_finish", "postprocess",
            {
                "chain_id": chain_id,
                "clip_paths": [str(path) for path in clips],
                "destination_path": str(destination),
                "options": options,
                "music_path": str(music_path) if music_path else None,
                "voiceover_path": str(voice_path) if voice_path else None,
                "logo_path": str(logo_path) if logo_path else None,
            },
            max_attempts=2,
        )
        db.update_chain_finish_job(chain_id, job["id"])
        return {"chain": db.get_chain(chain_id), "job": db.get_job(job["id"])}

    @app.get("/chain/{chain_id}/finished")
    async def finished_chain_output(chain_id: str) -> FileResponse:
        chain = db.get_chain(chain_id)
        if not chain or not chain.get("finished_path"):
            raise HTTPException(404, "Finished social export is unavailable")
        path = Path(chain["finished_path"])
        if not path.exists():
            raise HTTPException(410, "Finished social export is no longer present")
        return FileResponse(
            path, media_type="video/mp4", filename=f"vidbangergen-{chain_id}.mp4"
        )

    # Direct/raw generation compatibility -----------------------------

    @app.post("/generate/prompt-preview")
    async def prompt_preview(payload: PromptPreviewRequest) -> dict[str, Any]:
        """Return the compiled visual/audio contract without queueing ComfyUI.

        Quick Generate uses this for a collapsed "Sending to LTX" preview and to
        surface multi-shot screenplay paste before the operator burns a GPU job.
        """
        return preview_audio_prompt(
            payload.prompt,
            payload.negative_prompt,
            mode=payload.audio_mode,
            duration_seconds=payload.duration_seconds,
            dialogue=payload.dialogue,
            speaker=payload.speaker,
            language=payload.language,
            accent=payload.accent,
            ambience=payload.ambience,
        )

    @app.post("/generate/t2v", response_model=QueueResponse)
    async def legacy_t2v(request: Request) -> QueueResponse:
        """Queue direct T2V from either the Studio or the legacy quick UI.

        The original quick generator submitted multipart form data, while the
        production Studio initially submitted JSON. Supporting both keeps old
        browser bundles usable during a rolling local restart.
        """
        try:
            content_type = request.headers.get("content-type", "").lower()
            raw = (
                await request.json()
                if "application/json" in content_type
                else dict(await request.form())
            )
            payload = LegacyT2VRequest.model_validate(raw)
        except ValueError as exc:
            details = exc.errors() if hasattr(exc, "errors") else str(exc)
            raise HTTPException(422, detail=details) from exc
        await require_inference_capacity("comfy", payload.execution_target)
        client = await choose_client(payload.execution_target, "comfy")
        try:
            compiled_prompt, compiled_negative, _audio_contract = compile_audio_prompt(
                payload.prompt, payload.negative_prompt,
                mode=payload.audio_mode,
                duration_seconds=payload.duration_seconds,
                dialogue=payload.dialogue, speaker=payload.speaker,
                language=payload.language, accent=payload.accent,
            )
            graph, metadata = adapter.build(
                prompt=compiled_prompt, negative_prompt=compiled_negative,
                width=payload.width, height=payload.height,
                duration_seconds=payload.duration_seconds, fps=payload.fps,
                seed=payload.seed, filename_prefix=f"vbg/legacy/{new_id('clip')}",
                profile=payload.profile.value,
                include_audio=payload.audio_mode != "silent",
            )
            prompt_id = await client.queue(graph)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (WorkflowError, ComfyError, httpx.HTTPError, OSError) as exc:
            raise HTTPException(502, str(exc)) from exc
        return QueueResponse(
            prompt_id=prompt_id,
            estimated_seconds=_estimate_seconds(payload.width, payload.height, metadata["frames"]),
        )

    @app.post("/generate/i2v", response_model=QueueResponse)
    async def legacy_i2v(
        prompt: str = Form(..., min_length=1, max_length=5_000),
        negative_prompt: str = Form("", max_length=2_000),
        width: int = Form(640, ge=256, le=1024, multiple_of=8),
        height: int = Form(360, ge=256, le=1024, multiple_of=8),
        duration_seconds: float = Form(5.0, ge=1.0, le=20.0),
        fps: float = Form(24.0, ge=8.0, le=60.0),
        ic_lora_strength: float = Form(0.5, ge=0, le=1.5),
        img_cond_strength: float = Form(0.9, ge=0, le=1),
        audio_mode: str = Form("ambient", pattern="^(prompt|ambient|silent|native-dialogue)$"),
        dialogue: str = Form("", max_length=1_000),
        speaker: str = Form("", max_length=120),
        language: str = Form("English", max_length=80),
        accent: str = Form("", max_length=120),
        execution_target: str = Form(
            "auto", pattern=r"^(auto|[a-z0-9][a-z0-9-]{0,47})$"
        ),
        seed: int = Form(-1, ge=-1),
        image: UploadFile = File(...),
    ) -> QueueResponse:
        await require_inference_capacity("comfy_upload", execution_target)
        selected_client = await choose_client(execution_target, "comfy_upload")
        filename = safe_filename(image.filename or "reference.png", "reference.png")
        temporary = settings.upload_dir / "legacy" / f"{new_id('image')}_{filename}"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        content = await image.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(413, "Reference image is too large")
        temporary.write_bytes(content)
        try:
            remote_name = await selected_client.upload(temporary)
            compiled_prompt, compiled_negative, _audio_contract = compile_audio_prompt(
                prompt, negative_prompt, mode=audio_mode,
                duration_seconds=duration_seconds,
                dialogue=dialogue, speaker=speaker, language=language, accent=accent,
            )
            graph, metadata = adapter.build(
                prompt=compiled_prompt, negative_prompt=compiled_negative,
                width=width, height=height,
                duration_seconds=duration_seconds, fps=fps, seed=seed,
                filename_prefix=f"vbg/legacy/{new_id('clip')}", image_name=remote_name,
                ic_lora_strength=ic_lora_strength,
                image_condition_strength=img_cond_strength,
                profile="motion-draft-4x3",
                include_audio=audio_mode != "silent",
            )
            prompt_id = await selected_client.queue(graph)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (WorkflowError, ComfyError, httpx.HTTPError, OSError) as exc:
            raise HTTPException(502, str(exc)) from exc
        finally:
            temporary.unlink(missing_ok=True)
        return QueueResponse(
            prompt_id=prompt_id,
            estimated_seconds=_estimate_seconds(width, height, metadata["frames"]),
        )

    @app.get("/status/{prompt_id}")
    async def legacy_status(prompt_id: str) -> dict[str, Any]:
        for client in clients:
            try:
                value = await client.history(prompt_id)
            except (ComfyError, httpx.HTTPError, OSError):
                continue
            if value:
                return value
        return {"prompt_id": prompt_id, "status": "running", "files": []}

    @app.get("/output/{filename:path}")
    async def legacy_output(filename: str) -> StreamingResponse:
        artifact = remote_artifact(filename)
        params = urlencode({**artifact, "type": "output"})
        for client in clients:
            try:
                async with httpx.AsyncClient() as http:
                    response = await http.get(f"{client.base_url}/view?{params}", timeout=90)
                if response.status_code == 200:
                    return StreamingResponse(
                        iter([response.content]), media_type="video/mp4",
                        headers={"Content-Disposition": f'attachment; filename="{artifact["filename"]}"'},
                    )
            except httpx.HTTPError:
                continue
        raise HTTPException(404, "Output not found")

    @app.get("/history")
    async def legacy_history(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
        results = await asyncio.gather(
            *(client.all_history() for client in clients), return_exceptions=True
        )
        generations = [
            item for result in results if isinstance(result, list)
            for item in result if item["files"]
        ]
        generations.sort(key=lambda value: value.get("created_at_ms") or 0, reverse=True)
        return {"generations": generations[:limit], "chains": db.list_chains(limit)}

    @app.get("/models")
    async def models() -> dict[str, Any]:
        return {
            "workflow_version": store.version,
            "resolutions": [
                {"label": "Vertical 720×1280", "w": 360, "h": 640, "aspect": "9:16"},
                {"label": "Landscape 1280×720", "w": 640, "h": 360, "aspect": "16:9"},
                {"label": "Square 1024×1024", "w": 512, "h": 512, "aspect": "1:1"},
                {"label": "Draft vertical 512×896", "w": 256, "h": 448, "aspect": "9:16"},
            ],
            "workers": list(settings.comfyui_urls),
            "ollama": {
                "director": settings.director_model,
                "visual_judge": settings.vision_model,
                "visual_scoring_enabled": settings.vision_scoring_enabled,
                "manual_prompt_mode": True,
            },
            "profiles": {
                name: store.profile_status(name)
                for name in store.manifest.get("profiles", {})
            },
            "features": [
                "creative-planning", "manual-prompts", "best-of-n", "automatic-scoring",
                "continuity-assets", "motion-segment-guidance", "masked-reference-audio",
                "persistent-jobs", "platform-exports", "captions", "audio-mixing",
                "durable-chains", "feedback-analytics", "visual-bible-assets",
                "creative-lab-catalog", "direct-t2v", "direct-i2v",
                "script-to-video", "script-element-extraction",
                "brand-product-kits", "per-shot-visual-references",
                "exact-logo-export-overlay",
                "read-only-gpu-monitor",
                "explicit-audio-modes", "dialogue-budget-guard", "voiceover-export",
                "pixel-spatial-upscale-gguf",
                "creative-lab-video-transforms", "creative-lab-cinemagraph",
                "processed-pass-chaining",
                "silent-safe-timeline", "quick-social-finishing",
            ],
            "creative_lab": {
                "catalog": "/creative-lab",
                "policy": "inference-only; optional modes remain operator-gated",
            },
        }

    # Production studio assets ---------------------------------------

    if WEB_DIST.joinpath("index.html").exists():
        @app.get("/studio", response_class=HTMLResponse, include_in_schema=False)
        @app.get("/studio/", response_class=HTMLResponse, include_in_schema=False)
        async def built_studio() -> HTMLResponse:
            return HTMLResponse(WEB_DIST.joinpath("index.html").read_text())

        @app.get("/studio/assets/{filename}", include_in_schema=False)
        async def built_studio_asset(filename: str) -> Response:
            safe = Path(filename).name
            path = WEB_DIST / "assets" / safe
            if not path.exists():
                raise HTTPException(404, "Studio asset not found")
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            return Response(path.read_bytes(), media_type=media_type)

    return app


app = create_app()
