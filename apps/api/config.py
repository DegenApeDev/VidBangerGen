from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parents[1]
PROJECT_ROOT = (
    SOURCE_ROOT
    if (SOURCE_ROOT / "package.json").is_file()
    else Path(os.getenv("VBG_PROJECT_ROOT", Path.cwd())).expanduser().resolve()
)
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class ExecutionTarget:
    """A logical machine or machine pool that can receive production work."""

    id: str
    label: str
    kind: str
    capabilities: tuple[str, ...]
    urls: tuple[str, ...] = ()
    ssh_host: str | None = None
    gpu_index: int = 0
    description: str = ""


def _target_id(value: Any) -> str:
    clean = re.sub(r"[^a-z0-9-]+", "-", str(value or "").strip().lower()).strip("-")
    if not clean or len(clean) > 48:
        raise ValueError("Execution target IDs must contain 1-48 letters, numbers, or dashes")
    return clean


def _execution_targets(legacy_urls: tuple[str, ...], ssh_host: str) -> tuple[ExecutionTarget, ...]:
    """Load an optional portable target map while preserving the existing setup.

    JSON configuration is intentionally environment-owned: adding an arbitrary
    browser-supplied SSH host or inference endpoint would turn a personal studio
    into a network pivot. The UI selects configured targets but cannot create
    credentials or silently administer another machine.
    """
    raw = os.getenv("VBG_EXECUTION_TARGETS_JSON", "").strip()
    if not raw:
        return (
            ExecutionTarget(
                id="primary",
                label="Primary ComfyUI",
                kind="comfyui",
                capabilities=(
                    "ltx-generation", "continuation", "latent-upscale", "post-upscale",
                ),
                urls=legacy_urls,
                description="Configured LTX 2.3 GGUF + Pixel Spatial inference pool",
            ),
            ExecutionTarget(
                id="local",
                label="Local Video2X",
                kind="local-video2x",
                capabilities=("post-upscale",),
                gpu_index=_env_int("VBG_LOCAL_UPSCALE_GPU", 0),
                description="Optional local Real-ESRGAN post-upscale worker",
            ),
        )
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"VBG_EXECUTION_TARGETS_JSON is invalid JSON: {exc}") from exc
    if not isinstance(values, list) or not values:
        raise ValueError("VBG_EXECUTION_TARGETS_JSON must be a non-empty JSON array")
    targets: list[ExecutionTarget] = []
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Every execution target must be a JSON object")
        target_id = _target_id(value.get("id"))
        if target_id == "auto" or target_id in seen_ids:
            raise ValueError(f"Duplicate or reserved execution target ID: {target_id}")
        kind = str(value.get("kind") or "").strip().lower()
        if kind not in {"comfyui", "local-video2x", "ssh-video2x"}:
            raise ValueError(f"Unsupported execution target kind for {target_id}: {kind}")
        default_capabilities = (
            ("ltx-generation", "continuation", "latent-upscale", "post-upscale")
            if kind == "comfyui" else ("post-upscale",)
        )
        capabilities = tuple(
            str(item).strip() for item in value.get("capabilities", default_capabilities)
            if str(item).strip()
        )
        urls = tuple(
            str(item).strip().rstrip("/") for item in value.get("urls", [])
            if str(item).strip()
        )
        if kind == "comfyui" and not urls:
            raise ValueError(f"ComfyUI execution target {target_id} requires at least one URL")
        duplicate_urls = seen_urls.intersection(urls)
        if duplicate_urls:
            raise ValueError(f"ComfyUI URLs may belong to only one target: {sorted(duplicate_urls)}")
        seen_urls.update(urls)
        target_ssh_host = str(value.get("ssh_host") or "").strip() or None
        if kind == "ssh-video2x" and not target_ssh_host:
            raise ValueError(f"SSH upscale target {target_id} requires ssh_host")
        targets.append(ExecutionTarget(
            id=target_id,
            label=str(value.get("label") or target_id).strip()[:80],
            kind=kind,
            capabilities=capabilities,
            urls=urls,
            ssh_host=target_ssh_host,
            gpu_index=int(value.get("gpu_index", 0)),
            description=str(value.get("description") or "").strip()[:240],
        ))
        seen_ids.add(target_id)
    return tuple(targets)


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    database_path: Path
    artifact_dir: Path
    upload_dir: Path
    workflow_manifest: Path
    comfyui_urls: tuple[str, ...]
    comfyui_remote_workflow: str
    ssh_host: str
    ollama_url: str
    director_model: str
    vision_model: str
    max_upload_bytes: int
    generation_timeout_seconds: int
    cors_origins: tuple[str, ...]
    vision_scoring_enabled: bool = False
    legacy_chains_path: Path | None = None
    upload_capable_urls: tuple[str, ...] = ()
    exclusive_comfy_url: str | None = None
    managed_comfy_urls: tuple[str, ...] = ()
    interrupt_capable_urls: tuple[str, ...] = ()
    execution_targets: tuple[ExecutionTarget, ...] = ()
    local_upscaler_binary: str = "video2x"
    remote_lora_dir: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = _env_path("VBG_DATA_DIR", PROJECT_ROOT / "data")
        legacy_urls = tuple(
            value.strip().rstrip("/")
            for value in os.getenv("VBG_COMFYUI_URLS", "http://127.0.0.1:8188").split(",")
            if value.strip()
        )
        ssh_host = os.getenv("VBG_SSH_HOST", "").strip()
        execution_targets = _execution_targets(legacy_urls, ssh_host)
        configured_urls = tuple(
            url for target in execution_targets if target.kind == "comfyui" for url in target.urls
        )
        urls = configured_urls or legacy_urls
        origins = tuple(
            value.strip()
            for value in os.getenv(
                "VBG_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
            ).split(",")
            if value.strip()
        )
        upload_urls = tuple(
            value.strip().rstrip("/")
            for value in os.getenv("VBG_UPLOAD_CAPABLE_URLS", ",".join(urls)).split(",")
            if value.strip()
        )
        managed_urls = tuple(
            value.strip().rstrip("/")
            for value in os.getenv("VBG_MANAGED_COMFY_URLS", "").split(",")
            if value.strip()
        )
        exclusive_url = os.getenv("VBG_EXCLUSIVE_COMFY_URL", "").strip().rstrip("/")
        interrupt_urls = tuple(
            value.strip().rstrip("/")
            for value in os.getenv("VBG_INTERRUPT_CAPABLE_URLS", "").split(",")
            if value.strip()
        )
        return cls(
            data_dir=data_dir,
            database_path=_env_path("VBG_DATABASE_PATH", data_dir / "vidbangergen.sqlite3"),
            artifact_dir=_env_path("VBG_ARTIFACT_DIR", data_dir / "output"),
            upload_dir=_env_path("VBG_UPLOAD_DIR", data_dir / "uploads"),
            workflow_manifest=_env_path(
                "VBG_WORKFLOW_MANIFEST", PACKAGE_ROOT / "workflows/ltx23.json"
            ),
            comfyui_urls=urls,
            comfyui_remote_workflow=os.getenv(
                "VBG_COMFYUI_REMOTE_WORKFLOW", ""
            ),
            ssh_host=ssh_host,
            ollama_url=os.getenv("VBG_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
            director_model=os.getenv("VBG_DIRECTOR_MODEL", "bonsai-27b:latest"),
            vision_model=os.getenv("VBG_VISION_MODEL", "qwen3-vl:8b"),
            max_upload_bytes=_env_int("VBG_MAX_UPLOAD_MB", 20) * 1024 * 1024,
            generation_timeout_seconds=_env_int("VBG_GENERATION_TIMEOUT", 900),
            cors_origins=origins,
            vision_scoring_enabled=_env_bool("VBG_VISION_SCORING_ENABLED", False),
            legacy_chains_path=_env_path(
                "VBG_LEGACY_CHAINS_PATH", PROJECT_ROOT / "apps/api/_chains.json"
            ),
            upload_capable_urls=upload_urls,
            exclusive_comfy_url=exclusive_url or None,
            managed_comfy_urls=managed_urls,
            interrupt_capable_urls=interrupt_urls,
            execution_targets=execution_targets,
            local_upscaler_binary=os.getenv("VBG_LOCAL_UPSCALER_BINARY", "video2x").strip()
            or "video2x",
            remote_lora_dir=os.getenv("VBG_REMOTE_LORA_DIR", "").strip() or None,
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def execution_target(self, target_id: str) -> ExecutionTarget | None:
        return next((target for target in self.resolved_execution_targets() if target.id == target_id), None)

    def resolved_execution_targets(self) -> tuple[ExecutionTarget, ...]:
        if self.execution_targets:
            return self.execution_targets
        return (
            ExecutionTarget(
                id="primary", label="Primary ComfyUI", kind="comfyui",
                capabilities=(
                    "ltx-generation", "continuation", "latent-upscale", "post-upscale",
                ),
                urls=self.comfyui_urls,
                description="Configured LTX 2.3 GGUF + Pixel Spatial inference pool",
            ),
            ExecutionTarget(
                id="local", label="Local GPU", kind="local-video2x",
                capabilities=("post-upscale",),
                description="Local Real-ESRGAN post-upscale worker",
            ),
        )

    def target_for_url(self, url: str) -> ExecutionTarget | None:
        clean = url.rstrip("/")
        return next(
            (target for target in self.resolved_execution_targets() if clean in target.urls), None
        )


SETTINGS = Settings.from_env()

# Backwards-compatible names used by older scripts.
COMFYUI_URL = SETTINGS.comfyui_urls[0]
SSH_HOST = SETTINGS.ssh_host
OUTPUT_DIR = str(SETTINGS.artifact_dir)
WORKFLOW_PATH = SETTINGS.comfyui_remote_workflow
