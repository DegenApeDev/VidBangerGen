from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = (
    SOURCE_ROOT
    if (SOURCE_ROOT / "package.json").is_file()
    else Path(os.getenv("VBG_PROJECT_ROOT", Path.cwd())).expanduser().resolve()
)
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_LOCAL_COMFY_URLS = (
    "http://127.0.0.1:8188",
    "http://127.0.0.1:8189",
)
REQUIRED_LTX_NODES = {
    "EmptyLTXVLatentVideo",
    "GetVideoComponents",
    "LoadVideo",
    "LTXICLoRALoaderModelOnly",
    "LTXVAddGuide",
    "LTXVAudioVAEEncode",
    "LTXVCropGuides",
    "LTXVLatentUpsampler",
    "LTXVSetAudioVideoMaskByTime",
    "LTXVTiledVAEDecode",
    "SaveVideo",
    "UnetLoaderGGUF",
}
MANAGED_ENV_KEYS = (
    "VBG_SETUP_COMPLETE",
    "VBG_SETUP_VERSION",
    "VBG_DATA_DIR",
    "VBG_COMFYUI_URLS",
    "VBG_UPLOAD_CAPABLE_URLS",
    "VBG_MANAGED_COMFY_URLS",
    "VBG_INTERRUPT_CAPABLE_URLS",
    "VBG_EXECUTION_TARGETS_JSON",
    "VBG_SSH_HOST",
    "VBG_REMOTE_LORA_DIR",
    "VBG_OLLAMA_URL",
    "VBG_VISION_SCORING_ENABLED",
    "VBG_CORS_ORIGINS",
    "VBG_GENERATION_TIMEOUT",
    "VBG_MAX_UPLOAD_MB",
)
ENV_ASSIGNMENT_RE = re.compile(
    r"^(?P<prefix>\s*(?:export\s+)?)(?P<key>[A-Za-z_][A-Za-z0-9_]*)="
)
SAFE_ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_./:@,+%-]*$")


class SetupCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ComfyProbe:
    url: str
    healthy: bool
    version: str | None = None
    devices: tuple[str, ...] = ()
    missing_ltx_nodes: tuple[str, ...] = ()
    node_inventory_checked: bool = False
    error: str | None = None

    @property
    def ltx_ready(self) -> bool | None:
        if not self.healthy or not self.node_inventory_checked:
            return None
        return not self.missing_ltx_nodes


class Console:
    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self._input = input_fn
        self._output = output_fn

    def write(self, value: str = "") -> None:
        self._output(value)

    def ask(
        self,
        question: str,
        *,
        default: str | None = None,
        allow_empty: bool = False,
        validator: Callable[[str], str] | None = None,
    ) -> str:
        suffix = f" [{default}]" if default not in {None, ""} else ""
        while True:
            try:
                raw = self._input(f"{question}{suffix}: ")
            except (EOFError, KeyboardInterrupt) as exc:
                raise SetupCancelled("Setup cancelled") from exc
            value = raw.strip()
            if not value and default is not None:
                value = default
            if not value and not allow_empty:
                self.write("Please enter a value.")
                continue
            if validator and value:
                try:
                    value = validator(value)
                except ValueError as exc:
                    self.write(str(exc))
                    continue
            return value

    def confirm(self, question: str, *, default: bool) -> bool:
        prompt = "Y/n" if default else "y/N"
        while True:
            try:
                raw = self._input(f"{question} [{prompt}]: ").strip().lower()
            except (EOFError, KeyboardInterrupt) as exc:
                raise SetupCancelled("Setup cancelled") from exc
            if not raw:
                return default
            if raw in {"y", "yes"}:
                return True
            if raw in {"n", "no"}:
                return False
            self.write("Enter yes or no.")

    def choose(
        self,
        question: str,
        options: Sequence[tuple[str, str]],
        *,
        default: str,
    ) -> str:
        self.write(question)
        keys = {key for key, _ in options}
        for index, (key, label) in enumerate(options, start=1):
            marker = " (default)" if key == default else ""
            self.write(f"  {index}. {label}{marker}")
        while True:
            value = self.ask("Choose", default=default).lower()
            if value in keys:
                return value
            if value.isdigit() and 1 <= int(value) <= len(options):
                return options[int(value) - 1][0]
            self.write("Choose one of the listed numbers or names.")


def normalize_comfy_url(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("ComfyUI URL cannot be empty")
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("ComfyUI URLs must use http:// or https://")
    if not parsed.hostname:
        raise ValueError(f"Invalid ComfyUI URL: {value}")
    if parsed.username or parsed.password:
        raise ValueError(
            "Do not embed credentials in a ComfyUI URL; use a private network or reverse proxy"
        )
    if parsed.query or parsed.fragment:
        raise ValueError("ComfyUI URLs cannot contain a query string or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("ComfyUI URL must point to the server root")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid ComfyUI port in {value}") from exc
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{port or 8188}"
    return urlunsplit((parsed.scheme, netloc, "", "", "")).rstrip("/")


def parse_comfy_urls(value: str) -> tuple[str, ...]:
    urls: list[str] = []
    seen: set[str] = set()
    for item in value.split(","):
        if not item.strip():
            continue
        url = normalize_comfy_url(item)
        if url not in seen:
            urls.append(url)
            seen.add(url)
    if not urls:
        raise ValueError("Enter at least one ComfyUI URL")
    return tuple(urls)


def _read_json(url: str, path: str, timeout: float) -> dict:
    request = Request(
        f"{url.rstrip('/')}{path}",
        headers={"Accept": "application/json", "User-Agent": "VidBangerGen-Setup/1"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path} returned a non-object JSON response")
    return value


def probe_comfyui(
    url: str,
    *,
    include_nodes: bool = False,
    timeout: float = 3.0,
    read_json: Callable[[str, str, float], dict] = _read_json,
) -> ComfyProbe:
    clean_url = normalize_comfy_url(url)
    try:
        stats = read_json(clean_url, "/system_stats", timeout)
        system = stats.get("system") if isinstance(stats.get("system"), dict) else {}
        raw_devices = stats.get("devices") if isinstance(stats.get("devices"), list) else []
        devices = tuple(
            str(device.get("name") or device.get("type") or "Unknown device")
            for device in raw_devices
            if isinstance(device, dict)
        )
        missing: tuple[str, ...] = ()
        checked = False
        if include_nodes:
            inventory = read_json(clean_url, "/object_info", max(timeout, 8.0))
            missing = tuple(sorted(REQUIRED_LTX_NODES.difference(inventory)))
            checked = True
        return ComfyProbe(
            url=clean_url,
            healthy=True,
            version=str(system.get("comfyui_version") or "").strip() or None,
            devices=devices,
            missing_ltx_nodes=missing,
            node_inventory_checked=checked,
        )
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        message = getattr(exc, "reason", None) or str(exc) or type(exc).__name__
        return ComfyProbe(url=clean_url, healthy=False, error=str(message)[:240])


def discover_local_comfyui(
    *,
    candidates: Iterable[str] = DEFAULT_LOCAL_COMFY_URLS,
    probe: Callable[..., ComfyProbe] = probe_comfyui,
) -> tuple[ComfyProbe, ...]:
    values = tuple(candidates)
    with ThreadPoolExecutor(max_workers=min(4, len(values) or 1)) as executor:
        results = tuple(executor.map(lambda url: probe(url, timeout=1.0), values))
    return tuple(result for result in results if result.healthy)


def validate_ssh_target(value: str) -> str:
    clean = value.strip()
    if not clean:
        return ""
    if clean.startswith("-") or any(character.isspace() for character in clean):
        raise ValueError("SSH target must look like user@hostname without options")
    if "://" in clean or "/" in clean:
        raise ValueError("SSH target must look like user@hostname, not a URL")
    return clean


def probe_ssh(
    target: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[bool, str]:
    if not target:
        return False, "No SSH target configured"
    try:
        result = runner(
            [
                "ssh",
                "-F",
                "/dev/null",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                target,
                "printf vbg-ready",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)[:240]
    if result.returncode == 0 and result.stdout.strip() == "vbg-ready":
        return True, "SSH connection ready"
    message = (result.stderr or result.stdout or "SSH connection failed").strip()
    return False, message[:240]


def is_local_url(url: str) -> bool:
    hostname = (urlsplit(url).hostname or "").lower()
    return hostname in {"127.0.0.1", "localhost", "::1"}


def _decode_env_value(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith('"') and value.endswith('"'):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, str) else str(decoded)
        except json.JSONDecodeError:
            return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ENV_ASSIGNMENT_RE.match(line)
        if not match:
            continue
        key = match.group("key")
        values[key] = _decode_env_value(line[match.end():])
    return values


def configuration_complete(path: Path) -> bool:
    values = read_env_file(path)
    marker = values.get("VBG_SETUP_COMPLETE", "").strip().lower()
    if marker in {"1", "true", "yes", "on"}:
        return True
    # Existing installations predate the marker. A deliberate worker setting
    # is enough to avoid surprising them with an interactive prompt.
    return bool(values.get("VBG_COMFYUI_URLS", "").strip())


def _encode_env_value(value: str) -> str:
    if "\n" in value or "\r" in value or "\x00" in value:
        raise ValueError("Environment values cannot contain control characters")
    if SAFE_ENV_VALUE_RE.fullmatch(value):
        return value
    return json.dumps(value)


def update_env_file(path: Path, values: dict[str, str]) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    existing_lines: list[str] = []
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup_path = path.with_name(f"{path.name}.setup-backup-{stamp}")
        shutil.copy2(path, backup_path)
        backup_path.chmod(0o600)

    remaining = dict(values)
    rendered: list[str] = []
    seen_keys: set[str] = set()
    for line in existing_lines:
        match = ENV_ASSIGNMENT_RE.match(line)
        if not match:
            rendered.append(line)
            continue
        key = match.group("key")
        if key not in values:
            rendered.append(line)
            continue
        if key in seen_keys:
            continue
        rendered.append(f"{key}={_encode_env_value(values[key])}")
        remaining.pop(key, None)
        seen_keys.add(key)

    if remaining:
        if rendered and rendered[-1].strip():
            rendered.append("")
        rendered.append("# Managed by VidBangerGen first-run setup")
        for key in MANAGED_ENV_KEYS:
            if key in remaining:
                rendered.append(f"{key}={_encode_env_value(remaining.pop(key))}")
        for key in sorted(remaining):
            rendered.append(f"{key}={_encode_env_value(remaining[key])}")

    temporary = path.with_name(f".{path.name}.setup-{os.getpid()}.tmp")
    temporary.write_text("\n".join(rendered).rstrip() + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)
    return backup_path


def _display_probe(console: Console, result: ComfyProbe) -> None:
    if not result.healthy:
        console.write(f"  ✗ {result.url} — {result.error or 'unreachable'}")
        return
    details = list(result.devices)
    if result.version:
        details.append(f"ComfyUI {result.version}")
    console.write(f"  ✓ {result.url}" + (f" — {', '.join(details)}" if details else ""))
    if result.ltx_ready is False:
        console.write(
            "    Reachable, but missing required LTX nodes: "
            + ", ".join(result.missing_ltx_nodes)
        )
    elif result.ltx_ready is True:
        console.write("    Required LTX workflow nodes detected.")


def _default_remote_urls(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if "://" in value or "," in value:
        return value
    return f"http://{value}:8188"


def _select_urls(
    console: Console,
    *,
    local_results: tuple[ComfyProbe, ...],
    probe: Callable[..., ComfyProbe],
) -> tuple[tuple[str, ...], tuple[ComfyProbe, ...], bool]:
    default_mode = "local" if local_results else "remote"
    mode = console.choose(
        "Where does ComfyUI run?",
        (
            ("local", "On this computer"),
            ("remote", "On another computer or server"),
        ),
        default=default_mode,
    )
    remote = mode == "remote"
    if local_results and not remote:
        default_urls = ",".join(result.url for result in local_results)
    elif not remote:
        default_urls = DEFAULT_LOCAL_COMFY_URLS[0]
    else:
        remote_value = console.ask(
            "Remote hostname, IP, or comma-separated ComfyUI URLs",
            validator=_default_remote_urls,
        )
        default_urls = remote_value

    while True:
        raw_urls = console.ask(
            "ComfyUI URL(s), separated by commas",
            default=default_urls,
        )
        try:
            urls = parse_comfy_urls(raw_urls)
        except ValueError as exc:
            console.write(str(exc))
            continue
        console.write("Validating ComfyUI and LTX nodes…")
        results = tuple(probe(url, include_nodes=True, timeout=4.0) for url in urls)
        for result in results:
            _display_probe(console, result)
        healthy = tuple(result for result in results if result.healthy)
        if healthy:
            if len(healthy) != len(results) and not console.confirm(
                "Keep the unreachable workers in the configuration?", default=False
            ):
                urls = tuple(result.url for result in healthy)
                results = healthy
            return urls, results, remote
        if console.confirm(
            "No worker is reachable. Save these URLs for a worker that will start later?",
            default=False,
        ):
            return urls, results, remote
        default_urls = raw_urls


def _install_dependencies(
    root: Path,
    console: Console,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> bool:
    if sys.version_info < (3, 11):
        raise RuntimeError(
            f"Python 3.11+ is required; this setup is running on {sys.version.split()[0]}"
        )
    venv_dir = root / ".venv"
    venv_python = venv_dir / "bin" / "python"
    if os.name == "nt":
        venv_python = venv_dir / "Scripts" / "python.exe"

    ready = True
    install_python = False
    if not venv_python.exists():
        if not console.confirm(
            "Create .venv and install the Python application dependencies?", default=True
        ):
            console.write("Skipped Python dependency installation.")
            ready = False
        else:
            console.write("Creating Python environment…")
            runner([sys.executable, "-m", "venv", str(venv_dir)], cwd=root, check=True)
            install_python = True
    else:
        console.write(f"✓ Python environment detected at {venv_dir}")
        dependency_check = runner(
            [
                str(venv_python),
                "-c",
                "import fastapi, httpx, pydantic, uvicorn",
            ],
            cwd=root,
            capture_output=True,
        )
        install_python = dependency_check.returncode != 0
        if install_python:
            install_python = console.confirm(
                "The Python environment is incomplete. Install application dependencies?",
                default=True,
            )
            if not install_python:
                ready = False
    if install_python:
        console.write("Installing VidBangerGen…")
        runner(
            [str(venv_python), "-m", "pip", "install", "-e", str(root)],
            cwd=root,
            check=True,
        )

    npm = shutil.which("npm")
    if not npm:
        console.write("! npm was not found. Install Node.js 20+ before building the web app.")
        return False
    if (root / "node_modules").is_dir() and (root / "apps/web/node_modules").is_dir():
        console.write("✓ Node dependencies detected.")
    elif console.confirm("Install Node.js dependencies now?", default=True):
        runner([npm, "install"], cwd=root, check=True)
        runner([npm, "--prefix", "apps/web", "install"], cwd=root, check=True)
    else:
        console.write("Skipped Node.js dependency installation.")
        ready = False

    if not (root / "apps/web/dist/index.html").exists() and console.confirm(
        "Build the production web interface now?", default=True
    ):
        runner([npm, "run", "build"], cwd=root, check=True)
    return ready


def run_setup(
    *,
    root: Path = PROJECT_ROOT,
    env_path: Path | None = None,
    console: Console | None = None,
    force: bool = False,
    install_dependencies: bool = True,
    probe: Callable[..., ComfyProbe] = probe_comfyui,
    discover: Callable[..., tuple[ComfyProbe, ...]] = discover_local_comfyui,
    ssh_probe: Callable[[str], tuple[bool, str]] = probe_ssh,
    dependency_installer: Callable[[Path, Console], bool] = _install_dependencies,
) -> int:
    console = console or Console()
    root = root.resolve()
    env_path = (env_path or root / ".env").resolve()

    console.write("VidBangerGen first-run setup")
    console.write("============================")
    console.write("This configures the controller only; it never installs models or changes ComfyUI.")
    console.write()

    if configuration_complete(env_path) and not force:
        console.write(f"✓ Existing configuration detected at {env_path}")
        console.write("Run `vbg doctor` to validate it or `vbg setup --force` to reconfigure.")
        return 0
    if env_path.exists() and not force and not console.confirm(
        f"Update the existing environment file at {env_path}?", default=False
    ):
        console.write("Existing configuration left unchanged.")
        return 0

    console.write("Looking for local ComfyUI workers on ports 8188 and 8189…")
    local_results = discover(probe=probe)
    if local_results:
        for result in local_results:
            _display_probe(console, result)
    else:
        console.write("  No local ComfyUI worker responded. No LAN scan was performed.")

    urls, results, remote = _select_urls(
        console,
        local_results=local_results,
        probe=probe,
    )
    default_label = "Remote ComfyUI" if remote else "Local ComfyUI"
    target_label = console.ask("Name this inference target", default=default_label)

    upload_urls: tuple[str, ...]
    if console.confirm(
        "Can every configured worker accept image/video uploads?", default=True
    ):
        upload_urls = urls
    else:
        while True:
            try:
                upload_urls = parse_comfy_urls(
                    console.ask("Upload-capable ComfyUI URL(s), separated by commas")
                )
            except ValueError as exc:
                console.write(str(exc))
                continue
            unknown_upload_urls = set(upload_urls).difference(urls)
            if not unknown_upload_urls:
                break
            console.write(
                "Upload-capable URLs must also be configured workers: "
                + ", ".join(sorted(unknown_upload_urls))
            )

    owned_workers = console.confirm(
        "Are these workers exclusively controlled by this VidBangerGen installation?",
        default=False,
    )
    pixel_spatial = console.confirm(
        "Is the optional LTX Pixel Spatial model pack installed on this target?",
        default=False,
    )

    ssh_target = ""
    if remote:
        ssh_target = console.ask(
            "Optional SSH target for GPU monitoring (user@hostname; blank to skip)",
            default="",
            allow_empty=True,
            validator=validate_ssh_target,
        )
        if ssh_target:
            ready, detail = ssh_probe(ssh_target)
            console.write(("✓ " if ready else "! ") + detail)
            if not ready and not console.confirm(
                "Save this SSH target anyway?", default=False
            ):
                ssh_target = ""

    existing = read_env_file(env_path)
    data_default = existing.get("VBG_DATA_DIR") or str(root / "data")
    data_dir = Path(console.ask("Runtime data directory", default=data_default)).expanduser()
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    data_dir = data_dir.resolve()
    cors_default = existing.get(
        "VBG_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    )
    cors_origins = console.ask("Allowed browser origins", default=cors_default)

    capabilities = [
        "ltx-generation",
        "continuation",
        "latent-upscale",
    ]
    if pixel_spatial:
        capabilities.append("post-upscale")
    target = {
        "id": "primary",
        "label": target_label[:80],
        "kind": "comfyui",
        "urls": list(urls),
        "capabilities": capabilities,
        "description": (
            "ComfyUI inference pool configured by VidBangerGen first-run setup"
        ),
    }
    values = {
        "VBG_SETUP_COMPLETE": "true",
        "VBG_SETUP_VERSION": "1",
        "VBG_DATA_DIR": str(data_dir),
        "VBG_COMFYUI_URLS": ",".join(urls),
        "VBG_UPLOAD_CAPABLE_URLS": ",".join(upload_urls),
        "VBG_MANAGED_COMFY_URLS": ",".join(urls) if owned_workers else "",
        "VBG_INTERRUPT_CAPABLE_URLS": ",".join(urls) if owned_workers else "",
        "VBG_EXECUTION_TARGETS_JSON": json.dumps(
            [target], separators=(",", ":"), ensure_ascii=True
        ),
        "VBG_SSH_HOST": ssh_target,
        "VBG_REMOTE_LORA_DIR": "",
        "VBG_OLLAMA_URL": existing.get("VBG_OLLAMA_URL", "http://127.0.0.1:11434"),
        "VBG_VISION_SCORING_ENABLED": existing.get(
            "VBG_VISION_SCORING_ENABLED", "false"
        ),
        "VBG_CORS_ORIGINS": cors_origins,
        "VBG_GENERATION_TIMEOUT": existing.get("VBG_GENERATION_TIMEOUT", "1200"),
        "VBG_MAX_UPLOAD_MB": existing.get("VBG_MAX_UPLOAD_MB", "100"),
    }
    backup = update_env_file(env_path, values)
    for directory in (data_dir, data_dir / "output", data_dir / "uploads"):
        directory.mkdir(parents=True, exist_ok=True)

    console.write()
    console.write(f"✓ Configuration saved to {env_path}")
    if backup:
        console.write(f"✓ Previous configuration backed up to {backup}")
    console.write(f"✓ {len(urls)} ComfyUI worker(s) configured")
    ready_count = sum(result.healthy for result in results)
    console.write(f"✓ {ready_count}/{len(results)} worker connection(s) ready now")

    if install_dependencies:
        try:
            dependencies_ready = dependency_installer(root, console)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            console.write(f"! Dependency installation stopped: {exc}")
            console.write(
                "  Configuration was saved; rerun `vbg setup --force` after "
                "resolving the error."
            )
            return 1
        if not dependencies_ready:
            console.write(
                "! Configuration is ready, but required local dependencies are incomplete."
            )
            return 1

    console.write()
    console.write("Setup complete. Start VidBangerGen with:")
    console.write("  npm run dev")
    console.write("Then open http://127.0.0.1:5173")
    return 0


def run_doctor(
    *,
    root: Path = PROJECT_ROOT,
    env_path: Path | None = None,
    console: Console | None = None,
    probe: Callable[..., ComfyProbe] = probe_comfyui,
    ssh_probe: Callable[[str], tuple[bool, str]] = probe_ssh,
) -> int:
    console = console or Console()
    env_path = (env_path or root / ".env").resolve()
    values = read_env_file(env_path)
    console.write("VidBangerGen setup doctor")
    console.write("==========================")
    if not configuration_complete(env_path):
        console.write(f"✗ First-run configuration is incomplete: {env_path}")
        console.write("  Run `vbg setup` or `python3 scripts/setup.py setup`.")
        return 1
    console.write(f"✓ Configuration: {env_path}")

    required_tools = ("ffmpeg", "ffprobe")
    missing_tools = [tool for tool in required_tools if not shutil.which(tool)]
    for tool in required_tools:
        console.write(
            f"{'✓' if tool not in missing_tools else '✗'} {tool}"
            + ("" if tool not in missing_tools else " not found")
        )

    try:
        urls = parse_comfy_urls(values.get("VBG_COMFYUI_URLS", ""))
    except ValueError as exc:
        console.write(f"✗ {exc}")
        return 1
    results = tuple(probe(url, include_nodes=True, timeout=4.0) for url in urls)
    for result in results:
        _display_probe(console, result)

    ssh_target = values.get("VBG_SSH_HOST", "")
    if ssh_target:
        ready, detail = ssh_probe(ssh_target)
        console.write(("✓ " if ready else "! ") + f"SSH: {detail}")

    healthy = [result for result in results if result.healthy]
    ltx_ready = [result for result in healthy if result.ltx_ready is not False]
    if missing_tools or not healthy or not ltx_ready:
        console.write("Doctor found blocking setup problems.")
        return 1
    console.write("Doctor found no blocking setup problems.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vbg",
        description="Configure and validate a VidBangerGen installation.",
    )
    subparsers = parser.add_subparsers(dest="command")
    setup_parser = subparsers.add_parser("setup", help="run interactive first-time setup")
    setup_parser.add_argument("--force", action="store_true", help="reconfigure an existing .env")
    setup_parser.add_argument(
        "--no-install",
        action="store_true",
        help="configure the app without installing Python/Node dependencies",
    )
    setup_parser.add_argument("--env-file", type=Path, help="override the .env path")

    first_parser = subparsers.add_parser(
        "first-run", help="run setup only when no configuration exists"
    )
    first_parser.add_argument(
        "--no-install",
        action="store_true",
        help="do not install dependencies when setup is required",
    )
    first_parser.add_argument("--env-file", type=Path, help="override the .env path")

    doctor_parser = subparsers.add_parser("doctor", help="validate the saved configuration")
    doctor_parser.add_argument("--env-file", type=Path, help="override the .env path")

    serve_parser = subparsers.add_parser(
        "serve", help="run first-time setup when needed, then start the API"
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8090)
    serve_parser.add_argument("--reload", action="store_true")
    return parser


def run_server(
    *,
    host: str,
    port: int,
    reload: bool,
    root: Path = PROJECT_ROOT,
) -> int:
    env_path = root / ".env"
    if not configuration_complete(env_path):
        result = run_setup(root=root, env_path=env_path)
        if result:
            return result
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "apps.api.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        command.append("--reload")
    os.chdir(root)
    os.execv(sys.executable, command)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = args.command or "setup"
    try:
        if command == "doctor":
            return run_doctor(env_path=args.env_file)
        if command == "serve":
            return run_server(host=args.host, port=args.port, reload=args.reload)
        return run_setup(
            env_path=getattr(args, "env_file", None),
            force=getattr(args, "force", False),
            install_dependencies=not getattr(args, "no_install", False),
        )
    except SetupCancelled:
        print("\nSetup cancelled; no further changes were made.", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
