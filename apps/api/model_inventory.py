from __future__ import annotations

import subprocess
from pathlib import PurePosixPath


def query_remote_model_inventory(
    ssh_host: str, remote_lora_dir: str | None,
) -> dict[str, int] | None:
    """Read the permitted user's LoRA directory without changing the worker.

    `None` means inventory could not be observed; an empty dict means the
    directory was observed and contains no model files.  Keeping these states
    distinct prevents a transient SSH problem from being reported as an
    installation problem.
    """
    if not remote_lora_dir:
        return None
    directory = str(PurePosixPath(remote_lora_dir))
    command = (
        "find " + _shell_quote(directory)
        + " -maxdepth 1 -type f -printf '%f|%s\\n' 2>/dev/null"
    )
    try:
        result = subprocess.run(
            [
                "ssh", "-F", "/dev/null", "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=5", ssh_host, command,
            ],
            capture_output=True, text=True, timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    inventory: dict[str, int] = {}
    for line in result.stdout.splitlines():
        name, separator, raw_size = line.rpartition("|")
        if not separator or not name:
            continue
        try:
            inventory[name] = int(raw_size)
        except ValueError:
            continue
    return inventory


def required_model_files(mode: dict) -> list[str]:
    value = mode.get("model_files") or mode.get("model_file") or []
    if isinstance(value, dict):
        return [str(item) for item in value.values()]
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
