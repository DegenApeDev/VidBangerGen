from __future__ import annotations

import csv
import subprocess
from datetime import UTC, datetime
from typing import Any


GPU_FIELDS = (
    "index", "name", "driver_version", "temperature.gpu", "fan.speed", "pstate",
    "utilization.gpu", "memory.used", "memory.total", "power.draw", "power.limit",
)


def _number(value: str, *, integer: bool = False) -> int | float | None:
    clean = value.strip()
    if not clean or clean.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        number = float(clean)
    except ValueError:
        return None
    return int(round(number)) if integer else round(number, 2)


def parse_nvidia_smi_csv(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) != len(GPU_FIELDS):
            continue
        index, name, driver, temperature, fan, pstate, utilization, used, total, power, limit = (
            value.strip() for value in row
        )
        used_mib = _number(used, integer=True)
        total_mib = _number(total, integer=True)
        utilization_percent = _number(utilization, integer=True)
        memory_percent = (
            round(float(used_mib) / float(total_mib) * 100, 1)
            if used_mib is not None and total_mib else None
        )
        gpus.append(
            {
                "index": _number(index, integer=True),
                "name": name,
                "driver_version": driver,
                "temperature_c": _number(temperature, integer=True),
                "fan_percent": _number(fan, integer=True),
                "performance_state": pstate,
                "utilization_percent": utilization_percent,
                "memory_used_mib": used_mib,
                "memory_total_mib": total_mib,
                "memory_percent": memory_percent,
                "power_draw_w": _number(power),
                "power_limit_w": _number(limit),
            }
        )
    if not gpus:
        raise RuntimeError("nvidia-smi returned no readable GPU rows")
    return gpus


def query_gpu_status(ssh_host: str) -> dict[str, Any]:
    """Read remote telemetry without mutating drivers, processes, or ComfyUI."""
    result = subprocess.run(
        [
            "ssh", "-F", "/dev/null", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            ssh_host,
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "nvidia-smi failed").strip()
        raise RuntimeError(message[:300])
    return {
        "host": ssh_host.split("@")[-1],
        "updated_at": datetime.now(UTC).isoformat(),
        "polling": "read-only nvidia-smi over SSH",
        "gpus": parse_nvidia_smi_csv(result.stdout),
    }
