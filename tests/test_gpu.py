from __future__ import annotations

import subprocess

import pytest

from apps.api.gpu import parse_nvidia_smi_csv, query_gpu_status


SAMPLE = """0, NVIDIA Test GPU, 999.1, 52, 0, P8, 0, 411, 24576, 32.13, 300.00
1, NVIDIA Test GPU, 999.1, 68, 66, P2, 0, 22275, 24576, 128.44, 300.00
"""


def test_nvidia_smi_csv_is_normalized_for_the_ui(monkeypatch):
    gpus = parse_nvidia_smi_csv(SAMPLE)
    assert gpus[0]["temperature_c"] == 52
    assert gpus[0]["memory_percent"] == 1.7
    assert gpus[1]["fan_percent"] == 66
    assert gpus[1]["memory_used_mib"] == 22275
    assert gpus[1]["memory_percent"] == 90.6
    assert gpus[1]["power_draw_w"] == 128.44

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, SAMPLE, ""),
    )
    status = query_gpu_status("operator@render.test")
    assert status["host"] == "render.test"
    assert status["polling"] == "read-only nvidia-smi over SSH"
    assert len(status["gpus"]) == 2


def test_nvidia_smi_ssh_failure_is_bounded_and_readable(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 255, "", "SSH agent has no key"
        ),
    )
    with pytest.raises(RuntimeError, match="SSH agent has no key"):
        query_gpu_status("operator@render.test")
