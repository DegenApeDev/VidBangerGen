from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.api.setup_wizard import (
    ComfyProbe,
    Console,
    configuration_complete,
    normalize_comfy_url,
    parse_comfy_urls,
    probe_comfyui,
    read_env_file,
    run_doctor,
    run_setup,
    update_env_file,
)


def scripted_console(answers: list[str]) -> tuple[Console, list[str]]:
    values = iter(answers)
    output: list[str] = []

    def read(_prompt: str) -> str:
        return next(values)

    return Console(input_fn=read, output_fn=output.append), output


def ready_probe(url: str, **_kwargs) -> ComfyProbe:
    return ComfyProbe(
        url=normalize_comfy_url(url),
        healthy=True,
        version="0.test",
        devices=("cuda:0 Test GPU",),
        node_inventory_checked=True,
    )


def test_normalize_comfy_urls_defaults_ports_and_rejects_credentials():
    assert normalize_comfy_url("localhost") == "http://localhost:8188"
    assert normalize_comfy_url("https://renderbox.local:9443/") == (
        "https://renderbox.local:9443"
    )
    assert parse_comfy_urls(
        "localhost:8188, http://localhost:8188, renderbox.local:8189"
    ) == ("http://localhost:8188", "http://renderbox.local:8189")

    with pytest.raises(ValueError, match="credentials"):
        normalize_comfy_url("http://user:secret@renderbox.local:8188")
    with pytest.raises(ValueError, match="server root"):
        normalize_comfy_url("http://renderbox.local:8188/comfy")


def test_probe_comfyui_reports_missing_ltx_nodes():
    def read_json(_url: str, path: str, _timeout: float) -> dict:
        if path == "/system_stats":
            return {
                "system": {"comfyui_version": "0.99"},
                "devices": [{"name": "cuda:0 RTX Test"}],
            }
        return {"SaveVideo": {}, "UnetLoaderGGUF": {}}

    result = probe_comfyui(
        "http://localhost:8188",
        include_nodes=True,
        read_json=read_json,
    )

    assert result.healthy is True
    assert result.version == "0.99"
    assert result.devices == ("cuda:0 RTX Test",)
    assert result.node_inventory_checked is True
    assert "EmptyLTXVLatentVideo" in result.missing_ltx_nodes
    assert result.ltx_ready is False


def test_update_env_is_private_preserves_unknown_values_and_creates_backup(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text("# operator value\nCUSTOM_SETTING=keep-me\nVBG_COMFYUI_URLS=old\n")

    backup = update_env_file(
        env_path,
        {
            "VBG_SETUP_COMPLETE": "true",
            "VBG_COMFYUI_URLS": "http://127.0.0.1:8188",
            "VBG_EXECUTION_TARGETS_JSON": '[{"id":"primary","label":"Creator\'s GPU"}]',
        },
    )

    values = read_env_file(env_path)
    assert values["CUSTOM_SETTING"] == "keep-me"
    assert values["VBG_COMFYUI_URLS"] == "http://127.0.0.1:8188"
    assert json.loads(values["VBG_EXECUTION_TARGETS_JSON"])[0]["label"] == "Creator's GPU"
    assert env_path.stat().st_mode & 0o777 == 0o600
    assert backup is not None and backup.read_text().endswith("VBG_COMFYUI_URLS=old\n")


def test_first_run_setup_detects_local_worker_and_writes_owned_pool(tmp_path: Path):
    console, output = scripted_console([
        "",  # local mode
        "",  # detected URL
        "",  # target label
        "",  # uploads supported
        "yes",  # exclusively owned
        "",  # Pixel Spatial unavailable
        "",  # data directory
        "",  # CORS
    ])
    detected = ready_probe("http://127.0.0.1:8188")

    result = run_setup(
        root=tmp_path,
        console=console,
        install_dependencies=False,
        probe=ready_probe,
        discover=lambda **_kwargs: (detected,),
    )

    assert result == 0
    values = read_env_file(tmp_path / ".env")
    assert configuration_complete(tmp_path / ".env")
    assert values["VBG_COMFYUI_URLS"] == "http://127.0.0.1:8188"
    assert values["VBG_UPLOAD_CAPABLE_URLS"] == "http://127.0.0.1:8188"
    assert values["VBG_MANAGED_COMFY_URLS"] == "http://127.0.0.1:8188"
    assert values["VBG_INTERRUPT_CAPABLE_URLS"] == "http://127.0.0.1:8188"
    target = json.loads(values["VBG_EXECUTION_TARGETS_JSON"])[0]
    assert target["label"] == "Local ComfyUI"
    assert target["urls"] == ["http://127.0.0.1:8188"]
    assert "post-upscale" not in target["capabilities"]
    assert (tmp_path / "data/output").is_dir()
    assert any("1/1 worker connection" in line for line in output)


def test_remote_setup_validates_ssh_and_defaults_to_shared_workers(tmp_path: Path):
    console, _output = scripted_console([
        "",  # remote mode is the default when discovery is empty
        "renderbox.local",  # remote hostname
        "",  # normalized URL
        "Render box",
        "",  # uploads supported
        "",  # shared worker (remote default)
        "",  # Pixel Spatial unavailable
        "creator@renderbox.local",
        "",  # data directory
        "",  # CORS
    ])

    result = run_setup(
        root=tmp_path,
        console=console,
        install_dependencies=False,
        probe=ready_probe,
        discover=lambda **_kwargs: (),
        ssh_probe=lambda _target: (True, "SSH connection ready"),
    )

    assert result == 0
    values = read_env_file(tmp_path / ".env")
    assert values["VBG_COMFYUI_URLS"] == "http://renderbox.local:8188"
    assert values["VBG_MANAGED_COMFY_URLS"] == ""
    assert values["VBG_INTERRUPT_CAPABLE_URLS"] == ""
    assert values["VBG_SSH_HOST"] == "creator@renderbox.local"
    assert json.loads(values["VBG_EXECUTION_TARGETS_JSON"])[0]["label"] == "Render box"


def test_first_run_preserves_pre_marker_install_without_prompting(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text("VBG_COMFYUI_URLS=http://existing.test:8188\n")

    def unexpected_input(_prompt: str) -> str:
        raise AssertionError("configured first-run check must not prompt")

    console = Console(input_fn=unexpected_input, output_fn=lambda _value: None)
    result = run_setup(
        root=tmp_path,
        console=console,
        install_dependencies=False,
    )

    assert result == 0
    assert env_path.read_text() == "VBG_COMFYUI_URLS=http://existing.test:8188\n"


def test_doctor_is_read_only_and_fails_when_required_nodes_are_missing(tmp_path: Path):
    env_path = tmp_path / ".env"
    update_env_file(
        env_path,
        {
            "VBG_SETUP_COMPLETE": "true",
            "VBG_COMFYUI_URLS": "http://127.0.0.1:8188",
        },
    )
    console, output = scripted_console([])

    def incomplete_probe(url: str, **_kwargs) -> ComfyProbe:
        return ComfyProbe(
            url=url,
            healthy=True,
            missing_ltx_nodes=("LTXVLatentUpsampler",),
            node_inventory_checked=True,
        )

    before = env_path.read_bytes()
    result = run_doctor(
        root=tmp_path,
        console=console,
        probe=incomplete_probe,
    )

    assert result == 1
    assert env_path.read_bytes() == before
    assert any("missing required LTX nodes" in line for line in output)
