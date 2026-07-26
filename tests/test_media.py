from __future__ import annotations

import json
import subprocess
from pathlib import Path

from apps.api.media import (
    extract_continuity_assets, merge_chain_clips, merge_upscale_chunks,
    plan_upscale_chunks, prepare_reference_audio, prepare_upscale_chunks,
    probe_duration, probe_stream_types, probe_video_info, render_timeline,
    restore_source_audio,
)
from apps.api.scoring import (
    MediaScorer, normalize_visual_judgment, parse_ollama_structured_message,
)


def make_clip(path: Path, color: str, duration: float = 1.2) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c={color}:s=640x640:r=24:d={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={duration}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
        ],
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode()


def make_silent_clip(path: Path, color: str, duration: float = 1.2) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c={color}:s=640x640:r=24:d={duration}",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode()


def make_logo(path: Path) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=lime:s=120x60:d=0.1",
            "-frames:v", "1", str(path),
        ],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_probe_continuity_and_platform_export(test_settings, tmp_path):
    first, second = tmp_path / "one.mp4", tmp_path / "two.mp4"
    make_clip(first, "red")
    make_clip(second, "blue")
    probe = MediaScorer(test_settings)._technical_probe(first)
    assert probe["valid"] is True
    assert probe["width"] == 640
    assets = extract_continuity_assets(first, tmp_path / "continuity")
    assert Path(assets["motion_tail_path"]).exists()
    assert Path(assets["last_frame_path"]).exists()
    assert assets["guide_frames"] == 17
    assert 0.7 <= assets["overlap_seconds"] <= 0.71

    logo = tmp_path / "logo.png"
    make_logo(logo)
    destination = tmp_path / "vertical.mp4"
    metadata = render_timeline(
        [first, second], destination,
        {
            "platform": "reels", "width": 360, "height": 640,
            "captions": "STOP SCROLLING\nTHE REVEAL",
            "burn_captions": True, "original_audio_volume": 0.8,
            "logo_position": "top-left", "logo_width_percent": 12,
            "logo_opacity": 0.8,
        },
        logo_path=logo,
    )
    assert destination.exists()
    assert metadata["width"] == 360
    assert metadata["height"] == 640
    assert metadata["clip_count"] == 2
    assert metadata["captions_burned"] is True
    assert metadata["logo_overlaid"] is True
    assert metadata["logo_position"] == "top-left"
    assert metadata["transition_seconds"] == [0.0, 0.15]
    assert metadata["video_encode_passes"] == 1
    assert metadata["video_crf"] == 16
    assert 2.2 <= metadata["duration_seconds"] <= 2.3
    audio_probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate", "-of", "json", str(destination),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert json.loads(audio_probe.stdout)["streams"][0]["sample_rate"] == "48000"

    padded_audio = tmp_path / "voice_seed.wav"
    audio_metadata = prepare_reference_audio(first, padded_audio, 3.0, 0.75)
    assert 2.95 <= probe_duration(padded_audio) <= 3.05
    assert audio_metadata["seed_seconds"] == 0.75

    merged = tmp_path / "chain.mp4"
    chain_metadata = merge_chain_clips([first, second], merged)
    assert merged.exists()
    assert chain_metadata["clip_count"] == 2
    assert chain_metadata["copy_only"] is True

    blended = tmp_path / "chain-blended.mp4"
    blended_metadata = merge_chain_clips([first, second], blended, [0.0, 0.5])
    assert blended.exists()
    assert blended_metadata["copy_only"] is False
    assert 1.85 <= blended_metadata["duration_seconds"] <= 1.95


def test_silent_ltx_clips_can_be_finished_and_mixed_with_audio_clips(tmp_path):
    silent = tmp_path / "silent.mp4"
    voiced = tmp_path / "voiced.mp4"
    make_silent_clip(silent, "green")
    make_clip(voiced, "blue")
    assert probe_stream_types(silent) == {"video"}

    finished = tmp_path / "finished.mp4"
    metadata = render_timeline(
        [silent], finished,
        {
            "platform": "reels", "width": 360, "height": 640,
            "captions": "A SILENT VISUAL", "burn_captions": True,
        },
    )
    assert finished.exists()
    assert probe_stream_types(finished) == {"video", "audio"}
    assert metadata["captions_burned"] is True

    mixed = tmp_path / "mixed-chain.mp4"
    mixed_metadata = merge_chain_clips([silent, voiced], mixed)
    assert mixed.exists()
    assert probe_stream_types(mixed) == {"video", "audio"}
    assert mixed_metadata["copy_only"] is False


def test_long_pixel_upscale_is_chunked_blended_and_restores_audio(tmp_path):
    assert plan_upscale_chunks(193) == [(0, 121), (112, 193)]
    assert plan_upscale_chunks(241) == [(0, 121), (120, 241)]
    source = tmp_path / "eight-seconds.mp4"
    make_clip(source, "purple", duration=193 / 24)
    assert probe_video_info(source)["frames"] == 193

    prepared = prepare_upscale_chunks(source, tmp_path / "passes")
    assert [value["frames"] for value in prepared["chunks"]] == [121, 81]
    assert [value["overlap_frames"] for value in prepared["chunks"]] == [0, 9]

    destination = tmp_path / "merged-upscale.mp4"
    result = merge_upscale_chunks(
        [Path(value["path"]) for value in prepared["chunks"]],
        source, destination, [0.0, 9 / 24], 24.0,
    )
    assert destination.exists()
    assert result["source_audio_restored"] is True
    assert "audio" in probe_stream_types(destination)
    assert 192 <= probe_video_info(destination)["frames"] <= 194
    assert 8.0 <= probe_duration(destination) <= 8.1


def test_merged_video_is_normalized_for_two_pixel_passes(tmp_path):
    source = tmp_path / "two-merged-clips.mp4"
    make_clip(source, "orange", duration=242 / 24)
    assert probe_video_info(source)["frames"] == 242

    prepared = prepare_upscale_chunks(source, tmp_path / "merged-passes")

    assert prepared["source_frames"] == 242
    assert prepared["prepared_frames"] == 241
    assert prepared["frame_normalization"]["original_frames"] == 242
    assert prepared["frame_normalization"]["prepared_frames"] == 241
    assert [value["frames"] for value in prepared["chunks"]] == [121, 121]
    assert [value["overlap_frames"] for value in prepared["chunks"]] == [0, 1]


def test_single_pixel_pass_restores_approved_source_audio(tmp_path):
    source = tmp_path / "source-with-voice.mp4"
    rendered = tmp_path / "pixel-video-only.mp4"
    destination = tmp_path / "pixel-with-source-audio.mp4"
    make_clip(source, "teal", duration=1.2)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-map", "0:v:0", "-an", "-c:v", "copy", str(rendered),
        ],
        capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode()

    metadata = restore_source_audio(rendered, source, destination)

    assert metadata["source_audio_restored"] is True
    assert not rendered.exists()
    assert probe_stream_types(destination) == {"video", "audio"}


def test_qwen_vision_structured_output_accepts_thinking_field():
    value = parse_ollama_structured_message(
        {"message": {"content": "", "thinking": '{"prompt_alignment": 23}'}}
    )
    assert value["prompt_alignment"] == 23


def test_visual_judge_caps_attractive_wrong_subject():
    value = normalize_visual_judgment(
        {
            "primary_subject_present": False, "identity_consistent": False,
            "observed_primary_subject": "human puppet", "prompt_alignment": 25,
            "temporal_coherence": 20, "aesthetics": 15, "hook_strength": 10,
            "major_prompt_misses": ["robot is absent"], "issues": [],
        }
    )
    assert value["prompt_alignment"] == 10
    assert value["temporal_coherence"] == 12
    assert value["aesthetics"] == 15
    assert "robot is absent" in value["issues"]


def test_unavailable_visual_judge_never_publishes_a_fake_neutral_score(
    test_settings,
):
    scorer = MediaScorer(test_settings)
    technical = {
        "valid": True, "issues": [], "duration_seconds": 5.0,
        "width": 1280, "height": 704,
    }
    visual = {
        "available": False,
        "prompt_alignment": 16.0,
        "temporal_coherence": 12.0,
        "aesthetics": 9.0,
        "hook_strength": 6.0,
        "issues": ["vision judge unavailable"],
        "summary": "No creative score assigned.",
        "judge": "technical-fallback-v1",
    }
    score = scorer._combine_scores(technical, visual)

    assert score["available"] is False
    assert score["technical_score"] == 30
    assert score["total"] is None
    assert score["judge"] == "technical-fallback-v1"
