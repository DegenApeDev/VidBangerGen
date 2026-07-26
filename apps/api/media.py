from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


PLATFORM_SIZES = {
    "tiktok": (720, 1280),
    "reels": (720, 1280),
    "shorts": (720, 1280),
    "youtube": (1280, 720),
    "x": (1280, 720),
    "custom": (1280, 720),
}


def safe_filename(filename: str, fallback: str = "upload.bin") -> str:
    value = Path(filename or fallback).name
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value[:180] or fallback


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unable to probe {path.name}: {result.stderr[:200]}")
    return float(json.loads(result.stdout)["format"]["duration"])


def probe_stream_types(path: Path) -> set[str]:
    """Validate uploaded media from decoded stream metadata, not its filename."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe rejected the upload: {result.stderr[:200]}")
    return {
        str(value.get("codec_type")) for value in json.loads(result.stdout).get("streams", [])
        if value.get("codec_type")
    }


def timeline_inputs(
    clips: list[Path], durations: list[float],
) -> tuple[list[str], list[int], int, list[bool]]:
    """Build stable ffmpeg input indexes, synthesising silence when required.

    Structurally silent LTX outputs intentionally contain no audio stream. The
    timeline still needs an audio leg for transitions, captions-plus-voiceover,
    and loudness-normalised delivery. A finite `anullsrc` per silent clip keeps
    those operations deterministic without reintroducing generated speech.
    """
    if len(clips) != len(durations):
        raise RuntimeError("Timeline durations do not match its clips")
    inputs = list(sum((["-i", str(path)] for path in clips), []))
    audio_inputs: list[int] = []
    source_audio: list[bool] = []
    next_input = len(clips)
    for index, (clip, duration) in enumerate(zip(clips, durations, strict=True)):
        if "audio" in probe_stream_types(clip):
            audio_inputs.append(index)
            source_audio.append(True)
            continue
        inputs.extend([
            "-f", "lavfi", "-i",
            f"anullsrc=r=48000:cl=stereo:d={max(0.05, duration):.9f}",
        ])
        audio_inputs.append(next_input)
        source_audio.append(False)
        next_input += 1
    return inputs, audio_inputs, next_input, source_audio


def probe_video_info(path: Path) -> dict[str, Any]:
    """Return the geometry/frame count needed for memory-safe AI upscaling."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries",
            "stream=width,height,nb_read_frames,nb_frames,r_frame_rate",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=90,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unable to inspect {path.name}: {result.stderr[:200]}")
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"{path.name} has no video stream")
    stream = streams[0]
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"{path.name} has invalid video dimensions")
    raw_frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    frames = int(raw_frames) if str(raw_frames or "").isdigit() else 0
    rate = str(stream.get("r_frame_rate") or "0/1").split("/", 1)
    try:
        fps = float(rate[0]) / max(1.0, float(rate[1]))
    except (ValueError, IndexError):
        fps = 0.0
    return {"width": width, "height": height, "frames": frames, "fps": fps}


def prepare_reference_sheet(
    source: Path, destination: Path, width: int = 768, height: int = 448,
) -> dict[str, Any]:
    """Decode and fit a visual bible onto the Ingredients training canvas.

    The image is never cropped or stretched: unused space becomes black, which
    matches the documented panel-sheet format and keeps logos/product geometry
    intact. Decoding with ffmpeg also rejects spoofed or corrupt image uploads
    before any remote GPU work is queued.
    """
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("Reference-sheet dimensions must be positive multiples of 32")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vf",
            (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
            ),
            "-frames:v", "1", str(destination),
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0 or not destination.exists():
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to prepare the visual bible: {result.stderr[-500:]}")
    return {
        "path": str(destination), "width": width, "height": height,
        "fit": "contain", "background": "black", "source_path": str(source),
    }


def prepare_cinemagraph_image(source: Path, destination: Path) -> dict[str, Any]:
    """Decode a still and fit it to the closest Cinemagraph training canvas."""
    info = probe_video_info(source)
    width, height = (
        (704, 512) if int(info["width"]) >= int(info["height"]) else (512, 704)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vf",
            (
                f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={width}:{height}:(iw-ow)/2:(ih-oh)/2"
            ),
            "-frames:v", "1", str(destination),
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0 or not destination.exists():
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to prepare the Cinemagraph source: {result.stderr[-500:]}")
    return {
        "path": str(destination), "width": width, "height": height,
        "fit": "center-crop", "source_dimensions": [info["width"], info["height"]],
        "source_path": str(source),
    }


def prepare_lipdub_source(
    source: Path,
    destination: Path,
    *,
    fps: int = 24,
    max_frames: int = 121,
    landscape_size: tuple[int, int] = (768, 448),
) -> dict[str, Any]:
    """Normalize a one-speaker source for the 24 GB LipDub inference route."""
    streams = probe_stream_types(source)
    if "audio" not in streams:
        raise RuntimeError(
            "LipDub needs source speech audio to preserve the speaker's voice identity"
        )
    info = probe_video_info(source)
    duration = probe_duration(source)
    available_frames = int(math.floor(duration * fps + 1e-6))
    frames = min(max_frames, ((available_frames - 1) // 8) * 8 + 1)
    if frames < 17:
        raise RuntimeError("LipDub needs at least 17 source frames with speech audio")
    if int(info["height"]) > int(info["width"]):
        width, height = landscape_size[1], landscape_size[0]
    else:
        width, height = landscape_size
    output_duration = frames / float(fps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0",
            "-vf",
            (
                f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease:"
                f"flags=lanczos,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
            ),
            "-af", f"aresample=48000,apad=whole_dur={output_duration:.9f}",
            "-frames:v", str(frames), "-t", f"{output_duration:.9f}",
            "-c:v", "libx264", "-crf", "16", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
            "-movflags", "+faststart", str(destination),
        ],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0 or not destination.exists():
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to prepare the LipDub source: {result.stderr[-500:]}")
    prepared = probe_video_info(destination)
    if int(prepared["frames"]) != frames:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"LipDub preparation produced {prepared['frames']} frames instead of {frames}"
        )
    return {
        "path": str(destination), "width": width, "height": height,
        "frames": frames, "fps": fps, "duration_seconds": round(output_duration, 3),
        "source_dimensions": [info["width"], info["height"]],
        "source_frames": info["frames"], "source_fps": info["fps"],
        "fit": "contain", "audio": "reference-speech-preserved",
    }


def prepare_video_mask(
    mask_source: Path, source_video: Path, destination: Path,
) -> dict[str, Any]:
    """Normalize a static or animated binary mask to the source video timeline."""
    source = probe_video_info(source_video)
    mask = probe_video_info(mask_source)
    if (int(mask["width"]), int(mask["height"])) != (
        int(source["width"]), int(source["height"]),
    ):
        raise RuntimeError(
            "The inpaint mask must have the same pixel dimensions as its source video "
            f"({source['width']}x{source['height']}); this mask is "
            f"{mask['width']}x{mask['height']}"
        )
    mask_frames = int(mask.get("frames") or 0)
    source_frames = int(source.get("frames") or 0)
    static = mask_frames <= 1
    if not static and mask_frames != source_frames:
        raise RuntimeError(
            f"An animated mask must match the source frame count ({source_frames}); "
            f"this mask has {mask_frames} frames"
        )
    fps = float(source.get("fps") or 24.0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if static:
        command.extend(["-loop", "1", "-i", str(mask_source)])
    else:
        command.extend(["-i", str(mask_source)])
    command.extend([
        "-vf",
        (
            f"fps={fps:.12g},scale={source['width']}:{source['height']}:flags=neighbor,"
            "format=gray"
        ),
        "-frames:v", str(source_frames), "-an", "-c:v", "ffv1",
        "-pix_fmt", "gray", str(destination),
    ])
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode != 0 or not destination.exists():
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to prepare the video mask: {result.stderr[-500:]}")
    return {
        "path": str(destination), "kind": "static" if static else "animated",
        "width": source["width"], "height": source["height"],
        "frames": source_frames, "fps": fps,
    }


def prepare_outpaint_inputs(
    source: Path,
    output_dir: Path,
    direction: str,
    expansion_percent: int,
    max_side: int = 1024,
) -> dict[str, Any]:
    """Create a padded source and exact binary mask for model-side outpainting."""
    if direction not in {"left", "right", "top", "bottom", "all"}:
        raise ValueError("Unsupported outpaint direction")
    if not 10 <= int(expansion_percent) <= 100:
        raise ValueError("Outpaint expansion must be between 10 and 100 percent")
    info = probe_video_info(source)
    width, height = int(info["width"]), int(info["height"])
    ratio = float(expansion_percent) / 100.0
    left = right = top = bottom = 0
    if direction == "left": left = round(width * ratio)
    elif direction == "right": right = round(width * ratio)
    elif direction == "top": top = round(height * ratio)
    elif direction == "bottom": bottom = round(height * ratio)
    else:
        left = right = round(width * ratio)
        top = bottom = round(height * ratio)
    raw_width, raw_height = width + left + right, height + top + bottom
    scale = min(1.0, float(max_side) / max(raw_width, raw_height))
    base_width = max(64, round(width * scale))
    base_height = max(64, round(height * scale))
    left, right = round(left * scale), round(right * scale)
    top, bottom = round(top * scale), round(bottom * scale)
    target_width = max(64, math.ceil((base_width + left + right) / 64) * 64)
    target_height = max(64, math.ceil((base_height + top + bottom) / 64) * 64)
    # Put any modulus padding on the generated side, never through the source.
    if direction in {"left", "all"}: x = target_width - base_width - right
    else: x = left
    if direction in {"top", "all"}: y = target_height - base_height - bottom
    else: y = top
    x, y = max(0, x), max(0, y)
    output_dir.mkdir(parents=True, exist_ok=True)
    padded = output_dir / "outpaint-source.mp4"
    mask_png = output_dir / "outpaint-mask.png"
    mask_video = output_dir / "outpaint-mask.mkv"
    video = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vf",
            (
                f"scale={base_width}:{base_height}:flags=lanczos,"
                f"pad={target_width}:{target_height}:{x}:{y}:color=black"
            ),
            "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-crf", "16",
            "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
            "-movflags", "+faststart", str(padded),
        ],
        capture_output=True, text=True, timeout=600,
    )
    if video.returncode != 0 or not padded.exists():
        padded.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to prepare outpaint source: {video.stderr[-500:]}")
    mask = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", f"color=white:s={target_width}x{target_height}", "-vf",
            f"drawbox=x={x}:y={y}:w={base_width}:h={base_height}:color=black:t=fill",
            "-frames:v", "1", str(mask_png),
        ],
        capture_output=True, text=True, timeout=120,
    )
    if mask.returncode != 0 or not mask_png.exists():
        padded.unlink(missing_ok=True)
        mask_png.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to prepare outpaint mask: {mask.stderr[-500:]}")
    mask_metadata = prepare_video_mask(mask_png, padded, mask_video)
    mask_png.unlink(missing_ok=True)
    return {
        "source_path": str(padded), "mask_path": str(mask_video),
        "source_dimensions": [width, height],
        "output_dimensions": [target_width, target_height],
        "source_rect": [x, y, base_width, base_height],
        "direction": direction, "expansion_percent": int(expansion_percent),
        "mask": mask_metadata,
    }


def trim_silent_video(
    source: Path, destination: Path, duration_seconds: float,
) -> dict[str, Any]:
    """Create an exact, structurally silent story beat from a model output."""
    if duration_seconds <= 0:
        raise ValueError("Trim duration must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-t", f"{duration_seconds:.6f}", "-map", "0:v:0", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
        ],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0 or not destination.exists():
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to trim the Ingredients clip: {result.stderr[-500:]}")
    return {
        "path": str(destination), "duration_seconds": round(probe_duration(destination), 3),
        "video_codec": "h264", "audio": "omitted",
    }


def plan_upscale_chunks(
    total_frames: int, max_frames: int = 121, overlap_frames: int = 9,
) -> list[tuple[int, int]]:
    """Plan overlapping LTX-compatible frame windows for Pixel Spatial passes."""
    if total_frames <= 0:
        raise RuntimeError("The source video has no countable frames")
    if max_frames < 9 or max_frames % 8 != 1:
        raise ValueError("Pixel Spatial chunk size must be 1 modulo 8")
    if overlap_frames < 1 or overlap_frames >= max_frames or overlap_frames % 8 != 1:
        raise ValueError("Pixel Spatial overlap must be 1 modulo 8 and smaller than a pass")
    if total_frames % 8 != 1:
        raise RuntimeError(
            f"Pixel Spatial expects an LTX frame count of 8n+1; this clip has {total_frames}"
        )
    if total_frames <= max_frames:
        return [(0, total_frames)]
    # Use the requested overlap when possible, but reduce it as far as one
    # frame when that avoids an entire extra diffusion pass. For example, two
    # merged 121-frame clips normalize to 241 frames and fit in two passes with
    # a one-frame overlap instead of three passes with a fixed nine-frame one.
    pass_count = math.ceil((total_frames - 1) / (max_frames - 1))
    minimum_stride = math.ceil((total_frames - max_frames) / (pass_count - 1))
    minimum_stride = math.ceil(minimum_stride / 8) * 8
    stride = min(max_frames - 1, max(max_frames - overlap_frames, minimum_stride))
    windows: list[tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = min(total_frames, start + max_frames)
        windows.append((start, end))
        if end == total_frames:
            break
        start += stride
    return windows


def prepare_upscale_chunks(
    source: Path, output_dir: Path, max_frames: int = 121, overlap_frames: int = 9,
) -> dict[str, Any]:
    """Split a longer LTX clip into exact overlapping AV windows."""
    original_info = probe_video_info(source)
    info = original_info
    working_source = source
    frame_normalization: dict[str, Any] | None = None
    if int(info["frames"]) % 8 != 1:
        original_frames = int(info["frames"])
        prepared_frames = original_frames - ((original_frames - 1) % 8)
        if prepared_frames < 9:
            prepared_frames = 9
        duration = probe_duration(source)
        prepared_fps = prepared_frames / duration
        output_dir.mkdir(parents=True, exist_ok=True)
        working_source = output_dir / "ltx-compatible-source.mp4"
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-map", "0:v:0", "-an",
            "-vf", (
                f"fps={prepared_fps:.12f},trim=end_frame={prepared_frames},"
                "setpts=PTS-STARTPTS"
            ),
            "-frames:v", str(prepared_frames), "-c:v", "libx264", "-preset", "fast",
            "-crf", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(working_source),
        ]
        result = subprocess.run(command, capture_output=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                "Unable to normalize merged video for Pixel Spatial: "
                + result.stderr.decode(errors="replace")[:800]
            )
        info = probe_video_info(working_source)
        if int(info["frames"]) != prepared_frames:
            raise RuntimeError(
                "Pixel Spatial frame normalization produced "
                f"{info['frames']} frames; expected {prepared_frames}"
            )
        frame_normalization = {
            "original_frames": original_frames,
            "prepared_frames": prepared_frames,
            "original_fps": float(original_info["fps"] or 24.0),
            "prepared_fps": float(info["fps"] or prepared_fps),
            "duration_seconds": round(duration, 6),
        }
    fps = float(info["fps"] or 24.0)
    windows = plan_upscale_chunks(int(info["frames"]), max_frames, overlap_frames)
    has_audio = "audio" in probe_stream_types(working_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, Any]] = []
    for index, (start_frame, end_frame) in enumerate(windows):
        path = output_dir / f"source-{index:03d}.mp4"
        expected_frames = end_frame - start_frame
        valid_existing = False
        if path.exists():
            try:
                valid_existing = probe_video_info(path)["frames"] == expected_frames
            except (OSError, RuntimeError, ValueError):
                valid_existing = False
        if not valid_existing:
            path.unlink(missing_ok=True)
            start_seconds, end_seconds = start_frame / fps, end_frame / fps
            if has_audio:
                filters = (
                    f"[0:v]trim=start_frame={start_frame}:end_frame={end_frame},"
                    "setpts=PTS-STARTPTS[v];"
                    f"[0:a]atrim=start={start_seconds:.9f}:end={end_seconds:.9f},"
                    "asetpts=PTS-STARTPTS[a]"
                )
                command = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i",
                    str(working_source),
                    "-filter_complex", filters, "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "12",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
                    "-ar", "48000", "-movflags", "+faststart", str(path),
                ]
            else:
                command = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i",
                    str(working_source),
                    "-vf", (
                        f"trim=start_frame={start_frame}:end_frame={end_frame},"
                        "setpts=PTS-STARTPTS"
                    ),
                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "12",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
                ]
            result = subprocess.run(command, capture_output=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(
                    "Unable to prepare Pixel Spatial pass: "
                    + result.stderr.decode(errors="replace")[:800]
                )
            actual_frames = probe_video_info(path)["frames"]
            if actual_frames != expected_frames:
                raise RuntimeError(
                    f"Pixel Spatial chunk {index + 1} has {actual_frames} frames; "
                    f"expected {expected_frames}"
                )
        chunks.append({
            "path": str(path), "start_frame": start_frame, "end_frame": end_frame,
            "frames": expected_frames,
            "overlap_frames": 0 if index == 0 else windows[index - 1][1] - start_frame,
        })
    return {
        "chunks": chunks, "fps": fps,
        "source_frames": int(original_info["frames"]),
        "prepared_frames": int(info["frames"]),
        "has_audio": has_audio, "overlap_frames": overlap_frames,
        "frame_normalization": frame_normalization,
    }


def merge_upscale_chunks(
    clips: list[Path], source: Path, destination: Path, overlaps: list[float], fps: float,
) -> dict[str, Any]:
    """Blend AI-upscaled video windows and restore the untouched source audio."""
    if not clips or len(clips) != len(overlaps):
        raise RuntimeError("Pixel Spatial merge inputs are incomplete")
    for clip in clips:
        if not clip.exists():
            raise RuntimeError(f"Missing Pixel Spatial pass: {clip.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = probe_video_info(clips[0])
    durations = [probe_duration(path) for path in clips]
    normalised = [0.0]
    for index, value in enumerate(overlaps[1:], start=1):
        normalised.append(max(0.0, min(float(value), durations[index] - 0.05)))
    filters = [
        (
            f"[{index}:v]fps={fps:.9f},scale={target['width']}:{target['height']}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={target['width']}:{target['height']}:(ow-iw)/2:(oh-ih)/2:color=black,"
            "format=yuv420p,settb=AVTB,setpts=PTS-STARTPTS"
            f"[v{index}]"
        )
        for index in range(len(clips))
    ]
    video_label = "v0"
    timeline_duration = durations[0]
    for index in range(1, len(clips)):
        overlap = normalised[index]
        next_video = f"vx{index}"
        offset = max(0.0, timeline_duration - overlap)
        filters.append(
            f"[{video_label}][v{index}]xfade=transition=fade:"
            f"duration={overlap:.9f}:offset={offset:.9f}[{next_video}]"
        )
        timeline_duration += durations[index] - overlap
        video_label = next_video

    video_only = destination.with_name(f".{destination.stem}.video-only.mp4")
    rendered = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *sum((["-i", str(path)] for path in clips), []),
            "-filter_complex", ";".join(filters), "-map", f"[{video_label}]", "-an",
            "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(video_only),
        ],
        capture_output=True, timeout=1800,
    )
    if rendered.returncode != 0:
        raise RuntimeError(
            "Pixel Spatial seam blend failed: "
            + rendered.stderr.decode(errors="replace")[:800]
        )
    audio = restore_source_audio(video_only, source, destination)
    return {
        "path": str(destination), "clip_count": len(clips),
        "duration_seconds": round(probe_duration(destination), 3),
        "overlap_seconds": normalised,
        "source_audio_restored": audio["source_audio_restored"],
        "video_frames_preserved": audio["video_frames_preserved"],
        "video_codec": "libx264", "video_crf": 14,
    }


def restore_source_audio(
    video_only: Path, source: Path, destination: Path,
) -> dict[str, Any]:
    """Mux untouched source audio onto a Pixel-rendered video stream."""
    if not video_only.exists():
        raise RuntimeError(f"Missing Pixel Spatial video: {video_only.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    has_audio = "audio" in probe_stream_types(source)
    if not has_audio:
        video_only.replace(destination)
        return {
            "source_audio_restored": False,
            "video_frames_preserved": True,
            "audio_codec": None,
        }
    expected_frames = int(probe_video_info(video_only)["frames"])
    video_duration = probe_duration(video_only)
    base_command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_only), "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
    ]
    muxed = subprocess.run(
        [
            *base_command, "-c:a", "copy", "-t", f"{video_duration:.9f}",
            "-movflags", "+faststart", str(destination),
        ],
        capture_output=True, timeout=600,
    )
    preservation = "bitstream-copy"
    if muxed.returncode != 0:
        destination.unlink(missing_ok=True)
        muxed = subprocess.run(
            [
                *base_command, "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
                "-t", f"{video_duration:.9f}", "-movflags", "+faststart",
                str(destination),
            ],
            capture_output=True, timeout=600,
        )
        preservation = "aac-compatibility-fallback"
    if muxed.returncode != 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            "Unable to restore source audio after Pixel Spatial render: "
            + muxed.stderr.decode(errors="replace")[:800]
        )
    output_frames = int(probe_video_info(destination)["frames"])
    if output_frames != expected_frames:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            "Source-audio restoration changed the Pixel Spatial frame count "
            f"from {expected_frames} to {output_frames}"
        )
    video_only.unlink(missing_ok=True)
    return {
        "source_audio_restored": True,
        "video_frames_preserved": True,
        "audio_codec": "source-copy" if preservation == "bitstream-copy" else "aac",
        "audio_preservation": preservation,
        **({"audio_bitrate": "256k"} if preservation != "bitstream-copy" else {}),
    }


def mux_generated_foley(
    source: Path, generated_audio: Path, destination: Path,
) -> dict[str, Any]:
    """Replace a clip's soundtrack while bitstream-copying its video.

    The Foley graph returns only lossless audio. Keeping the mux local avoids a
    second video encode and guarantees that the V2A pass cannot alter approved
    frames. The generated track is padded/trimmed to the exact source duration.
    """
    if not source.exists() or not generated_audio.exists():
        raise RuntimeError("Foley mux inputs are incomplete")
    duration = probe_duration(source)
    if duration <= 0:
        raise RuntimeError("Foley source has no usable duration")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source), "-i", str(generated_audio),
            "-filter_complex",
            (
                f"[1:a]aresample=48000,aformat=sample_rates=48000:"
                f"channel_layouts=stereo,apad,atrim=0:{duration:.9f}[foley]"
            ),
            "-map", "0:v:0", "-map", "[foley]", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
            "-t", f"{duration:.9f}", "-movflags", "+faststart", str(destination),
        ],
        capture_output=True, timeout=600,
    )
    if result.returncode != 0 or not destination.exists():
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            "Unable to attach generated Foley: "
            + result.stderr.decode(errors="replace")[:800]
        )
    return {
        "path": str(destination), "duration_seconds": round(probe_duration(destination), 3),
        "video_codec": "copy", "video_frames_preserved": True,
        "audio_codec": "aac", "audio_bitrate": "256k",
        "source_audio_replaced": "audio" in probe_stream_types(source),
    }


def prepare_reference_audio(
    source: Path, destination: Path, duration_seconds: float, seed_seconds: float,
) -> dict[str, Any]:
    """Trim a voice seed and silence-pad it to the complete LTX generation length."""
    if not source.exists():
        raise RuntimeError(f"Reference audio is missing: {source}")
    source_duration = probe_duration(source)
    effective_seed = min(float(seed_seconds), float(duration_seconds), source_duration)
    if effective_seed <= 0:
        raise RuntimeError("Reference audio has no usable duration")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-af", (
                f"atrim=0:{effective_seed:.6f},asetpts=PTS-STARTPTS,"
                f"apad=whole_dur={duration_seconds:.6f},atrim=0:{duration_seconds:.6f}"
            ),
            "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(destination),
        ],
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Unable to prepare reference audio: "
            + result.stderr.decode(errors="replace")[:500]
        )
    return {
        "path": str(destination), "source_duration_seconds": round(source_duration, 3),
        "duration_seconds": round(duration_seconds, 3),
        "seed_seconds": round(effective_seed, 3), "sample_rate": 24000, "channels": 1,
    }


def extract_continuity_assets(
    video_path: Path, output_dir: Path, frames: int = 17, fps: float = 24.0,
) -> dict[str, Any]:
    """Create both a motion tail and final-frame fallback for the next shot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(video_path)
    tail_duration = min(duration, frames / fps)
    tail_path = output_dir / f"{video_path.stem}.tail{frames}.mp4"
    frame_path = output_dir / f"{video_path.stem}.last.png"
    tail = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-sseof", f"-{tail_duration:.4f}",
            "-i", str(video_path), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tail_path),
        ],
        capture_output=True,
        timeout=90,
    )
    frame = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-sseof", "-0.08",
            "-i", str(video_path), "-frames:v", "1", str(frame_path),
        ],
        capture_output=True,
        timeout=60,
    )
    if tail.returncode != 0 or frame.returncode != 0:
        raise RuntimeError("Unable to extract continuity assets")
    return {
        "motion_tail_path": str(tail_path), "last_frame_path": str(frame_path),
        "guide_frames": frames, "guide_fps": fps,
        "overlap_seconds": round(tail_duration, 6),
    }


def merge_chain_clips(
    clips: list[Path], destination: Path, overlaps: list[float] | None = None,
) -> dict[str, Any]:
    """Join clips, optionally crossfading their LTX motion-guide overlap.

    A continuation conditions its opening frames on the previous clip's tail.  A
    plain concat would replay that tail and then jump.  When overlap durations are
    supplied we align and blend those equivalent regions with one high-quality
    encode.  Unconditioned compatible clips retain the lossless concat path.
    """
    if not clips:
        raise RuntimeError("Chain has no clips to merge")
    for clip in clips:
        if not clip.exists():
            raise RuntimeError(f"Missing chain clip: {clip}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    overlaps = list(overlaps or [0.0] * len(clips))
    if len(overlaps) != len(clips):
        raise RuntimeError("Chain overlap metadata must match the clip count")
    overlaps[0] = 0.0
    clip_streams = [probe_stream_types(path) for path in clips]
    mixed_audio_layout = len({"audio" in streams for streams in clip_streams}) > 1
    if any(value > 0 for value in overlaps[1:]) or mixed_audio_layout:
        durations = [probe_duration(path) for path in clips]
        inputs, audio_inputs, _next_input, _source_audio = timeline_inputs(clips, durations)
        shape_probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "json", str(clips[0]),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if shape_probe.returncode != 0:
            raise RuntimeError("Unable to inspect the first chain clip")
        stream = json.loads(shape_probe.stdout).get("streams", [{}])[0]
        target_width = int(stream.get("width") or 0)
        target_height = int(stream.get("height") or 0)
        if target_width <= 0 or target_height <= 0:
            raise RuntimeError("First chain clip has no valid video dimensions")
        normalised = [0.0]
        for index, value in enumerate(overlaps[1:], start=1):
            normalised.append(max(0.0, min(float(value), durations[index] - 0.05)))

        filters = [
            (
                f"[{index}:v]fps=24,scale={target_width}:{target_height}:"
                f"force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:"
                f"(ow-iw)/2:(oh-ih)/2:color=black,"
                f"settb=AVTB,setpts=PTS-STARTPTS[v{index}];"
                f"[{audio_inputs[index]}:a]aresample=48000,aformat=sample_rates=48000:"
                f"channel_layouts=stereo,asetpts=PTS-STARTPTS[a{index}]"
            )
            for index in range(len(clips))
        ]
        video_label, audio_label = "v0", "a0"
        timeline_duration = durations[0]
        for index in range(1, len(clips)):
            overlap = normalised[index]
            next_video, next_audio = f"vx{index}", f"ax{index}"
            if overlap > 0:
                offset = max(0.0, timeline_duration - overlap)
                filters.append(
                    f"[{video_label}][v{index}]xfade=transition=fade:"
                    f"duration={overlap:.6f}:offset={offset:.6f}[{next_video}];"
                    f"[{audio_label}][a{index}]acrossfade=d={overlap:.6f}:"
                    f"c1=tri:c2=tri[{next_audio}]"
                )
                timeline_duration += durations[index] - overlap
            else:
                filters.append(
                    f"[{video_label}][v{index}]concat=n=2:v=1:a=0[{next_video}];"
                    f"[{audio_label}][a{index}]concat=n=2:v=0:a=1[{next_audio}]"
                )
                timeline_duration += durations[index]
            video_label, audio_label = next_video, next_audio

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                *inputs,
                "-filter_complex", ";".join(filters),
                "-map", f"[{video_label}]", "-map", f"[{audio_label}]",
                "-c:v", "libx264", "-preset", "slow", "-crf", "14",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
                "-movflags", "+faststart", str(destination),
            ],
            capture_output=True,
            timeout=1200,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Overlap-aware chain merge failed: "
                + result.stderr.decode(errors="replace")[:800]
            )
        return {
            "path": str(destination), "clip_count": len(clips),
            "duration_seconds": round(probe_duration(destination), 3),
            "copy_only": False, "video_codec": "libx264", "video_crf": 14,
            "overlap_seconds": normalised,
        }

    concat_file = destination.with_suffix(".concat.txt")
    concat_file.write_text(
        "".join(f"file '{str(path.resolve()).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for path in clips)
    )
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
            "-safe", "0", "-i", str(concat_file), "-c", "copy", "-movflags",
            "+faststart", str(destination),
        ],
        capture_output=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Chain merge failed: " + result.stderr.decode(errors="replace")[:800]
        )
    return {
        "path": str(destination), "clip_count": len(clips),
        "duration_seconds": round(probe_duration(destination), 3),
        "copy_only": True,
    }


def _write_srt(captions: str, durations: list[float], destination: Path) -> None:
    lines = [line.strip() for line in captions.splitlines() if line.strip()]
    if not lines:
        destination.write_text("")
        return
    total = sum(durations)
    segment = total / len(lines)

    def stamp(seconds: float) -> str:
        millis = round(seconds * 1000)
        hours, millis = divmod(millis, 3_600_000)
        minutes, millis = divmod(millis, 60_000)
        secs, millis = divmod(millis, 1_000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    blocks = []
    for index, line in enumerate(lines):
        start, end = index * segment, min(total, (index + 1) * segment)
        blocks.append(f"{index + 1}\n{stamp(start)} --> {stamp(end)}\n{line}\n")
    destination.write_text("\n".join(blocks))


def render_timeline(
    clips: list[Path], destination: Path, options: dict[str, Any],
    music_path: Path | None = None, voiceover_path: Path | None = None,
    logo_path: Path | None = None,
) -> dict[str, Any]:
    if not clips:
        raise RuntimeError("No selected clips are available to export")
    for clip in clips:
        if not clip.exists():
            raise RuntimeError(f"Missing selected clip: {clip}")
    width, height = PLATFORM_SIZES.get(options.get("platform", "custom"), (1280, 720))
    if options.get("width") and options.get("height"):
        width, height = int(options["width"]), int(options["height"])
    width -= width % 2
    height -= height % 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    work_dir = destination.parent / f".{destination.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    durations = [probe_duration(clip) for clip in clips]
    requested_transition = float(options.get("transition_seconds", 0.15))
    transitions = [0.0]
    for index in range(1, len(clips)):
        transitions.append(
            max(
                0.0,
                min(requested_transition, durations[index - 1] - 0.05, durations[index] - 0.05),
            )
        )

    inputs, audio_inputs, next_input, source_audio = timeline_inputs(clips, durations)
    filters = []
    for index in range(len(clips)):
        filters.extend(
            [
                (
                    f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                    f"crop={width}:{height},fps=24,format=yuv420p,settb=AVTB,"
                    f"setpts=PTS-STARTPTS[v{index}]"
                ),
                (
                    f"[{audio_inputs[index]}:a]aresample=48000,aformat=sample_rates=48000:"
                    f"channel_layouts=stereo,asetpts=PTS-STARTPTS[a{index}]"
                ),
            ]
        )

    video_label, audio_label = "v0", "a0"
    timeline_duration = durations[0]
    for index in range(1, len(clips)):
        transition = transitions[index]
        next_video, next_audio = f"vx{index}", f"ax{index}"
        if transition > 0:
            offset = max(0.0, timeline_duration - transition)
            filters.extend(
                [
                    (
                        f"[{video_label}][v{index}]xfade=transition=fade:"
                        f"duration={transition:.6f}:offset={offset:.6f}[{next_video}]"
                    ),
                    (
                        f"[{audio_label}][a{index}]acrossfade=d={transition:.6f}:"
                        f"c1=tri:c2=tri[{next_audio}]"
                    ),
                ]
            )
            timeline_duration += durations[index] - transition
        else:
            filters.extend(
                [
                    f"[{video_label}][v{index}]concat=n=2:v=1:a=0[{next_video}]",
                    f"[{audio_label}][a{index}]concat=n=2:v=0:a=1[{next_audio}]",
                ]
            )
            timeline_duration += durations[index]
        video_label, audio_label = next_video, next_audio

    mix_inputs = ["[original]"]
    if any(source_audio):
        filters.append(
            f"[{audio_label}]loudnorm=I=-16:TP=-1.5:LRA=11,"
            f"volume={float(options.get('original_audio_volume', 0.8))}[original]"
        )
    else:
        # EBU loudness measurement is undefined for digital silence and some
        # ffmpeg builds emit NaN samples. Keep the synthetic track untouched.
        filters.append(f"[{audio_label}]volume=0[original]")
    if music_path:
        inputs.extend(["-stream_loop", "-1", "-i", str(music_path)])
        filters.append(
            f"[{next_input}:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"volume={float(options.get('music_volume', 0.18))}[music]"
        )
        mix_inputs.append("[music]")
        next_input += 1
    if voiceover_path:
        inputs.extend(["-i", str(voiceover_path)])
        filters.append(
            f"[{next_input}:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"volume={float(options.get('voiceover_volume', 1.0))}[voice]"
        )
        mix_inputs.append("[voice]")
        next_input += 1

    visual_label = video_label
    if logo_path:
        if not logo_path.exists():
            raise RuntimeError(f"Missing logo asset: {logo_path}")
        inputs.extend(["-loop", "1", "-framerate", "24", "-i", str(logo_path)])
        logo_width = max(16, round(width * float(options.get("logo_width_percent", 14.0)) / 100))
        logo_opacity = float(options.get("logo_opacity", 1.0))
        margin = max(0, round(min(width, height) * float(
            options.get("logo_margin_percent", 3.0)
        ) / 100))
        position = options.get("logo_position", "bottom-right")
        coordinates = {
            "top-left": (str(margin), str(margin)),
            "top-right": (f"main_w-overlay_w-{margin}", str(margin)),
            "bottom-left": (str(margin), f"main_h-overlay_h-{margin}"),
            "bottom-right": (
                f"main_w-overlay_w-{margin}", f"main_h-overlay_h-{margin}"
            ),
        }
        x, y = coordinates.get(position, coordinates["bottom-right"])
        filters.extend(
            [
                (
                    f"[{next_input}:v]scale={logo_width}:-1:"
                    "force_original_aspect_ratio=decrease:flags=lanczos,format=rgba,"
                    f"colorchannelmixer=aa={logo_opacity:.4f}[brandlogo]"
                ),
                f"[{video_label}][brandlogo]overlay=x={x}:y={y}:eof_action=repeat[vbrand]",
            ]
        )
        visual_label = "vbrand"

    captions = options.get("captions", "").strip()
    srt_path = work_dir / "captions.srt"
    if captions and options.get("burn_captions", True):
        caption_durations = [durations[0], *[
            durations[index] - transitions[index] for index in range(1, len(durations))
        ]]
        _write_srt(captions, caption_durations, srt_path)
        escaped = str(srt_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        caption_size = max(22, round(height * 0.038))
        caption_margin = max(40, round(height * 0.08))
        filters.append(
            f"[{visual_label}]subtitles='{escaped}':force_style='FontName=DejaVu Sans,"
            f"FontSize={caption_size},"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,"
            f"Shadow=0,Alignment=2,MarginV={caption_margin}'[vout]"
        )
    else:
        filters.append(f"[{visual_label}]null[vout]")
    if any(source_audio) or music_path or voiceover_path:
        filters.append(
            "".join(mix_inputs)
            + f"amix=inputs={len(mix_inputs)}:duration=first:normalize=0,"
            "loudnorm=I=-14:TP=-1.5:LRA=11[aout]"
        )
    else:
        filters.append("[original]anull[aout]")
    filter_complex = ";".join(filters)
    final = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
            "-filter_complex", filter_complex, "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-c:a", "aac",
            "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", "-shortest",
            "-t", f"{timeline_duration:.9f}",
            str(destination),
        ],
        capture_output=True,
        timeout=1200,
    )
    if final.returncode != 0:
        raise RuntimeError(f"Final export failed: {final.stderr.decode(errors='replace')[:800]}")
    return {
        "path": str(destination),
        "width": width,
        "height": height,
        "duration_seconds": round(probe_duration(destination), 3),
        "clip_count": len(clips),
        "transition_seconds": transitions,
        "captions_burned": bool(captions and options.get("burn_captions", True)),
        "music_mixed": bool(music_path),
        "voiceover_mixed": bool(voiceover_path),
        "logo_overlaid": bool(logo_path),
        "logo_position": options.get("logo_position") if logo_path else None,
        "logo_width_percent": options.get("logo_width_percent") if logo_path else None,
        "audio_target_lufs": -14,
        "video_codec": "libx264", "video_crf": 16, "video_preset": "slow",
        "video_encode_passes": 1, "audio_codec": "aac", "audio_bitrate": "256k",
        "audio_sample_rate": 48000,
    }
