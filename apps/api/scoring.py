from __future__ import annotations

import asyncio
import base64
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx

from .config import Settings


def parse_ollama_structured_message(payload: dict[str, Any]) -> dict[str, Any]:
    message = payload.get("message", {})
    raw_text = message.get("content") or message.get("thinking") or "{}"
    value = json.loads(raw_text)
    if not isinstance(value, dict):
        raise ValueError("Vision judge response must be a JSON object")
    return value


def normalize_visual_judgment(raw: dict[str, Any]) -> dict[str, Any]:
    def clamp(value: Any, maximum: float) -> float:
        try:
            return round(max(0.0, min(maximum, float(value))), 2)
        except (TypeError, ValueError):
            return maximum * 0.6

    prompt_alignment = clamp(raw.get("prompt_alignment"), 25)
    temporal_coherence = clamp(raw.get("temporal_coherence"), 20)
    primary_subject_present = raw.get("primary_subject_present") is not False
    identity_consistent = raw.get("identity_consistent") is not False
    if not primary_subject_present:
        prompt_alignment = min(prompt_alignment, 10.0)
    if not identity_consistent:
        temporal_coherence = min(temporal_coherence, 12.0)
    major_misses = [str(value)[:300] for value in raw.get("major_prompt_misses", [])[:10]]
    issues = [str(value)[:300] for value in raw.get("issues", [])[:10]]
    if not primary_subject_present:
        issues.insert(0, "required primary subject is absent or the wrong entity")
    return {
        "available": True,
        "prompt_alignment": prompt_alignment,
        "temporal_coherence": temporal_coherence,
        "aesthetics": clamp(raw.get("aesthetics"), 15),
        "hook_strength": clamp(raw.get("hook_strength"), 10),
        "issues": list(dict.fromkeys([*issues, *major_misses])),
        "summary": str(raw.get("summary", ""))[:1_000],
        "primary_subject_present": primary_subject_present,
        "identity_consistent": identity_consistent,
        "observed_primary_subject": str(raw.get("observed_primary_subject", ""))[:300],
        "major_prompt_misses": major_misses,
    }


class MediaScorer:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def score(
        self, video_path: Path, prompt: str, brief: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        technical = await asyncio.to_thread(self._technical_probe, video_path)
        visual = await self._visual_judge(video_path, prompt, brief or {})
        return self._combine_scores(technical, visual)

    def _combine_scores(
        self, technical: dict[str, Any], visual: dict[str, Any]
    ) -> dict[str, Any]:
        technical_points = self._technical_points(technical)
        judge_available = bool(visual.get("available", True))
        total = (
            round(
                technical_points
                + float(visual["prompt_alignment"])
                + float(visual["temporal_coherence"])
                + float(visual["aesthetics"])
                + float(visual["hook_strength"]),
                2,
            )
            if judge_available else None
        )
        return {
            "available": judge_available,
            "total": max(0.0, min(100.0, total)) if total is not None else None,
            "technical_score": technical_points,
            "prompt_alignment": visual["prompt_alignment"],
            "temporal_coherence": visual["temporal_coherence"],
            "aesthetics": visual["aesthetics"],
            "hook_strength": visual["hook_strength"],
            "issues": list(dict.fromkeys(technical.get("issues", []) + visual.get("issues", []))),
            "summary": visual.get("summary", "Automated technical assessment completed."),
            "probe": technical,
            "judge": visual.get("judge", "technical-fallback-v1"),
            "subject_check": {
                "present": visual.get("primary_subject_present"),
                "identity_consistent": visual.get("identity_consistent"),
                "observed": visual.get("observed_primary_subject", ""),
                "major_misses": visual.get("major_prompt_misses", []),
            },
        }

    def _technical_probe(self, path: Path) -> dict[str, Any]:
        if not path.exists() or path.stat().st_size == 0:
            return {"valid": False, "issues": ["missing or empty output"]}
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration,size,bit_rate:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
                "-of", "json", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if probe.returncode != 0:
            return {"valid": False, "issues": [f"ffprobe failed: {probe.stderr[:160]}"]}
        data = json.loads(probe.stdout)
        streams = data.get("streams", [])
        video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
        audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
        duration = float(data.get("format", {}).get("duration") or 0)
        issues: list[str] = []
        if duration < 0.8:
            issues.append("video is shorter than one second")
        if not video:
            issues.append("no video stream")
        if int(video.get("width") or 0) < 512 or int(video.get("height") or 0) < 512:
            issues.append("low output resolution")
        if not audio:
            issues.append("no audio stream")

        detect = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                "-vf", "blackdetect=d=0.4:pix_th=0.08,freezedetect=n=-55dB:d=0.8",
                "-an", "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        black_seconds = sum(
            float(value) for value in re.findall(r"black_duration:([0-9.]+)", detect.stderr)
        )
        freeze_starts = len(re.findall(r"freeze_start:", detect.stderr))
        if duration and black_seconds / duration > 0.15:
            issues.append("more than 15% black frames")
        if freeze_starts:
            issues.append(f"detected {freeze_starts} frozen segment(s)")
        return {
            "valid": bool(video) and duration >= 0.8,
            "duration_seconds": round(duration, 3),
            "size_bytes": int(data.get("format", {}).get("size") or path.stat().st_size),
            "bit_rate": int(data.get("format", {}).get("bit_rate") or 0),
            "width": int(video.get("width") or 0),
            "height": int(video.get("height") or 0),
            "fps": video.get("r_frame_rate"),
            "video_codec": video.get("codec_name"),
            "audio_codec": audio.get("codec_name"),
            "audio_channels": audio.get("channels"),
            "black_seconds": round(black_seconds, 3),
            "frozen_segments": freeze_starts,
            "issues": issues,
        }

    def _technical_points(self, technical: dict[str, Any]) -> float:
        if not technical.get("valid"):
            return 0.0
        score = 30.0
        penalties = {
            "no audio stream": 5,
            "low output resolution": 3,
            "more than 15% black frames": 15,
        }
        for issue in technical.get("issues", []):
            score -= penalties.get(issue, 4 if "frozen" in issue else 2)
        return round(max(0.0, score), 2)

    async def _visual_judge(
        self, video_path: Path, prompt: str, brief: dict[str, Any]
    ) -> dict[str, Any]:
        contact_sheet = video_path.with_suffix(".contact.jpg")
        try:
            result = await asyncio.to_thread(self._make_contact_sheet, video_path, contact_sheet)
            if not result:
                raise RuntimeError("contact sheet failed")
            encoded = base64.b64encode(contact_sheet.read_bytes()).decode()
            rubric = {
                "prompt": prompt,
                "required_primary_subject": brief.get("subject", ""),
                "audience": brief.get("audience", "general audience"),
                "goal": brief.get("goal", "retention"),
                "hook_style": brief.get("hook_style", "curiosity"),
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.settings.ollama_url}/api/chat",
                    json={
                        "model": self.settings.vision_model,
                        "stream": False,
                        "format": "json",
                        "think": False,
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "Judge these ordered video frames strictly against the supplied prompt. "
                                    "First identify the visible primary subject and verify every required subject "
                                    "attribute. A beautiful video is still a prompt failure when the named subject "
                                    "is absent or becomes a different entity. Return JSON with: "
                                    "primary_subject_present (boolean), identity_consistent (boolean), "
                                    "observed_primary_subject (short string), major_prompt_misses (array of strings), "
                                    "prompt_alignment 0-25, temporal_coherence 0-20, aesthetics 0-15, "
                                    "hook_strength 0-10, issues (array of strings), and a concise summary. "
                                    "Cap prompt_alignment at 10 when the primary subject is absent or the wrong "
                                    "entity; cap at 15 for a major action or attribute miss. Rubric: "
                                    + json.dumps(rubric)
                                ),
                                "images": [encoded],
                            }
                        ],
                        "options": {
                            "temperature": 0.1, "num_predict": 320, "num_ctx": 4096,
                        },
                    },
                    timeout=httpx.Timeout(120, connect=5),
                )
            response.raise_for_status()
            # Ollama's Qwen3-VL currently places structured output in `thinking`
            # even when thinking is disabled, and can leave `content` empty.
            raw = parse_ollama_structured_message(response.json())
            return {**normalize_visual_judgment(raw), "judge": self.settings.vision_model}
        except (
            OSError, RuntimeError, httpx.HTTPError, json.JSONDecodeError, ValueError, TypeError
        ) as exc:
            return {
                "available": False,
                "prompt_alignment": 16.0,
                "temporal_coherence": 12.0,
                "aesthetics": 9.0,
                "hook_strength": 6.0,
                "issues": [
                    "vision judge unavailable; neutral creative scores applied",
                    f"judge diagnostic: {type(exc).__name__}: {str(exc)[:180]}",
                ],
                "summary": (
                    "Technical checks passed, but no creative score was assigned because the "
                    "configured vision judge could not be reached."
                ),
                "judge": "technical-fallback-v1",
            }
        finally:
            contact_sheet.unlink(missing_ok=True)

    def _make_contact_sheet(self, video_path: Path, destination: Path) -> bool:
        duration = max(0.1, float(self._technical_probe(video_path).get("duration_seconds") or 1))
        sample_fps = 4.0 / duration
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
                "-vf", f"fps={sample_fps:.6f},scale=384:-2,tile=4x1", "-frames:v", "1",
                str(destination),
            ],
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0 and destination.exists()

    @staticmethod
    def _clamp(value: Any, maximum: float) -> float:
        try:
            return round(max(0.0, min(maximum, float(value))), 2)
        except (TypeError, ValueError):
            return maximum * 0.6
