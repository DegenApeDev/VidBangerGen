from __future__ import annotations

import json
import math
import re
from typing import Any

import httpx

from .config import Settings
from .audio_prompt import spoken_word_limit
from .scoring import parse_ollama_structured_message


DIRECTOR_SYSTEM_PROMPT = """You are the creative director for a short-form video studio.
Return strict JSON only. Develop visually executable concepts with an immediate first-second hook,
clear escalation, a payoff, precise camera/action directions, synchronized sound ideas, and no vague
adjectives standing in for visible action. Each shot must be independently renderable by a video model.
Every shot prompt must explicitly restate the brief's primary subject/continuity anchor and the relevant
central story event; never replace the user's required scene with an isolated secondary prop.
Use enough shots to cover the requested duration and keep every shot between 1 and 20 seconds.
Keep visual direction separate from speech. Never turn captions, a topic, or descriptive prose into
dialogue. Only put words in `dialogue` when they are deliberately meant to be spoken; copy them exactly,
identify one speaker, specify the language, and keep the line below 2.5 words per second of shot time.
The JSON shape is: {"concepts":[{"title":"...","hook":"...","treatment":"...",
"retention_reason":"...","shots":[{"title":"...","purpose":"hook|build|payoff|cta",
"duration_seconds":5,"prompt":"...","negative_prompt":"...","camera":"...",
"audio":"ambient/foley only","audio_mode":"ambient|native-dialogue","speaker":"...",
"dialogue":"exact spoken words or empty","language":"English","accent":"...",
"caption":"...","transition":"..."}]}]}.
"""

SCRIPT_DIRECTOR_SYSTEM_PROMPT = """You are the script breakdown director for a video studio.
Return strict JSON only. Preserve the supplied story, named characters, dialogue intent, locations,
chronology, and ending. Convert it into an editable production storyboard rather than inventing a
different treatment. Extract reusable Elements for every recurring character, important object, and
location. Each video shot must be independently renderable, restate the visible identities it needs,
contain one continuous action, specify camera movement, and last between 1 and 20 seconds. Keep the
combined shot duration equal to the requested runtime. A prose narration or advertising script is
voiceover copy, not native character dialogue: create visual shots with ambience/foley and preserve its
text in `voiceover_text` for deterministic export. Only use `native-dialogue` for words explicitly spoken
by a character in the source. Copy those words exactly, use one speaker per shot, specify language/accent,
and keep them below 2.5 words per second. Never paste a whole script excerpt into the visual prompt or
the audio field. Use this JSON shape:
{"concepts":[{"title":"...","hook":"...","treatment":"...","retention_reason":"...",
"elements":[{"type":"character|location|object","name":"...","description":"..."}],
"shots":[{"scene_number":1,"scene_heading":"...","script_excerpt":"...","title":"...",
"purpose":"hook|build|escalate|payoff|cta","duration_seconds":5,"prompt":"...",
"negative_prompt":"...","camera":"...","audio":"ambient/foley only",
"audio_mode":"ambient|native-dialogue","speaker":"...","dialogue":"exact words or empty",
"language":"English","accent":"...","voiceover_text":"exact narration or empty","caption":"...",
"transition":"..."}]}]}.
"""

SCENE_HEADING = re.compile(
    r"^(?:SCENE\s+\d+\s*[:.-]?\s*)?(?:INT\.?|EXT\.?|INT\.?/EXT\.?|I/E\.?)\s+.+$",
    re.IGNORECASE,
)


class CreativeDirector:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def plan(
        self, brief: dict[str, Any], concept_count: int = 3,
        learning_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        script_mode = brief.get("source_kind") == "script"
        if script_mode:
            concept_count = 1
            request = (
                "Break this script into one faithful, editable production storyboard. "
                f"The complete storyboard must total {brief['duration_seconds']} seconds. "
                "Condense beats when necessary, but do not replace the plot, cast, or ending. "
                "Project settings and script:\n"
                + json.dumps(brief, ensure_ascii=False)
            )
        else:
            request = (
                f"Create {concept_count} distinct concepts for this brief. The total duration of every "
                f"concept must be approximately {brief['duration_seconds']} seconds. Brief:\n"
                + json.dumps(brief, ensure_ascii=False)
            )
        if learning_context and learning_context.get("example_count"):
            request += (
                "\n\nHistorical studio learning signals follow. Infer reusable principles; "
                "do not copy old subjects or prompts. Preserve what users marked excellent, "
                "avoid rejected failure patterns, and use score issues as explicit constraints:\n"
                + json.dumps(learning_context, ensure_ascii=False)
            )
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.settings.ollama_url}/api/chat",
                    json={
                        "model": self.settings.director_model,
                        "stream": False,
                        "format": "json",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    SCRIPT_DIRECTOR_SYSTEM_PROMPT
                                    if script_mode else DIRECTOR_SYSTEM_PROMPT
                                ),
                            },
                            {"role": "user", "content": request},
                        ],
                        "options": {
                            "temperature": 0.35 if script_mode else 0.85,
                            "top_p": 0.9,
                            "num_ctx": 8192 if script_mode else 4096,
                        },
                    },
                    timeout=90,
                )
            response.raise_for_status()
            generated = parse_ollama_structured_message(response.json())
            concepts = self._normalise(generated.get("concepts", []), brief, concept_count)
            if len(concepts) >= concept_count:
                return {
                    "provider": "ollama", "model": self.settings.director_model,
                    "concepts": concepts,
                    "learning_examples": int((learning_context or {}).get("example_count", 0)),
                }
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        return {
            "provider": "built-in-director",
            "model": "deterministic-fallback-v1",
            "concepts": (
                self._script_fallback(brief) if script_mode
                else self._fallback(brief, concept_count)
            ),
            "learning_examples": int((learning_context or {}).get("example_count", 0)),
        }

    def _normalise(
        self, concepts: list[Any], brief: dict[str, Any], concept_count: int
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for raw in concepts[:concept_count]:
            if not isinstance(raw, dict) or not isinstance(raw.get("shots"), list):
                continue
            shots: list[dict[str, Any]] = []
            for index, shot in enumerate(raw["shots"][:24]):
                if not isinstance(shot, dict) or not str(shot.get("prompt", "")).strip():
                    continue
                duration = max(1.0, min(10.0, float(shot.get("duration_seconds", 5.0))))
                audio_fields = self._normalise_audio(shot, brief, duration)
                shots.append(
                    {
                        **({
                            "scene_number": max(1, int(shot.get("scene_number", index + 1))),
                            "scene_heading": str(
                                shot.get("scene_heading") or f"Scene {index + 1}"
                            )[:180],
                            "script_excerpt": str(shot.get("script_excerpt", ""))[:2_000],
                        } if brief.get("source_kind") == "script" or shot.get("scene_heading") else {}),
                        "title": str(shot.get("title") or f"Shot {index + 1}")[:120],
                        "purpose": str(shot.get("purpose") or ("hook" if index == 0 else "build")),
                        "duration_seconds": duration,
                        "prompt": self._enrich_prompt(str(shot["prompt"]), shot, brief),
                        "negative_prompt": self._negative_prompt(
                            str(shot.get("negative_prompt", "")), brief
                        ),
                        "camera": str(shot.get("camera", ""))[:300],
                        **audio_fields,
                        "caption": str(shot.get("caption", ""))[:300],
                        "transition": str(shot.get("transition", "hard cut"))[:100],
                    }
                )
            if not shots:
                continue
            self._fit_duration(shots, float(brief["duration_seconds"]))
            result.append(
                {
                    "title": str(raw.get("title", "Untitled concept"))[:120],
                    "hook": str(raw.get("hook", shots[0]["prompt"]))[:500],
                    "treatment": str(raw.get("treatment", ""))[:1_500],
                    "retention_reason": str(raw.get("retention_reason", ""))[:700],
                    "elements": self._normalise_elements(raw.get("elements", [])),
                    "shots": shots,
                }
            )
        return result

    def _normalise_audio(
        self, shot: dict[str, Any], brief: dict[str, Any], duration: float
    ) -> dict[str, str]:
        dialogue = re.sub(r"\s+", " ", str(shot.get("dialogue", ""))).strip()
        source = str(brief.get("script", "")) if brief.get("source_kind") == "script" else ""
        brief_dialogue = str(brief.get("dialogue", ""))
        # Assisted planning may format text, but it may never invent or
        # paraphrase words that the operator did not supply.
        supplied = source or brief_dialogue
        if dialogue and supplied and dialogue.casefold() not in supplied.casefold():
            dialogue = ""
        if len(re.findall(r"\b[\w’'-]+\b", dialogue)) > spoken_word_limit(duration):
            dialogue = ""

        audio = re.sub(r"\s+", " ", str(shot.get("audio", ""))).strip()
        if (
            not audio
            or re.search(
                r"[\"“”]|\b(?:dialogue|voiceover|narrat|speaks?|says?|words?)\b",
                audio,
                flags=re.IGNORECASE,
            )
        ):
            audio = "natural location ambience and synchronized scene Foley"
        excerpt = str(shot.get("script_excerpt", "")).strip()
        voiceover = str(shot.get("voiceover_text", "")).strip()
        if not voiceover and brief.get("source_kind") == "script" and not dialogue:
            voiceover = excerpt
        return {
            "audio": audio[:500],
            "audio_mode": "native-dialogue" if dialogue else "ambient",
            "dialogue": dialogue[:1_000],
            "speaker": str(shot.get("speaker", "")).strip()[:120],
            "language": str(shot.get("language", "English") or "English")[:80],
            "accent": str(shot.get("accent", "")).strip()[:120],
            "voiceover_text": voiceover[:2_000],
        }

    def _normalise_elements(self, values: Any) -> list[dict[str, str]]:
        if not isinstance(values, list):
            return []
        result: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for value in values[:60]:
            if not isinstance(value, dict):
                continue
            kind = str(value.get("type") or value.get("kind") or "object").lower()
            if kind not in {"character", "location", "object"}:
                kind = "object"
            name = str(value.get("name", "")).strip()[:120]
            if not name:
                continue
            key = kind, name.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append({
                "type": kind,
                "name": name,
                "description": str(value.get("description", "")).strip()[:700],
            })
        return result

    def _enrich_prompt(
        self, prompt: str, shot: dict[str, Any], brief: dict[str, Any]
    ) -> str:
        pieces = [prompt.strip()]
        subject = str(brief.get("subject", "")).strip()
        if subject and subject.casefold() not in prompt.casefold():
            pieces.append(
                f"Primary continuity subject that must remain clearly visible and unchanged: {subject}"
            )
        topic = str(brief.get("topic", "")).strip()
        if (
            brief.get("source_kind") != "script"
            and topic and topic.casefold() not in prompt.casefold()
        ):
            pieces.append(f"Required central story event: {topic}")
        camera = str(shot.get("camera", "")).strip()
        if camera and camera.lower() not in prompt.lower():
            pieces.append(f"Camera: {camera}")
        pieces.extend(
            [
                f"Visual style: {brief.get('style', 'cinematic')}",
                "Maintain anatomically coherent subjects and continuous physically plausible motion",
                "The shot contains one clear action with a visible change before it ends",
            ]
        )
        brand_notes = str(brief.get("brand_notes", "")).strip()
        if brand_notes:
            pieces.append(f"Non-negotiable production constraints: {brand_notes}")
        return (". ".join(piece.rstrip(". ") for piece in pieces if piece) + ".")[:5_000]

    def _negative_prompt(self, existing: str, brief: dict[str, Any]) -> str:
        defaults = [
            "flicker", "temporal jitter", "warped anatomy", "extra limbs", "melted faces",
            "unreadable text", "watermark", "logo", "camera teleportation", "frozen motion",
        ]
        values = [existing] if existing.strip() else []
        values.extend(defaults)
        values.extend(str(value) for value in brief.get("forbidden_elements", []))
        return ", ".join(
            dict.fromkeys(value.strip() for value in values if value.strip())
        )[:2_000]

    def _fit_duration(self, shots: list[dict[str, Any]], target: float) -> None:
        target = max(1.0, float(target))
        # At a one-second minimum, an overlong model response cannot fit a
        # short brief. Preserve the opening sequence and its final payoff.
        maximum_shots = max(1, int(math.floor(target)))
        if len(shots) > maximum_shots:
            shots[:] = (
                shots[: maximum_shots - 1] + [shots[-1]]
                if maximum_shots > 1 else [shots[-1]]
            )
        total = sum(float(shot["duration_seconds"]) for shot in shots)
        if total <= 0:
            return
        ratio = target / total
        expanded: list[dict[str, Any]] = []
        for shot in shots:
            scaled = float(shot["duration_seconds"]) * ratio
            parts = max(1, math.ceil(scaled / 20.0))
            for part_index in range(parts):
                value = dict(shot)
                value["duration_seconds"] = round(max(1.0, scaled / parts), 2)
                if parts > 1:
                    value["title"] = f"{shot.get('title', 'Shot')} · {part_index + 1}/{parts}"[:120]
                    value["prompt"] = (
                        f"{shot['prompt']} Continuous phase {part_index + 1} of {parts}: "
                        "advance the same action without resetting the subject, scene, or camera axis."
                    )[:5_000]
                expanded.append(value)
        shots[:] = expanded

        # Correct rounding and one-second clamps while preserving the exact
        # requested project duration and the 20-second generation ceiling.
        delta = round(target - sum(float(shot["duration_seconds"]) for shot in shots), 2)
        while abs(delta) >= 0.01:
            changed = False
            for shot in reversed(shots):
                duration = float(shot["duration_seconds"])
                room = 20.0 - duration if delta > 0 else duration - 1.0
                if room <= 0:
                    continue
                amount = min(abs(delta), room)
                shot["duration_seconds"] = round(
                    duration + amount if delta > 0 else duration - amount, 2
                )
                delta = round(
                    target - sum(float(value["duration_seconds"]) for value in shots), 2
                )
                changed = True
                if abs(delta) < 0.01:
                    break
            if not changed:
                break

    def _fallback(self, brief: dict[str, Any], concept_count: int) -> list[dict[str, Any]]:
        treatments = [
            (
                "Pattern Interrupt",
                "Open on the impossible result, rewind through escalating visual clues, then reveal the cause.",
                "The audience sees the payoff before receiving the explanation, creating an immediate information gap.",
            ),
            (
                "Transformation Escalation",
                "Begin with an ordinary state, trigger a visible transformation, and intensify it on every cut.",
                "Every shot changes the visual state, continually renewing attention.",
            ),
            (
                "One-Take Discovery",
                "Use a continuous-feeling camera move that uncovers progressively more surprising details.",
                "Forward camera momentum promises another discovery just outside the frame.",
            ),
            (
                "Expectation Flip",
                "Establish a familiar expectation, interrupt it with an incongruous action, and land on a visual punchline.",
                "Recognition followed by contradiction creates surprise and replay value.",
            ),
            (
                "Sensory Spectacle",
                "Build around macro texture, rhythmic motion, dramatic sound and an oversized final reveal.",
                "Dense sensory change creates a strong stop-scroll opening and satisfying payoff.",
            ),
        ]
        shot_count = max(1, min(24, math.ceil(float(brief["duration_seconds"]) / 5.0)))
        per_shot = float(brief["duration_seconds"]) / shot_count
        results: list[dict[str, Any]] = []
        for concept_index in range(concept_count):
            name, treatment, reason = treatments[concept_index % len(treatments)]
            shots = []
            for index in range(shot_count):
                is_first, is_last = index == 0, index == shot_count - 1
                purpose = "hook" if is_first else ("payoff" if is_last else "build")
                action = (
                    "Begin in the middle of a startling, high-motion visual event that instantly raises a question"
                    if is_first
                    else (
                        "Deliver the clearest, largest visual payoff and resolve the opening question"
                        if is_last
                        else "Escalate the action with a new visible consequence and stronger motion"
                    )
                )
                camera = (
                    "aggressive macro push-in with a stable subject lock"
                    if is_first
                    else ("smooth reveal into a composed hero frame" if is_last else "fast lateral tracking move")
                )
                caption = (
                    brief.get("call_to_action", "") if is_last else (brief["title"] if is_first else "")
                )
                shot = {
                    "title": f"{purpose.title()} {index + 1}",
                    "purpose": purpose,
                    "duration_seconds": round(per_shot, 2),
                    "prompt": (
                        f"{action}. Subject: {brief.get('subject') or brief['topic']}. "
                        f"Scene communicates: {brief['topic']}. Camera: {camera}. "
                        f"Lighting and palette: {brief.get('style')}. "
                        "Precise continuous movement, strong foreground-background separation, realistic detail."
                    ),
                    "negative_prompt": self._negative_prompt("", brief),
                    "camera": camera,
                    "audio": brief.get("music_style") or "rhythmic sound design matched to visible impacts",
                    "audio_mode": "ambient",
                    "dialogue": "",
                    "speaker": "",
                    "language": "English",
                    "accent": "",
                    "voiceover_text": "",
                    "caption": caption,
                    "transition": "sound-matched hard cut" if not is_last else "hold hero frame",
                }
                shots.append(shot)
            self._fit_duration(shots, float(brief["duration_seconds"]))
            results.append(
                {
                    "title": name,
                    "hook": shots[0]["prompt"],
                    "treatment": treatment,
                    "retention_reason": reason,
                    "shots": shots,
                }
            )
        return results

    def _script_fallback(self, brief: dict[str, Any]) -> list[dict[str, Any]]:
        """Build an editable, faithful storyboard even when Bonsai is offline.

        This deliberately performs structure extraction rather than creative
        rewriting. The operator can improve any generated prompt before GPU work.
        """
        script = str(brief.get("script", "")).strip()
        target_shots = max(1, min(24, math.ceil(float(brief["duration_seconds"]) / 5.0)))
        beats = self._script_beats(script, target_shots)
        elements = self._script_elements(script)
        shots: list[dict[str, Any]] = []
        for index, beat in enumerate(beats):
            is_first, is_last = index == 0, index == len(beats) - 1
            purpose = "hook" if is_first else ("payoff" if is_last else "build")
            camera = (
                "Immediate subject-led push-in that makes the opening action readable"
                if is_first else (
                    "Controlled reveal that resolves on the story's final visual beat"
                    if is_last else "Motivated tracking shot with stable screen direction"
                )
            )
            excerpt = beat["text"][:2_000]
            prompt = (
                f"Create a visual-only cinematic interpretation of this narration beat in "
                f"{beat['heading']}: {excerpt}. Do not display or speak the narration text. "
                f"Keep every named character, location, and important object visually consistent. "
                f"Camera: {camera}. Visual style: {brief.get('style', 'cinematic')}. "
                "One continuous physically plausible action with a clear visual change and no reset. "
                "No person speaks or mouths words."
            )
            shots.append({
                "scene_number": beat["scene_number"],
                "scene_heading": beat["heading"],
                "script_excerpt": excerpt,
                "title": self._beat_title(excerpt, index),
                "purpose": purpose,
                "duration_seconds": 5.0,
                "prompt": prompt[:5_000],
                "negative_prompt": self._negative_prompt("", brief),
                "camera": camera,
                "audio": "natural location ambience and synchronized scene Foley",
                "audio_mode": "ambient",
                "dialogue": "",
                "speaker": "",
                "language": "English",
                "accent": "",
                "voiceover_text": excerpt,
                "caption": "",
                "transition": "hard cut" if not is_last else "hold on final story image",
            })
        self._fit_duration(shots, float(brief["duration_seconds"]))
        return [{
            "title": str(brief.get("title") or "Imported script")[:120],
            "hook": shots[0]["script_excerpt"] if shots else script[:500],
            "treatment": (
                "A faithful script breakdown with editable shots. Bonsai was unavailable, so scene "
                "structure and recurring Elements were extracted locally without rewriting the plot."
            ),
            "retention_reason": "Preserves the supplied story order while fitting the selected runtime.",
            "elements": elements,
            "shots": shots,
        }]

    def _script_beats(self, script: str, target: int) -> list[dict[str, Any]]:
        lines = [line.rstrip() for line in script.replace("\r\n", "\n").split("\n")]
        scenes: list[dict[str, Any]] = []
        heading = "Opening"
        body: list[str] = []
        scene_number = 1

        def flush() -> None:
            nonlocal body, scene_number
            text = "\n".join(body).strip()
            if text:
                scenes.append({
                    "scene_number": scene_number,
                    "heading": heading,
                    "text": text,
                })
                scene_number += 1
            body = []

        for line in lines:
            stripped = line.strip()
            if stripped and SCENE_HEADING.match(stripped):
                flush()
                heading = stripped
            else:
                body.append(line)
        flush()
        if not scenes:
            scenes = [{"scene_number": 1, "heading": "Opening", "text": script}]

        paragraphs: list[dict[str, Any]] = []
        for scene in scenes:
            chunks = [value.strip() for value in re.split(r"\n\s*\n+", scene["text"]) if value.strip()]
            for chunk in chunks or [scene["text"]]:
                paragraphs.append({**scene, "text": chunk})
        beats = paragraphs or scenes
        # Prose scripts often arrive as one paragraph. Feeding that entire
        # paragraph to every expanded phase duplicates narration and makes LTX
        # attempt impossible speech. Split long prose into ordered, contiguous
        # sentence/word beats before duration fitting.
        if len(beats) < target:
            expanded: list[dict[str, Any]] = []
            for beat in beats:
                text = str(beat["text"]).strip()
                sentences = [
                    value.strip()
                    for value in re.split(r"(?<=[.!?])\s+|\n+", text)
                    if value.strip()
                ]
                expanded.extend({**beat, "text": value} for value in (sentences or [text]))
            beats = expanded or beats
        while len(beats) < target:
            longest_index = max(
                range(len(beats)), key=lambda index: len(str(beats[index]["text"]).split())
            )
            longest = beats[longest_index]
            text = str(longest["text"])
            word_matches = list(re.finditer(r"\S+", text))
            if len(word_matches) < 2:
                break
            split_at = word_matches[len(word_matches) // 2].start()
            left, right = text[:split_at].strip(), text[split_at:].strip()
            if not left or not right:
                break
            beats[longest_index:longest_index + 1] = [
                {**longest, "text": left}, {**longest, "text": right},
            ]
        if len(beats) <= target:
            return beats

        grouped: list[dict[str, Any]] = []
        for index in range(target):
            start = round(index * len(beats) / target)
            end = round((index + 1) * len(beats) / target)
            group = beats[start:max(start + 1, end)]
            grouped.append({
                "scene_number": group[0]["scene_number"],
                "heading": group[0]["heading"],
                "text": "\n\n".join(value["text"] for value in group),
            })
        return grouped

    def _script_elements(self, script: str) -> list[dict[str, str]]:
        characters: list[str] = []
        locations: list[str] = []
        for raw in script.splitlines():
            line = raw.strip()
            if not line:
                continue
            if SCENE_HEADING.match(line):
                location = re.split(r"\s+-\s+", line, maxsplit=1)[0]
                location = re.sub(
                    r"^(?:SCENE\s+\d+\s*[:.-]?\s*)?(?:INT\.?/EXT\.?|INT\.?|EXT\.?|I/E\.?)\s+",
                    "", location, flags=re.IGNORECASE,
                ).strip()
                if location and location.casefold() not in {v.casefold() for v in locations}:
                    locations.append(location[:120])
                continue
            cue = re.sub(r"\s*\([^)]*\)\s*$", "", line).strip()
            if (
                cue == cue.upper() and 1 < len(cue) <= 40
                and re.search(r"[A-Z]", cue)
                and not cue.endswith(("TO:", "IN:", "OUT:"))
                and cue not in {"CUT", "FADE", "DISSOLVE", "THE END"}
            ):
                if cue.casefold() not in {v.casefold() for v in characters}:
                    characters.append(cue.title())
        elements = [
            {
                "type": "character", "name": name,
                "description": f"Recurring character from the imported script: {name}. Keep appearance, wardrobe, and voice consistent.",
            }
            for name in characters[:20]
        ]
        elements.extend(
            {
                "type": "location", "name": name,
                "description": f"Recurring script location: {name}. Preserve layout, lighting logic, and production design.",
            }
            for name in locations[:20]
        )
        return elements

    @staticmethod
    def _beat_title(text: str, index: int) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        first = re.split(r"(?<=[.!?])\s+", compact, maxsplit=1)[0]
        return (first[:90] or f"Script beat {index + 1}").rstrip(". ")
