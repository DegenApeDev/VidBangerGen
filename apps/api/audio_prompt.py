from __future__ import annotations

import math
import re
from typing import Any


SPEECH_NEGATIVES = (
    "speech", "human voice", "spoken words", "talking", "dialogue", "narration",
    "voiceover", "singing", "vocals", "lip sync", "mouth movements", "phonemes",
    "mumbled words", "gibberish", "unintelligible language",
    "reading aloud", "narrating action", "stage directions", "parentheticals",
)
EXTRA_SPEECH_NEGATIVES = (
    "narration", "voiceover", "off-screen voice", "additional dialogue",
    "improvised dialogue", "extra spoken words", "unintelligible speech", "gibberish",
    "wrong words", "paraphrased dialogue", "repeated words", "extra syllables",
    "reading stage directions", "speaking action lines",
)
STAGE_DIRECTION_NEGATIVES = (
    "stage directions", "parentheticals", "beat", "cut to", "reading aloud",
    "narrating the scene", "speaking the prompt",
)

_QUOTED_DIALOGUE = re.compile(r'[“\"]([^”\"]+)[”\"]')
_LABELED_AUDIO = re.compile(
    r"\s*(?:Synchronized audio|Audio|Dialogue|Voiceover|Narration)\s*:\s*[^.\n]*(?:\.|$)",
    re.IGNORECASE,
)
_SPEECH_WITH_QUOTE = re.compile(
    r"(?:\b(?:and|then)\s+)?\b(?:says?|speaks?|talks?|asks?|replies?|shouts?|"
    r"whispers?|mutters?|narrates?|announces?)\b[^\n.!?“\"]*[“\"][^”\"]+[”\"]",
    re.IGNORECASE,
)
_SPEECH_CLAUSE = re.compile(
    r"\b(?:says?|speaks?|talks?|asks?|replies?|shouts?|whispers?|mutters?|"
    r"narrates?|announces?)\b[^.!?]*(?:[.!?]|$)",
    re.IGNORECASE,
)
# Assisted visual prompts often describe a subject as "talking about X" without
# providing a real line of dialogue. In a joint audio/video model that phrase is
# still a strong phonetic cue, even when the end of the prompt asks for no
# speech. Replace the activity while keeping the visual subject and sentence.
_SPEECH_ACTIVITY = re.compile(
    r"\b(?:talking|speaking|narrating|singing|chanting|vocalizing|vocalising|"
    r"reading aloud|delivering (?:a |the )?(?:speech|monologue|narration))\b"
    r"[^.!?]*(?=[.!?]|$)",
    re.IGNORECASE,
)
# Fountain-style speaker cues only. Must not match prompt field labels such as
# "Subject:" / "Scene communicates:" that appear in assisted storyboard text.
_CHARACTER_CUE = re.compile(
    r"(?m)^[ \t]*(?P<name>"
    r"[A-Z]{2,24}(?:\s+[A-Z]{2,24}){0,2}"
    r"|[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,2}"
    r")(?P<paren>[ \t]*\([^)\n]{0,60}\))?[ \t]*:[ \t]*"
)
_INLINE_CHARACTER_CUE = re.compile(
    r"(?<![A-Za-z])(?P<name>"
    r"[A-Z]{2,24}(?:\s+[A-Z]{2,24}){0,2}"
    r"|[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,2}"
    r")(?P<paren>[ \t]*\([^)\n]{0,60}\))?[ \t]*:[ \t]*"
)
# Labels that look like Title Case "Name:" but are storyboard / prompt fields.
_CUE_DENYLIST = frozenset({
    "subject", "scene", "scene communicates", "audio", "camera", "visual",
    "style", "note", "notes", "location", "setting", "lighting", "mood",
    "action", "description", "prompt", "negative", "duration", "title",
    "topic", "goal", "audience", "hook", "begin", "end", "start", "cut",
    "fade", "time", "day", "night", "int", "ext", "movement", "reference",
    "requirement", "visual reference", "visual reference requirement",
})
_PARENTHETICAL = re.compile(
    r"\((?:beat|pause|whispering|mouth full|aside|o\.?s\.?|v\.?o\.?|"
    r"[^)\n]{0,40}(?:whisper|shout|mutter|aside|quietly|loudly|deadpan)[^)\n]{0,20})\)",
    re.IGNORECASE,
)
_SLUGLINE = re.compile(
    r"(?im)\b(?:INT\.|EXT\.|EST\.|I/E\.|INT/EXT\.)[^\n.!?]{0,120}"
)
# Only the transition keyword itself — never the following action prose.
# Greedy "cut to …" spans were swallowing real visual description and leaving
# fragments like "e behind the glass" that LTX then tried to vocalise.
_TRANSITION = re.compile(
    r"(?i)\b(?:"
    r"(?:smash |match |jump |hard )?cut to(?:\s+(?:side|wide|close|close-?up|reaction|insert))?(?:\s+view)?"
    r"|quick zoom(?:\s+(?:in|out|back))?"
    r"|zoom (?:in|out|back)"
    r"|dolly (?:in|out)|whip pan|crash zoom"
    r"|fade (?:in|out|to black)|dissolve to|wipe to"
    r"|angle on|pov shot|close on"
    r")\b"
)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def _cue_name_key(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def _is_character_cue_name(name: str) -> bool:
    key = _cue_name_key(name)
    if not key or key in _CUE_DENYLIST:
        return False
    if any(token in key for token in ("communicates", "requirement", "reference", "describes")):
        return False
    # Reject long descriptive noun phrases posing as cues.
    if len(key.split()) > 3 or len(key) > 40:
        return False
    return True


def _strip_character_cues(value: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        return " " if _is_character_cue_name(match.group("name")) else match.group(0)

    cleaned = _CHARACTER_CUE.sub(replacer, value)
    return _INLINE_CHARACTER_CUE.sub(replacer, cleaned)


def spoken_word_limit(duration_seconds: float) -> int:
    """A conservative intelligibility budget for native generated dialogue."""
    return max(3, math.floor(float(duration_seconds) * 2.5))


def quoted_dialogue(prompt: str) -> str:
    return " ".join(match.strip() for match in _QUOTED_DIALOGUE.findall(prompt) if match.strip())


def quoted_dialogue_lines(prompt: str) -> list[str]:
    return [match.strip() for match in _QUOTED_DIALOGUE.findall(prompt) if match.strip()]


def explicit_dialogue_intent(prompt: str) -> list[str]:
    """Return quoted lines that are directly attached to a speech verb.

    Ambient mode must not silently discard an unmistakable request such as
    ``Speaks only "Far Out Dude"`` and then spend a GPU render on a conflicting
    no-speech contract. Plain quoted signs or screen text are not treated as
    dialogue unless a speech verb anchors them.
    """
    lines: list[str] = []
    for match in _SPEECH_WITH_QUOTE.finditer(prompt or ""):
        prefix = (prompt or "")[max(0, match.start() - 32):match.start()]
        if re.search(
            r"(?i)(?:no\s+one|nobody|never|does\s+not|doesn't|without)\s*$",
            prefix,
        ):
            continue
        lines.extend(quoted_dialogue_lines(match.group(0)))
    return list(dict.fromkeys(line for line in lines if line))


def _clean_dialogue(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    if len(cleaned) >= 2 and cleaned[0] in {'"', '“'} and cleaned[-1] in {'"', '”'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _normalize_prose(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _MULTI_NEWLINE.sub("\n\n", value)
    value = _MULTI_SPACE.sub(" ", value)
    # Collapse separator dashes only — never word-internal hyphens (high-motion).
    value = re.sub(r"\s+[—–]\s*", " ", value)
    value = re.sub(r"\s+-\s+", " ", value)
    value = re.sub(r"\s+([,.!?;:])", r"\1", value)
    value = re.sub(r"([,;:])\s*([,;:])+", r"\1", value)
    value = re.sub(r"\.{2,}", ".", value)
    value = re.sub(r"[ \t]*\n[ \t]*", "\n", value)
    value = re.sub(r"\s+", " ", value)
    # Drop leading leftovers from slug removal; keep sentence-final periods.
    value = value.lstrip(" \t\n.·")
    return value.rstrip(" \t\n")


def _strip_screenplay_residue(value: str) -> str:
    """Remove Fountain-style cues, parentheticals, and cut language.

    After dialogue quotes are stripped, residual text like
    ``Baker (whispering): … Baker: (beat) Cut to side view`` is still
    phonetically attractive to joint AV models. Strip that scaffolding so
    ambient/silent compiles keep only camera and action prose.
    """
    cleaned = _strip_character_cues(value)
    cleaned = _PARENTHETICAL.sub(" ", cleaned)
    cleaned = _SLUGLINE.sub(" ", cleaned)
    # Preserve an actual requested ending while turning screenplay syntax into
    # natural visual prose that the joint AV model is less likely to vocalize.
    cleaned = re.sub(
        r"(?i)\b(?:then\s+)?fade\s+out(?:\s+the\s+camera)?\s+to\s+black\b",
        " The image gradually darkens until fully black ",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\b(?:then\s+)?fade\s+to\s+black\b",
        " The image gradually darkens until fully black ",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\band\s+only\s+show\b",
        " then shows only ",
        cleaned,
    )
    cleaned = _TRANSITION.sub(" ", cleaned)
    # Lone stage-beat tokens left after parenthetical removal.
    cleaned = re.sub(r"(?i)(?:^|[\s,;:—-])beat(?:$|[\s,;:.!?—-])", " ", cleaned)
    return cleaned


def _without_speech_direction(prompt: str) -> str:
    # Remove the protected dialogue first. Otherwise a period inside a quoted
    # line can terminate the labelled-audio matcher and leak the remainder back
    # into the visual prose that LTX may try to vocalise.
    value = _SPEECH_WITH_QUOTE.sub(" ", prompt)
    value = _QUOTED_DIALOGUE.sub("", value)
    value = _LABELED_AUDIO.sub(" ", value)
    value = _SPEECH_ACTIVITY.sub("gesturing silently", value)
    value = _SPEECH_CLAUSE.sub(" ", value)
    value = _strip_screenplay_residue(value)
    # A clause like `Speaks "..." as he points` intentionally preserves the
    # simultaneous visual action. Repair the sentence fragment after removing
    # the speech request.
    value = re.sub(
        r"(?i)(^|[.!?]\s+)(?:then\s+)?as\s+(he|she|they|it)\b",
        lambda match: f"{match.group(1)}{match.group(2).capitalize()}",
        value,
    )
    value = re.sub(r"\b(?:and|then|while)\s*(?=[.!?]|$)", " ", value, flags=re.IGNORECASE)
    return _normalize_prose(value)


def visual_only_prompt(prompt: str) -> str:
    """Return scene/action prose with dialogue and audio directions removed."""
    return _without_speech_direction(prompt) or "A natural cinematic shot"


def detect_script_shape(prompt: str) -> dict[str, Any]:
    """Flag multi-beat screenplay paste that is a poor Quick Generate input."""
    text = prompt or ""
    quotes = quoted_dialogue_lines(text)
    quote_count = len(quotes)
    has_slugline = bool(_SLUGLINE.search(text))
    has_character_cue = any(
        _is_character_cue_name(match.group("name"))
        for match in list(_CHARACTER_CUE.finditer(text)) + list(_INLINE_CHARACTER_CUE.finditer(text))
    )
    has_parenthetical = bool(_PARENTHETICAL.search(text))
    has_transition = bool(_TRANSITION.search(text))
    line_count = len([line for line in text.splitlines() if line.strip()])
    word_count = len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))
    multi_speaker = has_character_cue and quote_count >= 2
    multi_beat = has_transition or (quote_count >= 2 and (has_character_cue or has_parenthetical))
    looks_like_script = bool(
        (has_slugline and (has_character_cue or quote_count >= 1))
        or multi_speaker
        or multi_beat
        or (quote_count >= 3)
        or (has_character_cue and has_parenthetical and quote_count >= 1)
        or (word_count >= 120 and quote_count >= 2)
    )
    reasons: list[str] = []
    if has_slugline:
        reasons.append("scene slugline")
    if has_character_cue:
        reasons.append("character dialogue cues")
    if quote_count >= 2:
        reasons.append(f"{quote_count} quoted lines")
    if has_parenthetical:
        reasons.append("stage parentheticals")
    if has_transition:
        reasons.append("cut/zoom transitions")
    if word_count >= 120 and quote_count >= 1:
        reasons.append("long multi-beat prose")

    hint = ""
    if looks_like_script:
        hint = (
            "Looks like a multi-shot script. Quick Generate works best with one visual beat; "
            "use Studio → Script to video to split beats, or paste a single camera beat here. "
            "Put comedy dialogue in Finish social cut voiceover for verbatim timing."
        )
    return {
        "looks_like_script": looks_like_script,
        "reasons": reasons,
        "quote_count": quote_count,
        "quoted_lines": quotes,
        "has_slugline": has_slugline,
        "has_character_cue": has_character_cue,
        "has_parenthetical": has_parenthetical,
        "has_transition": has_transition,
        "line_count": line_count,
        "word_count": word_count,
        "hint": hint,
    }


def _add_negative(negative_prompt: str, values: tuple[str, ...]) -> str:
    parts = [part.strip() for part in negative_prompt.split(",") if part.strip()]
    parts.extend(values)
    return ", ".join(dict.fromkeys(parts))[:2_000]


def compile_audio_prompt(
    prompt: str,
    negative_prompt: str,
    *,
    mode: str,
    duration_seconds: float,
    dialogue: str = "",
    speaker: str = "",
    language: str = "English",
    accent: str = "",
    ambience: str = "",
) -> tuple[str, str, dict[str, Any]]:
    """Compile an explicit LTX audio contract into the shared AV prompt.

    LTX generates audio and video jointly. Leaving speech intent ambiguous can
    turn prose, captions, or stage directions into phonetic material. This
    compiler makes the operator's choice explicit while keeping the untouched
    `prompt` mode available for exact/manual prompts.
    """
    selected = str(mode or "prompt")
    script_shape = detect_script_shape(prompt)
    if selected == "prompt":
        return prompt, negative_prompt, {
            "mode": "prompt",
            "prompt_preserved": True,
            "dialogue_words": None,
            "visual_prompt": prompt,
            "script_shape": script_shape,
        }

    # Never fall back to the original prompt here: it may consist entirely of
    # dialogue, and reusing it would duplicate those words outside quotation
    # marks. The joint AV text conditioning has no hard modality boundary.
    visual_prompt = visual_only_prompt(prompt)
    if selected == "ambient":
        requested_lines = explicit_dialogue_intent(prompt)
        if requested_lines:
            detected = requested_lines[0]
            raise ValueError(
                "Ambience + Foley is a no-speech mode, but this prompt explicitly requests "
                f"“{detected}”. Choose Native quoted speech to use that line, or remove the "
                "speech direction. For guaranteed verbatim wording, use uploaded voiceover."
            )
        sound = ambience.strip() or "natural environmental ambience and synchronized scene Foley"
        compiled = (
            f"{visual_prompt.rstrip('. ')}. Audio: {sound} only. "
            "The soundtrack contains zero human voices and zero spoken words. "
            "The visual description is not narration and none of its words are heard. "
            "Every visible mouth remains closed; nobody speaks, mouths words, narrates, "
            "sings, whispers, or vocalizes. Do not read stage directions or action lines aloud."
        )
        return compiled[:5_000], _add_negative(
            negative_prompt, SPEECH_NEGATIVES + STAGE_DIRECTION_NEGATIVES
        ), {
            "mode": "ambient",
            "speech_allowed": False,
            "dialogue_words": 0,
            "visual_prompt": visual_prompt,
            "script_shape": script_shape,
        }

    if selected == "silent":
        compiled = (
            f"{visual_prompt.rstrip('. ')}. Audio: silence only. "
            "No speech, voices, vocalization, music, or environmental sound. "
            "Do not read stage directions or action lines aloud."
        )
        return compiled[:5_000], _add_negative(
            negative_prompt,
            SPEECH_NEGATIVES + STAGE_DIRECTION_NEGATIVES + ("music", "ambient sound", "foley"),
        ), {
            "mode": "silent",
            "speech_allowed": False,
            "dialogue_words": 0,
            "visual_prompt": visual_prompt,
            "script_shape": script_shape,
        }

    if selected != "native-dialogue":
        raise ValueError(f"Unsupported audio mode: {selected}")

    explicit_dialogue = _clean_dialogue(dialogue)
    # Prefer the operator's dialogue field. When it is empty, refuse to auto-join
    # every quoted line from a multi-beat screenplay paste into one speech take.
    if (
        not explicit_dialogue
        and script_shape["looks_like_script"]
        and script_shape["quote_count"] >= 2
    ):
        raise ValueError(
            "This looks like a multi-line script with several quoted lines. Enter only the "
            "single short line for this clip in the dialogue field, split the script in Studio, "
            "or use uploaded voiceover at Finish social cut."
        )
    exact_dialogue = explicit_dialogue or quoted_dialogue(prompt)
    if not exact_dialogue:
        raise ValueError(
            "Native LTX dialogue requires exact spoken words. Add dialogue to the shot "
            "or put it in quotation marks."
        )
    words = len(re.findall(r"\b[\w’'-]+\b", exact_dialogue, flags=re.UNICODE))
    limit = spoken_word_limit(duration_seconds)
    if words > limit:
        raise ValueError(
            f'Native dialogue has {words} words but this {float(duration_seconds):g}s shot '
            f"has a reliability budget of {limit}. Shorten the line, lengthen/split the shot, "
            "or use an uploaded voiceover at export."
        )
    speaker_name = speaker.strip() or "The single visible speaker"
    language_name = language.strip() or "English"
    delivery = f" with a {accent.strip()} accent" if accent.strip() else ""
    ambient_suffix = (
        f" Under the voice: {ambience.strip()}." if ambience.strip() else ""
    )
    # Follow LTX's native training-style syntax: a flowing visual paragraph with
    # the spoken line in quotes. Avoid instruction-heavy text such as "say
    # exactly"; those meta-instructions can themselves become phonetic material
    # in a joint audio/video model.
    compiled = (
        f'{speaker_name} speaks one short {language_name} line{delivery}: "{exact_dialogue}" '
        f"The speaker says no other words before or after this line. "
        f"After the final word, the speaker closes their mouth. "
        f"{visual_prompt.rstrip('. ')}. The voice is crisp and clearly articulated. "
        f"There is no narrator and no off-screen speaker.{ambient_suffix}"
    )
    guarded_negative = _add_negative(
        negative_prompt, EXTRA_SPEECH_NEGATIVES + STAGE_DIRECTION_NEGATIVES
    )
    return compiled[:5_000], guarded_negative, {
        "mode": "native-dialogue",
        "speech_allowed": True,
        "speaker": speaker_name,
        "language": language_name,
        "accent": accent.strip(),
        "dialogue": exact_dialogue,
        "dialogue_words": words,
        "word_limit": limit,
        "visual_prompt": visual_prompt,
        "script_shape": script_shape,
    }


def preview_audio_prompt(
    prompt: str,
    negative_prompt: str = "",
    *,
    mode: str = "ambient",
    duration_seconds: float = 5.0,
    dialogue: str = "",
    speaker: str = "",
    language: str = "English",
    accent: str = "",
    ambience: str = "",
) -> dict[str, Any]:
    """Return a UI-safe preview of the compiled LTX contract without generating."""
    script_shape = detect_script_shape(prompt)
    requested_lines = explicit_dialogue_intent(prompt)
    detected_dialogue = requested_lines[0] if len(requested_lines) == 1 else None
    visual = visual_only_prompt(prompt) if mode != "prompt" else (prompt or "")
    try:
        compiled, negative, contract = compile_audio_prompt(
            prompt,
            negative_prompt,
            mode=mode,
            duration_seconds=duration_seconds,
            dialogue=dialogue,
            speaker=speaker,
            language=language,
            accent=accent,
            ambience=ambience,
        )
        return {
            "ok": True,
            "mode": contract.get("mode", mode),
            "visual_prompt": contract.get("visual_prompt", visual),
            "compiled_prompt": compiled,
            "negative_prompt": negative,
            "dialogue": contract.get("dialogue"),
            "dialogue_words": contract.get("dialogue_words"),
            "word_limit": contract.get("word_limit") or spoken_word_limit(duration_seconds),
            "speech_allowed": contract.get("speech_allowed"),
            "prompt_preserved": contract.get("prompt_preserved", False),
            "script_shape": contract.get("script_shape", script_shape),
            "detected_dialogue": detected_dialogue,
            "suggested_audio_mode": None,
            "error": None,
            "summary": _preview_summary(contract.get("mode", mode), contract, script_shape),
        }
    except ValueError as exc:
        return {
            "ok": False,
            "mode": mode,
            "visual_prompt": visual,
            "compiled_prompt": None,
            "negative_prompt": negative_prompt,
            "dialogue": _clean_dialogue(dialogue) or None,
            "dialogue_words": None,
            "word_limit": spoken_word_limit(duration_seconds),
            "speech_allowed": mode == "native-dialogue",
            "prompt_preserved": mode == "prompt",
            "script_shape": script_shape,
            "detected_dialogue": detected_dialogue,
            "suggested_audio_mode": (
                "native-dialogue"
                if mode == "ambient" and detected_dialogue
                else None
            ),
            "error": str(exc),
            "summary": "Cannot compile this audio contract yet.",
        }


def _preview_summary(mode: str, contract: dict[str, Any], script_shape: dict[str, Any]) -> str:
    if mode == "prompt":
        base = "Advanced · prompt text is sent unchanged (speech may follow the prose)."
    elif mode == "silent":
        base = "Silent · picture only; generated audio is disconnected."
    elif mode == "native-dialogue":
        words = contract.get("dialogue_words")
        limit = contract.get("word_limit")
        base = f"Native dialogue · {words}/{limit} words · generative speech."
    else:
        base = "Ambience + Foley · zero-speech contract on the visual prose."
    if script_shape.get("looks_like_script"):
        return f"{base} Script-shaped input detected."
    return base
