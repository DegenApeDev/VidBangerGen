from __future__ import annotations

import pytest

from apps.api.audio_prompt import (
    compile_audio_prompt,
    detect_script_shape,
    explicit_dialogue_intent,
    preview_audio_prompt,
    spoken_word_limit,
    visual_only_prompt,
)
from apps.api.schemas import GenerationSettings, ManualPlanShot, ShotCreate


OVEN_SCREENPLAY = """
INT. OVEN – DAY. Static camera from inside the oven, looking outward through the slightly fogged glass door. Warm golden light glows around freshly baked cookies. The baker’s face fills the frame, eyes wide with focus, his breath fogging the glass as he leans in. Subtle reflections move across the glass as steam rises.
Baker (whispering dramatically): “Today… I achieve perfection.”
He leans even closer, nose nearly touching the glass.
“Golden edges. Soft center. The gods themselves will smell these cookies and weep.”
Baker: “Wait—”
(beat)
“Did I… forget the chocolate chips?”
Cut to side view — coworker pops into frame, chewing casually.
Coworker (mouth full): “Nope. You forgot the sugar.”
Quick zoom back to the baker’s horrified face, pressed against the oven door, as cookies deflate behind the glass. Steam drifts upward in slow motion.
pixar style acting and timing
""".strip()

HIPPIE_DIALOGUE_PROMPT = (
    "Pixar-style 3D animation of a 40-year-old hippie man with long hair, "
    "a colorful headband, and a cozy casual outfit sitting at his desktop PC "
    "computer in a warm, ambient-lit room. He is looking at the glowing monitor "
    "screen where AI video generations are appearing. Bright expressive eyes, "
    "stylized character design, soft rim lighting, colorful cozy aesthetic, no text. "
    'Speaks only "Far Out Dude" as he points at the screen. then fade out the '
    "camera to black and only show LTX logo at the end."
)


def test_generation_surfaces_default_to_guarded_ambient_audio():
    assert GenerationSettings().audio_mode == "ambient"
    assert ManualPlanShot(prompt="A visual shot").audio_mode == "ambient"
    assert ShotCreate(prompt="A visual shot").audio_mode == "ambient"


def test_ambient_mode_rejects_an_explicit_quoted_speech_request():
    with pytest.raises(ValueError, match="no-speech mode.*Native quoted speech"):
        compile_audio_prompt(
            'A reporter faces camera and says: "This line is deliberate."',
            "flicker",
            mode="ambient",
            duration_seconds=5,
            ambience="quiet street ambience",
        )


def test_hippie_prompt_routes_far_out_dude_to_native_dialogue_without_prose_leak():
    assert explicit_dialogue_intent(HIPPIE_DIALOGUE_PROMPT) == ["Far Out Dude"]

    preview = preview_audio_prompt(
        HIPPIE_DIALOGUE_PROMPT,
        mode="ambient",
        duration_seconds=5,
    )
    assert preview["ok"] is False
    assert preview["detected_dialogue"] == "Far Out Dude"
    assert preview["suggested_audio_mode"] == "native-dialogue"
    assert "Ambience + Foley is a no-speech mode" in preview["error"]

    prompt, negative, contract = compile_audio_prompt(
        HIPPIE_DIALOGUE_PROMPT,
        "",
        mode="native-dialogue",
        duration_seconds=5,
        dialogue=preview["detected_dialogue"],
    )
    assert prompt.count('"Far Out Dude"') == 1
    assert "one short English line" in prompt
    assert "no other words before or after" in prompt
    assert "He points at the screen" in prompt
    assert "gradually darkens until fully black" in prompt
    assert "video generations are then" not in prompt
    assert "wrong words" in negative
    assert "paraphrased dialogue" in negative
    assert contract["dialogue"] == "Far Out Dude"
    assert contract["dialogue_words"] == 3


def test_ambient_mode_rewrites_unanchored_talking_activity():
    source = (
        "Subject: A simple cartoon talking about Local AI and promoting it. "
        "The character points toward a glowing computer."
    )

    visual = visual_only_prompt(source)
    prompt, negative, contract = compile_audio_prompt(
        source, "", mode="ambient", duration_seconds=5,
    )

    assert "talking about" not in visual.lower()
    assert "local ai" not in visual.lower()
    assert "gesturing silently" in visual.lower()
    assert "glowing computer" in visual.lower()
    assert "talking about" not in prompt.lower()
    assert contract["visual_prompt"] == visual
    assert "gibberish" in negative


def test_advanced_prompt_mode_is_the_only_unchanged_audio_path():
    original = "A host reads this exact prompt aloud if the operator really wants that."
    prompt, negative, contract = compile_audio_prompt(
        original, "flicker", mode="prompt", duration_seconds=5
    )

    assert prompt == original
    assert negative == "flicker"
    assert contract["prompt_preserved"] is True


def test_native_dialogue_uses_exact_single_line_and_reports_budget():
    prompt, negative, contract = compile_audio_prompt(
        "A reporter looks directly into the lens.",
        "flicker",
        mode="native-dialogue",
        duration_seconds=5,
        dialogue="We can build this together.",
        speaker="The reporter",
        language="English",
        accent="Canadian",
    )

    assert (
        'speaks one short English line with a Canadian accent: '
        '"We can build this together."'
    ) in prompt
    assert "There is no narrator and no off-screen speaker" in prompt
    assert "narration" in negative
    assert "gibberish" in negative
    assert contract["dialogue_words"] == 5
    assert contract["word_limit"] == spoken_word_limit(5)


def test_native_dialogue_rejects_words_that_do_not_fit_the_clip():
    with pytest.raises(ValueError, match="reliability budget"):
        compile_audio_prompt(
            "A presenter speaks.", "", mode="native-dialogue", duration_seconds=2,
            dialogue="one two three four five six seven eight nine",
        )


def test_native_dialogue_separates_inline_quotes_from_visual_prose():
    prompt, _negative, contract = compile_audio_prompt(
        'A presenter points to the skyline and says, "Build it here." '
        "The camera pushes toward the illuminated bridge.",
        "",
        mode="native-dialogue",
        duration_seconds=5,
    )

    assert prompt.count("Build it here.") == 1
    assert "camera pushes toward the illuminated bridge" in prompt
    assert "says," not in prompt
    assert contract["dialogue"] == "Build it here."


def test_native_dialogue_never_duplicates_a_quote_only_prompt():
    prompt, _negative, contract = compile_audio_prompt(
        '“Keep moving.”', "", mode="native-dialogue", duration_seconds=2,
    )

    assert prompt.count("Keep moving.") == 1
    assert "A natural cinematic shot" in prompt
    assert contract["dialogue"] == "Keep moving."


def test_separate_dialogue_field_wins_and_accepts_wrapping_quotes():
    prompt, _negative, contract = compile_audio_prompt(
        'A reporter says "old wording" beside a quiet street.',
        "",
        mode="native-dialogue",
        duration_seconds=5,
        dialogue='“New wording only.”',
    )

    assert "old wording" not in prompt
    assert prompt.count("New wording only.") == 1
    assert contract["dialogue"] == "New wording only."


def test_screenplay_residue_is_stripped_from_visual_and_ambient_contracts():
    visual = visual_only_prompt(OVEN_SCREENPLAY)
    prompt, negative, contract = compile_audio_prompt(
        OVEN_SCREENPLAY, "", mode="ambient", duration_seconds=5
    )

    for forbidden in (
        "Baker:",
        "(beat)",
        "Cut to",
        "whispering dramatically",
        "Today… I achieve perfection",
        "chocolate chips",
        "forgot the sugar",
        "INT.",
    ):
        assert forbidden not in visual
        assert forbidden not in prompt

    assert "inside the oven" in visual
    assert "leans even closer" in visual
    assert "horrified face" in visual
    assert "deflate behind the glass" in visual
    assert "pixar style" in visual.lower()
    assert "zero human voices" in prompt
    assert "stage directions" in negative
    assert "reading aloud" in negative
    assert contract["visual_prompt"] == visual
    assert contract["script_shape"]["looks_like_script"] is True


def test_detect_script_shape_flags_multi_beat_screenplay():
    shape = detect_script_shape(OVEN_SCREENPLAY)
    assert shape["looks_like_script"] is True
    assert shape["quote_count"] >= 3
    assert shape["has_character_cue"] is True
    assert shape["has_transition"] is True
    assert "Script to video" in shape["hint"]


def test_simple_visual_prompt_is_not_flagged_as_script():
    prompt = "A baker leans toward fogged oven glass while warm light glows on cookies."
    shape = detect_script_shape(prompt)
    assert shape["looks_like_script"] is False
    assert visual_only_prompt(prompt) == prompt


def test_native_dialogue_rejects_multi_quote_scripts_without_explicit_line():
    with pytest.raises(ValueError, match="multi-line script"):
        compile_audio_prompt(
            OVEN_SCREENPLAY, "", mode="native-dialogue", duration_seconds=5
        )


def test_native_dialogue_accepts_explicit_line_on_script_paste():
    prompt, _negative, contract = compile_audio_prompt(
        OVEN_SCREENPLAY,
        "",
        mode="native-dialogue",
        duration_seconds=5,
        dialogue="Did I forget the chocolate chips?",
        speaker="The baker",
    )
    assert contract["dialogue"] == "Did I forget the chocolate chips?"
    assert prompt.count("Did I forget the chocolate chips?") == 1
    assert "Baker:" not in prompt
    assert "(beat)" not in prompt


def test_preview_audio_prompt_reports_compile_errors_without_raising():
    preview = preview_audio_prompt(
        OVEN_SCREENPLAY, mode="native-dialogue", duration_seconds=5
    )
    assert preview["ok"] is False
    assert "multi-line script" in (preview["error"] or "")
    assert preview["script_shape"]["looks_like_script"] is True

    ambient = preview_audio_prompt(OVEN_SCREENPLAY, mode="ambient", duration_seconds=5)
    assert ambient["ok"] is True
    assert ambient["visual_prompt"]
    assert "Script-shaped" in ambient["summary"]
