from __future__ import annotations

from apps.api.creative import CreativeDirector


def _shot(index: int, duration: float = 5.0) -> dict:
    return {
        "title": f"Shot {index}", "prompt": f"Action {index}",
        "duration_seconds": duration,
    }


def test_duration_fit_trims_overfull_short_storyboard(test_settings):
    shots = [_shot(index) for index in range(8)]
    CreativeDirector(test_settings)._fit_duration(shots, 5)
    assert len(shots) == 5
    assert sum(value["duration_seconds"] for value in shots) == 5
    assert all(1 <= value["duration_seconds"] <= 20 for value in shots)
    assert shots[-1]["title"] == "Shot 7"


def test_duration_fit_splits_long_storyboard_into_safe_generations(test_settings):
    shots = [_shot(0), _shot(1)]
    CreativeDirector(test_settings)._fit_duration(shots, 120)
    assert len(shots) == 6
    assert sum(value["duration_seconds"] for value in shots) == 120
    assert all(value["duration_seconds"] == 20 for value in shots)
    assert all("Continuous phase" in value["prompt"] for value in shots)


def test_prompt_enrichment_cannot_drop_primary_subject_or_story_event(test_settings):
    brief = {
        "subject": "an obsidian desk with a glowing split mechanical keyboard",
        "topic": "the desk, monitors, and coffee mug rise together into zero gravity",
        "style": "cinematic dark-mode realism",
        "brand_notes": "photorealistic gravity; never cartoonish",
    }
    prompt = CreativeDirector(test_settings)._enrich_prompt(
        "Slow-motion coffee ripples inside a floating mug",
        {"camera": "macro orbit", "audio": "deep synth impact"},
        brief,
    )
    assert brief["subject"] in prompt
    assert brief["topic"] in prompt
    assert brief["brand_notes"] in prompt
    assert "macro orbit" in prompt


def test_script_fallback_extracts_scenes_elements_and_exact_runtime(test_settings):
    brief = {
        "title": "Chrome Bird",
        "topic": "Mara releases a mechanical bird",
        "source_kind": "script",
        "duration_seconds": 15,
        "style": "cinematic workshop realism",
        "forbidden_elements": [],
        "script": """INT. WORKSHOP - NIGHT

MARA tightens the last bolt on a chrome bird.

MARA
Let's see if you remember how to fly.

EXT. ROOFTOP - DAWN

MARA opens her hands. The chrome bird launches into the sunrise.""",
    }
    concepts = CreativeDirector(test_settings)._script_fallback(brief)

    assert len(concepts) == 1
    assert sum(shot["duration_seconds"] for shot in concepts[0]["shots"]) == 15
    assert all(shot["scene_heading"] for shot in concepts[0]["shots"])
    assert all(shot["script_excerpt"] in brief["script"] for shot in concepts[0]["shots"])
    element_pairs = {(item["type"], item["name"]) for item in concepts[0]["elements"]}
    assert ("character", "Mara") in element_pairs
    assert ("location", "WORKSHOP") in element_pairs
    assert ("location", "ROOFTOP") in element_pairs


def test_script_fallback_splits_prose_without_repeating_whole_narration(test_settings):
    script = (
        "Families feel the pressure every day. Small businesses face another stack of forms. "
        "Neighbours start organizing practical local solutions. Volunteers gather in the town square. "
        "The community ends with a clear invitation to participate."
    )
    brief = {
        "title": "Prose narration", "topic": script, "source_kind": "script",
        "duration_seconds": 30, "style": "documentary realism",
        "forbidden_elements": [], "script": script,
    }

    shots = CreativeDirector(test_settings)._script_fallback(brief)[0]["shots"]

    assert len(shots) == 6
    assert sum(shot["duration_seconds"] for shot in shots) == 30
    assert all(shot["audio_mode"] == "ambient" for shot in shots)
    assert all(not shot["dialogue"] for shot in shots)
    assert all(shot["voiceover_text"] in script for shot in shots)
    assert all(script not in shot["prompt"] for shot in shots)
