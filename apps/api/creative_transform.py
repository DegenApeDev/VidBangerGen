from __future__ import annotations

import secrets
from typing import Any


CREATIVE_TRANSFORMS: dict[str, dict[str, Any]] = {
    "water-simulation": {
        "name": "Water Simulation",
        "lora": "ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors",
        "default_negative": "",
        "strength": 1.2, "guidance": 1.0, "steps": 8,
        "recipe": "distilled-fixed", "required_fps": 24.0,
        "base_model": "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
        "components_checkpoint": "ltx-2.3-22b-distilled.safetensors",
        "require_ltx_frame_count": True,
        "max_frames": 121, "max_side": 768, "source_audio_policy": "discard",
    },
    "instant-shave": {
        "name": "Instant Shave",
        "lora": "ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors",
        "scene": (
            "the same visible person, completely smooth and clean-shaven with bare natural "
            "skin, no beard, no mustache, no stubble, and no facial hair"
        ),
        "default_negative": (
            "beard, mustache, facial hair, stubble, beard shadow, changed identity, changed "
            "expression, changed motion, changed lighting, worst quality, inconsistent motion, "
            "blurry, jittery, distorted, speech"
        ),
        "strength": 1.0, "guidance": 4.0, "steps": 30,
        "recipe": "dev-stg", "stg": 1.0, "stg_rescale": 0.7, "stg_blocks": "29",
        "base_model": "ltx-2.3-22b-dev-Q4_K_M.gguf",
        "components_checkpoint": "ltx-2.3-22b-dev.safetensors",
        "max_frames": 121, "max_side": 768, "source_audio_policy": "preserve",
    },
    "cross-eyed": {
        "name": "Cross-Eyed Character Effect",
        "lora": "ltx-2.3-22b-ic-lora-cross-eyed-0.9.safetensors",
        "scene": (
            "a close-up portrait of the same visible person with permanent severe convergent "
            "strabismus, both eyes continuously turned inward toward the nose throughout the shot"
        ),
        "default_negative": (
            "worst quality, inconsistent motion, blurry, jittery, distorted, normal eyes, "
            "straight eyes, corrected eyes, gaze correction, shifting identity, speech"
        ),
        "strength": 1.0, "guidance": 4.0, "steps": 30,
        "recipe": "dev-stg", "stg": 1.0, "stg_rescale": 0.7, "stg_blocks": "29",
        "base_model": "ltx-2.3-22b-dev-Q4_K_M.gguf",
        "components_checkpoint": "ltx-2.3-22b-dev.safetensors",
        "require_ltx_frame_count": True,
        "max_frames": 121, "max_side": 768, "source_audio_policy": "preserve",
    },
    "day-to-night": {
        "name": "Day to Night",
        "lora": "ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors",
        "default_prompt": (
            "A realistic nighttime scene with coherent practical lighting, deep natural "
            "shadows, restrained highlights, and a dark sky. Only the lighting changes "
            "from day to night; composition, framing, camera movement, subject identity, "
            "and motion remain identical to the reference."
        ),
        "default_negative": (
            "daytime, bright sunlight, blue sky, overexposed, changed composition, "
            "changed identity, inconsistent motion, blurry, jittery, distorted, speech"
        ),
        "strength": 1.0,
        "guidance": 3.5,
        "steps": 30,
        "max_frames": 121,
        "max_side": 768,
        "source_audio_policy": "preserve",
    },
    "deblur": {
        "name": "Video Deblur",
        "lora": "ltx-2.3-22b-ic-lora-deblur-0.9.safetensors",
        "scene": "the exact source scene, subjects, setting, action, and camera movement",
        "default_negative": (
            "worst quality, blurry, out of focus, defocused, soft, hazy, smeared, "
            "low detail, jittery, distorted, oversharpened, haloing, ringing, speech"
        ),
        "strength": 1.0, "guidance": 3.5, "steps": 30,
        "max_frames": 121, "max_side": 768, "source_audio_policy": "preserve",
    },
    "decompression": {
        "name": "Video Decompression",
        "lora": "ltx-2.3-22b-ic-lora-decompression-0.9.safetensors",
        "scene": "the exact source scene, subjects, setting, action, and camera movement",
        "default_negative": (
            "worst quality, macroblocking, compression artifacts, chroma bleed, ringing, "
            "banding, mosquito noise, blurry, changed identity, changed geometry, jittery, speech"
        ),
        "strength": 1.0, "guidance": 3.5, "steps": 30,
        "max_frames": 121, "max_side": 768, "source_audio_policy": "preserve",
    },
    "colorization": {
        "name": "Video Colorization",
        "lora": "ltx-2.3-22b-ic-lora-colorization-0.9.safetensors",
        "scene": "the exact source scene with natural, restrained, historically plausible colors",
        "default_negative": (
            "worst quality, monochrome, grayscale, desaturated, color bleed, oversaturated, "
            "changed identity, changed geometry, inconsistent motion, jittery, distorted, speech"
        ),
        "strength": 1.0, "guidance": 3.5, "steps": 30,
        "max_frames": 121, "max_side": 768, "source_audio_policy": "preserve",
    },
    "clean-plate": {
        "name": "Clean Plate",
        "lora": "ltx-2.3-22b-ic-lora-clean-plate-1.0.safetensors",
        "scene": (
            "an empty clean plate of the exact same location, background, structures, "
            "lighting, framing, and camera movement"
        ),
        "default_negative": (
            "people, humans, figures, faces, bodies, arms, hands, legs, vehicles, moving "
            "foreground objects, changed background, changed camera, blurry, jittery, distorted"
        ),
        "strength": 1.0, "guidance": 3.5, "steps": 30,
        "max_frames": 121, "max_side": 768, "source_audio_policy": "discard",
    },
}


class CreativeTransformAdapter:
    """Build isolated, source-preserving LTX 2.3 Creative Lab transforms.

    These graphs are submitted through ComfyUI's API and never alter the
    server's saved workflows.  The approved source video is attached as an
    IC-LoRA guide for the complete stage-one denoise.  Audio is intentionally
    excluded from the diffusion graph and restored bit-for-bit from the source
    by the local worker after the visual pass.
    """

    max_frames = 121
    max_side = 768
    distilled_sigmas = (
        "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
    )

    def limits(self, mode: str) -> dict[str, Any]:
        config = CREATIVE_TRANSFORMS.get(mode)
        if not config:
            raise ValueError(f"Unsupported Creative Lab transform: {mode}")
        return {
            "max_frames": int(config.get("max_frames", self.max_frames)),
            "max_side": int(config.get("max_side", self.max_side)),
            "required_fps": config.get("required_fps"),
            "require_ltx_frame_count": bool(config.get("require_ltx_frame_count", False)),
        }

    def validate_source(self, mode: str, info: dict[str, Any]) -> None:
        limits = self.limits(mode)
        frames = int(info.get("frames") or 0)
        if frames <= 0:
            raise ValueError("Creative Lab source must have a countable video frame stream")
        required_fps = limits.get("required_fps")
        actual_fps = float(info.get("fps") or 0)
        if required_fps and abs(actual_fps - float(required_fps)) > 0.05:
            raise ValueError(
                f"{mode.replace('-', ' ').title()} requires a {float(required_fps):g} fps "
                f"source; this clip is {actual_fps:g} fps"
            )
        if limits.get("require_ltx_frame_count") and (frames - 1) % 8:
            raise ValueError(
                f"{mode.replace('-', ' ').title()} requires an LTX frame count of 8n+1; "
                f"this clip has {frames} frames"
            )

    def _positive_prompt(self, mode: str, prompt: str, config: dict[str, Any]) -> str:
        supplied = prompt.strip()
        if mode == "water-simulation":
            if not supplied:
                raise ValueError(
                    "Water Simulation needs a concrete description of the water, its motion, "
                    "and how it interacts with the scene"
                )
            return (
                "Reference shows the exact dry source scene, subjects, clothing, pose, camera "
                "framing, and background geometry. Edited shows the same scene with water added. "
                f"ADD WATER {supplied.rstrip('. ')}. Subject identity, clothing, framing, camera "
                "movement, and background geometry are identical to the reference; only "
                "water-related elements differ between reference and edited."
            )
        if mode == "instant-shave":
            scene = supplied or str(config["scene"])
            return (
                f"REMOVEBEARD {scene.rstrip('. ')}. Subject identity, facial structure, "
                "expression, motion, lighting, camera framing, clothing, and the surrounding "
                "scene remain identical to the reference; only facial hair is removed."
            )
        if mode == "cross-eyed":
            scene = supplied or str(config["scene"])
            return (
                f"{scene.rstrip('. ')}. The subject maintains their natural facial expression, "
                "head movement, identity, lighting, clothing, surrounding scene, and camera "
                "framing; only the direction of both eyes changes."
            )
        if mode == "day-to-night":
            return supplied or str(config["default_prompt"])
        scene = supplied or str(config["scene"])
        if mode == "deblur":
            return (
                f"Reference shows {scene}, heavily out of focus with soft defocused blur and "
                f"no fine detail. Edited shows the same scene in sharp focus with crisp detail "
                f"and clean edges. DEBLUR {scene}. Subject identity, framing, motion, and "
                "background geometry are identical to the reference; only focus and sharpness differ."
            )
        if mode == "decompression":
            return (
                f"Reference shows {scene}, heavily compressed with visible macroblocking, "
                f"chroma bleed, ringing, and banding. Edited shows the same scene restored to "
                f"high quality with clean edges and no compression artifacts. ENHANCE QUALITY {scene}."
            )
        if mode == "colorization":
            return (
                f"Reference shows {scene}, rendered in monochrome with identical motion and "
                f"geometry. Edited shows the same scene with natural colors restored. COLORIZE {scene}."
            )
        if mode == "clean-plate":
            return (
                f"{scene}. No people, humans, figures, faces, body parts, vehicles, or unwanted "
                "foreground objects appear anywhere in the frame. The static environment and "
                "camera movement remain identical to the source video."
            )
        raise ValueError(f"Unsupported Creative Lab transform: {mode}")

    def build(
        self,
        *,
        mode: str,
        video_name: str,
        prompt: str = "",
        negative_prompt: str = "",
        strength: float | None = None,
        seed: int = -1,
        filename_prefix: str = "vbg/creative-lab/output",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        config = CREATIVE_TRANSFORMS.get(mode)
        if not config:
            raise ValueError(f"Unsupported Creative Lab transform: {mode}")
        actual_seed = seed if seed >= 0 else secrets.randbelow(2**63 - 2)
        lora_strength = float(config["strength"] if strength is None else strength)
        if not 0.5 <= lora_strength <= 1.25:
            raise ValueError("Creative transform strength must be between 0.5 and 1.25")
        positive = self._positive_prompt(mode, str(prompt or ""), config)
        negative = str(negative_prompt or config["default_negative"]).strip()
        max_side = int(config.get("max_side", self.max_side))
        recipe = str(config.get("recipe") or "dev-cfg")
        base_model = str(
            config.get("base_model") or "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"
        )
        components_checkpoint = str(
            config.get("components_checkpoint") or "ltx-2.3-22b-distilled.safetensors"
        )

        graph: dict[str, Any] = {
            "1": {"class_type": "LoadVideo", "inputs": {"file": video_name}},
            "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
            "3": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["2", 0], "resize_type": "scale longer dimension",
                    "resize_type.longer_size": max_side, "scale_method": "area",
                },
            },
            "4": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["3", 0], "resize_type": "scale to multiple",
                    "resize_type.multiple": 32, "scale_method": "lanczos",
                },
            },
            "5": {"class_type": "GetImageSize", "inputs": {"image": ["4", 0]}},
            "6": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "ltx-2.3-22b-distilled_video_vae.safetensors"},
            },
            "7": {
                "class_type": "VAEEncodeTiled",
                "inputs": {
                    "pixels": ["4", 0], "vae": ["6", 0], "tile_size": 512,
                    "overlap": 64, "temporal_size": 32, "temporal_overlap": 8,
                },
            },
            "8": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": base_model},
            },
            "9": {
                "class_type": "LTXICLoRALoaderModelOnly",
                "inputs": {
                    "model": ["8", 0], "lora_name": config["lora"],
                    "strength_model": lora_strength,
                },
            },
            "10": {
                "class_type": "LTXAVTextEncoderLoader",
                "inputs": {
                    "text_encoder": "comfy_gemma_3_12B_it.safetensors",
                    "ckpt_name": components_checkpoint, "device": "default",
                },
            },
            "11": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["10", 0]}},
            "12": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["10", 0]}},
            "13": {
                "class_type": "LTXVConditioning",
                "inputs": {"positive": ["11", 0], "negative": ["12", 0], "frame_rate": ["2", 2]},
            },
            "14": {
                "class_type": "LTXAddVideoICLoRAGuide",
                "inputs": {
                    "positive": ["13", 0], "negative": ["13", 1], "vae": ["6", 0],
                    "latent": ["7", 0], "image": ["4", 0], "frame_idx": 0,
                    "strength": 1.0, "latent_downscale_factor": ["9", 1],
                    "crop": "disabled", "use_tiled_encode": True,
                    "tile_size": 256, "tile_overlap": 64,
                },
            },
            "15": {
                "class_type": "LTXVAudioVAELoader",
                "inputs": {"ckpt_name": components_checkpoint},
            },
            "16": {"class_type": "LTXFloatToInt", "inputs": {"a": ["2", 2]}},
            "17": {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": ["5", 2], "frame_rate": ["16", 0],
                    "batch_size": 1, "audio_vae": ["15", 0],
                },
            },
            "18": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": ["14", 2], "audio_latent": ["17", 0]},
            },
            "19": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["9", 0], "positive": ["14", 0],
                    "negative": ["14", 1], "cfg": float(config["guidance"]),
                },
            },
            "20": {"class_type": "RandomNoise", "inputs": {"noise_seed": actual_seed}},
            "21": {
                "class_type": "KSamplerSelect",
                "inputs": {
                    "sampler_name": (
                        "euler_ancestral_cfg_pp"
                        if recipe in {"distilled-fixed", "dev-stg"} else "euler_cfg_pp"
                    )
                },
            },
            "22": {
                "class_type": "LTXVScheduler",
                "inputs": {
                    "steps": int(config["steps"]), "max_shift": 2.05,
                    "base_shift": 0.95, "stretch": True, "terminal": 0.1,
                    "latent": ["18", 0],
                },
            },
            "23": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["20", 0], "guider": ["19", 0], "sampler": ["21", 0],
                    "sigmas": ["22", 0], "latent_image": ["18", 0],
                },
            },
            "24": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["23", 0]}},
            "25": {
                "class_type": "LTXVCropGuides",
                "inputs": {"positive": ["14", 0], "negative": ["14", 1], "latent": ["24", 0]},
            },
            "26": {
                "class_type": "LTXVTiledVAEDecode",
                "inputs": {
                    "vae": ["6", 0], "latents": ["25", 2], "horizontal_tiles": 2,
                    "vertical_tiles": 2, "overlap": 6, "last_frame_fix": False,
                    "working_device": "auto", "working_dtype": "auto",
                },
            },
            "27": {"class_type": "CreateVideo", "inputs": {"images": ["26", 0], "fps": ["2", 2]}},
            "28": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["27", 0], "filename_prefix": filename_prefix,
                    "format": "mp4", "codec": "h264",
                },
            },
        }
        if recipe == "distilled-fixed":
            graph["22"] = {
                "class_type": "ManualSigmas",
                "inputs": {"sigmas": self.distilled_sigmas},
            }
        elif recipe == "dev-stg":
            graph["29"] = {
                "class_type": "LTXVApplySTG",
                "inputs": {"model": ["9", 0], "block_indices": config["stg_blocks"]},
            }
            graph["19"] = {
                "class_type": "STGGuider",
                "inputs": {
                    "model": ["29", 0], "positive": ["14", 0],
                    "negative": ["14", 1], "cfg": float(config["guidance"]),
                    "stg": float(config["stg"]),
                    "rescale": float(config["stg_rescale"]),
                },
            }
        return graph, {
            "engine": f"LTX 2.3 Creative Lab — {config['name']}",
            "mode": mode, "workflow": "isolated-stage-one-gguf",
            "base_model": base_model,
            "lora": config["lora"], "strength": lora_strength,
            "guidance": float(config["guidance"]), "steps": int(config["steps"]),
            "recipe": recipe,
            "sampler": graph["21"]["inputs"]["sampler_name"],
            "sigmas": self.distilled_sigmas if recipe == "distilled-fixed" else "scheduled",
            "stg": (
                {
                    "scale": float(config["stg"]),
                    "rescale": float(config["stg_rescale"]),
                    "blocks": [int(value) for value in str(config["stg_blocks"]).split(",")],
                }
                if recipe == "dev-stg" else None
            ),
            "max_frames": int(config.get("max_frames", self.max_frames)),
            "max_side": max_side,
            "source_audio_policy": config.get("source_audio_policy", "preserve"),
            "audio_mode": (
                "source-restored-locally"
                if config.get("source_audio_policy", "preserve") == "preserve"
                else "silent-source-audio-discarded"
            ),
            "seed": actual_seed,
            "prompt": positive, "negative_prompt": negative,
        }
