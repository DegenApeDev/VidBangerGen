from __future__ import annotations

import secrets
from typing import Any


class LipDubAdapter:
    """Build the official two-stage LTX 2.3 LipDub recipe on the GGUF base."""

    max_frames = 121
    base_landscape_size = (768, 448)
    stage_one_sigmas = (
        "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
    )
    stage_two_sigmas = "0.909375, 0.725, 0.421875, 0.0"

    @staticmethod
    def compile_prompt(scene_prompt: str, dialogue: str, language: str) -> str:
        words = dialogue.strip().replace('"', "'")
        if not words:
            raise ValueError("LipDub requires the exact desired dialogue")
        scene = scene_prompt.strip() or (
            "The source video shows one visible speaker; preserve the speaker, identity, "
            "performance, camera, lighting, and background"
        )
        tongue = language.strip() or "the source language"
        return f'{scene.rstrip(". ")}. The single visible speaker says in {tongue}: "{words}"'

    def validate_source(self, info: dict[str, Any]) -> None:
        frames = int(info.get("frames") or 0)
        if frames < 17:
            raise ValueError("LipDub needs at least 17 prepared source frames")
        if frames > self.max_frames:
            raise ValueError(
                f"LipDub supports up to {self.max_frames} frames per pass; this clip has {frames}"
            )
        if (frames - 1) % 8:
            raise ValueError(f"LipDub requires an LTX frame count of 8n+1; this clip has {frames}")

    def build(
        self,
        *,
        video_name: str,
        scene_prompt: str,
        dialogue: str,
        language: str,
        width: int,
        height: int,
        frames: int,
        fps: int = 24,
        negative_prompt: str = "",
        strength: float = 1.0,
        seed: int = -1,
        filename_prefix: str = "vbg/lipdub/output",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.validate_source({"frames": frames})
        if not 0.5 <= strength <= 1.25:
            raise ValueError("LipDub strength must be between 0.5 and 1.25")
        if width % 64 or height % 64:
            raise ValueError("LipDub source dimensions must be divisible by 64")
        actual_seed = seed if seed >= 0 else secrets.randbelow(2**63 - 3)
        positive = self.compile_prompt(scene_prompt, dialogue, language)
        negative = negative_prompt.strip() or (
            "multiple speakers, overlapping speech, incorrect lip sync, missing words, "
            "skipped words, gibberish, distorted voice, subtitles, text, identity drift, "
            "blurry face, jitter, worst quality"
        )
        graph: dict[str, Any] = {
            "1": {"class_type": "LoadVideo", "inputs": {"file": video_name}},
            "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
            "3": {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {
                    "width": width, "height": height, "length": frames, "batch_size": 1,
                },
            },
            "4": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "ltx-2.3-22b-distilled_video_vae.safetensors"},
            },
            "5": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": "ltx-2.3-22b-dev-Q4_K_M.gguf"},
            },
            "6": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["5", 0],
                    "lora_name": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
                    "strength_model": 0.5,
                },
            },
            "7": {
                "class_type": "LTXICLoRALoaderModelOnly",
                "inputs": {
                    "model": ["6", 0],
                    "lora_name": "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors",
                    "strength_model": strength,
                },
            },
            "8": {
                "class_type": "LTXAVTextEncoderLoader",
                "inputs": {
                    "text_encoder": "comfy_gemma_3_12B_it.safetensors",
                    "ckpt_name": "ltx-2.3-22b-dev.safetensors", "device": "default",
                },
            },
            "9": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": positive, "clip": ["8", 0]},
            },
            "10": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative, "clip": ["8", 0]},
            },
            "11": {
                "class_type": "LTXVConditioning",
                "inputs": {
                    "positive": ["9", 0], "negative": ["10", 0],
                    "frame_rate": ["2", 2],
                },
            },
            "12": {
                "class_type": "LTXAddVideoICLoRAGuide",
                "inputs": {
                    "positive": ["11", 0], "negative": ["11", 1],
                    "vae": ["4", 0], "latent": ["3", 0], "image": ["2", 0],
                    "frame_idx": 0, "strength": 1.0, "latent_downscale_factor": 1,
                    "crop": "disabled", "use_tiled_encode": True,
                    "tile_size": 256, "tile_overlap": 64,
                },
            },
            "13": {
                "class_type": "LTXVAudioVAELoader",
                "inputs": {"ckpt_name": "ltx-2.3-22b-dev.safetensors"},
            },
            "14": {
                "class_type": "LTXVAudioVAEEncode",
                "inputs": {"audio": ["2", 1], "audio_vae": ["13", 0]},
            },
            "15": {
                "class_type": "LTXVSetAudioRefTokens",
                "inputs": {
                    "positive": ["12", 0], "negative": ["12", 1],
                    "audio_latent": ["14", 0],
                },
            },
            "16": {"class_type": "LTXFloatToInt", "inputs": {"a": ["2", 2]}},
            "17": {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": frames, "frame_rate": ["16", 0],
                    "batch_size": 1, "audio_vae": ["13", 0],
                },
            },
            "18": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": ["12", 2], "audio_latent": ["17", 0]},
            },
            "19": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["7", 0], "positive": ["15", 0],
                    "negative": ["15", 1], "cfg": 1.0,
                },
            },
            "20": {"class_type": "RandomNoise", "inputs": {"noise_seed": actual_seed}},
            "21": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "22": {"class_type": "ManualSigmas", "inputs": {"sigmas": self.stage_one_sigmas}},
            "23": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["20", 0], "guider": ["19", 0],
                    "sampler": ["21", 0], "sigmas": ["22", 0],
                    "latent_image": ["18", 0],
                },
            },
            "24": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["23", 0]}},
            "25": {
                "class_type": "LTXVCropGuides",
                "inputs": {
                    "positive": ["12", 0], "negative": ["12", 1],
                    "latent": ["24", 0],
                },
            },
            "26": {
                "class_type": "LatentUpscaleModelLoader",
                "inputs": {"model_name": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"},
            },
            "27": {
                "class_type": "LTXVLatentUpsampler",
                "inputs": {"samples": ["25", 2], "upscale_model": ["26", 0], "vae": ["4", 0]},
            },
            "28": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["2", 0], "resize_type": "scale by multiplier",
                    "resize_type.multiplier": 2.0, "scale_method": "area",
                },
            },
            "29": {
                "class_type": "LTXVSetAudioRefTokens",
                "inputs": {
                    "positive": ["25", 0], "negative": ["25", 1],
                    "audio_latent": ["24", 1],
                },
            },
            "30": {
                "class_type": "LTXAddVideoICLoRAGuide",
                "inputs": {
                    "positive": ["29", 0], "negative": ["29", 1],
                    "vae": ["4", 0], "latent": ["27", 0], "image": ["28", 0],
                    "frame_idx": 0, "strength": 1.0, "latent_downscale_factor": 1,
                    "crop": "disabled", "use_tiled_encode": True,
                    "tile_size": 256, "tile_overlap": 64,
                },
            },
            "31": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": ["30", 2], "audio_latent": ["29", 2]},
            },
            "32": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["7", 0], "positive": ["30", 0],
                    "negative": ["30", 1], "cfg": 1.0,
                },
            },
            "33": {"class_type": "RandomNoise", "inputs": {"noise_seed": actual_seed + 1}},
            "34": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "35": {"class_type": "ManualSigmas", "inputs": {"sigmas": self.stage_two_sigmas}},
            "36": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["33", 0], "guider": ["32", 0],
                    "sampler": ["34", 0], "sigmas": ["35", 0],
                    "latent_image": ["31", 0],
                },
            },
            "37": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["36", 0]}},
            "38": {
                "class_type": "LTXVCropGuides",
                "inputs": {
                    "positive": ["30", 0], "negative": ["30", 1],
                    "latent": ["37", 0],
                },
            },
            "39": {
                "class_type": "LTXVTiledVAEDecode",
                "inputs": {
                    "vae": ["4", 0], "latents": ["38", 2],
                    "horizontal_tiles": 2, "vertical_tiles": 2, "overlap": 6,
                    "last_frame_fix": False, "working_device": "auto",
                    "working_dtype": "auto",
                },
            },
            "40": {
                "class_type": "LTXVAudioVAEDecode",
                "inputs": {"samples": ["37", 1], "audio_vae": ["13", 0]},
            },
            "41": {
                "class_type": "CreateVideo",
                "inputs": {"images": ["39", 0], "audio": ["40", 0], "fps": ["2", 2]},
            },
            "42": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["41", 0], "filename_prefix": filename_prefix,
                    "format": "mp4", "codec": "h264",
                },
            },
        }
        return graph, {
            "engine": "LTX 2.3 LipDub",
            "workflow": "official-two-stage-lipdub-gguf",
            "base_model": "ltx-2.3-22b-dev-Q4_K_M.gguf",
            "lora": "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors",
            "distilled_lora": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            "reference_downscale_factor": 1,
            "strength": strength, "width": width, "height": height,
            "frames": frames, "fps": fps,
            "stage_one": {"steps": 8, "guidance": 1.0, "sigmas": self.stage_one_sigmas},
            "stage_two": {
                "steps": 3, "guidance": 1.0, "sigmas": self.stage_two_sigmas,
                "spatial_upscale": 2, "audio": "frozen-from-stage-one",
            },
            "source_audio_policy": "voice-identity-reference",
            "audio_mode": "generated-dialogue-authoritative",
            "speaker_limit": 1, "language": language.strip(),
            "dialogue": dialogue.strip(), "prompt": positive,
            "negative_prompt": negative, "seed": actual_seed,
        }
