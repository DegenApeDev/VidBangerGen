from __future__ import annotations

import secrets
from typing import Any

from .audio_prompt import visual_only_prompt


FOLEY_NEGATIVE = (
    "music, melody, song, singing, vocals, score, soundtrack, beat, rhythm bed, "
    "instrumental backing, speech, dialogue, talking, narration, voiceover, whispering, "
    "mumbling, gibberish, unintelligible language, tinny, thin, harsh, clipped, distorted, "
    "low bitrate"
)


class FoleyAdapter:
    """Build a video-to-audio-only LTX 2.3 Foley graph.

    The source video latent receives a zero noise mask and therefore remains
    frozen. Only the empty audio latent is denoised. ComfyUI saves lossless
    audio; the worker later muxes it onto the untouched local source video so
    this pass cannot lower picture quality or make visual edits.
    """

    max_frames = 121
    max_side = 768
    fps = 24
    lora = "ltx-2.3-22b-lora-foley-v2a-1.0.safetensors"
    model = "ltx-2.3-22b-dev-Q4_K_M.gguf"

    @staticmethod
    def compile_prompt(prompt: str) -> str:
        scene = visual_only_prompt(str(prompt or "").strip())
        return (
            f"{scene.rstrip('. ')}. The soundtrack contains close, physically accurate, "
            "synchronized environmental ambience and Foley matching every visible action, "
            "impact, surface, and material. No speech is present. No music is present."
        )[:5_000]

    def build(
        self,
        *,
        video_name: str,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 1.0,
        seed: int = -1,
        duration_seconds: float = 5.0,
        filename_prefix: str = "vbg/foley/audio",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not 0.8 <= float(strength) <= 1.0:
            raise ValueError("Foley LoRA strength must be between 0.8 and 1.0")
        if duration_seconds <= 0:
            raise ValueError("Foley source duration must be positive")
        actual_seed = seed if seed >= 0 else secrets.randbelow(2**63 - 2)
        positive = self.compile_prompt(prompt)
        negative = str(negative_prompt or FOLEY_NEGATIVE).strip()[:2_000]

        graph: dict[str, Any] = {
            "1": {"class_type": "LoadVideo", "inputs": {"file": video_name}},
            "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
            "3": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["2", 0], "resize_type": "scale longer dimension",
                    "resize_type.longer_size": self.max_side, "scale_method": "area",
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
                "inputs": {"unet_name": self.model},
            },
            "9": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["8", 0], "lora_name": self.lora,
                    "strength_model": float(strength),
                },
            },
            "10": {
                "class_type": "LTXAVTextEncoderLoader",
                "inputs": {
                    "text_encoder": "comfy_gemma_3_12B_it.safetensors",
                    "ckpt_name": "ltx-2.3-22b-dev.safetensors", "device": "default",
                },
            },
            "11": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": positive, "clip": ["10", 0]},
            },
            "12": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative, "clip": ["10", 0]},
            },
            "13": {
                "class_type": "LTXVConditioning",
                "inputs": {
                    "positive": ["11", 0], "negative": ["12", 0],
                    "frame_rate": ["2", 2],
                },
            },
            "14": {
                "class_type": "LTXVAudioVAELoader",
                "inputs": {"ckpt_name": "ltx-2.3-22b-dev.safetensors"},
            },
            "15": {"class_type": "LTXFloatToInt", "inputs": {"a": ["2", 2]}},
            "16": {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": ["5", 2], "frame_rate": ["15", 0],
                    "batch_size": 1, "audio_vae": ["14", 0],
                },
            },
            "17": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": ["7", 0], "audio_latent": ["16", 0]},
            },
            "18": {
                "class_type": "LTXVSetAudioVideoMaskByTime",
                "inputs": {
                    "av_latent": ["17", 0], "positive": ["13", 0],
                    "negative": ["13", 1], "model": ["9", 0], "vae": ["6", 0],
                    "audio_vae": ["14", 0], "start_time": 0.0,
                    "end_time": float(duration_seconds) + 1.0, "video_fps": ["2", 2],
                    "mask_video": False, "mask_audio": True,
                    "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0,
                    "slope_len": 1,
                },
            },
            "19": {
                "class_type": "GuiderParameters",
                "inputs": {
                    "modality": "AUDIO", "cfg": 6.0, "stg": 1.0,
                    "perturb_attn": True, "rescale": 0.7, "modality_scale": 1.0,
                    "skip_step": 0, "cross_attn": True,
                },
            },
            "20": {
                "class_type": "MultimodalGuider",
                "inputs": {
                    "model": ["9", 0], "positive": ["18", 0],
                    "negative": ["18", 1], "parameters": ["19", 0],
                    "skip_blocks": "29",
                },
            },
            "21": {"class_type": "RandomNoise", "inputs": {"noise_seed": actual_seed}},
            "22": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler_ancestral_cfg_pp"},
            },
            "23": {
                "class_type": "LTXVScheduler",
                "inputs": {
                    "steps": 30, "max_shift": 2.05, "base_shift": 0.95,
                    "stretch": True, "terminal": 0.1, "latent": ["18", 2],
                },
            },
            "24": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["21", 0], "guider": ["20", 0],
                    "sampler": ["22", 0], "sigmas": ["23", 0],
                    "latent_image": ["18", 2],
                },
            },
            "25": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["24", 0]}},
            "26": {
                "class_type": "LTXVAudioVAEDecode",
                "inputs": {"samples": ["25", 1], "audio_vae": ["14", 0]},
            },
            "27": {
                "class_type": "SaveAudio",
                "inputs": {"audio": ["26", 0], "filename_prefix": filename_prefix},
            },
        }
        return graph, {
            "engine": "LTX 2.3 Creative Lab — Foley V2A",
            "mode": "foley-v2a", "workflow": "isolated-video-to-audio-gguf",
            "base_model": self.model, "lora": self.lora,
            "strength": float(strength), "guidance": 6.0, "steps": 30,
            "stg": {"scale": 1.0, "blocks": [29], "mode": "stg_av"},
            "max_frames": self.max_frames, "max_side": self.max_side,
            "video_noise_mask": 0.0, "audio_noise_mask": 1.0,
            "source_video_policy": "bitstream-copied-locally",
            "source_audio_policy": "replace",
            "audio_output": "lossless-flac-then-local-aac-mux",
            "speech_allowed": False, "music_allowed": False,
            "seed": actual_seed, "prompt": positive, "negative_prompt": negative,
        }
