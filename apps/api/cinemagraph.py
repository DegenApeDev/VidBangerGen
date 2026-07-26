from __future__ import annotations

import secrets
from typing import Any


class CinemagraphAdapter:
    """Build the LTX 2.3 still-image selective-motion recipe.

    Cinemagraph is a normal LoRA over the dev transformer, not an IC-LoRA
    video filter.  Keeping it in a dedicated graph avoids mixing it with the
    stable Quick Generate workflow or the source-video transform graph.
    """

    frames = 25
    fps = 25
    steps = 30
    guidance = 4.0
    stg = 1.0
    stg_rescale = 0.7
    stg_blocks = "29"

    def build(
        self,
        *,
        image_name: str,
        prompt: str,
        width: int,
        height: int,
        negative_prompt: str = "",
        strength: float = 1.0,
        seed: int = -1,
        filename_prefix: str = "vbg/cinemagraph/output",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        motion = prompt.strip()
        if not motion:
            raise ValueError(
                "Cinemagraph needs a precise description of the one element that moves"
            )
        if width % 32 or height % 32 or min(width, height) < 256:
            raise ValueError("Cinemagraph dimensions must be practical multiples of 32")
        if not 0.7 <= strength <= 3.0:
            raise ValueError("Cinemagraph strength must be between 0.7 and 3.0")
        actual_seed = seed if seed >= 0 else secrets.randbelow(2**63 - 2)
        positive = (
            "CINEMAGRAPH_MOTION, tripod locked-off static camera, zero camera movement, "
            f"{motion.rstrip('. ')}, everything else remains completely frozen, seamless "
            "natural loop"
        )
        negative = negative_prompt.strip() or (
            "cars moving, camera movement, pan, tilt, zoom, parallax, whole image moving, "
            "background sliding, person moving, flicker on entire image, noisy texture, "
            "crawling texture, distorted, blurry, low quality, speech, dialogue"
        )

        graph: dict[str, Any] = {
            "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "2": {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["1", 0], "upscale_method": "lanczos",
                    "width": width, "height": height, "crop": "disabled",
                },
            },
            "3": {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {
                    "width": width, "height": height, "length": self.frames,
                    "batch_size": 1,
                },
            },
            "4": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "ltx-2.3-22b-distilled_video_vae.safetensors"},
            },
            "5": {
                "class_type": "LTXVImgToVideoConditionOnly",
                "inputs": {
                    "image": ["2", 0], "latent": ["3", 0], "strength": 1.0,
                    "vae": ["4", 0], "bypass": False,
                },
            },
            "6": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": "ltx-2.3-22b-dev-Q4_K_M.gguf"},
            },
            "7": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["6", 0],
                    "lora_name": "ltx-2.3-22b-lora-cinemagraph-0.9.safetensors",
                    "strength_model": strength,
                },
            },
            "8": {
                "class_type": "LTXVApplySTG",
                "inputs": {"model": ["7", 0], "block_indices": self.stg_blocks},
            },
            "9": {
                "class_type": "LTXAVTextEncoderLoader",
                "inputs": {
                    "text_encoder": "comfy_gemma_3_12B_it.safetensors",
                    "ckpt_name": "ltx-2.3-22b-dev.safetensors", "device": "default",
                },
            },
            "10": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": positive, "clip": ["9", 0]},
            },
            "11": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative, "clip": ["9", 0]},
            },
            "12": {
                "class_type": "LTXVConditioning",
                "inputs": {
                    "positive": ["10", 0], "negative": ["11", 0],
                    "frame_rate": float(self.fps),
                },
            },
            "13": {
                "class_type": "LTXVAudioVAELoader",
                "inputs": {"ckpt_name": "ltx-2.3-22b-dev.safetensors"},
            },
            "14": {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": self.frames, "frame_rate": self.fps,
                    "batch_size": 1, "audio_vae": ["13", 0],
                },
            },
            "15": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": ["5", 0], "audio_latent": ["14", 0]},
            },
            "16": {
                "class_type": "STGGuider",
                "inputs": {
                    "model": ["8", 0], "positive": ["12", 0],
                    "negative": ["12", 1], "cfg": self.guidance,
                    "stg": self.stg, "rescale": self.stg_rescale,
                },
            },
            "17": {"class_type": "RandomNoise", "inputs": {"noise_seed": actual_seed}},
            "18": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler_ancestral_cfg_pp"},
            },
            "19": {
                "class_type": "LTXVScheduler",
                "inputs": {
                    "steps": self.steps, "max_shift": 2.05, "base_shift": 0.95,
                    "stretch": True, "terminal": 0.1, "latent": ["15", 0],
                },
            },
            "20": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["17", 0], "guider": ["16", 0],
                    "sampler": ["18", 0], "sigmas": ["19", 0],
                    "latent_image": ["15", 0],
                },
            },
            "21": {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {"av_latent": ["20", 0]},
            },
            "22": {
                "class_type": "LTXVTiledVAEDecode",
                "inputs": {
                    "vae": ["4", 0], "latents": ["21", 0],
                    "horizontal_tiles": 2, "vertical_tiles": 2, "overlap": 6,
                    "last_frame_fix": False, "working_device": "auto",
                    "working_dtype": "auto",
                },
            },
            # The cinemagraph route is intentionally silent.  Audio diffusion
            # stays in the joint latent but is never decoded into the container.
            "23": {
                "class_type": "CreateVideo",
                "inputs": {"images": ["22", 0], "fps": float(self.fps)},
            },
            "24": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["23", 0], "filename_prefix": filename_prefix,
                    "format": "mp4", "codec": "h264",
                },
            },
        }
        return graph, {
            "engine": "LTX 2.3 Creative Lab — Cinemagraph",
            "workflow": "isolated-image-to-video-dev-stg-gguf",
            "base_model": "ltx-2.3-22b-dev-Q4_K_M.gguf",
            "lora": "ltx-2.3-22b-lora-cinemagraph-0.9.safetensors",
            "strength": strength, "steps": self.steps, "guidance": self.guidance,
            "stg": {"scale": self.stg, "rescale": self.stg_rescale, "blocks": [29]},
            "frames": self.frames, "fps": self.fps,
            "dimensions": [width, height], "audio_mode": "silent-disconnected",
            "seed": actual_seed, "prompt": positive, "negative_prompt": negative,
        }
