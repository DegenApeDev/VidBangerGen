from __future__ import annotations

import secrets
from typing import Any


class IngredientsAdapter:
    """Build the official LTX 2.3 Ingredients reference-sheet workflow.

    The upstream distilled graph layers the distilled 384 LoRA and Ingredients
    IC-LoRA over the dev transformer, then supplies the reference sheet as a
    static video matching the generated clip. The configured worker uses the equivalent
    dev Q4_K_M GGUF transformer so this remains independent of the unreliable
    FP8 route and does not mutate the stable Quick Generate graph.
    """

    width = 768
    height = 448
    frames = 121
    fps = 24
    max_duration_seconds = 5.0
    sigmas = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

    def build(
        self,
        *,
        image_name: str,
        reference_description: str,
        shot_prompt: str,
        negative_prompt: str = "",
        strength: float = 1.0,
        seed: int = -1,
        filename_prefix: str = "vbg/ingredients/output",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        reference = reference_description.strip()
        action = shot_prompt.strip()
        if not reference:
            raise ValueError("Ingredients requires a description of the reference-sheet panels")
        if not action:
            raise ValueError("Ingredients requires a generated-video action prompt")
        if not 0.5 <= strength <= 1.5:
            raise ValueError("Ingredients strength must be between 0.5 and 1.5")
        actual_seed = seed if seed >= 0 else secrets.randbelow(2**63 - 2)
        positive = f"Reference sheet: {reference}\n\nGenerated video: {action}"
        negative = negative_prompt.strip() or (
            "worst quality, inconsistent identity, inconsistent costume, missing product, "
            "incorrect logo, inconsistent motion, blurry, jittery, distorted, duplicate subject, "
            "speech, dialogue, captions, watermark"
        )

        graph: dict[str, Any] = {
            "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "2": {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["1", 0], "upscale_method": "lanczos",
                    "width": self.width, "height": self.height, "crop": "disabled",
                },
            },
            "3": {
                "class_type": "RepeatImageBatch",
                "inputs": {"image": ["2", 0], "amount": self.frames},
            },
            "4": {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {
                    "width": self.width, "height": self.height,
                    "length": self.frames, "batch_size": 1,
                },
            },
            "5": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "ltx-2.3-22b-distilled_video_vae.safetensors"},
            },
            "6": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": "ltx-2.3-22b-dev-Q4_K_M.gguf"},
            },
            "7": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["6", 0],
                    "lora_name": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
                    "strength_model": 0.5,
                },
            },
            "8": {
                "class_type": "LTXICLoRALoaderModelOnly",
                "inputs": {
                    "model": ["7", 0],
                    "lora_name": "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
                    "strength_model": strength,
                },
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
                "class_type": "LTXAddVideoICLoRAGuide",
                "inputs": {
                    "positive": ["12", 0], "negative": ["12", 1], "vae": ["5", 0],
                    "latent": ["4", 0], "image": ["3", 0], "frame_idx": 0,
                    "strength": 1.0, "latent_downscale_factor": ["8", 1],
                    "crop": "disabled", "use_tiled_encode": True,
                    "tile_size": 256, "tile_overlap": 64,
                },
            },
            "14": {
                "class_type": "LTXVAudioVAELoader",
                "inputs": {"ckpt_name": "ltx-2.3-22b-dev.safetensors"},
            },
            "15": {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": self.frames, "frame_rate": self.fps,
                    "batch_size": 1, "audio_vae": ["14", 0],
                },
            },
            "16": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": ["13", 2], "audio_latent": ["15", 0]},
            },
            "17": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["8", 0], "positive": ["13", 0],
                    "negative": ["13", 1], "cfg": 1.0,
                },
            },
            "18": {"class_type": "RandomNoise", "inputs": {"noise_seed": actual_seed}},
            "19": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler_ancestral_cfg_pp"},
            },
            "20": {"class_type": "ManualSigmas", "inputs": {"sigmas": self.sigmas}},
            "21": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["18", 0], "guider": ["17", 0], "sampler": ["19", 0],
                    "sigmas": ["20", 0], "latent_image": ["16", 0],
                },
            },
            "22": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["21", 0]}},
            "23": {
                "class_type": "LTXVCropGuides",
                "inputs": {
                    "positive": ["13", 0], "negative": ["13", 1],
                    "latent": ["22", 0],
                },
            },
            "24": {
                "class_type": "LTXVTiledVAEDecode",
                "inputs": {
                    "vae": ["5", 0], "latents": ["23", 2],
                    "horizontal_tiles": 2, "vertical_tiles": 2, "overlap": 6,
                    "last_frame_fix": False, "working_device": "auto",
                    "working_dtype": "auto",
                },
            },
            # Ingredients is a visual-consistency render. Generated audio is
            # deliberately disconnected so it cannot read or mumble the prompt.
            "25": {
                "class_type": "CreateVideo",
                "inputs": {"images": ["24", 0], "fps": float(self.fps)},
            },
            "26": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["25", 0], "filename_prefix": filename_prefix,
                    "format": "mp4", "codec": "h264",
                },
            },
        }
        return graph, {
            "engine": "LTX 2.3 Creative Lab — Ingredients",
            "workflow": "official-distilled-reference-sheet-gguf",
            "official_workflow": (
                "LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json"
            ),
            "base_model": "ltx-2.3-22b-dev-Q4_K_M.gguf",
            "distilled_lora": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            "ingredients_lora": "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
            "strength": strength, "seed": actual_seed, "steps": 8, "cfg": 1.0,
            "dimensions": [self.width, self.height], "frames": self.frames,
            "fps": self.fps, "audio_mode": "silent",
            "reference_description": reference, "shot_prompt": action,
            "prompt": positive, "negative_prompt": negative,
        }
