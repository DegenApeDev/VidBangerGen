from __future__ import annotations

import secrets
from typing import Any


PIXEL_SPATIAL_LORAS = {
    2: "ltx-2.3-22b-ic-lora-pixel-spatial-upscaler-x2-0.9.safetensors",
    4: "ltx-2.3-22b-ic-lora-pixel-spatial-upscaler-x4-0.9.safetensors",
}


class PixelSpatialUpscaleAdapter:
    """Build the isolated LTX 2.3 Pixel Spatial Upscaler API graph.

    The official distilled workflow layers the distilled 384 LoRA and Pixel
    Spatial IC-LoRA over the 22B dev transformer. The configured worker uses the equivalent
    Q4_K_M GGUF transformer here so this pass does not depend on the unstable
    FP8 checkpoint or alter VidBangerGen's established generation graph.
    """

    max_output_side = 1536
    max_frames = 121
    sigmas = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

    def build(
        self,
        *,
        video_name: str,
        prompt: str,
        negative_prompt: str,
        scale: int,
        source_width: int,
        source_height: int,
        has_audio: bool,
        seed: int = -1,
        filename_prefix: str = "vbg/upscales/output",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if scale not in PIXEL_SPATIAL_LORAS:
            raise ValueError("Pixel Spatial Upscaler supports only 2x or 4x")
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Source video dimensions are invalid")

        actual_seed = seed if seed >= 0 else secrets.randbelow(2**63 - 2)
        max_input_side = self.max_output_side / scale
        source_long_side = max(source_width, source_height)
        preparation_scale = min(1.0, max_input_side / source_long_side)
        multiple = scale * 32
        prepared_width = source_width * preparation_scale
        prepared_height = source_height * preparation_scale
        output_width = max(multiple, round(prepared_width * scale / multiple) * multiple)
        output_height = max(multiple, round(prepared_height * scale / multiple) * multiple)

        graph: dict[str, Any] = {
            "1": {"class_type": "LoadVideo", "inputs": {"file": video_name}},
            "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
            # The official app caps this creative pass to a 1536px long edge.
            # First prepare oversized inputs, then measure the LoRA's exact
            # output factor and round to its latent-grid multiple.
            "3": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["2", 0], "resize_type": "scale by multiplier",
                    "resize_type.multiplier": preparation_scale, "scale_method": "lanczos",
                },
            },
            "4": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["3", 0], "resize_type": "scale by multiplier",
                    "resize_type.multiplier": float(scale), "scale_method": "lanczos",
                },
            },
            "5": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["4", 0], "resize_type": "scale to multiple",
                    "resize_type.multiple": multiple, "scale_method": "lanczos",
                },
            },
            "6": {"class_type": "GetImageSize", "inputs": {"image": ["5", 0]}},
            # Resize once from the prepared source to the measured delivery
            # canvas instead of feeding two serial interpolation passes to VAE.
            "7": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["3", 0], "resize_type": "scale dimensions",
                    "resize_type.width": ["6", 0], "resize_type.height": ["6", 1],
                    "resize_type.crop": "disabled", "scale_method": "lanczos",
                },
            },
            "8": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "ltx-2.3-22b-distilled_video_vae.safetensors"},
            },
            "9": {
                "class_type": "VAEEncodeTiled",
                "inputs": {
                    "pixels": ["7", 0], "vae": ["8", 0], "tile_size": 512,
                    "overlap": 64, "temporal_size": 32, "temporal_overlap": 8,
                },
            },
            "10": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": "ltx-2.3-22b-dev-Q4_K_M.gguf"},
            },
            "11": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["10", 0],
                    "lora_name": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
                    "strength_model": 0.5,
                },
            },
            "12": {
                "class_type": "LTXICLoRALoaderModelOnly",
                "inputs": {
                    "model": ["11", 0], "lora_name": PIXEL_SPATIAL_LORAS[scale],
                    "strength_model": 1.0,
                },
            },
            "13": {
                "class_type": "LTXAVTextEncoderLoader",
                "inputs": {
                    "text_encoder": "comfy_gemma_3_12B_it.safetensors",
                    "ckpt_name": "ltx-2.3-22b-dev.safetensors", "device": "default",
                },
            },
            "14": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["13", 0]}},
            "15": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["13", 0]},
            },
            "16": {
                "class_type": "LTXVConditioning",
                "inputs": {
                    "positive": ["14", 0], "negative": ["15", 0], "frame_rate": ["2", 2],
                },
            },
            "17": {
                "class_type": "LTXAddVideoICLoRAGuide",
                "inputs": {
                    "positive": ["16", 0], "negative": ["16", 1], "vae": ["8", 0],
                    "latent": ["9", 0], "image": ["7", 0], "frame_idx": 0,
                    "strength": 1.0, "latent_downscale_factor": ["12", 1],
                    "crop": "disabled", "use_tiled_encode": True,
                    "tile_size": 256, "tile_overlap": 64,
                },
            },
            "18": {
                "class_type": "LTXVAudioVAELoader",
                "inputs": {"ckpt_name": "ltx-2.3-22b-dev.safetensors"},
            },
            "22": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["12", 0], "positive": ["17", 0],
                    "negative": ["17", 1], "cfg": 1.0,
                },
            },
            "23": {"class_type": "RandomNoise", "inputs": {"noise_seed": actual_seed}},
            "24": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_cfg_pp"}},
            "25": {"class_type": "ManualSigmas", "inputs": {"sigmas": self.sigmas}},
            "26": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["23", 0], "guider": ["22", 0], "sampler": ["24", 0],
                    "sigmas": ["25", 0], "latent_image": ["21", 0],
                },
            },
            "27": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["26", 0]}},
            "28": {
                "class_type": "LTXVCropGuides",
                "inputs": {
                    "positive": ["17", 0], "negative": ["17", 1], "latent": ["27", 0],
                },
            },
            "29": {
                "class_type": "LTXVTiledVAEDecode",
                "inputs": {
                    "vae": ["8", 0], "latents": ["28", 2], "horizontal_tiles": 2,
                    "vertical_tiles": 2, "overlap": 6, "last_frame_fix": False,
                    "working_device": "auto", "working_dtype": "auto",
                },
            },
            "31": {
                "class_type": "CreateVideo",
                "inputs": {"images": ["29", 0], "fps": ["2", 2]},
            },
            "32": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["31", 0], "filename_prefix": filename_prefix,
                    "format": "mp4", "codec": "h264",
                },
            },
        }

        if has_audio:
            graph.update({
                "19": {
                    "class_type": "VAEEncodeAudio",
                    "inputs": {"audio": ["2", 1], "vae": ["18", 0]},
                },
                "20": {
                    "class_type": "LTXVSetAudioRefTokens",
                    "inputs": {
                        "positive": ["17", 0], "negative": ["17", 1],
                        "audio_latent": ["19", 0],
                    },
                },
                "21": {
                    "class_type": "LTXVConcatAVLatent",
                    "inputs": {"video_latent": ["17", 2], "audio_latent": ["20", 2]},
                },
                "30": {
                    "class_type": "LTXVAudioVAEDecode",
                    "inputs": {"samples": ["27", 1], "audio_vae": ["18", 0]},
                },
            })
            graph["31"]["inputs"]["audio"] = ["30", 0]
        else:
            graph.update({
                "19": {"class_type": "LTXFloatToInt", "inputs": {"a": ["2", 2]}},
                "20": {
                    "class_type": "LTXVEmptyLatentAudio",
                    "inputs": {
                        "frames_number": ["6", 2], "frame_rate": ["19", 0],
                        "batch_size": 1, "audio_vae": ["18", 0],
                    },
                },
                "21": {
                    "class_type": "LTXVConcatAVLatent",
                    "inputs": {"video_latent": ["17", 2], "audio_latent": ["20", 0]},
                },
            })

        return graph, {
            "engine": "LTX 2.3 Pixel Spatial Upscaler",
            "workflow": "isolated-distilled-gguf",
            "base_model": "ltx-2.3-22b-dev-Q4_K_M.gguf",
            "distilled_lora": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            "pixel_spatial_lora": PIXEL_SPATIAL_LORAS[scale],
            "scale": scale,
            "source_dimensions": [source_width, source_height],
            "source_preparation_scale": round(preparation_scale, 6),
            "planned_output_dimensions": [output_width, output_height],
            "max_output_side": self.max_output_side,
            "audio_mode": "frozen-source" if has_audio else "silent",
            "seed": actual_seed,
            "steps": 8,
        }
