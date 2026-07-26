from __future__ import annotations

import secrets
from typing import Any


class InOutpaintAdapter:
    """Build the official two-stage LTX 2.3 masked-edit recipe with GGUF.

    Source and mask geometry are prepared locally.  Stage one generates the
    masked region at half resolution; the result is blended with the untouched
    source, encoded at 2x, refined in stage two, and blended again.  This keeps
    the missing ImagePadForOutpaintTargetSize convenience node out of the graph
    without changing the model-side in/outpainting contract.
    """

    max_frames = 121
    max_side = 1024
    stage_one_sigmas = (
        "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
    )
    stage_two_sigmas = "0.7250, 0.4219, 0.0"

    def validate_source(self, info: dict[str, Any]) -> None:
        frames = int(info.get("frames") or 0)
        if frames <= 0:
            raise ValueError("In/Outpainting source must have a countable video stream")
        if frames > self.max_frames:
            raise ValueError(
                f"In/Outpainting supports up to {self.max_frames} frames per pass; "
                f"this clip has {frames} frames"
            )
        if (frames - 1) % 8:
            raise ValueError(
                f"In/Outpainting requires an LTX frame count of 8n+1; this clip has {frames}"
            )

    def build(
        self,
        *,
        video_name: str,
        mask_name: str,
        prompt: str = "",
        negative_prompt: str = "",
        strength: float = 1.0,
        dilation: int = 15,
        seed: int = -1,
        filename_prefix: str = "vbg/in-outpaint/output",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not 0.5 <= strength <= 1.25:
            raise ValueError("In/Outpainting strength must be between 0.5 and 1.25")
        if not 0 <= dilation <= 15:
            raise ValueError("Mask dilation must be between 0 and 15 pixels at stage one")
        actual_seed = seed if seed >= 0 else secrets.randbelow(2**63 - 3)
        positive = prompt.strip()
        negative = negative_prompt.strip() or (
            "worst quality, inconsistent motion, blurry, jittery, duplicated objects, "
            "visible seams, hard mask boundary, halo, color mismatch, speech"
        )
        graph: dict[str, Any] = {
            "1": {"class_type": "LoadVideo", "inputs": {"file": video_name}},
            "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
            "3": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["2", 0], "resize_type": "scale longer dimension",
                    "resize_type.longer_size": self.max_side, "scale_method": "lanczos",
                },
            },
            "4": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["3", 0], "resize_type": "scale to multiple",
                    "resize_type.multiple": 64, "scale_method": "lanczos",
                },
            },
            "5": {"class_type": "LoadVideo", "inputs": {"file": mask_name}},
            "6": {"class_type": "GetVideoComponents", "inputs": {"video": ["5", 0]}},
            "7": {
                "class_type": "ImageToMask",
                "inputs": {"image": ["6", 0], "channel": "red"},
            },
            "8": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["7", 0], "resize_type": "match size",
                    "resize_type.match": ["4", 0], "resize_type.crop": "center",
                    "scale_method": "area",
                },
            },
            "9": {
                "class_type": "LTXVDilateVideoMask",
                "inputs": {
                    "mask": ["8", 0], "spatial_radius": dilation * 2,
                    "temporal_radius": 0,
                },
            },
            "10": {
                "class_type": "LTXVInpaintPreprocess",
                "inputs": {"images": ["4", 0], "mask": ["9", 0]},
            },
            "11": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["4", 0], "resize_type": "scale by multiplier",
                    "resize_type.multiplier": 0.5, "scale_method": "lanczos",
                },
            },
            "12": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["7", 0], "resize_type": "match size",
                    "resize_type.match": ["11", 0], "resize_type.crop": "center",
                    "scale_method": "area",
                },
            },
            "13": {
                "class_type": "LTXVDilateVideoMask",
                "inputs": {
                    "mask": ["12", 0], "spatial_radius": dilation,
                    "temporal_radius": 0,
                },
            },
            "14": {
                "class_type": "LTXVInpaintPreprocess",
                "inputs": {"images": ["11", 0], "mask": ["13", 0]},
            },
            "15": {"class_type": "GetImageSize", "inputs": {"image": ["14", 0]}},
            "16": {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {
                    "width": ["15", 0], "height": ["15", 1],
                    "length": ["15", 2], "batch_size": 1,
                },
            },
            "17": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "ltx-2.3-22b-distilled_video_vae.safetensors"},
            },
            "18": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": "ltx-2.3-22b-dev-Q4_K_M.gguf"},
            },
            "19": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["18", 0],
                    "lora_name": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
                    "strength_model": 0.5,
                },
            },
            "20": {
                "class_type": "LTXICLoRALoaderModelOnly",
                "inputs": {
                    "model": ["19", 0],
                    "lora_name": "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
                    "strength_model": strength,
                },
            },
            "21": {
                "class_type": "LTXAVTextEncoderLoader",
                "inputs": {
                    "text_encoder": "comfy_gemma_3_12B_it.safetensors",
                    "ckpt_name": "ltx-2.3-22b-dev.safetensors", "device": "default",
                },
            },
            "22": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": positive, "clip": ["21", 0]},
            },
            "23": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative, "clip": ["21", 0]},
            },
            "24": {
                "class_type": "LTXVConditioning",
                "inputs": {
                    "positive": ["22", 0], "negative": ["23", 0],
                    "frame_rate": ["2", 2],
                },
            },
            "25": {
                "class_type": "LTXAddVideoICLoRAGuideAdvanced",
                "inputs": {
                    "positive": ["24", 0], "negative": ["24", 1],
                    "vae": ["17", 0], "latent": ["16", 0], "image": ["14", 0],
                    "frame_idx": 0, "strength": 1.0, "latent_downscale_factor": 1,
                    "crop": "disabled", "use_tiled_encode": False,
                    "tile_size": 256, "tile_overlap": 64, "attention_strength": 1.0,
                },
            },
            "26": {
                "class_type": "LTXVAudioVAELoader",
                "inputs": {"ckpt_name": "ltx-2.3-22b-dev.safetensors"},
            },
            "27": {"class_type": "LTXFloatToInt", "inputs": {"a": ["2", 2]}},
            "28": {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": ["15", 2], "frame_rate": ["27", 0],
                    "batch_size": 1, "audio_vae": ["26", 0],
                },
            },
            "29": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": ["25", 2], "audio_latent": ["28", 0]},
            },
            "30": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["20", 0], "positive": ["25", 0],
                    "negative": ["25", 1], "cfg": 1.0,
                },
            },
            "31": {"class_type": "RandomNoise", "inputs": {"noise_seed": actual_seed}},
            "32": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler_ancestral_cfg_pp"},
            },
            "33": {
                "class_type": "ManualSigmas",
                "inputs": {"sigmas": self.stage_one_sigmas},
            },
            "34": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["31", 0], "guider": ["30", 0],
                    "sampler": ["32", 0], "sigmas": ["33", 0],
                    "latent_image": ["29", 0],
                },
            },
            "35": {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {"av_latent": ["34", 1]},
            },
            "36": {
                "class_type": "LTXVCropGuides",
                "inputs": {
                    "positive": ["25", 0], "negative": ["25", 1],
                    "latent": ["35", 0],
                },
            },
            "37": {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["36", 2], "vae": ["17", 0], "tile_size": 512,
                    "overlap": 64, "temporal_size": 512, "temporal_overlap": 64,
                },
            },
            "38": {
                "class_type": "LTXVLaplacianPyramidBlend",
                "inputs": {
                    "image_a": ["37", 0], "image_b": ["14", 0],
                    "mask": ["13", 0], "trim_to_shortest": True,
                    "mask_low_res_dilation": 5,
                },
            },
            "39": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["38", 0], "resize_type": "scale by multiplier",
                    "resize_type.multiplier": 2.0, "scale_method": "lanczos",
                },
            },
            "40": {
                "class_type": "VAEEncodeTiled",
                "inputs": {
                    "pixels": ["39", 0], "vae": ["17", 0], "tile_size": 512,
                    "overlap": 64, "temporal_size": 512, "temporal_overlap": 64,
                },
            },
            "41": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": ["40", 0], "audio_latent": ["35", 1]},
            },
            "42": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["20", 0], "positive": ["36", 0],
                    "negative": ["36", 1], "cfg": 1.0,
                },
            },
            "43": {"class_type": "RandomNoise", "inputs": {"noise_seed": actual_seed + 1}},
            "44": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler_cfg_pp"},
            },
            "45": {
                "class_type": "ManualSigmas",
                "inputs": {"sigmas": self.stage_two_sigmas},
            },
            "46": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["43", 0], "guider": ["42", 0],
                    "sampler": ["44", 0], "sigmas": ["45", 0],
                    "latent_image": ["41", 0],
                },
            },
            "47": {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {"av_latent": ["46", 0]},
            },
            "48": {
                "class_type": "LTXVTiledVAEDecode",
                "inputs": {
                    "vae": ["17", 0], "latents": ["47", 0],
                    "horizontal_tiles": 2, "vertical_tiles": 2, "overlap": 6,
                    "last_frame_fix": False, "working_device": "auto",
                    "working_dtype": "auto",
                },
            },
            "49": {
                "class_type": "LTXVLaplacianPyramidBlend",
                "inputs": {
                    "image_a": ["48", 0], "image_b": ["10", 0],
                    "mask": ["9", 0], "trim_to_shortest": True,
                    "mask_low_res_dilation": 6,
                },
            },
            "50": {
                "class_type": "CreateVideo",
                "inputs": {"images": ["49", 0], "fps": ["2", 2]},
            },
            "51": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["50", 0], "filename_prefix": filename_prefix,
                    "format": "mp4", "codec": "h264",
                },
            },
        }
        return graph, {
            "engine": "LTX 2.3 Creative Lab — Video In/Outpainting",
            "workflow": "official-two-stage-masked-edit-gguf",
            "base_model": "ltx-2.3-22b-dev-Q4_K_M.gguf",
            "lora": "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
            "distilled_lora": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            "strength": strength, "mask_dilation": dilation,
            "stage_one": {"steps": 8, "guidance": 1.0, "sigmas": self.stage_one_sigmas},
            "stage_two": {"steps": 2, "guidance": 1.0, "sigmas": self.stage_two_sigmas},
            "max_frames": self.max_frames, "max_side": self.max_side,
            "source_audio_policy": "preserve", "audio_mode": "source-restored-locally",
            "seed": actual_seed, "prompt": positive, "negative_prompt": negative,
        }
