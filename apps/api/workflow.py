from __future__ import annotations

import copy
import json
import secrets
from pathlib import Path
from typing import Any

from .config import Settings


class WorkflowError(RuntimeError):
    pass


class WorkflowStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.manifest = json.loads(settings.workflow_manifest.read_text())
        self._cached_graph: dict[str, Any] | None = None

    @property
    def version(self) -> str:
        return self.manifest["version"]

    def available_capabilities(self) -> set[str]:
        available = set(self.manifest.get("available_capabilities", []))
        if "fp8_exclusive_inference" in available:
            target = self.settings.exclusive_comfy_url
            urls = set(self.settings.comfyui_urls)
            peers = urls - ({target} if target else set())
            if (
                not target or target not in urls
                or not peers.issubset(set(self.settings.managed_comfy_urls))
            ):
                available.discard("fp8_exclusive_inference")
        return available

    def load_graph(self, refresh: bool = False) -> dict[str, Any]:
        if self._cached_graph is not None and not refresh:
            return copy.deepcopy(self._cached_graph)
        source = self.manifest.get("source", {})
        if source.get("kind") != "file":
            raise WorkflowError(
                "Workflow manifests must use a local file source. "
                "ComfyUI is an inference worker, not a runtime configuration store."
            )
        try:
            path = Path(source["path"]).expanduser()
        except (KeyError, TypeError) as exc:
            raise WorkflowError("Local workflow source is missing its path") from exc
        if not path.is_absolute():
            path = self.settings.workflow_manifest.parent / path
        try:
            raw = path.read_text()
        except OSError as exc:
            raise WorkflowError(f"Unable to load local workflow {path}: {exc}") from exc
        try:
            graph = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"Workflow is not valid JSON: {exc}") from exc
        self.validate(graph)
        self._cached_graph = graph
        return copy.deepcopy(graph)

    def profile_status(self, profile: str) -> dict[str, Any]:
        config = self.manifest.get("profiles", {}).get(profile)
        if not config:
            raise WorkflowError(f"Unknown generation profile: {profile}")
        required = sorted(
            {
                stage["required_capability"]
                for stage in (
                    config.get("model_loader", {}),
                    config.get("refinement", {}), config.get("final_upscale", {}),
                    config.get("encoder", {}), config.get("audio_seed", {}),
                )
                if stage.get("required_capability")
                and not stage.get("required_when_used")
            }
        )
        available = self.available_capabilities()
        missing = [value for value in required if value not in available]
        return {
            "name": profile, "ready": not missing, "required_capabilities": required,
            "available_capabilities": sorted(available), "missing_capabilities": missing,
            "configuration": config,
        }

    def validate(self, graph: dict[str, Any]) -> None:
        errors: list[str] = []
        for name, spec in self.manifest["nodes"].items():
            node = graph.get(spec["id"])
            if node is None:
                if not spec.get("optional"):
                    errors.append(f"{name}: missing node {spec['id']}")
                continue
            if node.get("class_type") != spec["class_type"]:
                errors.append(
                    f"{name}: node {spec['id']} expected {spec['class_type']}, "
                    f"got {node.get('class_type')}"
                )
        if errors:
            raise WorkflowError("Workflow manifest mismatch: " + "; ".join(errors))


class LTXWorkflowAdapter:
    def __init__(self, store: WorkflowStore):
        self.store = store
        self.nodes = store.manifest["nodes"]
        self.constraints = store.manifest["constraints"]

    def _node(self, graph: dict[str, Any], name: str) -> dict[str, Any]:
        return graph[self.nodes[name]["id"]]

    def _next_id(self, graph: dict[str, Any]) -> str:
        candidate = max(int(node_id) for node_id in graph if node_id.isdigit()) + 1
        while str(candidate) in graph:
            candidate += 1
        return str(candidate)

    def frame_count(self, duration_seconds: float, fps: float) -> int:
        modulus = int(self.constraints.get("frame_modulus", 8))
        offset = int(self.constraints.get("frame_offset", 1))
        minimum = offset + modulus
        desired = max(minimum, round(duration_seconds * fps))
        return max(minimum, ((desired - offset + modulus // 2) // modulus) * modulus + offset)

    def build(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        duration_seconds: float,
        fps: float,
        seed: int,
        filename_prefix: str,
        image_name: str | None = None,
        ic_lora_strength: float = 0.5,
        image_condition_strength: float = 0.9,
        profile: str = "motion-draft-4x3",
        video_name: str | None = None,
        motion_guide_strength: float = 0.85,
        retake_video_name: str | None = None,
        retake_start_seconds: float = 0.0,
        retake_end_seconds: float | None = None,
        audio_name: str | None = None,
        audio_seed_seconds: float = 4.0,
        include_audio: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        graph = self.store.load_graph()
        actual_seed = seed if seed >= 0 else secrets.randbelow(2**63 - 2)
        frames = self.frame_count(duration_seconds, fps)
        profiles = self.store.manifest.get("profiles", {})
        if profile not in profiles:
            raise WorkflowError(f"Unknown generation profile: {profile}")
        profile_config = profiles[profile]
        status = self.store.profile_status(profile)
        missing_capabilities = [
            value for value in status["missing_capabilities"]
            if value != "audio_video_time_mask"
        ]
        if missing_capabilities:
            raise WorkflowError(
                f"Profile {profile} requires unavailable worker capabilities: "
                + ", ".join(missing_capabilities)
            )

        latent = self._node(graph, "latent")["inputs"]
        latent["width"], latent["height"] = width, height
        self._node(graph, "frames")["inputs"]["value"] = frames
        self._node(graph, "fps")["inputs"]["value"] = fps
        self._node(graph, "positive_gemma")["inputs"]["prompt"] = prompt
        self._node(graph, "positive_clip")["inputs"]["text"] = prompt
        self._node(graph, "negative_clip")["inputs"]["text"] = negative_prompt
        optional_negative = graph.get(self.nodes["negative_gemma"]["id"])
        if optional_negative:
            optional_negative["inputs"]["prompt"] = negative_prompt
        self._node(graph, "seed_stage_one")["inputs"]["noise_seed"] = actual_seed
        self._node(graph, "seed_stage_two")["inputs"]["noise_seed"] = actual_seed + 1
        self._node(graph, "motion_sigmas")["inputs"]["sigmas"] = profile_config["motion"]["sigmas"]
        self._node(graph, "upscale_sigmas")["inputs"]["sigmas"] = profile_config[
            "latent_upscale"
        ]["sigmas"]
        self._node(graph, "output")["inputs"]["filename_prefix"] = filename_prefix
        if not include_audio:
            # Prompting for silence is probabilistic in a joint AV model. For
            # the explicit Silent mode, disconnect decoded audio from the
            # container so the delivered MP4 is guaranteed to have no voice.
            self._node(graph, "create_video")["inputs"].pop("audio", None)

        model_loader = profile_config.get("model_loader", {})
        if model_loader.get("class_type"):
            model_node = self._node(graph, "model")
            model_node["class_type"] = model_loader["class_type"]
            model_node["inputs"] = copy.deepcopy(model_loader.get("inputs", {}))

        if image_name:
            self._node(graph, "image")["inputs"]["image"] = image_name
            self._node(graph, "image_bypass")["inputs"]["value"] = False
            self._node(graph, "image_condition_stage_one")["inputs"][
                "strength"
            ] = image_condition_strength
            self._node(graph, "image_condition_stage_two")["inputs"][
                "strength"
            ] = image_condition_strength
        else:
            # Comfy validates output ancestors even when the custom bypass flag is true.
            # Remove the dormant image branch so fresh workers need no hidden example.png.
            self._node(graph, "av_concat_stage_one")["inputs"]["video_latent"] = [
                self.nodes["latent"]["id"], 0
            ]
            self._node(graph, "av_concat_stage_two")["inputs"]["video_latent"] = [
                self.nodes["latent_upsampler"]["id"], 0
            ]
            for name in (
                "image", "image_resize", "image_preprocess", "image_bypass",
                "image_condition_stage_one", "image_condition_stage_two",
            ):
                graph.pop(self.nodes[name]["id"], None)

        if image_name or video_name:
            lora_id = self._next_id(graph)
            lora = self.store.manifest["ic_lora"]
            graph[lora_id] = {
                "class_type": lora["class_type"],
                "inputs": {
                    "model": [self.nodes["model"]["id"], 0],
                    "lora_name": lora["name"],
                    "strength_model": ic_lora_strength,
                },
            }
            self._node(graph, "guider_stage_one")["inputs"]["model"] = [lora_id, 0]
            self._node(graph, "guider_stage_two")["inputs"]["model"] = [lora_id, 0]

        motion_guide_metadata: dict[str, Any] | None = None
        if video_name:
            available = set(self.store.manifest.get("available_capabilities", []))
            if "motion_segment_guide" not in available:
                raise WorkflowError(
                    "Motion-tail continuity requires unavailable capability: motion_segment_guide"
                )
            load_video_id = self._next_id(graph)
            graph[load_video_id] = {
                "class_type": "LoadVideo", "inputs": {"file": video_name},
            }
            components_id = self._next_id(graph)
            graph[components_id] = {
                "class_type": "GetVideoComponents",
                "inputs": {"video": [load_video_id, 0]},
            }
            guide_id = self._next_id(graph)
            graph[guide_id] = {
                "class_type": "LTXVAddGuide",
                "inputs": {
                    "positive": [self.nodes["conditioning"]["id"], 0],
                    "negative": [self.nodes["conditioning"]["id"], 1],
                    "vae": [self.nodes["video_vae"]["id"], 0],
                    "latent": [self.nodes["latent"]["id"], 0],
                    "image": [components_id, 0],
                    "frame_idx": 0,
                    "strength": motion_guide_strength,
                },
            }
            self._node(graph, "av_concat_stage_one")["inputs"]["video_latent"] = [guide_id, 2]
            self._node(graph, "guider_stage_one")["inputs"]["positive"] = [guide_id, 0]
            self._node(graph, "guider_stage_one")["inputs"]["negative"] = [guide_id, 1]
            # LTXVAddGuide appends reference tokens to the first-stage video
            # latent. The sampler output is a combined AV NestedTensor, so it
            # must first pass through LTXVSeparateAVLatent; LTXVCropGuides then
            # receives the ordinary video LATENT before spatial upscaling.
            # Cropping after the second sampler is too late because the latent
            # upscaler has already propagated guide tokens into the delivery
            # sequence.
            crop_id = self._next_id(graph)
            graph[crop_id] = {
                "class_type": "LTXVCropGuides",
                "inputs": {
                    "positive": [guide_id, 0], "negative": [guide_id, 1],
                    "latent": [self.nodes["separate_stage_one"]["id"], 0],
                },
            }
            self._node(graph, "latent_upsampler")["inputs"]["samples"] = [crop_id, 2]
            self._node(graph, "guider_stage_two")["inputs"]["positive"] = [crop_id, 0]
            self._node(graph, "guider_stage_two")["inputs"]["negative"] = [crop_id, 1]
            motion_guide_metadata = {
                "input": video_name, "frames": 17, "strength": motion_guide_strength,
                "node": "LTXVAddGuide", "frame_index": 0,
                "crop_node": "LTXVCropGuides",
                "crop_stage": "post_av_separation_pre_upscale",
            }

        retake_metadata: dict[str, Any] | None = None
        if retake_video_name:
            if audio_name or video_name or image_name:
                raise WorkflowError(
                    "Retake source cannot be combined with image, motion-tail, or audio-seed inputs"
                )
            available = set(self.store.manifest.get("available_capabilities", []))
            if "audio_video_time_mask" not in available:
                raise WorkflowError("AV retakes require audio_video_time_mask")
            total_duration = frames / fps
            start = max(0.0, min(float(retake_start_seconds), total_duration))
            end = max(start, min(float(retake_end_seconds or total_duration), total_duration))
            if end <= start:
                raise WorkflowError("Retake interval must have positive duration")
            load_source_id = self._next_id(graph)
            graph[load_source_id] = {
                "class_type": "LoadVideo", "inputs": {"file": retake_video_name},
            }
            source_components_id = self._next_id(graph)
            graph[source_components_id] = {
                "class_type": "GetVideoComponents",
                "inputs": {"video": [load_source_id, 0]},
            }
            scale_source_id = self._next_id(graph)
            graph[scale_source_id] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": [source_components_id, 0], "upscale_method": "lanczos",
                    "width": width, "height": height, "crop": "center",
                },
            }
            encode_video_id = self._next_id(graph)
            graph[encode_video_id] = {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": [scale_source_id, 0],
                    "vae": [self.nodes["video_vae"]["id"], 0],
                },
            }
            encode_source_audio_id = self._next_id(graph)
            graph[encode_source_audio_id] = {
                "class_type": "LTXVAudioVAEEncode",
                "inputs": {
                    "audio": [source_components_id, 1],
                    "audio_vae": [self.nodes["audio_vae"]["id"], 0],
                },
            }
            source_av_id = self._next_id(graph)
            graph[source_av_id] = {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {
                    "video_latent": [encode_video_id, 0],
                    "audio_latent": [encode_source_audio_id, 0],
                },
            }
            retake_mask_id = self._next_id(graph)
            stage_one_inputs = self._node(graph, "guider_stage_one")["inputs"]
            graph[retake_mask_id] = {
                "class_type": "LTXVSetAudioVideoMaskByTime",
                "inputs": {
                    "av_latent": [source_av_id, 0],
                    "positive": stage_one_inputs.get(
                        "positive", [self.nodes["conditioning"]["id"], 0]
                    ),
                    "negative": stage_one_inputs.get(
                        "negative", [self.nodes["conditioning"]["id"], 1]
                    ),
                    "model": stage_one_inputs["model"],
                    "vae": [self.nodes["video_vae"]["id"], 0],
                    "audio_vae": [self.nodes["audio_vae"]["id"], 0],
                    "start_time": start, "end_time": end, "video_fps": fps,
                    "mask_video": True, "mask_audio": True,
                    "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0,
                    "slope_len": 5,
                },
            }
            self._node(graph, "guider_stage_one")["inputs"]["positive"] = [
                retake_mask_id, 0
            ]
            self._node(graph, "guider_stage_one")["inputs"]["negative"] = [
                retake_mask_id, 1
            ]
            self._node(graph, "sampler_stage_one")["inputs"]["latent_image"] = [
                retake_mask_id, 2
            ]
            retake_metadata = {
                "input": retake_video_name, "start_seconds": start, "end_seconds": end,
                "preserved_outside_interval": True, "regenerate_audio": True,
                "node": "LTXVSetAudioVideoMaskByTime",
            }

        audio_seed_metadata: dict[str, Any] | None = None
        if audio_name:
            audio_config = profile_config.get("audio_seed") or {}
            capability = audio_config.get("required_capability", "audio_video_time_mask")
            available = set(self.store.manifest.get("available_capabilities", []))
            if capability not in available:
                raise WorkflowError(
                    f"Reference audio requires unavailable worker capability: {capability}"
                )
            seed_duration = max(0.01, min(float(audio_seed_seconds), frames / fps))
            load_audio_id = self._next_id(graph)
            graph[load_audio_id] = {
                "class_type": "LoadAudio", "inputs": {"audio": audio_name},
            }
            encode_audio_id = self._next_id(graph)
            graph[encode_audio_id] = {
                "class_type": "LTXVAudioVAEEncode",
                "inputs": {
                    "audio": [load_audio_id, 0],
                    "audio_vae": [self.nodes["audio_vae"]["id"], 0],
                },
            }
            self._node(graph, "av_concat_stage_one")["inputs"]["audio_latent"] = [
                encode_audio_id, 0
            ]
            mask_id = self._next_id(graph)
            stage_one_guider_inputs = self._node(graph, "guider_stage_one")["inputs"]
            graph[mask_id] = {
                "class_type": "LTXVSetAudioVideoMaskByTime",
                "inputs": {
                    "av_latent": [self.nodes["av_concat_stage_one"]["id"], 0],
                    "positive": stage_one_guider_inputs.get(
                        "positive", [self.nodes["conditioning"]["id"], 0]
                    ),
                    "negative": stage_one_guider_inputs.get(
                        "negative", [self.nodes["conditioning"]["id"], 1]
                    ),
                    "model": stage_one_guider_inputs["model"],
                    "vae": [self.nodes["video_vae"]["id"], 0],
                    "audio_vae": [self.nodes["audio_vae"]["id"], 0],
                    "start_time": seed_duration,
                    "end_time": frames / fps,
                    "video_fps": fps,
                    "mask_video": False,
                    "mask_audio": True,
                    "mask_init_value_video": 1.0,
                    "mask_init_value_audio": 0.0,
                    "slope_len": 5,
                },
            }
            self._node(graph, "guider_stage_one")["inputs"]["positive"] = [mask_id, 0]
            self._node(graph, "guider_stage_one")["inputs"]["negative"] = [mask_id, 1]
            self._node(graph, "sampler_stage_one")["inputs"]["latent_image"] = [mask_id, 2]
            audio_seed_metadata = {
                "input": audio_name, "preserved_seconds": seed_duration,
                "generated_from_seconds": seed_duration,
                "generated_through_seconds": frames / fps,
                "node": "LTXVSetAudioVideoMaskByTime",
            }

        metadata = {
            "workflow_version": self.store.version,
            "seed": actual_seed,
            "width": width,
            "height": height,
            "output_width": width * int(self.constraints.get("spatial_upscale", 2)),
            "output_height": height * int(self.constraints.get("spatial_upscale", 2)),
            "output_dimensions_source": "nominal spatial-upscale factor; artifact probe is authoritative",
            "frames": frames,
            "fps": fps,
            "duration_seconds": frames / fps,
            "mode": (
                "retake" if retake_video_name else
                ("v2v" if video_name else ("i2v" if image_name else "t2v"))
            ),
            "profile": profile,
            "model_loader": {
                "kind": profile_config.get("model_loader", {}).get("kind", "workflow-default"),
                "class_type": self._node(graph, "model")["class_type"],
                "exclusive_dual_gpu": bool(
                    profile_config.get("model_loader", {}).get("exclusive_dual_gpu", False)
                ),
            },
            "stages": {
                "motion_steps": profile_config["motion"]["steps"],
                "latent_upscale_steps": profile_config["latent_upscale"]["steps"],
                "refinement_steps": profile_config["refinement"]["steps"],
                "final_upscale": profile_config["final_upscale"],
                "encoder": profile_config["encoder"],
            },
            "capability_gates": [
                stage["required_capability"]
                for stage in (
                    profile_config.get("model_loader", {}),
                    profile_config.get("refinement", {}),
                    profile_config.get("final_upscale", {}),
                    profile_config.get("encoder", {}),
                    profile_config.get("audio_seed", {}),
                )
                if stage.get("required_capability")
            ],
            "audio_seed": audio_seed_metadata,
            "audio_output": "generated" if include_audio else "none",
            "motion_guide": motion_guide_metadata,
            "retake": retake_metadata,
        }
        return graph, metadata
