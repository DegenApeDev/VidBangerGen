from apps.api.pixel_upscale import PixelSpatialUpscaleAdapter


def test_pixel_spatial_graph_layers_distilled_and_x2_loras_over_dev_gguf():
    graph, metadata = PixelSpatialUpscaleAdapter().build(
        video_name="source.mp4",
        prompt="A detailed cinematic source shot",
        negative_prompt="flicker",
        scale=2,
        source_width=1280,
        source_height=720,
        has_audio=True,
        seed=42,
        filename_prefix="vbg/test/upscale",
    )

    assert graph["10"]["inputs"]["unet_name"] == "ltx-2.3-22b-dev-Q4_K_M.gguf"
    assert graph["11"]["inputs"]["model"] == ["10", 0]
    assert graph["11"]["inputs"]["lora_name"] == (
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    )
    assert graph["12"]["inputs"]["model"] == ["11", 0]
    assert "pixel-spatial-upscaler-x2" in graph["12"]["inputs"]["lora_name"]
    assert graph["17"]["inputs"]["latent_downscale_factor"] == ["12", 1]
    assert graph["9"]["class_type"] == "VAEEncodeTiled"
    assert graph["17"]["inputs"]["use_tiled_encode"] is True
    assert graph["20"]["class_type"] == "LTXVSetAudioRefTokens"
    assert graph["31"]["inputs"]["audio"] == ["30", 0]
    assert metadata["audio_mode"] == "frozen-source"
    assert metadata["planned_output_dimensions"] == [1536, 896]


def test_pixel_spatial_graph_uses_silent_av_latent_without_source_audio():
    graph, metadata = PixelSpatialUpscaleAdapter().build(
        video_name="silent.mp4",
        prompt="Preserve the source",
        negative_prompt="flicker",
        scale=4,
        source_width=384,
        source_height=256,
        has_audio=False,
    )

    assert "pixel-spatial-upscaler-x4" in graph["12"]["inputs"]["lora_name"]
    assert graph["20"]["class_type"] == "LTXVEmptyLatentAudio"
    assert "audio" not in graph["31"]["inputs"]
    assert metadata["audio_mode"] == "silent"
    assert metadata["planned_output_dimensions"] == [1536, 1024]


def test_pixel_spatial_graph_never_enlarges_past_its_vram_safe_long_edge():
    _, metadata = PixelSpatialUpscaleAdapter().build(
        video_name="large.mp4",
        prompt="Preserve the source",
        negative_prompt="flicker",
        scale=2,
        source_width=1920,
        source_height=1080,
        has_audio=True,
    )

    assert max(metadata["planned_output_dimensions"]) <= 1536
    assert metadata["source_preparation_scale"] < 1
