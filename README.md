# VidBangerGen

VidBangerGen is a local-first production studio for LTX 2.3 video generation.
It turns a creative brief into concepts, storyboards, best-of-N candidates,
selected final shots, and platform-ready exports while keeping projects and
media under the operator's control.

The application connects to one or more existing ComfyUI workers over HTTP. The
web/API controller can run on the inference host or on a separate computer.
Model weights are not bundled or downloaded automatically.

## Features

- Structured creative briefs for short-form and long-form social platforms.
- Assisted storyboarding through an optional local Ollama model, plus a fully
  manual prompt mode.
- Parallel candidate generation across configured ComfyUI workers.
- Durable SQLite jobs with retry, cancellation, restart recovery, and live
  progress.
- Text-to-video, image-to-video, multi-shot continuity, and keeper review.
- Reusable brand, product, character, wardrobe, location, and visual-bible
  references.
- LTX 2.3 Creative Lab workflows for Ingredients, in/outpainting, restoration,
  relighting, effects, cinemagraph, Foley, LipDub, and Pixel Spatial upscale.
- Local FFmpeg finishing with captions, voiceover, music, logo overlays,
  transitions, loudness normalization, and H.264 output.
- Optional local visual scoring that never replaces human review.
- Capability-aware routing for shared or exclusive inference workers.

## Requirements

The controller requires:

- Python 3.11 or newer;
- Node.js 20 or newer and npm;
- FFmpeg and ffprobe; and
- one or more reachable ComfyUI workers with the required LTX nodes and models.

The active workflow uses the LTX 2.3 22B Q4_K_M GGUF model and is demanding.
The original integration was validated on 24 GB NVIDIA cards. ComfyUI can run
on supported AMD GPUs through ROCm, but the complete LTX node stack must be
validated on the specific AMD GPU and operating system. NVIDIA telemetry,
the dormant dual-CUDA FP8 profile, and optional RTX Video Super Resolution are
NVIDIA-only.

## Quick start

Clone the repository and run the setup wizard:

```bash
git clone https://github.com/DegenApeDev/VidBangerGen.git
cd VidBangerGen
python3 scripts/setup.py setup
```

The wizard:

1. detects local ComfyUI workers on ports 8188 and 8189 or validates URLs you
   provide;
2. checks the required LTX node inventory;
3. records which workers accept conditioning uploads;
4. distinguishes shared workers from exclusively managed workers;
5. writes a private `.env`; and
6. optionally installs Python/Node dependencies and builds the web client.

Start the development services:

```bash
npm run dev
```

Open <http://127.0.0.1:5173>. For a built frontend served by the API, open
<http://127.0.0.1:8090/studio/>.

Reconfigure or validate an installation with:

```bash
.venv/bin/vbg setup --force
.venv/bin/vbg doctor
```

See [the setup guide](docs/SETUP.md) for unattended and remote-worker
configuration.

## ComfyUI preparation

VidBangerGen deliberately does not modify ComfyUI, install models, or restart
workers. Each configured endpoint must provide the node inventory checked by
the setup wizard, including:

- `ComfyUI-LTXVideo`;
- a compatible GGUF loader;
- the LTX 2.3 Q4_K_M model and associated text/audio/VAE components; and
- the optional IC-LoRAs required by any Creative Lab modes you enable.

The versioned API workflow and mappings live in:

- `apps/api/workflows/ltx23_base_graph.json`
- `apps/api/workflows/ltx23.json`
- `apps/api/workflows/creative_lab.json`

For multi-worker, service, shared-worker, and optional NVIDIA finishing-node
guidance, see [the deployment guide](docs/DEPLOYMENT.md).

## Configuration

Configuration is generated in `.env` and read by
`apps/api/config.py`. Launch-time environment variables take precedence.
The minimum unattended configuration is:

```dotenv
VBG_SETUP_COMPLETE=true
VBG_DATA_DIR=./data
VBG_COMFYUI_URLS=http://127.0.0.1:8188
VBG_UPLOAD_CAPABLE_URLS=http://127.0.0.1:8188
```

Use `VBG_EXECUTION_TARGETS_JSON` to describe named ComfyUI pools or optional
Video2X workers. Never put credentials in a ComfyUI URL.

Runtime state is stored under `data/` by default:

- generated media in `data/output`;
- uploads in `data/uploads`; and
- projects and durable jobs in `data/vidbangergen.sqlite3`.

The `.env`, runtime data, generated web bundle, caches, and local virtual
environments are ignored by Git.

## Development and verification

```bash
npm test
npm run build
.venv/bin/python scripts/check_installation.py
```

The worker integration check requires configured, reachable ComfyUI endpoints.
The normal test suite uses isolated temporary state and does not submit GPU
jobs.

## Security and privacy

VidBangerGen is intended for a trusted local network. ComfyUI commonly exposes
an unauthenticated API, so do not publish worker ports directly to the
internet. Use a private network or an authenticated reverse proxy.

Before sharing a fork:

- keep `.env`, `data/`, model inventories, logs, and generated media out of
  source control;
- replace machine-specific service paths and hostnames;
- scan the complete Git history, not only the latest files; and
- start a fresh history if secrets or personal infrastructure details were ever
  committed.

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities and deployment
guidance.

## Third-party components

This repository does not redistribute LTX model weights. Models, ComfyUI,
custom nodes, Ollama models, FFmpeg, and other dependencies retain their own
licenses and terms. See [THIRD_PARTY.md](THIRD_PARTY.md).

## License

Copyright 2026 VidBangerGen contributors.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
