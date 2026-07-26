# Deployment

VidBangerGen is an application controller for one or more ComfyUI workers. The
API and browser interface can run on the inference machine or on a separate
host. ComfyUI model weights, custom nodes, and GPU drivers remain owned by the
operator and are not installed automatically.

## Application service

The example unit in `deploy/vidbangergen-api.service` assumes:

- the repository is installed at `/opt/vidbangergen`;
- a dedicated `vidbangergen` system user owns that directory;
- private configuration is stored in
  `/etc/vidbangergen/vidbangergen.env`; and
- the web client has already been built with `npm run build`.

Copy and edit the unit for the target host, then install it using the service
manager appropriate for that system. Never commit the real environment file.

The API listens on loopback by default. Put an authenticated TLS reverse proxy
in front of it before exposing it outside a trusted network.

## ComfyUI workers

Every URL in `VBG_COMFYUI_URLS` creates an independent generation worker.
Multiple URLs may refer to separate GPUs on one host or to different hosts.
Each worker must expose the ComfyUI HTTP and WebSocket APIs and contain the
models and custom nodes reported by:

```bash
.venv/bin/vbg doctor
```

For NVIDIA multi-GPU hosts, `deploy/comfyui-gpu1.example.service` demonstrates
an isolated second worker on port 8189. Edit its user, paths, and CUDA device
before installing it. It is an example, not a portable service definition.

AMD/ROCm workers can be configured through the same ComfyUI URLs when the
required LTX nodes and models load successfully. NVIDIA telemetry and optional
RTX Video Super Resolution are unavailable on AMD.

## Shared and exclusive workers

Mark a worker exclusive only when no other person or application submits jobs
to it. VidBangerGen may interrupt an exclusive worker during cancellation.
Shared workers are never globally interrupted; completed work that was
cancelled locally is discarded safely.

The setup wizard records this distinction in:

- `VBG_MANAGED_COMFY_URLS`
- `VBG_INTERRUPT_CAPABLE_URLS`

## Upload-capable workers

Image-, audio-, and video-conditioned workflows require a ComfyUI worker with a
writable input directory. List only verified endpoints in
`VBG_UPLOAD_CAPABLE_URLS`.

Use separate input, output, and user directories for parallel workers. Verify a
small image, WAV, and MP4 upload before enabling conditioned production work.

## Optional NVIDIA finishing nodes

The capability-gated production profile can use NVIDIA RTX Video Super
Resolution and VideoHelperSuite. The installer in
`scripts/install_nvidia_production_nodes.sh` pins reviewed upstream revisions
and requires an explicit operator acknowledgement. Review it before use:

```bash
VBG_NVIDIA_NODES_APPROVED=yes \
  COMFY_DIR=/opt/ComfyUI \
  bash scripts/install_nvidia_production_nodes.sh
```

These nodes are optional. The normal GGUF draft and final profiles do not
require the RTX finishing pass.

## Validation

From the application checkout:

```bash
.venv/bin/vbg doctor
.venv/bin/python scripts/check_installation.py
npm test
npm run build
```

Do not publish the generated `.env`, `data/`, media outputs, database files,
private model inventories, service logs, or a Git history that previously
contained them.
