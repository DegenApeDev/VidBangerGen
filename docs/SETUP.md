# VidBangerGen setup

VidBangerGen is a controller for one or more existing ComfyUI workers. The
controller can run on the GPU machine or on a separate computer. Model weights
and ComfyUI custom nodes are intentionally not installed or changed by the
VidBangerGen setup wizard.

## Interactive first run

From a fresh checkout:

```bash
python3 scripts/setup.py setup
```

The setup module uses only the Python standard library until dependency
installation begins. It can therefore configure a fresh checkout before
VidBangerGen itself is installed.

The wizard:

1. Looks only at `127.0.0.1:8188` and `127.0.0.1:8189` for local workers. It
   never scans the LAN.
2. Accepts comma-separated HTTP(S) URLs for remote workers.
3. calls ComfyUI's read-only `/system_stats` and `/object_info` endpoints.
4. Reports the device, ComfyUI version, and missing required LTX nodes.
5. Asks which workers accept uploaded conditioning media.
6. Asks whether the workers are exclusively controlled by this installation.
7. Enables optional Pixel Spatial routing only when the operator confirms that
   its model pack is installed.
8. Optionally checks non-interactive SSH for the read-only GPU Monitor.
9. Writes `.env` with mode `0600` and creates the runtime data directories.
10. Optionally creates `.venv`, installs dependencies, and builds the web app.

When an existing `.env` is reconfigured, setup makes a timestamped ignored
backup before replacing managed values. Unrelated environment settings and
comments are preserved.

## Worker ownership

Choose **exclusive** only when no other application or person submits work to
the selected ComfyUI process and VidBangerGen is allowed to interrupt or release
that process. Exclusive workers are written to `VBG_MANAGED_COMFY_URLS` and
`VBG_INTERRUPT_CAPABLE_URLS`.

Choose **shared** for a household server, a worker also used from the ComfyUI
interface, or infrastructure managed by someone else. VidBangerGen can submit
and observe its own work but will not treat the worker as disposable.

The wizard never stops a process, installs a remote service, changes a GPU
assignment, or modifies a remote ComfyUI installation.

## Remote workers

ComfyUI HTTP access is sufficient for generation. SSH is optional and is used
only for read-only NVIDIA telemetry and, when explicitly configured, model
inventory.

Use key-based SSH:

```text
operator@render-host
```

The wizard invokes SSH with `BatchMode=yes`; it does not collect or store
passwords or private keys. If the connection check fails, setup can omit SSH
while retaining the working ComfyUI HTTP connection.

Do not include usernames, passwords, tokens, or API keys in a ComfyUI URL.

## Automatic first-run behavior

These commands check setup before starting:

```bash
npm run dev
npm run dev:api
npm run serve:api
```

After installing the Python package, the packaged entry point can perform the
same flow:

```bash
vbg serve
```

If `.env` contains either `VBG_SETUP_COMPLETE=true` or an explicit
`VBG_COMFYUI_URLS`, startup continues immediately. This compatibility rule
prevents older installations from receiving an unexpected interactive prompt.

## Validate or reconfigure

```bash
vbg doctor
vbg setup --force
```

`doctor` checks FFmpeg/ffprobe, every configured ComfyUI URL, required LTX
nodes, and optional SSH access. It is read-only.

## Manual and unattended configuration

Copy `.env.example` to `.env` only when interactive setup is unsuitable, then
replace every example URL and path. At minimum configure:

```dotenv
VBG_SETUP_COMPLETE=true
VBG_DATA_DIR=./data
VBG_COMFYUI_URLS=http://127.0.0.1:8188
VBG_UPLOAD_CAPABLE_URLS=http://127.0.0.1:8188
```

Launching-process environment variables take precedence over `.env`, so
container and orchestration deployments can inject configuration without
writing a file.
