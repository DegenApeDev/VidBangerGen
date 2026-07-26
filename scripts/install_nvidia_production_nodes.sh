#!/usr/bin/env bash
set -Eeuo pipefail

# Optional NVIDIA-only installation. The ComfyUI owner/operator must review the
# pinned changes and deliberately opt in before this script will do anything.
# It does not modify the LTX model graph and does not restart ComfyUI.

COMFY_DIR="${COMFY_DIR:-/opt/ComfyUI}"
CUSTOM_NODES_DIR="${CUSTOM_NODES_DIR:-${COMFY_DIR}/custom_nodes}"
PYTHON="${PYTHON:-${COMFY_DIR}/.venv/bin/python}"

NVIDIA_REPO="https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.git"
NVIDIA_COMMIT="892515e3eb9a4920a131a502a047e47adca9eb0d"
NVIDIA_DIR="${CUSTOM_NODES_DIR}/Nvidia_RTX_Nodes_ComfyUI"

VHS_REPO="https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
VHS_COMMIT="4ee72c065db22c9d96c2427954dc69e7b908444b"
VHS_DIR="${CUSTOM_NODES_DIR}/ComfyUI-VideoHelperSuite"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ "${VBG_NVIDIA_NODES_APPROVED:-}" == "yes" ]] || die \
  "Operator approval required (set VBG_NVIDIA_NODES_APPROVED=yes after review)"

install_repo() {
  local url="$1"
  local destination="$2"
  local commit="$3"

  if [[ -e "${destination}" && ! -d "${destination}/.git" ]]; then
    die "${destination} exists but is not a Git checkout"
  fi
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none --no-checkout "${url}" "${destination}"
  fi
  git -C "${destination}" fetch --depth=1 origin "${commit}"
  git -C "${destination}" checkout --detach "${commit}"
  test "$(git -C "${destination}" rev-parse HEAD)" = "${commit}" \
    || die "Failed to pin ${destination} to ${commit}"
}

[[ -d "${COMFY_DIR}" ]] || die "ComfyUI directory not found: ${COMFY_DIR}"
[[ -d "${CUSTOM_NODES_DIR}" ]] || die "Custom-nodes directory not found: ${CUSTOM_NODES_DIR}"
[[ -w "${CUSTOM_NODES_DIR}" ]] \
  || die "${CUSTOM_NODES_DIR} is not writable; run as the ComfyUI owner"
[[ -x "${PYTHON}" ]] || die "ComfyUI Python not executable: ${PYTHON}"

install_repo "${NVIDIA_REPO}" "${NVIDIA_DIR}" "${NVIDIA_COMMIT}"
install_repo "${VHS_REPO}" "${VHS_DIR}" "${VHS_COMMIT}"

"${PYTHON}" -m pip install -r "${NVIDIA_DIR}/requirements.txt"
# Keep the optional NVIDIA runtime deterministic.
"${PYTHON}" -m pip install "nvidia-vfx==0.1.0.1"
"${PYTHON}" -m pip install -r "${VHS_DIR}/requirements.txt"
"${PYTHON}" -m pip check

"${PYTHON}" - <<'PY'
import cv2
import imageio_ffmpeg
import nvvfx

print("Dependency imports passed:", cv2.__version__, imageio_ffmpeg.get_ffmpeg_version())
print("NVIDIA VFX module:", nvvfx.__file__)
PY

printf '\nInstalled pinned production finishing nodes.\n'
printf 'Restart the affected ComfyUI workers, then run scripts/check_installation.py.\n'
