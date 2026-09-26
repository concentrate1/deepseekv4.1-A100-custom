#!/usr/bin/env bash
# Restore the dependencies needed by run_mtp_local.sh after rebuilding Docker.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if (( EUID != 0 )); then
    echo "Run this setup script as root inside the container (apt packages are required)." >&2
    exit 1
fi

if [[ ! -f apt-requirements.txt || ! -f requirements.txt ]]; then
    echo "Dependency lists are missing from $(pwd)." >&2
    exit 1
fi

mapfile -t apt_packages < <(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' apt-requirements.txt)
missing_packages=()
for package in "${apt_packages[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -qx 'install ok installed'; then
        missing_packages+=("$package")
    fi
done

if (( ${#missing_packages[@]} > 0 )); then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing_packages[@]}"
fi

# Install into the container's system Python, including images with PEP 668.
/usr/bin/python3 -m pip install --break-system-packages -r requirements.txt
/usr/bin/python3 -m pip check

ckpt="${DSV41_CKPT:-/models/DeepSeek-V4.1-Flash}"
if [[ ! -d "$ckpt" ]]; then
    echo "Checkpoint directory not found: $ckpt" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; expose the NVIDIA GPUs to this container." >&2
    exit 1
fi
nvidia-smi --list-gpus >/dev/null
/usr/bin/python3 - <<'PYCHECK'
import torch

visible = torch.cuda.device_count()
if visible < 5:
    raise SystemExit(f"run_mtp_local.sh needs CUDA device indices 0-4; found {visible}")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {visible} visible GPU(s)")
PYCHECK

mkdir -p logs
echo "MTP environment ready. Start with ./run_mtp_local.sh --background"
