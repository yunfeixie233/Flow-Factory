#!/usr/bin/env bash
# scripts/install_geneval_deps.sh
# ─────────────────────────────────────────────────────────────────────────────
# Install GenEval reward model dependencies (mmcv + mmdet + open_clip)
#
# Requirements:
#   - Python 3.10 or 3.12 (tested)
#   - PyTorch >= 2.0 with CUDA
#   - CUDA toolkit (nvcc) for mmcv CUDA ops compilation
#   - uv (recommended) or pip
#
# Usage:
#   bash scripts/install_geneval_deps.sh
# ─────────────────────────────────────────────────────────────────────────────
set -Eeuo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/load_env.sh"
flowfactory_load_env "${FLOWFACTORY_ENV_FILE:-${REPO_ROOT}/.env}"
flowfactory_require_env CONDA_ENV CUDA_ROOT MMCV_WHEEL MAX_JOBS \
    TORCH_CUDA_ARCH_LIST MMCV_WITH_OPS FORCE_CUDA GENEVAL_BUILD_DIR \
    GENEVAL_MMCV_VERSION

PYTHON_BIN="${CONDA_ENV}/bin/python"
PIP=("${PYTHON_BIN}" -m pip)
[[ -x "${PYTHON_BIN}" ]] || {
    error "Python not found: ${PYTHON_BIN}"
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────

PY_VERSION=$("${PYTHON_BIN}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

if [[ "$PY_VERSION" != "3.10" && "$PY_VERSION" != "3.12" ]]; then
    warn "Python ${PY_VERSION} detected. This script has only been tested with Python 3.10 and 3.12."
    warn "Proceeding anyway..."
    echo ""
fi

if ! "${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    error "PyTorch with CUDA is required but not available."
    exit 1
fi

TORCH_VERSION=$("${PYTHON_BIN}" -c "import torch; print(torch.__version__)")
info "Python ${PY_VERSION}, PyTorch ${TORCH_VERSION}, installer: ${PIP[*]}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Install the pinned Python reward dependencies
# ─────────────────────────────────────────────────────────────────────────────
info "Step 1/3: Installing pinned GenEval Python dependencies..."
"${PIP[@]}" install -r "${REPO_ROOT}/config/runtime/geneval-reward-requirements.txt"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Compile mmcv with CUDA ops
# ─────────────────────────────────────────────────────────────────────────────
info "Step 2/3: Installing mmcv with CUDA ops..."

MMCV_BUILD_DIR="${GENEVAL_BUILD_DIR}"
mkdir -p "${MMCV_BUILD_DIR}"

if [[ -f "${MMCV_WHEEL}" ]]; then
    info "Installing prebuilt MMCV wheel: ${MMCV_WHEEL}"
    "${PIP[@]}" install --no-deps "${MMCV_WHEEL}"
else
    if [ ! -d "${MMCV_BUILD_DIR}/mmcv" ]; then
        git clone --depth 1 -b "v${GENEVAL_MMCV_VERSION}" \
            https://github.com/open-mmlab/mmcv.git "${MMCV_BUILD_DIR}/mmcv"
    fi
    CPATH="${CUDA_ROOT}/targets/x86_64-linux/include" \
    LIBRARY_PATH="${CUDA_ROOT}/targets/x86_64-linux/lib" \
    CUDA_HOME="${CUDA_ROOT}" MMCV_WITH_OPS="${MMCV_WITH_OPS}" \
    FORCE_CUDA="${FORCE_CUDA}" MAX_JOBS="${MAX_JOBS}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        "${PIP[@]}" install "${MMCV_BUILD_DIR}/mmcv" --no-build-isolation
fi
info "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}, MAX_JOBS=${MAX_JOBS}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Verification
# ─────────────────────────────────────────────────────────────────────────────
info "Step 3/3: Installing mmdet and verifying installation..."
"${PIP[@]}" install --no-deps \
    -r "${REPO_ROOT}/config/runtime/geneval-mmdet-requirement.txt"

if ! "${PYTHON_BIN}" - <<'PY'
import mmcv, mmdet, mmengine, open_clip
import torch
from mmcv.ops import nms
from mmdet.apis import inference_detector, init_detector  # noqa: F401

print(f'  mmcv:      {mmcv.__version__}')
print(f'  mmdet:     {mmdet.__version__}')
print(f'  mmengine:  {mmengine.__version__}')
print(f'  open_clip: {open_clip.__version__}')
boxes = torch.tensor(
    [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]],
    device='cuda',
)
scores = torch.tensor([0.9, 0.8, 0.7], device='cuda')
_, keep = nms(boxes, scores, 0.5)
torch.cuda.synchronize()
assert keep.cpu().tolist() == [0, 2], keep
print(f'  CUDA ops:  OK on {torch.cuda.get_device_name()} sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}')
PY
then
    error "Verification failed."
    exit 1
fi

info ""
info "GenEval dependencies installed successfully!"
info "Build artifacts: ${MMCV_BUILD_DIR}/"
info "Mask2Former checkpoint will be auto-downloaded on first use."
