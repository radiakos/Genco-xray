#!/usr/bin/env bash
# Environment for the GPU box. Usage: CUDA_TAG=cu126 bash scripts/00_setup_env.sh
set -euo pipefail
CUDA_TAG="${CUDA_TAG:-cu126}"           # cpu | cu118 | cu126 ... must match your driver
PY="${PY:-python3.12}"
GRAPHKIT_REF="${GRAPHKIT_REF:-lfe-tutorial}"   # @9d42d82 reproduces the released checkpoints (results/GRAPHKIT_COMMIT)

$PY -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install "torch==2.12.*" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

mkdir -p external
if [ ! -d external/gridfm-graphkit ]; then
  git clone -b "$GRAPHKIT_REF" https://github.com/gridfm/gridfm-graphkit external/gridfm-graphkit
fi
pip install -e external/gridfm-graphkit
pip install -e ".[dev,ac]"
pip install "huggingface_hub[cli]"

# Only needed to generate the IEEE 118-bus contingency sets. Needs Julia; see datakit's Containerfile if this fails.
if [ "${WITH_DATAKIT:-1}" = "1" ]; then
  pip install gridfm-datakit && gridfm_datakit setup_pm || echo "WARN: datakit/Julia setup failed; use the Containerfile or pf_solver: lightsim2grid"
fi

export MLFLOW_ALLOW_FILE_STORE=true
python - <<'PY'
import torch, torch_geometric, gridfm_graphkit, gencoxray
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("graphkit", getattr(gridfm_graphkit, "__version__", "?"))
PY
git -C external/gridfm-graphkit rev-parse HEAD > results/GRAPHKIT_COMMIT
echo "OK. graphkit @ $(cat results/GRAPHKIT_COMMIT)"
