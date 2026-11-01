#!/usr/bin/env bash
# Optional CPU embeddings; run setup_linux_environment.sh first.
set -euo pipefail

task_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
task_workspace_root=$(dirname -- "$task_repo_root")
task_env_root=${SHINKU_LINUX_ENV_ROOT:-"$task_workspace_root/.venv-linux"}
task_uv=${SHINKU_UV_BIN:-"$task_workspace_root/.tools-linux/uv"}
task_lock="$task_repo_root/requirements-ml-linux-cpu.lock"
if (($#)); then
    echo "Usage: $0 (optional SHINKU_LINUX_ENV_ROOT and SHINKU_UV_BIN)" >&2
    exit 2
fi
if [[ $(uname -s) != Linux || ! -f "$task_env_root/pyvenv.cfg" || ! -x "$task_uv" ]]; then
    echo "Set up the independent Linux environment and uv first." >&2
    exit 2
fi
"$task_env_root/bin/python" -B -c '
import platform, sys
assert sys.version_info[:2] == (3, 13) and platform.machine() == "x86_64", "This lock targets Linux x86_64 Python 3.13"
import shinku, fastapi
'
# uv routes torch to its official CPU index; other packages keep their normal
# index. The core constraints prevent an ML install from upgrading the backend.
"$task_uv" pip install --python "$task_env_root/bin/python" \
    --torch-backend cpu \
    --constraint "$task_repo_root/requirements-cleanenv.lock" --requirements "$task_lock"
"$task_uv" pip check --python "$task_env_root/bin/python"
"$task_env_root/bin/python" -B -c '
import importlib.metadata as m, torch
assert torch.version.cuda is None, "CPU-only PyTorch required"
from sentence_transformers import SentenceTransformer
print("Local embedding runtime:", {p: m.version(p) for p in ("torch", "sentence-transformers", "transformers", "scikit-learn")})
'
