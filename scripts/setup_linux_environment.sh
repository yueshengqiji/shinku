#!/usr/bin/env bash
# Install an independent Linux runtime without changing the system Python.
set -euo pipefail

task_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
task_workspace_root=$(dirname -- "$task_repo_root")
task_uv=${SHINKU_UV_BIN:-}
task_python_version=${SHINKU_LINUX_PYTHON_VERSION:-3.13}
task_python_root=${SHINKU_LINUX_PYTHON_ROOT:-"$task_workspace_root/.python-linux"}
task_env_root=${SHINKU_LINUX_ENV_ROOT:-"$task_workspace_root/.venv-linux"}
task_extra_requirements=()

while (($#)); do
    case "$1" in
        --extra-requirements)
            if (($# < 2)) || [[ ! -f "$2" ]]; then
                echo "--extra-requirements needs an existing requirements file" >&2
                exit 2
            fi
            task_extra_requirements+=(--requirements "$2")
            shift 2
            ;;
        *)
            echo "Usage: $0 [--extra-requirements PATH]" >&2
            exit 2
            ;;
    esac
done

if [[ $(uname -s) != Linux ]]; then
    echo "This installer requires Linux." >&2
    exit 2
fi
if [[ -z "$task_uv" ]]; then
    if command -v uv >/dev/null 2>&1; then
        task_uv=$(command -v uv)
    elif [[ -x "$task_workspace_root/.tools-linux/uv" ]]; then
        task_uv="$task_workspace_root/.tools-linux/uv"
    else
        echo "Install uv or set SHINKU_UV_BIN to its executable." >&2
        exit 2
    fi
fi
if [[ -e "$task_env_root" && ! -f "$task_env_root/pyvenv.cfg" ]]; then
    echo "Refusing to replace an existing non-venv directory: $task_env_root" >&2
    exit 2
fi

export UV_PYTHON_INSTALL_DIR="$task_python_root"
"$task_uv" python install --no-bin "$task_python_version"
if [[ ! -f "$task_env_root/pyvenv.cfg" ]]; then
    "$task_uv" venv --managed-python --python "$task_python_version" "$task_env_root"
fi
"$task_uv" pip install --python "$task_env_root/bin/python" \
    --requirements "$task_repo_root/requirements-cleanenv.lock" \
    --editable "$task_repo_root[dev,documents]" \
    "${task_extra_requirements[@]}"
"$task_uv" pip check --python "$task_env_root/bin/python"
"$task_env_root/bin/python" -c 'import sys, fastapi, openai, uvicorn, yaml; print("Linux runtime:", sys.version.split()[0]); print("Python:", sys.executable)'
