#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo 'Install uv: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
python3 -m hint_webshop.setup --framework
export UV_PROJECT_ENVIRONMENT="$PWD/.deps/hint-sd/.venv"
uv sync --frozen --project .deps/hint-sd
uv venv --python 3.10 .env-webshop
uv pip install --python .env-webshop/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .env-webshop/bin/python -r configs/environment-requirements.txt
echo 'Runtimes installed. Set JAVA_HOME to Java11, then prepare assets and task rows as documented.'
