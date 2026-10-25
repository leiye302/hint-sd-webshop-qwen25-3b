#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f .env ]]; then set -a; source .env; set +a; fi
# Single-backbone repository: compatibility alias only; no second simultaneous experiment.
exec python3 -m hint_webshop.run train "$@" --size 3b
