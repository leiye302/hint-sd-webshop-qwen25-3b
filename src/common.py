"""Public asset/split helper paths shared with the other WebShop baselines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hint_webshop.common import ROOT, HINT, SDAR, load, sha, atomic_json
UPSTREAM = SDAR
WEBSHOP = SDAR / 'agent_system/environments/env_package/webshop/webshop'
WORLD_SIZE = 4
