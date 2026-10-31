"""Repository-independent guard against per-method evaluation drift."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation_contract import (PROTOCOL, MANIFEST_SHA256, SPLIT_SHA256,
                                 verify_shared_evaluation)


def test_same_public_manifest_not_a_per_method_fixed_draw():
    assert hashlib.sha256((ROOT/'configs/fixed_validation140.json').read_bytes()).hexdigest() == MANIFEST_SHA256
    assert hashlib.sha256((ROOT/'configs/task_split.json').read_bytes()).hexdigest() == SPLIT_SHA256
    assert verify_shared_evaluation(ROOT) == PROTOCOL
    fixed = json.loads((ROOT/'configs/fixed_validation140.json').read_text(encoding='utf-8'))
    split = json.loads((ROOT/'configs/task_split.json').read_text(encoding='utf-8'))
    assert [r['task_id'] for r in fixed['selection']] == split['validation_task_ids']
    assert [r['worker_seed'] for r in fixed['selection']] == list(range(1000,1140))
    assert [r['goal_index'] for r in fixed['selection']] == list(range(140))


def test_generation_parameter_drift_is_not_accepted():
    name = 'opsd.json' if (ROOT/'configs/opsd.json').is_file() else 'experiment.json'
    cfg = json.loads((ROOT/'configs'/name).read_text(encoding='utf-8'))
    field = 'eval_temperature' if 'eval_temperature' in cfg else 'actor_rollout_ref.rollout.val_kwargs.temperature'
    changed = copy.deepcopy(cfg)
    changed[field] = 0.0
    with pytest.raises(RuntimeError, match='parameter differs'):
        verify_shared_evaluation(ROOT, changed)


def test_different_fixed140_rejected_even_if_local_lock_was_updated(tmp_path):
    target = tmp_path/'configs'
    target.mkdir()
    for name in ('fixed_validation140.json','task_split.json','assets.json','evaluation_protocol.json'):
        (target/name).write_bytes((ROOT/'configs'/name).read_bytes())
    fixed = json.loads((target/'fixed_validation140.json').read_text(encoding='utf-8'))
    fixed['selection'][0]['worker_seed'] += 1
    (target/'fixed_validation140.json').write_text(json.dumps(fixed))
    with pytest.raises(RuntimeError, match='shared140 identity differs'):
        verify_shared_evaluation(tmp_path, {})
