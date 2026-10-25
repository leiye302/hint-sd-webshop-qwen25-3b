"""One fixed140-task native validation batch, disjoint from training goals."""
import hashlib
import json
from pathlib import Path
import numpy as np
from common import ROOT, atomic_json, load, sha
from task_split import manifest as split_manifest


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def manifest():
    value = load('fixed_validation140.json')
    split = split_manifest()
    expected = [{'worker': i, 'task_id': key,
                 'worker_seed': 1000+i, 'goal_index': i}
                for i, key in enumerate(split['validation_task_ids'])]
    if (value['schema'] != 1 or value['count'] != 140 or value['batch_size'] != 140
            or value['base_seed'] != 1000 or value['selection'] != expected
            or value['upstream_commit'] != load('upstream.json')['commit']
            or value['task_split_sha256'] != sha(ROOT / 'configs/task_split.json')
            or value['assets_lock_sha256'] != sha(ROOT / 'configs/assets.json')
            or sha(ROOT / 'configs/fixed_validation140.json')
                != load('upstream.json')['validation_manifest_sha256']):
        raise RuntimeError('Fixed140 WebShop validation manifest/configuration differs')
    return value


def native_validation_selection(env, base_seed):
    """Diagnostic preview of the original sampler, without advancing it."""
    preview = np.random.RandomState()
    preview.set_state(env._rng.get_state())
    indices = np.repeat(preview.choice(env.goal_idxs, size=env.env_num, replace=False), env.group_n).tolist()
    return [{'worker': i, 'worker_seed': base_seed + i // env.group_n, 'goal_index': int(index)}
            for i, index in enumerate(indices)]


class FixedSelection:
    def __init__(self, indices):
        self.indices = indices

    def choice(self, candidates, size, replace):
        if list(candidates) != list(range(140)) or size != len(self.indices) or replace:
            raise RuntimeError('Native validation sampler shape changed')
        return np.array(self.indices, dtype=np.int64)


def reset_fixed_validation(env, original_reset, base_seed):
    fixed = manifest()
    if (getattr(env, 'is_train', None) is not False or env.env_num != fixed['batch_size']
            or env.group_n != 1 or base_seed != fixed['base_seed']
            or list(env.goal_idxs) != list(range(140))):
        raise RuntimeError('Fixed evaluation requires the disjoint140-worker validation environment')
    selected = fixed['selection']
    previous = env._rng
    try:
        env._rng = FixedSelection([row['goal_index'] for row in selected])
        result = original_reset()
    finally:
        env._rng = previous
    observations, infos = result
    if (len(observations) != len(selected) or len(infos) != len(selected)
            or [info.get('task_id') for info in infos] != [row['task_id'] for row in selected]):
        raise RuntimeError('Actual validation task identities differ from the fixed manifest')
    identity = {
        'protocol': fixed['protocol'],
        'manifest_sha256': sha(ROOT / 'configs/fixed_validation140.json'),
        'task_split_sha256': sha(ROOT / 'configs/task_split.json'),
        'task_ids': [row['task_id'] for row in selected],
        'selection_sha256': digest(selected),
        'initial_observations_sha256': digest(list(observations)),
    }
    return result, identity


def record_identity(run_dir, identity):
    path = Path(run_dir) / 'fixed_validation_identity.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != identity:
            raise RuntimeError('Fixed validation initial tasks changed; refusing incomparable evaluation')
    else:
        atomic_json(path, identity)
