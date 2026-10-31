"""Cross-method WebShop evaluation contract; stdlib only, no training changes."""
import hashlib
import json
from pathlib import Path

CANONICAL_UPSTREAM = '29dcb12a55d8fc17ba6ae03fd0a69048f43051f3'
MANIFEST_SHA256 = '706adb889929acc45bafd52fc144cd160b85ee7f752adf7b6dad8699f8c5ce4a'
SPLIT_SHA256 = '6055ef05de812132425060ad959d11bd6ca1cbe40a94f9a195f9f0647ae9547c'
ASSETS_SHA256 = 'f8313d36b83df707e2f44b85fd2a07aa765ad82e3ee9cc9f26620e92e34cc81a'

PROTOCOL = {
    'schema': 1, 'name': 'webshop-shared140-evaluation-v1',
    'validation_manifest_sha256': MANIFEST_SHA256,
    'task_split_sha256': SPLIT_SHA256, 'assets_lock_sha256': ASSETS_SHA256,
    'canonical_environment_commit': CANONICAL_UPSTREAM,
    'tasks': 140, 'rollouts_per_task': 1, 'worker_seeds': '1000 + worker_index (0..139)',
    'temperature': 0.4, 'do_sample': True, 'top_p': 1.0, 'top_k': -1,
    'max_steps': 15, 'history_length': 2, 'max_prompt_tokens': 4096,
    'max_response_tokens': 512, 'use_small': True, 'human_goals': False,
    'success': 'native terminal raw_task_score == 1.0',
    'score': 'mean native raw_task_score; multiply by 100 only for display',
    'evaluation_hints': False,
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_shared_evaluation(root, recipe=None):
    """Reject a different fixed140 set even if its local lock was edited too."""
    root = Path(root)
    expected = {'fixed_validation140.json': MANIFEST_SHA256,
                'task_split.json': SPLIT_SHA256, 'assets.json': ASSETS_SHA256}
    for name, value in expected.items():
        if _sha(root / 'configs' / name) != value:
            raise RuntimeError('Cross-method shared140 identity differs: ' + name)
    protocol = json.loads((root / 'configs/evaluation_protocol.json').read_text(encoding='utf-8'))
    if protocol != PROTOCOL:
        raise RuntimeError('Cross-method evaluation parameters differ')
    if recipe is None:
        name = 'opsd.json' if (root / 'configs/opsd.json').is_file() else 'experiment.json'
        recipe = json.loads((root / 'configs' / name).read_text(encoding='utf-8'))
    if 'eval_temperature' in recipe:  # HINT-SD's public trainer uses native SamplingParams defaults.
        fields = {'eval_temperature': .4, 'eval_rollout_n': 1, 'max_env_steps': 15,
                  'history_length': 2, 'max_prompt_tokens': 4096, 'max_completion_length': 512}
    else:
        fields = {'data.val_batch_size': 140, 'data.max_prompt_length': 4096,
                  'data.max_response_length': 512, 'env.max_steps': 15, 'env.history_length': 2,
                  'env.webshop.use_small': True, 'env.webshop.human_goals': False,
                  'actor_rollout_ref.rollout.val_kwargs.temperature': .4,
                  'actor_rollout_ref.rollout.val_kwargs.do_sample': True,
                  'actor_rollout_ref.rollout.val_kwargs.top_p': 1.0,
                  'actor_rollout_ref.rollout.val_kwargs.top_k': -1,
                  'actor_rollout_ref.rollout.val_kwargs.n': 1}
    for name, value in fields.items():
        if recipe.get(name) != value:
            raise RuntimeError('Shared140 evaluation parameter differs: ' + name)
    return protocol
