"""Pure CPU deployment, shared-split and synchronous inference transport contracts."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hint_webshop.common import verify_sources, task_rows, load, sha
from hint_webshop.engine import RemotePool
from task_split import task_identity, partition_goals, assert_allowed, validation_ids
from fixed_validation import manifest, digest, record_identity


def test_byte_locked_shared_task_split_and_native_sources():
    verify_sources()
    rows = task_rows()
    fixed = manifest()
    assert [r['task_id'] for r in rows] == [r['task_id'] for r in fixed['selection']]
    assert [r['seed'] for r in rows] == list(range(1000,1140))
    assert load('task_split.json')['diversity']['category_counts'] == {
        'beauty':28,'electronics':28,'fashion':28,'garden':28,'grocery':28}


def test_goal_ids_ignore_prices_and_exclude_equivalent_validation_objectives():
    goal = {'instruction_text':'buy a blue shirt, and price lower than 20.0 dollars',
            'attributes':['blue'], 'goal_options':{'size':'L'}}
    changed = copy.deepcopy(goal)
    changed['instruction_text'] = 'buy a blue shirt, and price lower than 30.0 dollars'
    assert task_identity(goal) == task_identity(changed)
    changed['goal_options']['size'] = 'S'
    assert task_identity(goal) != task_identity(changed)
    source = (ROOT/'src/webshop_worker.py').read_text()
    assert 'partition_goals(server.goals, self.is_train)' in source
    assert source.count('self._current_id()') >= 4
    assert 'user_sessions.clear()' in source


def test_fixed_identity_rejects_task_or_observation_drift(tmp_path):
    identity = {'task_ids':[r['task_id'] for r in task_rows()],
                'initial_observations_sha256':'same'}
    record_identity(tmp_path, identity)
    record_identity(tmp_path, identity)
    with pytest.raises(RuntimeError,match='changed'):
        record_identity(tmp_path,dict(identity, initial_observations_sha256='different'))


@pytest.mark.parametrize('count',[0,1,2,3,4,8,17])
def test_three_replica_transport_keeps_exact_adapter_params_and_request_order(count):
    calls = []
    class Engine:
        def generate(self, prompts, params, use_tqdm=False):
            calls.append((self.rank,list(prompts),params,self.adapter))
            return [SimpleNamespace(prompt_token_ids=[p], outputs=[p]) for p in prompts]
    pool = RemotePool.__new__(RemotePool)
    pool.engines = [Engine() for _ in range(3)]
    for index,engine in enumerate(pool.engines):
        engine.rank = index
    pool.adapter = {'name':'student-17','id':17,'path':'immutable-public-fixture'}
    params = [object() for _ in range(count)]
    returned = pool.generate(list(range(count)), params)
    assert [item.prompt_token_ids[0] for item in returned] == list(range(count))
    for rank,prompts,received,adapter in calls:
        assert prompts == list(range(rank,count,3))
        assert all(param is params[index] for param,index in zip(received,prompts))
        assert adapter is pool.adapter
    assert len(calls) == min(3,count)


def test_one_training_process_global_budget_preserved():
    from hint_webshop.run import resolved
    cfg = resolved('3b')
    assert cfg['inference_replicas'] == 3 and cfg['training_processes'] == 1
    recipe = cfg['recipe']
    assert recipe['batch_size']*recipe['gradient_accumulation_steps']*recipe['rollout_n'] == 8
    assert recipe['max_prompt_tokens'] == 4096 and recipe['max_completion_length'] == 512
    assert recipe['max_env_steps'] == 15 and recipe['eval_every'] == 5


def test_cli_dry_run_works_without_external_models_or_assets():
    env = dict(os.environ, PYTHONUTF8='1')
    result = subprocess.run([sys.executable,'-m','hint_webshop.run','train','--size','3b','--dry-run'],
                            cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',check=True)
    cfg = json.loads(result.stdout)
    assert cfg['model']['model_id'] == 'Qwen/Qwen2.5-3B-Instruct'
    assert cfg['devices'] == {'training':'0','inference':['1','2','3']}
    assert cfg['evaluation_manifest_sha256'] == sha(ROOT/'configs/fixed_validation140.json')


def test_official_loss_and_update_not_replaced_by_grpo():
    import ast
    tree = ast.parse((ROOT/'hint_webshop/trainer.py').read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef))
    methods = {n.name for n in cls.body if isinstance(n,ast.FunctionDef)}
    assert methods.isdisjoint({'compute_loss','training_step','_compute_distillation_loss',
        '_update_teacher_adapter_ema','_prepare_bfcl_multi_step_inputs'})
    source = (ROOT/'hint_webshop/engine.py').read_text()
    assert 'tensor_parallel_size=1' in source and 'LoRARequest' in source
    env = (ROOT/'hint_webshop/environment.py').read_text()
    assert "multiprocessing.get_context('spawn')" in env and 'native_class' in env


def test_checkpoint_published_and_verified_before_old_save_pruned(tmp_path):
    import zipfile
    from hint_webshop.checkpoint_io import VerifiedCheckpointMixin, verify
    class Native:
        fail = False
        def _save_checkpoint(self, model=None, trial=None):
            assert self.args.save_total_limit is None
            if self.state.global_step==60:
                assert (tmp_path/'checkpoint-20').exists() and (tmp_path/'checkpoint-40').exists()
            path = Path(self.args.output_dir)/('checkpoint-'+str(self.state.global_step))
            path.mkdir()
            names = ['adapter_config.json','adapter_model.safetensors',
                'teacher_adapter/adapter_config.json','teacher_adapter/adapter_model.safetensors',
                'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','training_args.bin']
            for name in names:
                target = path/name
                target.parent.mkdir(exist_ok=True)
                if target.suffix in ('.pt','.pth','.bin'):
                    with zipfile.ZipFile(target,'w') as stream:
                        stream.writestr('archive/data.pkl',b'fixture')
                else:
                    target.write_bytes(b'{}')
            if self.fail:
                (path/'optimizer.pt').unlink()
        def _load_from_checkpoint(self, path):
            return str(path)
    class Trainer(VerifiedCheckpointMixin,Native):
        pass
    tr = Trainer()
    tr.args = SimpleNamespace(output_dir=str(tmp_path),save_total_limit=2)
    tr.state = SimpleNamespace(global_step=0)
    for step in (20,40,60):
        tr.state.global_step=step
        tr._save_checkpoint()
        verify(tmp_path/('checkpoint-'+str(step)))
    assert sorted(p.name for p in tmp_path.glob('checkpoint-*')) == ['checkpoint-40','checkpoint-60']
    tr.state.global_step=80
    tr.fail=True
    with pytest.raises(ValueError,match='Incomplete'):
        tr._save_checkpoint()
    assert (tmp_path/'checkpoint-40').exists() and (tmp_path/'checkpoint-60').exists()
    assert list(tmp_path.glob('.saving-*'))
    assert tr.args.output_dir == str(tmp_path) and tr.args.save_total_limit==2
    assert tr._load_from_checkpoint(tmp_path/'checkpoint-60').endswith('checkpoint-60')
    (tmp_path/'checkpoint-60/adapter_model.safetensors').write_bytes(b'damaged')
    with pytest.raises(ValueError,match='checksum'):
        tr._load_from_checkpoint(tmp_path/'checkpoint-60')
