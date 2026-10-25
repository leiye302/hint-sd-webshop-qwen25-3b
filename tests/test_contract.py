"""CPU tests exercise the pinned native math, selectors and EMA, not a rewritten loss."""
import ast
import copy
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
import pytest
torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hint_webshop.common import HINT, SDAR, checkpoint_complete, devices, load, sha, task_rows, verify_sources
from hint_webshop.run import resolved
from hint_webshop.feedback import feedback_messages
from hint_webshop.environment import policy_messages, native_parts
from hint_webshop.engine import AdapterPublisher
from hint_webshop.report import protocol, summarize


def load_native(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


NATIVE_LOSS = load_native('hint_test_native_loss', HINT / 'src/train/self_distillation.py')


def selected_methods(path, source_class, names, *, bases=None, namespace=None):
    """Execute real method ASTs without importing optional CUDA/Trainer dependencies."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == source_class)
    body = [copy.deepcopy(node) for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in body} == set(names)
    shell = ast.ClassDef(name='Harness', bases=bases or [], keywords=[], body=body, decorator_list=[], type_params=[])
    parsed = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), shell], type_ignores=[])
    namespace = {'torch': torch, 'Sequence': Sequence, **(namespace or {})}
    exec(compile(ast.fix_missing_locations(parsed), str(path), 'exec'), namespace)
    return namespace['Harness']


def test_sources_unchanged_and_game_manifest():
    verify_sources()
    assert len(task_rows()) == 140


@pytest.mark.parametrize('size', ['1.5b', '3b'])
def test_only_backbone_differs_between_sizes(size):
    value = resolved(size, '0,1,2,3')
    assert value['recipe'] == load('experiment.json')
    assert value['model']['model_id'] == f'Qwen/Qwen2.5-{size[:-1].upper()}B-Instruct'
    assert len(value['model']['revision']) == 40
    assert value['training_processes'] == value['vllm_tensor_parallel_size'] == 1
    assert value['inference_replicas'] == 3


@pytest.mark.parametrize('spec', ['0,1,2', '0,1,1,2', '0,1,2,three', '0,1,2,3,4'])
def test_bad_gpu_lists_rejected(spec):
    with pytest.raises(ValueError):
        devices(spec)


def test_four_gpus_single_size_one_train_three_inference():
    ids = devices('4,5,6,7')
    value = resolved('1.5b', ','.join(ids))
    assert value['devices'] == {'training':'4','inference':['5','6','7']}
    with pytest.raises(ValueError):
        resolved('1.5b', '0,1,2,2')


def test_original_recipe_no_grpo_or_extra_kl():
    value = load('experiment.json')
    assert (value['lora_rank'], value['lora_alpha'], value['lora_dropout']) == (32, 64, 0.05)
    assert value['lora_target_modules'] == ['q_proj', 'v_proj']
    assert value['ema_decay'] == 0.999
    assert value['distillation_topk'] == 100 and value['distillation_add_tail'] is True
    assert value['batch_size'] * value['gradient_accumulation_steps'] * value['rollout_n'] == 8
    assert value['updates'] == 150 and value['eval_every'] == 5 and value['save_every'] == 20
    assert not any(name in value for name in ['grpo_coef', 'sdar_coef', 'gate_beta', 'reference_kl'])


def test_adapter_does_not_override_native_objective():
    tree = ast.parse((ROOT / 'hint_webshop/trainer.py').read_text())
    adapter = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    names = {node.name for node in adapter.body if isinstance(node, ast.FunctionDef)}
    forbidden = {'compute_loss', 'training_step', '_compute_distillation_loss',
                 '_update_teacher_adapter_ema', '_prepare_bfcl_multi_step_inputs'}
    assert names.isdisjoint(forbidden)
    native = ast.parse((HINT / 'src/train/trainer.py').read_text())
    trainer = next(node for node in native.body if isinstance(node, ast.ClassDef) and node.name == 'Trainer')
    compute = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == 'compute_loss')
    assert '_compute_distillation_loss' in ast.unparse(compute)
    assert 'advantage' not in ast.unparse(compute)


@pytest.mark.parametrize('topk,tail', [(None, False), (2, True), (2, False), (4, True)])
def test_native_reverse_kl_matches_direct_calculation(topk, tail):
    student = torch.tensor([[[0.5, -0.4, 1.1, -0.2]]], requires_grad=True)
    teacher = torch.tensor([[[0.1, 0.8, -0.3, 0.4]]])
    sq, tp = student.log_softmax(-1), teacher.log_softmax(-1)
    q, p = NATIVE_LOSS.select_student_topk_log_probs(sq, tp, topk, tail)
    expected = (q.exp() * (q - p)).sum()
    actual, _ = NATIVE_LOSS.compute_distillation_loss(student_log_probs=sq, teacher_log_probs=tp,
        completion_mask=torch.ones(1, 1), distillation_topk=topk, distillation_add_tail=tail)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0


def test_native_kl_student_gradient_and_teacher_stopgrad():
    student = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
    teacher = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
    with torch.no_grad():
        teacher_log_probs = teacher.log_softmax(-1)
    actual, _ = NATIVE_LOSS.compute_distillation_loss(student_log_probs=student.log_softmax(-1),
        teacher_log_probs=teacher_log_probs, completion_mask=torch.ones(1, 1),
        distillation_topk=None, distillation_add_tail=False)
    actual.backward()
    assert student.grad[0, 0, 0] < 0 and student.grad[0, 0, 1] > 0
    assert teacher.grad is None


def test_unselected_and_padding_positions_have_no_gradient():
    student = torch.randn(2, 3, 7, requires_grad=True)
    teacher = torch.randn(2, 3, 7).log_softmax(-1)
    loss, _ = NATIVE_LOSS.compute_distillation_loss(student_log_probs=student.log_softmax(-1),
        teacher_log_probs=teacher, completion_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        distillation_topk=None, distillation_add_tail=False, self_distillation_mask=torch.tensor([1., 0.]))
    loss.backward()
    assert student.grad[0, :2].abs().sum() > 0
    assert student.grad[0, 2].abs().sum() == 0
    assert student.grad[1].abs().sum() == 0


def test_empty_target_batch_is_finite_zero():
    value = torch.randn(1, 2, 4, requires_grad=True).log_softmax(-1)
    loss, metrics = NATIVE_LOSS.compute_distillation_loss(student_log_probs=value,
        teacher_log_probs=value.detach(), completion_mask=torch.ones(1, 2),
        distillation_topk=2, distillation_add_tail=True, self_distillation_mask=torch.tensor([0.]))
    assert loss.item() == 0 and metrics['self_distillation/empty_target_batch'] == 1


@pytest.mark.parametrize('skip', [False, True])
def test_native_ema_tracks_student_but_skips_overflow(skip):
    cls = selected_methods(HINT / 'src/train/trainer.py', 'Trainer', ['_update_teacher_adapter_ema'])
    instance = cls()
    student, teacher = torch.nn.Parameter(torch.tensor([2.])), torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
    instance.accelerator = SimpleNamespace(optimizer_step_was_skipped=skip)
    instance._iter_student_teacher_parameter_pairs = lambda: [('weight', student, teacher)]
    instance._mark_vllm_weights_stale = lambda: None
    updated = instance._update_teacher_adapter_ema(0.999)
    torch.testing.assert_close(teacher, torch.tensor([1. if skip else 1.001]))
    assert updated is not skip and teacher.grad is None


def test_native_selected_step_consumes_exact_ids_and_full_completion():
    cls = selected_methods(HINT / 'src/train/trainer.py', 'Trainer',
        ['_prepare_bfcl_multi_step_inputs', '_pad_sequences', '_pad_mask_sequences', '_coerce_rollout_token_ids'])
    harness = cls()
    harness.accelerator = SimpleNamespace(device='cpu')
    harness.processing_class = SimpleNamespace(pad_token_id=99, eos_token_id=99)
    harness.args = SimpleNamespace(bfcl_feedback_multi_step_max=3)
    harness._detect_bfcl_failure_steps_multi = lambda *args, **kwargs: [[(1, 'correct the state')], None]
    harness._build_bfcl_per_step_teacher_prompt = lambda *args: 'teacher-with-feedback'
    harness._apply_chat_template = lambda messages, **kwargs: messages
    harness._encode_text = lambda text: [20, 21]
    step = SimpleNamespace(agent_metadata={'prompt_token_ids': [5, 6], 'completion_token_ids': [10, 11, 12, 13]})
    failed = SimpleNamespace(success=False, steps=[step, step])
    succeeded = SimpleNamespace(success=True, steps=[step])
    placeholder = SimpleNamespace(prompt_ids=[1], completion_ids=[2], completion_attention_mask=[1], completion_text='')
    batch = SimpleNamespace(episodes=[failed, succeeded], tokenized=[placeholder, placeholder],
                            feedback=[None, None], metrics={}, scores=[0., 1.])
    result = harness._prepare_bfcl_multi_step_inputs(batch)
    assert result['prompt_ids'].tolist() == [[5, 6]]
    assert result['completion_ids'].tolist() == [[10, 11, 12, 13]]
    assert result['completion_loss_mask'].tolist() == [[1, 1, 1, 1]]
    assert result['self_distillation_mask'].tolist() == [1.]
    assert result['self_distillation_active_fraction'] == 0.5


def test_original_feedback_parser_and_max_three():
    sys.path.insert(0, str(HINT))
    from src.train.appworld_feedback import parse_multi_step_failure_json
    raw = json.dumps({'failures': [{'step': i, 'feedback': 'correct this state'} for i in [4, 2, 1, 3]]})
    assert parse_multi_step_failure_json(raw, 4, max_steps=3) == [(0, 'correct this state'), (1, 'correct this state'), (2, 'correct this state')]
    assert parse_multi_step_failure_json('invalid JSON', 10) is None


def test_domain_prompt_preserves_feedback_schema():
    prompt = feedback_messages('full failed trajectory')
    assert 'up to 3' in prompt[0]['content']
    assert 'WebShop' in prompt[0]['content']
    assert 'failures' in prompt[1]['content']
    assert 'RETHINK' not in str(prompt)


def test_policy_observation_is_already_native_formatted_not_rebuilt():
    rendered = 'Exact native WebShop template, task, current observation and history'
    assert policy_messages(rendered, ['search[<your query>]'], 'buy', [], 2) == [
        {'role':'user','content':rendered}]


def test_teacher_only_hint_preserves_actual_truncated_prefix():
    class Base:
        def _apply_chat_template(self, messages, add_generation_prompt):
            result = ''.join('<|im_start|>' + item['role'] + '\n' + item['content'] + '<|im_end|>\n' for item in messages)
            return result + ('<|im_start|>assistant\n' if add_generation_prompt else '')
    cls = selected_methods(ROOT / 'hint_webshop/trainer.py', 'WebshopTrainer',
                           ['_apply_chat_template', '_build_bfcl_per_step_teacher_prompt'],
                           bases=[ast.Name(id='Base', ctx=ast.Load())], namespace={'Base': Base})
    harness = cls()
    harness.processing_class = SimpleNamespace(decode=lambda ids, **kwargs: ''.join(map(chr, ids)))
    actual = 'left-truncated-visible-history<|im_end|>\n<|im_start|>assistant\n'
    episode = SimpleNamespace(steps=[SimpleNamespace(agent_metadata={'prompt_token_ids': list(map(ord, actual))})])
    messages = harness._build_bfcl_per_step_teacher_prompt(episode, 0, 'PRIVATE-CORRECTION')
    teacher = harness._apply_chat_template(messages, True)
    assert teacher.startswith(actual.split('<|im_start|>assistant')[0])
    assert 'PRIVATE-CORRECTION' in teacher and 'PRIVATE-CORRECTION' not in actual
    assert teacher.endswith('<|im_start|>assistant\n')


def test_adapter_sync_versions_are_unique_and_bounded(tmp_path):
    class Trainer:
        def _save_adapter_to_dir(self, name, path):
            folder = Path(path)
            folder.mkdir()
            (folder / 'adapter_config.json').write_text('{}')
    publisher = AdapterPublisher(tmp_path / 'copies')
    versions = [publisher.publish(Trainer(), name) for name in ['student', 'teacher', 'student', 'teacher']]
    assert len({item['id'] for item in versions}) == 4
    assert len(list((tmp_path / 'copies').iterdir())) == 2
    assert all(Path(item['path']).exists() for item in versions[-2:])


def test_incomplete_training_checkpoint_rejected(tmp_path):
    with pytest.raises(ValueError, match='Incomplete'):
        checkpoint_complete(tmp_path)
    names = ['adapter_config.json', 'adapter_model.safetensors', 'teacher_adapter/adapter_config.json',
             'teacher_adapter/adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt',
             'rng_state.pth', 'trainer_state.json', 'training_args.bin']
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'x')
    assert checkpoint_complete(tmp_path) == tmp_path


def test_protocol_is_telemetry_not_replacement_parser():
    assert protocol('<think>plan</think><action>look</action>')['strict']
    assert not protocol('<action>look</action><think>plan</think>')['strict']
    assert not protocol('<think></think><action>look</action>')['reasoning']


def test_dry_run_independent_of_cuda_and_assets():
    environment = dict(os.environ)
    environment.pop('HINTSD_GPUS', None)
    value = subprocess.run([sys.executable, '-m', 'hint_webshop.run', 'train', '--size', '1.5b', '--dry-run'],
                            cwd=ROOT, env=environment, capture_output=True, text=True, check=True)
    result = json.loads(value.stdout)
    assert result['devices'] == {'training':'0', 'inference':['1','2','3']}
    assert result['recipe']['max_env_steps'] == 15


def test_native_teacher_forward_is_no_grad():
    source = (HINT / 'src/train/trainer.py').read_text()
    assert 'with self._use_adapter(self.teacher_adapter_name), torch.no_grad():' in source
    assert 'student_param.detach()' in source


def test_loopback_client_works_without_requests_or_environment_proxy(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from hint_webshop.environment import Client
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            encoded = json.dumps({'received': request, 'path': self.path}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv('HTTP_PROXY', 'http://invalid.local:1')
    try:
        result = Client(f'http://127.0.0.1:{server.server_port}').call('health', check='cpu')
        assert result == {'received': {'check': 'cpu'}, 'path': '/health'}
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_rollout_preserves_generated_ids_and_releases_every_game(monkeypatch):
    sys.path.insert(0, str(HINT))
    from hint_webshop import rollout
    class LocalService:
        instances = {}
        def __init__(self, url):
            pass
        def call(self, operation, **values):
            if operation == 'create':
                identity = values['task']['task_id']
                self.instances[identity] = False
                return {'instance_id': identity, 'observation': 'Your task is to: inspect desk',
                        'admissible_commands': ['look'], 'task': 'inspect desk', 'history': []}
            if operation == 'release':
                self.instances[values['instance_id']] = True
                return {'released': True}
            return {'observation': 'looked at desk', 'admissible_commands': ['look'], 'task': 'inspect desk',
                    'history': [], 'command': 'look', 'action_valid': True, 'executable': True,
                    'won': True, 'terminated': True, 'reward': 10.0}
    class RolloutHarness:
        recipe = load('experiment.json')
        environment_url = 'not-a-network-request'
        student_adapter_name = 'student'
        def _ensure_vllm_adapter_weights(self, adapter):
            assert adapter == 'student'
        def _apply_chat_template(self, messages, add_generation_prompt):
            return messages[0]['content'] + '\nassistant-header'
        def _encode_text(self, text):
            return list(map(ord, text))
        def _sample_model_outputs(self, prompts, **kwargs):
            return [SimpleNamespace(prompt_token_ids=value['prompt_token_ids'],
                outputs=[SimpleNamespace(text='<think>check</think><action>look</action>',
                                         token_ids=[7, 8, 9], finish_reason='stop')]) for value in prompts]
    monkeypatch.setattr(rollout, 'Client', LocalService)
    rows = [{'task_id': 'game-one', 'seed': 1, 'split':'train'}, {'task_id': 'game-two', 'seed': 2, 'split':'train'}]
    batch = rollout.run_rollouts(rows, RolloutHarness(), rollout_n=1, temperature=0.4)
    assert all(LocalService.instances.values()) and all(episode.released for episode in batch.episodes)
    assert batch.scores == [1., 1.]
    for episode in batch.episodes:
        metadata = episode.steps[0].agent_metadata
        assert metadata['completion_token_ids'] == [7, 8, 9]
        assert metadata['prompt_token_ids'][-len('assistant-header'):] == list(map(ord, 'assistant-header'))


def test_no_private_dependencies_or_absolute_author_paths():
    import re
    for folder in [ROOT / 'hint_webshop', ROOT / 'scripts', ROOT / 'configs']:
        for path in folder.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            raw = path.read_text(encoding='utf-8')
            assert not re.search(r'(?<!\w)[A-Za-z]:[/\\]', raw), str(path)
            assert 'connect.' + 'seetacloud.com' not in raw
            assert '.ssh/' not in raw and 'BEGIN ' + 'PRIVATE KEY' not in raw


def test_feedback_does_not_inject_an_extra_implicit_system_message():
    class Base:
        def _apply_chat_template(self,messages,add_generation_prompt):
            result = 'IMPLICIT-DEFAULT-SYSTEM\n'
            result += ''.join('<|im_start|>'+item['role']+'\n'+item['content']+'<|im_end|>\n' for item in messages)
            return result+('<|im_start|>assistant\n' if add_generation_prompt else '')
    cls = selected_methods(ROOT/'hint_webshop/trainer.py','WebshopTrainer',
        ['_apply_chat_template','_build_bfcl_per_step_teacher_prompt'],
        bases=[ast.Name(id='Base',ctx=ast.Load())],namespace={'Base':Base})
    harness=cls()
    harness.processing_class=SimpleNamespace(decode=lambda ids,**kwargs:''.join(map(chr,ids)))
    actual='truncated-visible-history<|im_end|>\n<|im_start|>assistant\n'
    episode=SimpleNamespace(steps=[SimpleNamespace(agent_metadata={'prompt_token_ids':list(map(ord,actual))})])
    message=harness._build_bfcl_per_step_teacher_prompt(episode,0,'CORRECT-THIS-STEP')
    teacher=harness._apply_chat_template(message,True)
    assert teacher.startswith(actual[:-len('<|im_start|>assistant\n')])
    assert 'CORRECT-THIS-STEP' in teacher
    assert 'IMPLICIT-DEFAULT-SYSTEM' not in teacher
    assert teacher.count('<|im_start|>user\n') == 1
