"""Small, CPU-only source locks, task bindings and output helpers."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HINT = ROOT / '.deps/hint-sd'
SDAR = ROOT / '.deps/sdar'


def load(name):
    return json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def git(path, *args):
    return subprocess.run(['git', '-c', 'core.autocrlf=false', '-C', str(path), *args],
                          check=True, capture_output=True, text=True,
                          encoding='utf-8', errors='replace').stdout.strip()


def devices(spec):
    result = [x.strip() for x in spec.split(',')]
    if len(result) != 4 or len(set(result)) != 4 or any(not x.isdigit() for x in result):
        raise ValueError('HINTSD_GPUS must contain four distinct allocated physical GPU indices')
    return result


def assets():
    return SDAR / 'agent_system/environments/env_package/webshop/webshop'


def task_rows():
    value = load('fixed_validation140.json')
    rows = [dict(task_id=row['task_id'], seed=row['worker_seed'],
                 goal_index=row['goal_index'], split='validation')
            for row in value['selection']]
    if len(rows) != 140 or len({row['task_id'] for row in rows}) != 140:
        raise ValueError('Expected the shared fixed140 WebShop tasks')
    if any(row['seed'] != 1000+i or row['goal_index'] != i for i,row in enumerate(rows)):
        raise ValueError('Fixed task seeds/indices changed')
    return rows


def verify_tasks(root=None):
    del root
    for name, field in [('fixed_validation140.json', 'validation_manifest_sha256'),
                        ('task_split.json', 'task_split_sha256'), ('assets.json', 'assets_lock_sha256')]:
        if sha(ROOT / 'configs' / name) != load('upstream.json')[field]:
            raise ValueError('Shared WebShop task/asset manifest changed: ' + name)
    task_rows()


def verify_sources():
    lock = load('upstream.json')
    for key, directory in [('hint_sd', HINT), ('sdar_environment', SDAR)]:
        source = lock[key]
        if git(directory, 'rev-parse', 'HEAD') != source['commit']:
            raise ValueError('Upstream revision differs: ' + key)
        if git(directory, 'diff', '--name-only', 'HEAD'):
            raise ValueError('Upstream tracked files are dirty: ' + key)
        for name, expected in source['files'].items():
            content = (directory / name).read_bytes().replace(b'\r\n', b'\n')
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('Locked upstream source changed: ' + name)
    verify_tasks()
    return lock


def checkpoint_complete(path):
    path = Path(path)
    required = ['adapter_config.json', 'adapter_model.safetensors',
                'teacher_adapter/adapter_config.json', 'teacher_adapter/adapter_model.safetensors',
                'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json',
                'training_args.bin']
    missing = [name for name in required if not (path / name).is_file()
               or (path / name).stat().st_size == 0]
    if missing:
        raise ValueError('Incomplete resumable checkpoint: ' + ', '.join(missing))
    return path
