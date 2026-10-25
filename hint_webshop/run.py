"""CPU-only launch/config inspection and owned process lifecycle; never touches other jobs."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from .common import ROOT, HINT, assets, atomic_json, checkpoint_complete, devices, load, sha, verify_sources, verify_tasks


def resolved(size, gpu_spec='0,1,2,3', smoke=False):
    ids = devices(gpu_spec)
    recipe = load('experiment.json')
    if smoke:
        recipe['updates'] = 1
    return {'method':'HINT-SD-Multi', 'model':load('models.json')[size], 'recipe':recipe,
        'devices':{'training':ids[0], 'inference':ids[1:]},
        'training_processes':1, 'inference_replicas':3, 'vllm_tensor_parallel_size':1,
        'algorithm':'official selected-turn reverse KL; EMA Teacher; no GRPO or RETHINK',
        'evaluation_manifest_sha256':sha(ROOT/'configs/fixed_validation140.json')}


def preflight(gpus):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    inventory = {parts[0]: (parts[1], int(parts[2]), int(parts[3])) for line in rows.splitlines()
                 if (parts := [p.strip() for p in line.split(',')])}
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
    for gpu in gpus:
        if gpu not in inventory or inventory[gpu][1] < 70*1024 or inventory[gpu][2] < 60000 or inventory[gpu][0] in occupied:
            raise ValueError('Allocated GPU is unavailable or occupied: ' + gpu)


def launch(args):
    config = resolved(args.size, args.gpus, args.smoke)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    verify_sources()
    verify_tasks(assets())
    preflight(devices(args.gpus))
    output = ROOT / 'runs' / ('hint_sd_' + args.size + ('_smoke' if args.smoke else '') +
                             ('_eval' if args.eval_checkpoint or args.eval_base else ''))
    if args.output:
        output = Path(args.output).resolve()
    # Runtime outputs must be private task files, never overwrite the checkout itself.
    if output == ROOT or not output.is_relative_to(ROOT / 'runs'):
        raise ValueError('Output must be a child of this checkout\'s runs directory')
    binding = {'model': config['model'], 'recipe': config['recipe'],
               'validation_manifest_sha256':config['evaluation_manifest_sha256'],
               'task_split_sha256':sha(ROOT/'configs/task_split.json'),
               'asset_receipt_sha256':sha(assets()/'asset_receipt.json'),
               'training_tasks_sha256': sha(ROOT / 'data/train_tasks.json'),
               'upstream_lock_sha256': sha(ROOT / 'configs/upstream.json'),
               'adapter_code': {p.name: sha(p) for p in sorted((ROOT / 'hint_webshop').glob('*.py'))}}
    if args.resume:
        saved = json.loads((output / 'binding.json').read_text())
        if saved != binding:
            raise ValueError('Model, data, configuration or code differs from the saved run')
        candidates = sorted(output.glob('checkpoint-*'), key=lambda p: int(p.name.rsplit('-', 1)[1]), reverse=True)
        resume = next((p for p in candidates if (p / 'complete.json').exists()), None)
        if resume is None:
            raise ValueError('No complete resumable checkpoint; start a new output only with explicit approval')
        from .checkpoint_io import verify as verify_checkpoint
        verify_checkpoint(resume)
        receipt = json.loads((resume / 'complete.json').read_text())
        if any(sha(resume / name) != expected for name, expected in receipt['files'].items()):
            raise ValueError('Checkpoint checksum mismatch')
    else:
        if output.exists():
            raise FileExistsError('Output already exists; use --resume or a different --output')
        resume = None
    model_directory = ROOT / 'models' / config['model']['model_id'].split('/')[-1]
    model_receipt = json.loads((model_directory / 'download_receipt.json').read_text())
    if model_receipt['model_id'] != config['model']['model_id'] or model_receipt['revision'] != config['model']['revision']:
        raise ValueError('Downloaded model receipt is not the pinned base model')
    if any(sha(model_directory / name) != item['sha256'] for name, item in model_receipt['files'].items()):
        raise ValueError('Base model file checksum differs from download receipt')
    if args.eval_checkpoint:
        from .checkpoint_io import verify as verify_checkpoint
        verify_checkpoint(args.eval_checkpoint)
        source_binding = json.loads((args.eval_checkpoint.parent / 'binding.json').read_text())
        if source_binding['model'] != config['model']:
            raise ValueError('Evaluation checkpoint belongs to a different backbone/revision')
    training_python = HINT / '.venv/bin/python'
    environment_python = Path(os.environ.get('HINTSD_ENV_PYTHON', ROOT / '.env-webshop/bin/python'))
    if not training_python.is_file() or not environment_python.is_file():
        raise ValueError('Missing runtime. Run scripts/setup.sh first')
    subprocess.run([str(environment_python), str(ROOT/'src/assets.py'), '--verify'], check=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'src')))
    if not args.resume:
        output.mkdir(parents=True)
        atomic_json(output / 'binding.json', binding)
    import fcntl
    lock_file = (output/'controller.lock').open('a+')
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise RuntimeError('This run already has an active controller') from None
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'src')]),
               TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES=config['devices']['training'],
               HINTSD_INFERENCE_GPUS=','.join(config['devices']['inference']),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
               _JAVA_OPTIONS='-XX:+UseSerialGC -Xss512k -Xmx1g')
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(name, None)
    port = args.port or (18760 if args.size == '1.5b' else 18761)
    service = trainer = None
    with (output / 'environment.log').open('a') as env_log, (output / 'training.log').open('a') as train_log:
        try:
            service_env = dict(env, CUDA_VISIBLE_DEVICES='')
            service = subprocess.Popen([str(environment_python), '-m', 'hint_webshop.environment', '--port', str(port)],
                                       cwd=ROOT, env=service_env, stdout=env_log, stderr=subprocess.STDOUT)
            from .environment import Client
            client = Client(f'http://127.0.0.1:{port}')
            for attempt in range(120):
                if service.poll() is not None:
                    raise RuntimeError('CPU environment service failed; inspect environment.log')
                try:
                    if client.call('health', timeout=2).get('pid') == service.pid:
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Environment service did not become ready')
            command = [str(training_python), '-m', 'hint_webshop.train', '--size', args.size,
                       '--environment-url', client.url, '--output', str(output)]
            if args.smoke:
                command.append('--smoke')
            if resume:
                command += ['--resume', str(resume)]
            if args.eval_checkpoint:
                command += ['--eval-checkpoint', str(args.eval_checkpoint.resolve())]
            if args.eval_base:
                command.append('--eval-base')
            trainer = subprocess.Popen(command, cwd=ROOT, env=env, stdout=train_log, stderr=subprocess.STDOUT)
            atomic_json(output / 'processes.json', {'launcher': os.getpid(), 'trainer': trainer.pid,
                                                   'environment': service.pid, 'gpus': config['devices']})
            print('Started', args.size, 'on GPUs', args.gpus,
                  '; log:', str(output.relative_to(ROOT) / 'training.log'), flush=True)
            result = trainer.wait()
            if result:
                raise RuntimeError(f'Training/evaluation exited with status {result}; outputs retained')
        finally:
            if trainer is not None and trainer.poll() is None:
                trainer.send_signal(signal.SIGINT)
                try:
                    trainer.wait(60)
                except subprocess.TimeoutExpired:
                    trainer.terminate()
                    trainer.wait(30)
            if service is not None and service.poll() is None:
                service.terminate()
                service.wait(30)
            lock_file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train','both','eval'])
    parser.add_argument('--size', choices=['1.5b','3b'], required=True)
    parser.add_argument('--gpus', default=os.environ.get('HINTSD_GPUS','0,1,2,3'))
    parser.add_argument('--port', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--eval-checkpoint', type=Path)
    parser.add_argument('--eval-base', action='store_true')
    args = parser.parse_args()
    if args.command == 'eval' and not (args.eval_base or args.eval_checkpoint):
        parser.error('eval requires --eval-base or --eval-checkpoint')
    if args.command != 'eval' and (args.eval_base or args.eval_checkpoint):
        parser.error('Evaluation options require the eval command')
    if args.eval_base and args.eval_checkpoint or args.resume and args.command=='eval':
        parser.error('Conflicting evaluation/resume options')
    launch(args)  # "both" is a compatibility alias for this repo's ONE backbone, never two jobs.


if __name__ == '__main__':
    main()
