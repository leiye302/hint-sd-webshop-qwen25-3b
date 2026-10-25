"""Fetch public pinned sources/models and prepare the WebShop training task list."""
import argparse
import json
import uuid
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from .common import ROOT, HINT, SDAR, assets, atomic_json, git, load, sha, verify_sources, verify_tasks


def framework(mirrors=None):
    lock = load('upstream.json')
    for key, target in [('hint_sd', HINT), ('sdar_environment', SDAR)]:
        source = lock[key]
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + '-' + uuid.uuid4().hex[:8])
        mirror = (mirrors or {}).get(key)
        if mirror:
            git(ROOT, 'clone', '--no-checkout', '--no-hardlinks', str(Path(mirror).resolve()), str(staging))
            git(staging, 'checkout', '--detach', source['commit'])
            git(staging, 'remote', 'set-url', 'origin', source['repository'])
        else:
            staging.mkdir()
            git(staging, 'init')
            git(staging, 'remote', 'add', 'origin', source['repository'])
            git(staging, 'fetch', '--depth', '1', 'origin', source['commit'])
            git(staging, 'checkout', '--detach', 'FETCH_HEAD')
        staging.rename(target)
    verify_sources()
    print('Unmodified official HINT-SD and public WebShop environment sources verified.')


def prepare():
    verify_sources()
    from .environment import native_parts, env_kwargs
    from task_split import manifest, task_identity, partition_goals
    Worker, _, _, _ = native_parts()
    worker = Worker(0, env_kwargs())
    try:
        goals = partition_goals(worker.env.unwrapped.server.goals, True)
        records = [dict(task_id=task_identity(goal), goal_index=i, seed=0, split='train')
                   for i, goal in enumerate(goals)]
        if len(records) != manifest()['training_goal_entries']:
            raise ValueError('Training pool size differs from shared140 split')
        atomic_json(ROOT / 'data/train_tasks.json', records)
    finally:
        worker.close()
    print('Prepared', len(records), 'native train entries; all140 validation families excluded.')


def download(size):
    from huggingface_hub import snapshot_download
    spec = load('models.json')[size]
    target = ROOT / 'models' / spec['model_id'].split('/')[-1]
    snapshot_download(spec['model_id'], revision=spec['revision'], local_dir=target,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE', 'README.md'])
    atomic_json(target / 'download_receipt.json', dict(spec, files={
        path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)}
        for path in sorted(target.iterdir()) if path.is_file() and path.name != 'download_receipt.json'}))
    print('Downloaded and hashed pinned base model:', spec['model_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--prepare-data', action='store_true')
    parser.add_argument('--download-model', choices=['1.5b', '3b', 'both'])
    parser.add_argument('--hint-source', type=Path, help='Optional read-only public Git mirror for offline source preparation')
    parser.add_argument('--sdar-source', type=Path, help='Optional read-only public Git mirror for offline source preparation')
    args = parser.parse_args()
    if not any([args.framework, args.verify, args.prepare_data, args.download_model]):
        parser.error('Choose an operation')
    if args.framework:
        framework({'hint_sd': args.hint_source, 'sdar_environment': args.sdar_source})
    if args.verify:
        verify_sources()
        print('Source and task-manifest locks passed.')
    if args.prepare_data:
        prepare()
    if args.download_model:
        for size in (['1.5b', '3b'] if args.download_model == 'both' else [args.download_model]):
            download(size)


if __name__ == '__main__':
    main()
