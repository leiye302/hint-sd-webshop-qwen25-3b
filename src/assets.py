"""Download pinned public WebShop assets and build the unchanged Lucene index."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request
import uuid
from common import ROOT, WEBSHOP, atomic_json, load, sha


PACKAGES = {'numpy': '1.26.4', 'spacy': '3.7.2', 'pyserini': '0.17.0',
            'faiss-cpu': '1.7.4', 'en-core-web-sm': '3.7.1', 'cymem': '2.0.8',
            'preshed': '3.0.9', 'murmurhash': '1.0.10', 'srsly': '2.4.8'}


def download_file(destination, urls, expected):
    destination = Path(destination)
    if destination.is_symlink():
        raise RuntimeError('Refuse linked asset destination')
    if destination.exists():
        if destination.stat().st_size == expected['bytes'] and sha(destination) == expected['sha256']:
            return
        raise RuntimeError('Existing asset differs; preserve and inspect it: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for url in urls:
        temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.pending')
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'WebShop-GRPO-reproduction/1'})
            with urllib.request.urlopen(request, timeout=120) as source, temporary.open('xb') as target:
                while block := source.read(1024**2):
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            if temporary.stat().st_size != expected['bytes'] or sha(temporary) != expected['sha256']:
                raise RuntimeError('Public mirror checksum mismatch')
            temporary.rename(destination)
            return
        except (OSError, ValueError, RuntimeError) as error:
            failures.append(type(error).__name__)
            # Failed downloads stay as evidence; never replace an accepted asset.
    raise RuntimeError('Public asset download failed (' + ', '.join(failures) + '): ' + destination.name)


def download():
    lock = load('assets.json')
    for name, expected in lock['files'].items():
        urls = [f'https://huggingface.co/datasets/{m["repository"]}/resolve/{m["revision"]}/{name}?download=true'
                for m in lock['mirrors']]
        download_file(WEBSHOP / 'data' / name, urls, expected)
    verify_data()


def install_nlp():
    model = load('assets.json')['nlp']
    target = ROOT / '.deps/downloads' / model['filename']
    download_file(target, [model['url']], model)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', str(target)], check=True)


def verify_data():
    lock = load('assets.json')
    for name, expected in lock['files'].items():
        path = WEBSHOP / 'data' / name
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != expected['bytes']
                or sha(path) != expected['sha256']):
            raise RuntimeError('Missing/mismatched WebShop data: ' + name)
    products = json.loads((WEBSHOP / 'data/items_shuffle_1000.json').read_text(encoding='utf-8'))
    if not isinstance(products, list) or len(products) != 1000:
        raise RuntimeError('Expected the exact public 1000-product raw catalog')
    return {name: info['sha256'] for name, info in lock['files'].items()}


def java_version():
    command = str(Path(os.environ['JAVA_HOME']) / 'bin/java') if os.environ.get('JAVA_HOME') else 'java'
    result = subprocess.run([command, '-version'], text=True, capture_output=True, check=True)
    version = result.stdout + result.stderr
    if not re.search(r'version "11[."]', version):
        raise RuntimeError('Native pyserini0.17 requires the documented OpenJDK11 runtime')
    return version.strip()


def versions():
    found = {name: importlib.metadata.version(name) for name in PACKAGES}
    if found != PACKAGES:
        raise RuntimeError('WebShop dependency/NLP model version mismatch: ' + str(found))
    return found


def index_files():
    root = WEBSHOP / 'search_engine'
    paths = [root / 'resources/documents.jsonl']
    paths += sorted(p for p in (root / 'indexes').glob('*') if p.name != 'write.lock')
    if not any(path.name.startswith('segments_') for path in paths):
        raise RuntimeError('No complete native Lucene index')
    if any(p.is_symlink() or not p.is_file() or not p.stat().st_size for p in paths):
        raise RuntimeError('Missing/linked/empty native index files')
    return {p.relative_to(WEBSHOP).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in paths}


def verify():
    data = verify_data()
    current = versions()
    java_version()
    receipt_path = WEBSHOP / 'asset_receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (receipt['asset_lock_sha256'] != sha(ROOT / 'configs/assets.json')
            or receipt['corpus_sha256'] != data or receipt['packages'] != current
            or receipt['index_files'] != index_files()):
        raise RuntimeError('WebShop assets/index/receipt changed; do not resume with different data')
    return receipt


def build_index():
    verify_data()
    versions()
    java = java_version()
    receipt_path = WEBSHOP / 'asset_receipt.json'
    if receipt_path.exists():
        verify()
        return
    search = WEBSHOP / 'search_engine'
    for name in ('indexes', 'indexes_100', 'indexes_1k', 'indexes_100k'):
        target = search / name
        if target.is_symlink() or (target.exists() and any(target.iterdir())):
            raise RuntimeError('Unverified index exists; move it aside before rebuilding: ' + name)
    for name in ('resources', 'resources_100', 'resources_1k', 'resources_100k'):
        (search / name).mkdir(exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OMP_NUM_THREADS='1', _JAVA_OPTIONS='-XX:+UseSerialGC -Xss512k -Xmx1g')
    subprocess.run([sys.executable, 'convert_product_file_format.py'], cwd=search, env=environment, check=True)
    # Invoke the original four indexing commands, not a replacement retriever.
    subprocess.run(['bash', 'run_indexing.sh'], cwd=search, env=environment, check=True)
    files = index_files()
    receipt = {'asset_lock_sha256': sha(ROOT / 'configs/assets.json'), 'corpus_sha256': verify_data(),
               'packages': versions(), 'java': java, 'index_files': files,
               'documents_sha256': sha(search / 'resources/documents.jsonl')}
    atomic_json(receipt_path, receipt)
    verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--nlp', action='store_true')
    parser.add_argument('--index', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error('Choose --download, --nlp, --index or --verify')
    if args.download:
        download()
    if args.nlp:
        install_nlp()
    if args.index:
        build_index()
    if args.verify:
        verify()
    print('WebShop asset preparation/checks completed; no Actor/GPU training started.')


if __name__ == '__main__':
    main()
