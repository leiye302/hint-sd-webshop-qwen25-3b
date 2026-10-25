"""Storage-only wrapper: native serialization, verified publish, then latest-two rotation."""
import json
from pathlib import Path
import shutil
import tempfile
import zipfile
from .common import atomic_json, checkpoint_complete, sha


def binding(root):
    path = Path(root)/'binding.json'
    return sha(path) if path.is_file() else None


def verify(path, require_receipt=True):
    path = checkpoint_complete(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError('Linked checkpoint refused')
    receipt_path = path/'complete.json'
    if not receipt_path.is_file():
        if require_receipt:
            raise ValueError('No verified completion receipt')
        return path
    receipt = json.loads(receipt_path.read_text())
    actual = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file() and p.name!='complete.json'}
    if actual != set(receipt['files']):
        raise ValueError('Checkpoint inventory differs')
    for name,expected in receipt['files'].items():
        target = (path/name).resolve()
        if not target.is_relative_to(path.resolve()) or sha(target) != expected:
            raise ValueError('Checkpoint checksum/path mismatch')
    if receipt.get('binding_sha256') != binding(path.parent):
        raise ValueError('Checkpoint run binding differs')
    return path


def seal(path, run_root):
    path = verify(path, require_receipt=False)
    for item in path.rglob('*'):
        if item.suffix in ('.pt','.pth','.bin'):
            with zipfile.ZipFile(item) as archive:
                if archive.testzip() is not None:
                    raise ValueError('Native checkpoint CRC failed')
    atomic_json(path/'complete.json', {'update':int(path.name.split('-')[-1]),
        'student_and_ema_teacher':True, 'binding_sha256':binding(run_root),
        'files':{p.relative_to(path).as_posix():sha(p) for p in path.rglob('*')
                 if p.is_file() and p.name!='complete.json'}})
    return path


class VerifiedCheckpointMixin:
    def _save_checkpoint(self, *args, **kwargs):
        root = Path(self.args.output_dir).resolve()
        name = 'checkpoint-'+str(self.state.global_step)
        destination = root/name
        if destination.exists():
            raise FileExistsError('Refuse to overwrite an existing checkpoint')
        staging = Path(tempfile.mkdtemp(prefix='.saving-',dir=root))
        old_dir,old_limit = self.args.output_dir,self.args.save_total_limit
        try:
            self.args.output_dir,self.args.save_total_limit = str(staging),None
            super()._save_checkpoint(*args,**kwargs)  # Original native HF/HINT serializer.
            seal(staging/name, root)
            (staging/name).rename(destination)
            verify(destination)
        finally:
            self.args.output_dir,self.args.save_total_limit = old_dir,old_limit
        try:
            staging.rmdir()
        except OSError:
            pass  # Partial evidence is not deleted.
        healthy = []
        for path in root.glob('checkpoint-*'):
            try:
                verify(path)
                healthy.append(path)
            except (OSError,ValueError,KeyError,zipfile.BadZipFile):
                continue
        healthy.sort(key=lambda p:int(p.name.split('-')[-1]))
        for path in healthy[:-2]:
            if path.resolve().parent != root or path.is_symlink():
                raise ValueError('Unsafe checkpoint rotation target')
            shutil.rmtree(path)

    def _load_from_checkpoint(self, resume_from_checkpoint, *args, **kwargs):
        verify(resume_from_checkpoint)
        return super()._load_from_checkpoint(resume_from_checkpoint,*args,**kwargs)
