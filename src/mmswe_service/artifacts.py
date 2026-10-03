"""Freeze untrusted inputs into private, content-addressed copies (never hardlinks)."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import time


class Rejected(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def relative(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise Rejected('invalid relative path')
    parts = value.split('/')
    if any(p in ('', '.', '..') for p in parts) or PurePosixPath(value).is_absolute():
        raise Rejected('path must stay inside the registered workspace')
    return parts


def open_dir(path):
    """Walk from / using directory descriptors, never following symlinks."""
    p = Path(path)
    if not p.is_absolute():
        raise Rejected('registered path must be absolute')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in p.parts[1:]:
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new
        return fd
    except BaseException:
        os.close(fd)
        raise


def discard_private_tree(path):
    """Remove a broker-created tree, including its read-only frozen directories.

    Only call on private copies after their writer has stopped, never on agent
    source paths or a live container output.
    """
    path = Path(path)
    if not path.exists():
        return
    for root, dirs, _ in os.walk(path, followlinks=False):
        os.chmod(root, 0o700)
    shutil.rmtree(path)


def freeze(root, name, destination, max_bytes, max_files=20000, deadline=None):
    """A concurrent writer cannot retain access to the copy used for evaluation.

    File size, inode and mtime are checked around each read. This is not an
    atomic snapshot of a changing source tree; the digest binds the bytes that
    were actually copied. Model loading and index checks happen on that copy.
    """
    fd = open_dir(root)
    try:
        for part in relative(name):
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new
        return freeze_fd(fd, destination, max_bytes, max_files, deadline)
    finally:
        os.close(fd)


def freeze_fd(fd, destination, max_bytes, max_files, deadline=None):
    dest = Path(destination)
    dest.mkdir(mode=0o700)
    files, used, entries = [], 0, 0

    def copy_tree(src, target, prefix='', depth=0):
        nonlocal used, entries
        if depth > 20:
            raise Rejected('tree nesting limit')
        for name in sorted(os.listdir(src)):
            entries += 1
            if entries > max_files:
                raise Rejected('artifact entry limit')
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("artifact freeze deadline")
            relative(name)
            old = os.stat(name, dir_fd=src, follow_symlinks=False)
            rel = prefix + name
            if stat.S_ISDIR(old.st_mode):
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=src)
                try:
                    (target/name).mkdir(mode=0o700)
                    copy_tree(child, target/name, rel+'/', depth+1)
                finally:
                    os.close(child)
            elif stat.S_ISREG(old.st_mode):
                if len(files) >= max_files or old.st_size > max_bytes-used:
                    raise Rejected('artifact size/file limit')
                srcfd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=src)
                try:
                    before = os.fstat(srcfd)
                    if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (old.st_dev, old.st_ino):
                        raise Rejected('source changed during open')
                    h, size = hashlib.sha256(), 0
                    with (target/name).open('xb') as out:
                        while True:
                            if deadline is not None and time.monotonic() >= deadline:
                                raise TimeoutError("artifact freeze deadline")
                            block = os.read(srcfd, min(4*1024*1024, max_bytes-used+1))
                            if not block:
                                break
                            size += len(block)
                            used += len(block)
                            if used > max_bytes:
                                raise Rejected('artifact grew beyond quota')
                            h.update(block)
                            out.write(block)
                    after = os.fstat(srcfd)
                    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns) or size != before.st_size:
                        raise Rejected('source changed while copying')
                    (target/name).chmod(0o444)
                    files.append({'path': rel, 'size': size, 'sha256': h.hexdigest()})
                finally:
                    os.close(srcfd)
            else:
                raise Rejected('symlinks and special files are not accepted')
        target.chmod(0o555)

    try:
        copy_tree(fd, dest)
        if not files:
            raise Rejected('empty artifact')
    except BaseException:
        discard_private_tree(dest)
        raise
    return {'sha256': hashlib.sha256(canonical(files)).hexdigest(), 'bytes': used, 'files': files}


def validate_model(path):
    """Accept data-only Qwen3-Omni exports, not pickle or custom model Python."""
    path = Path(path)
    files = {p.name for p in path.iterdir()}
    if any(p.is_dir() for p in path.iterdir()):
        raise Rejected('export a flat inference model without optimizer directories')
    allowed = {'.json', '.safetensors', '.jinja', '.txt', '.model'}
    if any(Path(n).suffix not in allowed for n in files):
        raise Rejected('model contains executable or unsupported files')
    if not {'config.json', 'tokenizer_config.json'} <= files:
        raise Rejected('missing model/tokenizer configuration')
    for p in path.glob('*.json'):
        if p.stat().st_size > 32*1024*1024:
            raise Rejected('oversized model metadata')
        obj = json.loads(p.read_text())
        if isinstance(obj, dict) and any(obj.get(k) for k in ('auto_map', 'custom_pipelines')):
            raise Rejected('remote/custom model code is not accepted')
    if json.loads((path/'config.json').read_text()).get('model_type') != 'qwen3_omni_moe':
        raise Rejected('unexpected model family')
    shards = {p.name for p in path.glob('*.safetensors')}
    if not shards:
        raise Rejected('missing safetensors weights')
    index = path/'model.safetensors.index.json'
    if index.exists():
        refs = set(json.loads(index.read_text())['weight_map'].values())
        if refs != shards:
            raise Rejected('index does not match weight shards')
    elif shards != {'model.safetensors'}:
        raise Rejected('sharded model needs an index')
    for name in shards:
        p = path/name
        with p.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise Rejected('truncated safetensors')
            n = struct.unpack('<Q', raw)[0]
            if not 2 <= n <= min(64*1024*1024, p.stat().st_size-8):
                raise Rejected('invalid safetensors header')
            header = json.loads(f.read(n))
        offsets = sorted(v['data_offsets'] for k, v in header.items() if k != '__metadata__')
        if not offsets or offsets[0][0] != 0 or any(a > b for a, b in offsets) or any(a[1] != b[0] for a, b in zip(offsets, offsets[1:])) or offsets[-1][1] != p.stat().st_size-8-n:
            raise Rejected('invalid safetensors extents')
