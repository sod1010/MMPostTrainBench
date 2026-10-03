"""Pinned static guard plus host handshake before any candidate execution."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import struct
import subprocess
import tarfile
import time

PROFILE = 'entry-guard-v1'
GUARD_PATH = '/opt/mmptb-guard/guard'
PIN_ENV = 'MMSWE_RUNTIME_GUARD'
SHA_ENV = 'MMSWE_RUNTIME_GUARD_SHA256'
INSTANCE_CAPS = 0xa00405fb


def load_guard(pin):
    from .policy import private_file
    if (not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}
            or not isinstance(pin['path'],str) or not isinstance(pin['sha256'],str)
            or not re.fullmatch('[a-f0-9]{64}',pin['sha256'])):
        raise ValueError('entry guard requires a private binary path and SHA256')
    path = private_file(pin['path'])
    if not 0 < path.stat().st_size <= 5*1024*1024 or not os.access(path, os.X_OK):
        raise ValueError('invalid entry guard executable')
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != pin['sha256']:
        raise ValueError('entry guard bytes changed')
    # This portable instance guard is intentionally limited to Linux x86-64.
    if len(body) < 64 or body[:7] != b'\x7fELF\x02\x01\x01' or struct.unpack_from('<H', body, 18)[0] != 62:
        raise ValueError('entry guard must be a Linux x86-64 ELF')
    offset = struct.unpack_from('<Q', body, 32)[0]
    size, count = struct.unpack_from('<HH', body, 54)
    if size != 56 or not count or offset+size*count > len(body):
        raise ValueError('invalid entry guard ELF headers')
    # A static executable must have neither an interpreter nor a dynamic segment.
    if any(struct.unpack_from('<I', body, offset+i*size)[0] in (2, 3) for i in range(count)):
        raise ValueError('entry guard must be statically linked')
    return dict(pin)


def guard_from_environment(env, profile):
    if profile != PROFILE:
        if PIN_ENV in env or SHA_ENV in env:
            raise ValueError('entry guard pin requires entry-guard-v1')
        return None
    return load_guard({'path':env.get(PIN_ENV), 'sha256':env.get(SHA_ENV)})


def stage_guard(rootfs, pin):
    load_guard(pin)
    rootfs = Path(rootfs)
    opt = rootfs/'opt'
    if opt.is_symlink() or (opt.exists() and not opt.is_dir()):
        raise ValueError('unsafe guard destination')
    opt.mkdir(exist_ok=True)
    target = rootfs/GUARD_PATH.lstrip('/')
    target.parent.mkdir(mode=0o755, exist_ok=False)
    shutil.copyfile(pin['path'], target)
    target.chmod(0o555)
    if hashlib.sha256(target.read_bytes()).hexdigest() != pin['sha256']:
        raise ValueError('entry guard changed while staging')


def command(kind, purpose, nonce):
    if kind not in ('workload', 'instance') or not isinstance(nonce,str) or not re.fullmatch('[a-f0-9]{64}', nonce):
        raise ValueError('invalid entry guard invocation')
    allowed = ('train', 'generate', 'probe') if kind == 'workload' else ('grade', 'probe')
    if purpose not in allowed:
        raise ValueError('invalid entry guard purpose')
    return [kind, purpose, nonce]


def validate_launch(info, kind, purpose, nonce, image):
    config = info.get('Config', {})
    if (config.get('Entrypoint') != [GUARD_PATH] or config.get('Cmd') != command(kind, purpose, nonce)
            or config.get('OpenStdin') is not True or config.get('Tty') is not False
            or image not in (info.get('Image'), config.get('Image'))):
        raise ValueError('entry guard launch or image differs from fixed request')
    if kind == 'workload':
        for mount in info.get('Mounts', []):
            target = mount.get('Destination', '')
            if target != '/output' and not target.startswith('/input/'):
                raise ValueError('unexpected workload mount may replace trusted entry')


def validate_attestation(value, kind, purpose, nonce):
    if (not isinstance(value, dict) or type(value.get('protocol')) is not int or value.get('protocol') != 1 or value.get('kind') != kind
            or value.get('purpose') != purpose or value.get('nonce') != nonce):
        raise ValueError('entry guard identity mismatch')
    uid = 65534 if kind == 'workload' else 0
    if any(type(value.get(k)) is not int or value[k] != v for k,v in
           [('uid',uid),('gid',uid),('groups',0),('nnp',1)]):
        raise ValueError('entry guard identity or NNP not enforced')
    caps = value.get('caps')
    if not isinstance(caps, dict) or set(caps) != {'eff','prm','inh','bnd','amb'}:
        raise ValueError('incomplete capability attestation')
    allowed = 0 if kind == 'workload' else INSTANCE_CAPS
    for key, text in caps.items():
        if not isinstance(text, str) or not re.fullmatch('[a-f0-9]{16}', text):
            raise ValueError('invalid capability attestation')
        bits = int(text,16)
        if bits & ~allowed or (key in ('inh','amb') and bits):
            raise ValueError('entry guard capabilities not restricted')
    return value


def docker_env(home):
    return {'PATH':'/usr/bin:/bin', 'HOME':str(home)}


def verify_container_guard(binary, home, cid, pin, timeout=30):
    load_guard(pin)
    # No candidate has run. Only the fixed guard from a stopped container is read.
    proc = subprocess.Popen([binary,'cp',cid+':'+GUARD_PATH,'-'], env=docker_env(home),
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    blocks, size = [], 0
    end = time.monotonic()+timeout
    try:
        with selectors.DefaultSelector() as poll:
            poll.register(proc.stdout, selectors.EVENT_READ)
            while True:
                if time.monotonic() >= end: raise TimeoutError('guard copy timeout')
                if not poll.select(min(1, max(.01,end-time.monotonic()))): continue
                block = os.read(proc.stdout.fileno(), 65536)
                if not block: break
                size += len(block)
                if size > 6*1024*1024: raise ValueError('oversized guard copy')
                blocks.append(block)
        if proc.wait(timeout=max(.01,end-time.monotonic())):
            raise ValueError('missing guard in pinned image')
        with tarfile.open(fileobj=io.BytesIO(b''.join(blocks)),mode='r:*') as tar:
            files = tar.getmembers()
            if len(files) != 1 or not files[0].isfile() or files[0].size > 5*1024*1024:
                raise ValueError('invalid guard archive')
            digest = hashlib.sha256(tar.extractfile(files[0]).read()).hexdigest()
        if digest != pin['sha256']:
            raise ValueError('image guard does not match operator pin')
    finally:
        if proc.poll() is None: proc.kill()
        proc.wait(); proc.stdout.close()


def guarded_start(binary, home, cid, kind, purpose, nonce, log, deadline):
    """Attach stdin, validate the first trusted line, ACK, then drain bounded logs.

    Caller must validate launch and binary BEFORE calling, and remove the Docker
    container in finally even if attaching, protocol validation or logging fails.
    """
    command(kind,purpose,nonce)
    proc = subprocess.Popen([binary,'start','--attach','--interactive',cid], env=docker_env(home),
                            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    attestation, buffer, total = None, b'', 0
    startup = min(deadline,time.monotonic()+30)
    try:
        with Path(log).open('xb') as output, selectors.DefaultSelector() as poll:
            poll.register(proc.stdout,selectors.EVENT_READ)
            while True:
                end = deadline if attestation is not None else startup
                if time.monotonic() >= end: raise TimeoutError('entry guard execution timeout')
                if not poll.select(min(1,max(.01,end-time.monotonic()))): continue
                block = os.read(proc.stdout.fileno(),65536)
                if not block: break
                total += len(block)
                if total > 64*1024*1024: raise ValueError('container output limit')
                output.write(block); output.flush()
                if attestation is None:
                    buffer += block
                    if len(buffer)>4096: raise ValueError('oversized guard handshake')
                    if b'\n' not in buffer: continue
                    line, remainder = buffer.split(b'\n',1)
                    if remainder: raise ValueError('unexpected output before host approval')
                    attestation = validate_attestation(json.loads(line),kind,purpose,nonce)
                    proc.stdin.write(('GO '+nonce+'\n').encode()); proc.stdin.flush()
                    proc.stdin.close()
        code = proc.wait(timeout=max(.01,deadline-time.monotonic()))
        if attestation is None: raise ValueError('guard exited before verification; no approval sent')
        return {'attestation':attestation, 'attach_returncode':code}
    finally:
        if not proc.stdin.closed: proc.stdin.close()
        if proc.poll() is None: proc.kill()
        proc.wait(); proc.stdout.close()
