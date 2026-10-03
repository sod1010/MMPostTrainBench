#!/usr/bin/env python3
"""Candidate ordinary-container backend for the existing native grader.

Keep official image verification, patch application, assets and parsing. Replace
ONLY the execution function: import the verified, staged rootfs and run its
script as an ordinary container. No extra capabilities, host mounts, host PID
namespace, Docker socket, or privileged mode are passed to the instance.

Requires a platform-approved Docker daemon. It does not grant capabilities or
retry an authorization denial with weaker isolation. Cold import per instance
is intentionally conservative; throughput acceptance is still required.
"""
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tarfile
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmswe_service.offline_resources import snapshot_from_environment, host_args, check_hosts, stage_snapshot
from mmswe_service import runtime_guard as guard
try:
    from mmswe_service.container_policy import profile_name, security_args, inspect_security
    from mmswe_service.resources import resources_from_environment, resource_args, inspect_resources
except ModuleNotFoundError:
    from container_policy import profile_name, security_args, inspect_security
    from resources import resources_from_environment, resource_args, inspect_resources

HERE = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location('mmswe_native', HERE.parents[1]/'eval/tasks/mmswe/dlc_native_grade.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
BLOCKED = threading.Event()
EXECUTION_LOCK = threading.Lock()


def cli(args, timeout=60, check=True, **kwargs):
    env = {'PATH': '/usr/bin:/bin', 'HOME': os.environ.get('MMSWE_DOCKER_HOME', '/nonexistent')}
    result = subprocess.run([os.environ.get('MMSWE_DOCKER', '/usr/bin/docker'), *args],
                            env=env, timeout=timeout, **kwargs)
    if check and result.returncode:
        raise RuntimeError('ordinary Docker operation rejected or failed')
    return result


def instance_args(name, image, service, job, profile='strict-v1', resources=None, snapshot=None, nonce=None):
    guarded = profile == guard.PROFILE
    args = ['create', '--name', name, '--label', 'mmptb.service='+service,
            '--label', 'mmptb.job='+job, '--pull', 'never', '--init',
            '--user', '0:0', '--network', 'none',
            '--log-driver', 'local', '--log-opt', 'max-size=10m', '--log-opt', 'max-file=2',
            '--entrypoint', guard.GUARD_PATH if guarded else '/bin/bash']
    if guarded:
        args += ['--interactive']
    return args + resource_args(resources) + host_args(snapshot) + security_args('instance', profile) + [image] + (guard.command('instance','grade',nonce) if guarded else ['/run_in_chroot.sh'])


def extract_log(archive, maximum=64*1024*1024):
    """Never extract a container-controlled tar to the broker filesystem."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:*') as tar:
        entries = tar.getmembers()
        if len(entries) != 1 or not entries[0].isfile() or entries[0].size > maximum:
            raise RuntimeError('invalid container log archive')
        return tar.extractfile(entries[0]).read(maximum+1)


def copy_log(cid, filename, dest, timeout):
    # Use a capped spool rather than unbounded subprocess capture of test logs.
    binary = os.environ.get('MMSWE_DOCKER', '/usr/bin/docker')
    env = {'PATH': '/usr/bin:/bin', 'HOME': os.environ.get('MMSWE_DOCKER_HOME', '/nonexistent')}
    import selectors
    proc = subprocess.Popen([binary, 'cp', cid+':/tmp/'+filename, '-'], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, env=env)
    chunks, size = [], 0
    deadline = time.monotonic()+timeout
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('container log copy timed out')
                if not selector.select(min(remaining, 1)):
                    continue
                block = os.read(proc.stdout.fileno(), 1024*1024)
                if not block:
                    break
                size += len(block)
                if size > 65*1024*1024:
                    raise RuntimeError('container log too large')
                chunks.append(block)
        rc = proc.wait(timeout=max(.01, deadline-time.monotonic()))
        if rc:
            if filename in ('ns_diag.log',):
                return
            raise RuntimeError('missing container execution evidence')
        Path(dest).write_bytes(extract_log(b''.join(chunks)))
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdout.close()


def run_in_docker(rootfs, logf, timeout=None):
    # Native grade_one records exceptions and continues. A failed cleanup must
    # still prevent subsequent instances from launching in this invocation.
    with EXECUTION_LOCK:
        if BLOCKED.is_set():
            raise RuntimeError('Docker backend blocked after cleanup failure')
        return _run_in_docker(rootfs, logf, timeout)


def operator_settings():
    """Reject incompatible launchers before expensive image preparation."""
    if 'MMSWE_DOCKER_HARDENING' in os.environ:
        raise RuntimeError('MMSWE_DOCKER_HARDENING is unsupported; select an operator container profile')
    if os.environ.get('MMSWE_DOCKER_NETWORK', 'none') != 'none':
        raise RuntimeError('MMSWE_DOCKER_NETWORK is unsupported except none; unrestricted egress is not a release profile')
    profile = profile_name(os.environ.get('MMSWE_CONTAINER_PROFILE', 'strict-v1'))
    guard.guard_from_environment(os.environ,profile)
    service = os.environ.get('MMSWE_SERVICE_ID', 'standalone-acceptance')
    job = os.environ.get('MMSWE_JOB_ID', uuid.uuid4().hex)
    if not re.fullmatch('[a-zA-Z0-9_-]{1,64}', service) or not re.fullmatch('[a-zA-Z0-9_-]{1,64}', job):
        raise RuntimeError('invalid operator identity')
    return profile, service, job, resources_from_environment(os.environ), snapshot_from_environment(os.environ)


def _run_in_docker(rootfs, logf, timeout=None):
    profile, service, job, resources, snapshot = operator_settings()
    pin = guard.guard_from_environment(os.environ,profile)
    nonce = secrets.token_hex(32) if pin else None
    suffix = uuid.uuid4().hex
    name = 'mmptb-grade-'+suffix
    tag = 'mmptb-grade:'+suffix
    archive = rootfs.parent/'docker-rootfs.tar'
    # Runtime timeout starts after import; the service separately bounds the whole job.
    cid = None
    cleanup_errors = []
    try:
        if pin:
            guard.stage_guard(rootfs,pin)
        stage_snapshot(rootfs, snapshot)
        if snapshot:
            logf.write('offline_snapshot='+json.dumps(snapshot['pin'],sort_keys=True)+'\n')
        with tarfile.open(archive, 'w', dereference=False) as tar:
            for p in sorted(rootfs.iterdir()):
                tar.add(p, arcname=p.name, recursive=True)
        logf.write('backend=ordinary-docker; no cap-add/privileged/host mounts\n')
        logf.flush()
        with archive.open('rb') as src:
            result = cli(['import', '--change', 'LABEL mmptb.service='+service,
                          '--change', 'LABEL mmptb.job='+job, '-', tag],
                         timeout=1800, stdin=src, capture_output=True, text=False)
        image = result.stdout.decode().strip()
        if not re.fullmatch('sha256:[a-f0-9]{64}', image):
            raise RuntimeError('invalid imported image identity')
        result = cli(instance_args(name, image, service, job, profile, resources, snapshot, nonce), capture_output=True, text=True)
        cid = result.stdout.strip()
        if not re.fullmatch('[a-f0-9]{64}', cid):
            raise RuntimeError('invalid instance container identity')
        inspected = json.loads(cli(['inspect', cid], capture_output=True, text=True).stdout)
        if not isinstance(inspected, list) or len(inspected) != 1:
            raise RuntimeError('invalid instance container inspection')
        receipt = inspect_security(inspected[0], 'instance', profile)
        logf.write('container_security='+json.dumps(receipt, sort_keys=True)+'\n')
        logf.flush()
        if not receipt['compliant']:
            raise RuntimeError('container security policy mismatch; official tests not started')
        resource_receipt = inspect_resources(inspected[0], resources)
        logf.write('container_resources='+json.dumps(resource_receipt, sort_keys=True)+'\n')
        logf.flush()
        if not resource_receipt['compliant']:
            raise RuntimeError('container resource policy mismatch; official tests not started')
        check_hosts(inspected[0], snapshot)
        if pin:
            guard.validate_launch(inspected[0],'instance','grade',nonce,image)
            binary = os.environ.get('MMSWE_DOCKER','/usr/bin/docker')
            home = os.environ.get('MMSWE_DOCKER_HOME','/nonexistent')
            guard.verify_container_guard(binary,home,cid,pin)
            evidence = guard.guarded_start(binary,home,cid,'instance','grade',nonce,
                                          rootfs.parent/'guard-runtime.log',time.monotonic()+(timeout or 1800))
            logf.write('runtime_guard='+json.dumps({**evidence,'guard_sha256':pin['sha256']},sort_keys=True)+'\n')
            logf.flush()
            if evidence['attach_returncode']:
                raise RuntimeError('guarded instance failed; no valid score')
        else:
            cli(['start', cid], capture_output=True, text=True)
        result = cli(['wait', cid], timeout=timeout or 1800, capture_output=True, text=True)
        rc = int(result.stdout.strip())
        for filename in ('test_output.txt', 'apply.log', 'preflight.log'):
            copy_log(cid, filename, rootfs/'tmp'/filename, 60)
        (rootfs/'tmp/ns_diag.log').write_text('ordinary Docker instance; no nested unshare\n')
        return rc
    finally:
        # Never fall back to running directly in the caller's namespace.
        for args, absent in ((['rm', '-f', cid or name], 'No such container'),
                             (['image', 'rm', '-f', tag], 'No such image')):
            try:
                p = cli(args, check=False, capture_output=True, text=True)
                if p.returncode and absent not in p.stderr:
                    cleanup_errors.append('rejected')
            except Exception as error:
                cleanup_errors.append(type(error).__name__)
        try:
            archive.unlink(missing_ok=True)
        except OSError as error:
            cleanup_errors.append(type(error).__name__)
        if cleanup_errors:
            BLOCKED.set()
            raise RuntimeError('ordinary Docker cleanup failed')


def main():
    operator_settings()
    native.run_in_namespace = run_in_docker
    return native.main()


if __name__ == '__main__':
    raise SystemExit(main())
