"""Opt-in HTTPS snapshots served on loopback; no outbound proxy or fallback."""
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlsplit

MANIFEST_ENV = 'MMSWE_OFFLINE_RESOURCE_MANIFEST'
SHA_ENV = 'MMSWE_OFFLINE_RESOURCE_SHA256'
DEST = '/opt/mmptb-offline'


def load_snapshot(pin):
    from mmswe_service.policy import private_file
    if pin is None:
        return None
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise ValueError('offline snapshot needs an operator path and sha256')
    manifest = private_file(pin['path'])
    raw = manifest.read_bytes()
    if len(raw) > 1024*1024 or hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('offline snapshot manifest changed')
    data = json.loads(raw)
    if data.get('version') != 1 or data.get('mode') != 'loopback-https-snapshot':
        raise ValueError('unsupported offline resource snapshot')
    entries = data.get('resources')
    if not isinstance(entries, dict) or not 1 <= len(entries) <= 32:
        raise ValueError('snapshot requires 1..32 exact resource URLs')
    total, hosts = 0, set()
    for url, record in entries.items():
        u = urlsplit(url)
        if (u.scheme != 'https' or u.port not in (None,443) or u.username or u.password
                or u.fragment or not u.hostname or not re.fullmatch(r'[a-z0-9]+(?:[.-][a-z0-9]+)*\.[a-z]{2,63}',u.hostname)
                or u.hostname.endswith(('.localhost','.local')) or u.hostname=='localhost'
                or url != 'https://'+u.hostname+(u.path or '/')+('?' + u.query if u.query else '')):
            raise ValueError('only canonical exact HTTPS public hostname URLs are supported')
        hosts.add(u.hostname)
        digest = record['sha256']
        if not re.fullmatch('[a-f0-9]{64}', digest):
            raise ValueError('invalid resource digest')
        size=record['size']; total += size
        if type(size) is not int or not 0 < size <= 5*1024*1024 or total > 64*1024*1024:
            raise ValueError('snapshot resource size exceeds limit')
        content_type=record['content_type']
        if not isinstance(content_type,str) or not re.fullmatch(r'[a-zA-Z0-9.+/-]+(?:; charset=[a-zA-Z0-9-]+)?',content_type):
            raise ValueError('invalid snapshot content type')
        file = private_file(manifest.parent/'blobs'/(digest+'.bin'))
        body=file.read_bytes()
        if len(body)!=size or hashlib.sha256(body).hexdigest()!=digest:
            raise ValueError('snapshot resource bytes changed')
    for name in ('ca.pem','server.pem','server.key'):
        file=private_file(manifest.parent/name)
        body=file.read_bytes()
        if len(body)>65536 or hashlib.sha256(body).hexdigest()!=data['tls_sha256'][name]:
            raise ValueError('snapshot TLS material changed')
    browser = data.get('browser_nss')
    if browser is not None:
        # Opt-in for the root-run stock p5.js Puppeteer harness. This does not
        # change browser launch flags or disable certificate verification.
        if (not isinstance(browser,dict) or set(browser) != {'homes','sha256'}
                or browser['homes'] != ['/root'] or not isinstance(browser['sha256'],dict)
                or set(browser['sha256']) != {'cert9.db','key4.db'}):
            raise ValueError('unsupported browser NSS trust scope')
        for name, digest in browser['sha256'].items():
            file=private_file(manifest.parent/'nssdb'/name)
            body=file.read_bytes()
            if len(body)>1024*1024 or hashlib.sha256(body).hexdigest()!=digest:
                raise ValueError('browser NSS trust bytes changed')
    return {'pin': dict(pin), 'manifest': str(manifest), 'data': data, 'hosts': sorted(hosts)}


def snapshot_from_environment(env):
    if MANIFEST_ENV not in env and SHA_ENV not in env:
        return None
    if MANIFEST_ENV not in env or SHA_ENV not in env:
        raise ValueError('offline snapshot requires path and SHA256 together')
    return load_snapshot({'path':env[MANIFEST_ENV], 'sha256':env[SHA_ENV]})


def host_args(snapshot):
    result=[]
    for host in snapshot['hosts'] if snapshot else []:
        result += ['--add-host',host+':127.0.0.1']
    return result


def check_hosts(info, snapshot):
    expected={h+':127.0.0.1' for h in snapshot['hosts']} if snapshot else set()
    actual=info.get('HostConfig',{}).get('ExtraHosts') or []
    # Docker serializes add-host as either hostname:IP or hostname=IP.
    actual={v.replace('=',':',1) for v in actual}
    if actual != expected:
        raise ValueError('offline snapshot host mapping differs from requested policy')


def stage_snapshot(rootfs, snapshot):
    if snapshot is None:
        return
    rootfs=Path(rootfs)
    opt=rootfs/'opt'
    if opt.is_symlink() or (opt.exists() and not opt.is_dir()):
        raise ValueError('unsafe offline snapshot destination')
    opt.mkdir(exist_ok=True)
    target=rootfs/DEST.lstrip('/')
    target.mkdir(mode=0o755,exist_ok=False)
    (target/'blobs').mkdir(mode=0o755)
    source=Path(snapshot['manifest']).parent
    for name in ['manifest.json','ca.pem','server.pem','server.key']:
        src=Path(snapshot['manifest']) if name=='manifest.json' else source/name
        shutil.copyfile(src,target/name);(target/name).chmod(0o444)
    for entry in snapshot['data']['resources'].values():
        name=entry['sha256']+'.bin'
        shutil.copyfile(source/'blobs'/name,target/'blobs'/name)
        (target/'blobs'/name).chmod(0o444)
    if hashlib.sha256((target/'manifest.json').read_bytes()).hexdigest()!=snapshot['pin']['sha256']:
        raise ValueError('snapshot manifest changed while staging')
    for entry in snapshot['data']['resources'].values():
        body=(target/'blobs'/(entry['sha256']+'.bin')).read_bytes()
        if len(body)!=entry['size'] or hashlib.sha256(body).hexdigest()!=entry['sha256']:
            raise ValueError('snapshot resource changed while staging')
    for name,digest in snapshot['data']['tls_sha256'].items():
        if name not in ('ca.pem','server.pem','server.key') or hashlib.sha256((target/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('snapshot TLS material changed while staging')
    for name in ['offline_server.js','offline_run.sh']:
        shutil.copyfile(Path(__file__).with_name(name),target/name)
        (target/name).chmod(0o555)
    browser=snapshot['data'].get('browser_nss')
    if browser:
        # Do not replace an existing instance trust database or follow symlinks.
        for home in browser['homes']:
            current=rootfs
            for part in (home.lstrip('/')+'/.pki').split('/'):
                current=current/part
                if current.is_symlink() or (current.exists() and not current.is_dir()):
                    raise ValueError('unsafe browser NSS destination')
                current.mkdir(mode=0o700,exist_ok=True)
            db=current/'nssdb'
            db.mkdir(mode=0o700,exist_ok=False)
            for name,digest in browser['sha256'].items():
                shutil.copyfile(source/'nssdb'/name,db/name)
                (db/name).chmod(0o600)
                if hashlib.sha256((db/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('browser NSS trust changed while staging')
    script=rootfs/'run_in_chroot.sh'
    if script.is_symlink():
        raise ValueError('unsafe runtime script')
    text=script.read_text()
    original='  /bin/bash /eval.sh >> /tmp/test_output.txt 2>&1'
    if text.count(original)!=1:
        raise ValueError('cannot identify official test invocation for offline wrapper')
    script.write_text(text.replace(original,'  /bin/bash '+DEST+'/offline_run.sh /bin/bash /eval.sh >> /tmp/test_output.txt 2>&1'))
