#!/usr/bin/env python3
"""CPU-only operator preflight; never changes host policy or launches agent code.

Uses an existing immutable image. No pull, GPU, host mount, privileged mode,
security downgrade, or automatic profile fallback. An inspect failure prevents
start. Browser process checks are evidence for review, NOT official acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mmswe_service.container_policy import PROFILES, security_args, inspect_security


class ProbeStopped(Exception):
    pass


def probe_script(kind, probes, user, chrome=None, firefox=None):
    script = '''set -eu
id
sed -n '/^CapEff:/p;/^NoNewPrivs:/p;/^Seccomp:/p' /proc/self/status
cat /proc/self/uid_map /proc/self/gid_map
cat /proc/self/attr/current 2>/dev/null || true
command -v nproc >/dev/null && nproc || true
if command -v node >/dev/null; then node -e 'console.log("NODE_CPUS="+require("os").cpus().length)'; fi
probe_nnp=$(sed -n 's/^NoNewPrivs:[[:space:]]*//p' /proc/self/status)
[ "$probe_nnp" = 1 ] || { echo NO_NEW_PRIVS_MISMATCH; exit 43; }
probe_caps=$(sed -n 's/^CapEff:[[:space:]]*//p' /proc/self/status)
'''
    if kind == 'workload':
        script += '[ "$((0x$probe_caps))" = 0 ] || { echo CAPABILITY_MISMATCH; exit 44; }\n'
    else:
        script += '[ "$((0x$probe_caps & 0x08002000))" = 0 ] || { echo CAPABILITY_MISMATCH; exit 44; }\n'
    if kind == 'instance':
        script += 'su '+shlex.quote(user)+" -s /bin/sh -c 'id'\necho MMPTB_SU_OK\n"
    if probes == 'browsers':
        browser = '''set -eu
probe_dir=$(mktemp -d /tmp/mmptb-browser-XXXXXX)
trap 'rm -rf "$probe_dir"' EXIT HUP INT TERM
export HOME="$probe_dir"
'''
        browser += 'chrome='+shlex.quote(chrome or '')+'\n'
        browser += 'firefox='+shlex.quote(firefox or '')+'\n'
        browser += '''if [ -z "$chrome" ]; then
  for candidate in google-chrome-stable google-chrome chromium chromium-browser; do
    if command -v "$candidate" >/dev/null; then chrome=$(command -v "$candidate"); break; fi
  done
fi
if [ -z "$firefox" ]; then firefox=$(command -v firefox || true); fi
[ -n "$chrome" ] && [ -x "$chrome" ] || { echo MISSING_CHROME; exit 40; }
[ -n "$firefox" ] && [ -x "$firefox" ] || { echo MISSING_FIREFOX; exit 41; }
command -v timeout >/dev/null || { echo MISSING_TIMEOUT_TOOL; exit 42; }
echo "CHROME_BIN=$chrome"
echo "FIREFOX_BIN=$firefox"
timeout 15 "$chrome" --version
timeout 15 "$firefox" --version
timeout 45 "$chrome" --headless --disable-gpu --dump-dom 'data:text/html,<p>MMPTB_BROWSER_OK</p>' > "$probe_dir/chrome.html"
grep -F MMPTB_BROWSER_OK "$probe_dir/chrome.html"
echo MMPTB_CHROME_PAGE_OK
echo CHROME_SANDBOX_STATUS_BEGIN
timeout 45 "$chrome" --headless --disable-gpu --dump-dom chrome://sandbox
echo CHROME_SANDBOX_STATUS_END
timeout 45 "$firefox" --headless --screenshot "$probe_dir/firefox.png" 'data:text/html,<p>MMPTB_BROWSER_OK</p>'
[ -s "$probe_dir/firefox.png" ]
echo MMPTB_FIREFOX_PAGE_OK
'''
        script += 'su '+shlex.quote(user)+' -s /bin/sh -c '+shlex.quote(browser)+'\n'
    return script+'echo MMPTB_PREFLIGHT_FINISHED\n'


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    (out/'docker-home').mkdir(mode=0o700)
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(out/'docker-home')}
    name = 'mmptb-preflight-'+uuid.uuid4().hex
    result = {'version': 1, 'profile': args.profile, 'kind': args.kind,
              'probes': args.probes, 'image_requested': args.image, 'container': name,
              'official_acceptance': False, 'gpu_count': 0,
              'sandbox_review_required': args.probes == 'browsers', 'status': 'pending'}
    code, attempted = 3, False

    def cli(arguments, timeout=30):
        return subprocess.run([args.docker, *arguments], env=env, capture_output=True,
                              text=True, errors='replace', timeout=timeout)

    try:
        script = probe_script(args.kind, args.probes, args.test_user, args.chrome, args.firefox)
        (out/'probe.sh').write_text(script)
        flags = ['create', '--name', name, '--label', 'mmptb.purpose=runtime-preflight',
                 '--pull', 'never', '--init', '--user',
                 '0:0' if args.kind == 'instance' else '65534:65534',
                 '--network', 'none', '--cpus', '8', '--memory', '16g',
                 '--memory-swap', '16g', '--pids-limit', '512',
                 '--log-driver', 'local', '--log-opt', 'max-size=1m', '--log-opt', 'max-file=1']
        if args.kind == 'workload':
            flags += ['--read-only', '--tmpfs', '/tmp:rw,nosuid,nodev,size=256m']
        flags += security_args(args.kind, args.profile)
        flags += ['--entrypoint', '/bin/sh', args.image, '-c', script]
        attempted = True
        created = cli(flags)
        (out/'create.stderr').write_text(created.stderr)
        result['create_returncode'] = created.returncode
        if created.returncode:
            result['status'] = 'create_rejected_or_failed'
            code = 2
            raise ProbeStopped()
        info = cli(['inspect', name])
        if info.returncode:
            raise RuntimeError('inspect failed')
        parsed = json.loads(info.stdout)
        if not isinstance(parsed, list) or len(parsed) != 1:
            raise RuntimeError('unexpected inspect response')
        result['security'] = inspect_security(parsed[0], args.kind, args.profile)
        if not result['security']['compliant']:
            result['status'] = 'configuration_rejected_before_start'
            code = 2
            raise ProbeStopped()
        started = cli(['start', name])
        if started.returncode:
            raise RuntimeError('start failed')
        waited = cli(['wait', name], timeout=args.timeout)
        if waited.returncode:
            raise RuntimeError('wait failed')
        result['container_exit_code'] = int(waited.stdout.strip())
        logs = cli(['logs', '--tail', '250', name])
        raw = logs.stdout+'\n'+logs.stderr
        (out/'probe.log').write_text(raw[-1024*1024:])
        if logs.returncode:
            raise RuntimeError('log collection failed')
        complete = 'MMPTB_PREFLIGHT_FINISHED' in raw
        if args.kind == 'instance':
            complete = complete and 'MMPTB_SU_OK' in raw
        if args.probes == 'browsers':
            complete = complete and all(x in raw for x in (
                'MMPTB_CHROME_PAGE_OK', 'MMPTB_FIREFOX_PAGE_OK', 'CHROME_SANDBOX_STATUS_END'))
        if result['container_exit_code'] == 0 and complete:
            result['status'], code = 'probes_passed_not_official_acceptance', 0
        else:
            result['status'], code = 'runtime_probe_failed', 3
    except ProbeStopped:
        pass
    except Exception as error:
        result.update(status='preflight_error', error_type=type(error).__name__, error=str(error)[:2000])
        if attempted:
            try:
                logs = cli(['logs', '--tail', '250', name])
                (out/'probe.log').write_text((logs.stdout+'\n'+logs.stderr)[-1024*1024:])
            except Exception:
                pass
    finally:
        if attempted:
            try:
                cleanup = cli(['rm', '-f', name])
                ok = cleanup.returncode == 0 or 'No such container' in cleanup.stderr
                result['cleanup_ok'] = ok
            except Exception:
                result['cleanup_ok'] = False
            if not result['cleanup_ok']:
                result.update(status='cleanup_failed', operator_cleanup_required=name)
                code = 4
        result['finished_unix'] = time.time()
        (out/'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps({'status': result['status'], 'result': str(out/'result.json')}))
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--docker', required=True)
    p.add_argument('--image', required=True, help='existing sha256:image-ID or repository@sha256:digest')
    p.add_argument('--out', required=True, help='new output directory; existing directories are refused')
    p.add_argument('--profile', choices=('strict-v1','platform-managed-v1'), default='strict-v1')
    p.add_argument('--kind', choices=('instance', 'workload'), default='instance')
    p.add_argument('--probes', choices=('identity', 'browsers'), default='identity')
    p.add_argument('--test-user', help='default nobody for identity; chromeuser for browsers')
    p.add_argument('--chrome', help='optional absolute browser executable path in the instance image')
    p.add_argument('--firefox', help='optional absolute browser executable path in the instance image')
    p.add_argument('--timeout', type=int, default=240)
    a = p.parse_args()
    if not Path(a.docker).is_absolute() or not Path(a.docker).is_file():
        p.error('--docker must name an existing absolute executable')
    if not re.fullmatch(r'(?:[a-zA-Z0-9][a-zA-Z0-9._/:-]*@)?sha256:[a-f0-9]{64}', a.image):
        p.error('an immutable image reference is required; tags are refused')
    a.test_user = a.test_user or ('chromeuser' if a.probes == 'browsers' else 'nobody')
    if not re.fullmatch(r'[a-z_][a-z0-9_-]{0,31}', a.test_user):
        p.error('invalid test user')
    if not 10 <= a.timeout <= 600:
        p.error('--timeout must be 10..600 seconds')
    if a.probes == 'browsers' and a.kind != 'instance':
        p.error('browser checks require an instance image')
    for executable in (a.chrome, a.firefox):
        if executable and (not executable.startswith('/') or '\x00' in executable or '\n' in executable):
            p.error('browser paths must be absolute')
    return run(a)


if __name__ == '__main__':
    raise SystemExit(main())
