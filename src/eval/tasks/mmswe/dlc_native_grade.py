#!/usr/bin/env python3
"""
Route B — daemon-free SWE-bench Multimodal grading (NO docker daemon).

Why this exists
  mmswe grading must run each instance's official image (swebench/sweb.eval.*)
  to apply the model patch and run the test suite. Many managed GPU pods (ours
  included) have NO docker daemon and can't nest one, yet CAN reach docker.io's
  registry v2 API over plain HTTPS. So we replace ONLY the container runtime:
     pull image via registry v2 (curl + anon token)
     -> unpack layers to a rootfs on local disk (/dev/shm)
     -> unshare + chroot to run the SAME apply-loop + eval.sh the official
        run_instance runs
     -> grade with swebench.get_eval_report (unchanged).
  eval_script generation (make_test_spec), patch-apply order (GIT_APPLY_CMDS),
  log parsing and resolved computation are all reused verbatim from the venv.

Contract reproduced from swebench.harness.run_evaluation.run_instance:
  CONTAINER_WORKDIR   = /testbed
  CONTAINER_PATCH_FILE= /tmp/patch.diff
  GIT_APPLY_CMDS      = git apply --verbose | ... --3way | ... --reject
                        (with `git checkout -- . ; git clean -fd` between tries),
                        then a `git apply --check --reverse` "already applied" check.
  eval.sh             = _inject_asset_restore(test_spec.eval_script, restore_cmds)
                        restore_cmds come from image_assets test_patch/patch lists
                        (empty for most instances incl. the gold probe).
  test_output.txt     = combined stdout+stderr of `/bin/bash /eval.sh` ONLY
                        (apply-loop output is kept OUT, exactly like official,
                        so get_logs_eval's bad-code scan isn't tripped).
  grade               = get_eval_report(test_spec, pred, test_output.txt, True)

Usage:
  python dlc_native_grade.py \
      --preds  <predictions.jsonl> \
      --out    <report_dir> \
      [--instance-ids id1 id2 ...] \
      [--dataset SWE-bench/SWE-bench_Multimodal] [--split dev] \
      [--scratch /dev/shm/mmswe] [--timeout 1800]
  Writes <out>/report.json  {iid: {resolved, patch_successfully_applied, ...}}
         <out>/summary.json {resolved_ids, n, ...}
         <out>/<iid>/{pull.log, run_in_chroot.sh, eval.sh, patch.diff, test_output.txt}
"""
import argparse, hashlib, json, os, re, shlex, shutil, subprocess, sys, tarfile, threading, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── split-trust image pull (approved design) ────────────────────────────────
# Registry metadata and layer downloads may have different network reachability.
# Verify public mirror content against digests from the official registry:
#   MANIFEST + config DIGESTS come from the OFFICIAL registry (the trust anchor);
#   BLOB BYTES (config + layers) are fetched from a reachable public mirror, then
#   every blob's sha256 is HARD-VERIFIED against the official digest before it is
#   unpacked or executed. Docker images are content-addressed, so a mirror that
#   alters a single byte fails verification and we abort. The mirror is thus a
#   dumb CDN: it can affect availability, never integrity.
REGISTRY_API = os.environ.get("MMSWE_REGISTRY_API", "https://registry-1.docker.io").rstrip("/")
BLOB_MIRROR = os.environ.get("MMSWE_BLOB_MIRROR", "https://docker.1panel.live").rstrip("/")
# Content-addressed blob cache on the SHARED filesystem. SWE-bench MM images share
# almost all layers (same base + node toolchain); the mirror is ~290KB/s and jittery
# (an identical 362MB layer measured 34.5s for one instance, 1239.7s for another).
# Caching each verified layer by digest on shared storage means shared layers pull
# ONCE ever — so an arm loop re-grading the same val subset each iteration reuses
# them at filesystem read speed instead of re-pulling. Default = $SWE_WORK/blob_cache
# (else next to this file); set MMSWE_BLOB_CACHE="" to disable (ephemeral only).
_bc_dflt = os.path.join(os.environ.get("SWE_WORK")
                        or os.path.dirname(os.path.abspath(__file__)), "blob_cache")
_bc = os.environ.get("MMSWE_BLOB_CACHE", _bc_dflt)
BLOB_CACHE = Path(_bc) if _bc else None
# Official registry needs a bearer token for the manifest API; the anon mirror
# does not (and may reject a docker.io token), so blobs are fetched token-less.
USE_TOKEN = "registry-1.docker.io" in REGISTRY_API
AUTH = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
MANIFEST_ACCEPT = [
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
]
# swebench apply order (mirror of run_evaluation.GIT_APPLY_CMDS)
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --3way",
    "git apply --verbose --reject",
]
# official markers (mirror of swebench.harness.constants). get_eval_report scans the
# graded log for APPLY_PATCH_FAIL to short-circuit an unapplied patch to unresolved;
# restore/inject uses START_TEST_OUTPUT as the injection point. Mirrored locally (like
# GIT_APPLY_CMDS) so build_chroot_script needs no swebench import at module load.
APPLY_PATCH_PASS = ">>>>> Applied Patch"
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
START_TEST_OUTPUT = ">>>>> Start Test Output"


def log(*a):
    print("[dlc_grade]", *a, flush=True)


# ─────────────────────────── registry v2 pull ───────────────────────────
def _curl_json(url, token=None, accepts=None):
    cmd = ["curl", "-sSL", "--connect-timeout", "15", "--max-time", "120"]
    if token:
        cmd += ["-H", f"Authorization: Bearer {token}"]
    for a in accepts or []:
        cmd += ["-H", f"Accept: {a}"]
    cmd.append(url)
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"curl failed ({out.returncode}) for {url}: {out.stderr[:400]}")
    return json.loads(out.stdout)


def _content_length(url, token, logf):
    """HEAD the blob to learn its full size, so the resume loop knows when the
    layer is actually complete (curl rc alone lied: it returned 0 mid-truncation
    on some mirror responses). Returns int bytes or None if the mirror won't say."""
    cmd = ["curl", "-sSI", "--http1.1", "--connect-timeout", "15",
           "--max-time", "60", url]
    if token:
        cmd[3:3] = ["-H", f"Authorization: Bearer {token}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _curl_blob(url, token, dest, logf):
    # The mirror serves clean HTTP/1.1 but DROPS the connection partway through
    # multi-hundred-MB layers (curl 18: "transfer closed with N bytes remaining"),
    # and a plain --retry restarts from byte 0 so it never converges. Instead we
    # RESUME with HTTP Range (`curl -C -`), accumulating bytes across drops until
    # the file reaches Content-Length. Each pass appends only what's missing.
    # Correctness is still anchored by the sha256 re-check in fetch_blob_verified,
    # so even a mis-resumed (206-vs-200) file just fails verification, never trusts.
    total = _content_length(url, token, logf)
    logf.write(f"blob {url} content-length={total}\n")
    hdr = ["-H", f"Authorization: Bearer {token}"] if token else []
    last_size = -1
    for attempt in range(1, 41):
        have = dest.stat().st_size if dest.exists() else 0
        if total is not None and have >= total:
            break
        cmd = (["curl", "-sS", "--http1.1", "-C", "-",
                "--retry", "4", "--retry-delay", "2", "--retry-all-errors",
                "--connect-timeout", "15", "--max-time", "1800"]
               + hdr + ["-o", str(dest), url])
        r = subprocess.run(cmd, capture_output=True, text=True)
        have = dest.stat().st_size if dest.exists() else 0
        logf.write(f"  attempt {attempt} rc={r.returncode} size={have}"
                   f"{'/' + str(total) if total else ''}\n")
        if r.returncode == 0 and (total is None or have >= total):
            break
        if have == last_size and total is not None:
            # no forward progress this pass -> mirror is stuck, stop wasting time
            raise RuntimeError(
                f"blob download stalled at {have}/{total} B after {attempt} "
                f"resume passes (rc={r.returncode}): {r.stderr[:200]}")
        last_size = have
    have = dest.stat().st_size if dest.exists() else 0
    if total is not None and have < total:
        raise RuntimeError(
            f"blob download incomplete {have}/{total} B after resume loop")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def fetch_blob_verified(repo, digest, dest, logf):
    """Download a blob's BYTES from the mirror, then HARD-VERIFY its sha256
    against `digest` (which came from the OFFICIAL manifest). Abort on mismatch —
    the mirror can never make us unpack/execute bytes docker.io wouldn't."""
    url = f"{BLOB_MIRROR}/v2/{repo}/blobs/{digest}"
    for scratch in range(2):
        _curl_blob(url, None, dest, logf)   # anonymous mirror (resume loop inside)
        got = _sha256_file(dest)
        if got == digest:
            logf.write(f"  verified {digest} == mirror bytes ({dest.stat().st_size} B) OK\n")
            return
        # mismatch: a bad resume (server ignored Range -> 200 appended onto partial)
        # can corrupt the file. Nuke and re-pull from byte 0 once before giving up.
        logf.write(f"  DIGEST MISMATCH {digest}: got {got} (attempt {scratch+1}), "
                   f"discarding and re-pulling from scratch\n")
        try:
            dest.unlink()
        except OSError:
            pass
    raise RuntimeError(
        f"DIGEST MISMATCH for {digest}: mirror {BLOB_MIRROR} served sha256 {got} "
        f"— refusing to unpack (possible tampering / corrupt mirror)")


def ensure_blob(repo, digest, scratch, logf):
    """Return (local_path, is_ephemeral) for verified blob bytes of `digest`.
    Uses the shared content-addressed cache when enabled: a hit returns the cached
    path (no download); a miss pulls from the mirror into a pid-suffixed temp,
    verifies sha256, then atomically publishes into the cache — concurrent graders
    converge on identical content-addressed bytes. Without a cache, pulls into
    ephemeral scratch. A cached path (is_ephemeral=False) must NOT be unlinked."""
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise ValueError("invalid registry blob digest")
    key = digest.replace(":", "_")
    if BLOB_CACHE is not None:
        cpath = BLOB_CACHE / key
        if cpath.exists():
            if cpath.is_symlink() or not cpath.is_file() or _sha256_file(cpath) != digest:
                raise RuntimeError("cached blob digest mismatch; refusing to unpack")
            logf.write(f"  cache HIT {digest} ({cpath.stat().st_size} B)\n"); logf.flush()
            return cpath, False
        BLOB_CACHE.mkdir(parents=True, exist_ok=True)
        # temp name is unique per (process, THREAD): parallel graders run as threads
        # sharing one pid, so pid alone would collide when two threads cold-pull the
        # SAME digest into the same temp file and corrupt it. The atomic os.replace
        # publish below still converges them onto one content-addressed cache entry.
        tmp = BLOB_CACHE / f"{key}.tmp.{os.getpid()}.{threading.get_ident()}"
        fetch_blob_verified(repo, digest, tmp, logf)   # verified bytes into temp
        try:
            os.replace(str(tmp), str(cpath))           # atomic publish
        except OSError:
            try: tmp.unlink()
            except OSError: pass
            if not cpath.is_file() or cpath.is_symlink() or _sha256_file(cpath) != digest:
                raise RuntimeError("verified blob cache publication failed")
        return (cpath if cpath.exists() else tmp), False
    dest = scratch / "blobs" / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    fetch_blob_verified(repo, digest, dest, logf)
    return dest, True


def parse_image_ref(image):
    # e.g. "swebench/sweb.eval.x86_64.foo:latest" -> ("swebench/sweb.eval.x86_64.foo", "latest")
    ref = image
    if "/" not in ref.split(":")[0] and "." not in ref.split("/")[0]:
        pass
    if ":" in ref.rsplit("/", 1)[-1]:
        repo, tag = ref.rsplit(":", 1)
    else:
        repo, tag = ref, "latest"
    # docker.io official/library short names would need library/ prefix; swebench uses user/repo
    return repo, tag


def get_manifest(repo, ref, token, logf):
    # trust anchor: manifests + digests come from the OFFICIAL registry API
    url = f"{REGISTRY_API}/v2/{repo}/manifests/{urllib.parse.quote(ref, safe='')}"
    m = _curl_json(url, token, MANIFEST_ACCEPT)
    mt = m.get("mediaType", "")
    if "manifest.list" in mt or "image.index" in mt or m.get("manifests"):
        # multi-arch index: pick linux/amd64
        chosen = None
        for e in m.get("manifests", []):
            plat = e.get("platform", {})
            if plat.get("architecture") == "amd64" and plat.get("os") == "linux":
                chosen = e
                break
        if chosen is None:
            raise RuntimeError("no linux/amd64 manifest in index")
        logf.write(f"index -> platform manifest {chosen['digest']}\n")
        return get_manifest(repo, chosen["digest"], token, logf)
    return m


def pull_and_unpack(image, rootfs, scratch, logf, deadline=None):
    """Split-trust pull: manifest+digests from OFFICIAL registry, blob bytes from
    the mirror (via the shared cache) with sha256 verified against the official
    digests. Unpack layers into rootfs. Returns config dict. `deadline` (epoch
    seconds) bounds a cold pull: if exceeded between layers the pull aborts so the
    caller records the instance unresolved and grading continues (a single
    unreachable/slow image can't hang the whole eval). Cached layers cost seconds,
    so the deadline only bites on a cold + genuinely stuck pull."""
    repo, tag = parse_image_ref(image)
    logf.write(f"manifest_api={REGISTRY_API} blob_mirror={BLOB_MIRROR} "
               f"use_token={USE_TOKEN} repo={repo} tag={tag} cache={BLOB_CACHE}\n")
    token = None
    if USE_TOKEN:
        token = _curl_json(AUTH.format(repo=urllib.parse.quote(repo, safe='/')))["token"]
        logf.write(f"token len={len(token)}\n")
    manifest = get_manifest(repo, tag, token, logf)   # trust anchor (official)
    rootfs.mkdir(parents=True, exist_ok=True)
    # config blob (has .config.Env, .config.WorkingDir): bytes from mirror, sha256
    # verified against the official manifest's config.digest.
    cfg_digest = manifest["config"]["digest"]
    cfg_blob, cfg_eph = ensure_blob(repo, cfg_digest, scratch, logf)
    cfg = json.loads(Path(cfg_blob).read_text())
    if cfg_eph:
        Path(cfg_blob).unlink(missing_ok=True)
    layers = manifest["layers"]
    logf.write(f"config={cfg_digest} n_layers={len(layers)}\n")
    for i, lyr in enumerate(layers):
        if deadline is not None and time.time() > deadline:
            raise RuntimeError(
                f"cold-pull deadline exceeded before layer {i+1}/{len(layers)} "
                f"(image={image}); recording instance unresolved")
        dg = lyr["digest"]                              # official digest
        t0 = time.time()
        blob, eph = ensure_blob(repo, dg, scratch, logf)  # cache or mirror, verified
        extract_layer(blob, rootfs, logf)
        if eph:
            Path(blob).unlink(missing_ok=True)  # free /dev/shm; cached blobs persist
        logf.write(f"layer {i+1}/{len(layers)} {dg} done in {time.time()-t0:.1f}s\n")
        logf.flush()
    return cfg


def _rm(path):
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except FileNotFoundError:
        pass


def extract_layer(blob, root, logf):
    """Extract one gzipped layer tar into root, honoring docker whiteouts."""
    root = str(root)
    skipped = 0
    with tarfile.open(blob, "r:gz") as tf:
        for m in tf:
            name = m.name.lstrip("./")
            if not name:
                continue
            bn = os.path.basename(name)
            dirn = os.path.dirname(name)
            if bn == ".wh..wh..opq":
                d = os.path.join(root, dirn)
                if os.path.isdir(d):
                    for c in os.listdir(d):
                        _rm(os.path.join(d, c))
                continue
            if bn.startswith(".wh."):
                _rm(os.path.join(root, dirn, bn[4:]))
                continue
            target = os.path.join(root, name)
            # remove conflicting node so files can replace dirs/symlinks and vice-versa
            if (m.isdir() and os.path.islink(target)) or (not m.isdir() and os.path.isdir(target) and not os.path.islink(target)):
                _rm(target)
            elif os.path.islink(target) or os.path.isfile(target):
                if not m.isdir():
                    try:
                        os.unlink(target)
                    except OSError:
                        pass
            try:
                tf.extract(m, root, set_attrs=True, numeric_owner=True, filter="fully_trusted")
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    logf.write(f"  skip {name}: {e}\n")
    if skipped:
        logf.write(f"  (extracted with {skipped} skipped members in {os.path.basename(blob)})\n")


# ─────────────────────────── chroot exec ───────────────────────────
def _env_exports(cfg):
    env = {}
    for kv in (cfg.get("config", {}) or {}).get("Env", []) or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
    env.setdefault("HOME", "/root")
    env.setdefault("TERM", "xterm")
    env.setdefault("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    lines = []
    for k, v in env.items():
        lines.append(f"export {k}={shlex.quote(v)}")
    return "\n".join(lines)


def build_chroot_script(cfg):
    """The script run INSIDE the chroot: apply model patch (GIT_APPLY_CMDS loop
    + reverse-check), write the official APPLY_PATCH_PASS/FAIL marker to the head of
    /tmp/test_output.txt, then run /eval.sh (appending) ONLY if the patch applied.
    The verbose apply-loop output stays in /tmp/apply.log, out of the graded log,
    like official -- so get_eval_report's apply gate fires on a failed apply."""
    apply_block = []
    for i, cmd in enumerate(GIT_APPLY_CMDS):
        pre = "git checkout -- . >/dev/null 2>&1; git clean -fd >/dev/null 2>&1; " if i else ""
        apply_block.append(
            f'if [ "$APPLIED" = 0 ]; then {pre}'
            f'if {cmd} /tmp/patch.diff; then APPLIED=1; fi; fi'
        )
    apply_block = "\n".join(apply_block)
    return f"""#!/bin/bash
set -uo pipefail
{_env_exports(cfg)}
cd /testbed
git config --global --add safe.directory /testbed >/dev/null 2>&1 || true

# ---- preflight: prove /proc is usable (jest's getMaxWorkers needs os.cpus()) ----
{{
  echo "PREFLIGHT nproc=$(nproc 2>&1)"
  echo "PREFLIGHT /proc/stat cpu lines=$(grep -c '^cpu' /proc/stat 2>/dev/null || echo MISSING)"
  NODE_BIN="$(command -v node || true)"
  if [ -n "$NODE_BIN" ]; then
    echo "PREFLIGHT node os.cpus=$("$NODE_BIN" -e 'var c=require("os").cpus();process.stdout.write(c===undefined?"undefined":String((c||[]).length))' 2>&1)"
  fi
}} > /tmp/preflight.log 2>&1

# Resource evidence stays separate from the official test log and score parser.
mmptb_resource_diag() {{
  echo "RESOURCE_PHASE=$1"
  echo "RLIMIT_NPROC=$(ulimit -u 2>&1)"
  sed -n '/^Cpus_allowed_list:/p;/^Threads:/p' /proc/self/status 2>/dev/null || true
  for resource_file in /sys/fs/cgroup/pids.current /sys/fs/cgroup/pids.max \\
    /sys/fs/cgroup/pids.events /sys/fs/cgroup/memory.events \\
    /sys/fs/cgroup/memory.peak /sys/fs/cgroup/cpu.max \\
    /sys/fs/cgroup/pids/pids.current /sys/fs/cgroup/pids/pids.max \\
    /sys/fs/cgroup/pids/pids.events /sys/fs/cgroup/memory/memory.failcnt; do
    if [ -r "$resource_file" ]; then
      echo "RESOURCE_FILE=$resource_file"
      cat "$resource_file" 2>/dev/null || true
    fi
  done
}}
mmptb_resource_diag before >> /tmp/preflight.log 2>&1

# ---- apply model patch (GIT_APPLY_CMDS order; output NOT in test_output.txt) ----
APPLIED=0
{{
{apply_block}
if [ "$APPLIED" = 0 ]; then
  if git apply --check --reverse /tmp/patch.diff; then APPLIED=1; fi
fi
}} > /tmp/apply.log 2>&1
echo "APPLIED=$APPLIED" >> /tmp/apply.log

# ---- apply gate: reproduce official get_eval_report semantics ----
# Upstream writes APPLY_PATCH_PASS / APPLY_PATCH_FAIL into the SAME log get_logs_eval
# scans, so a patch that fails to apply short-circuits to unresolved. Route B kept
# the apply-loop output in a SEPARATE apply.log, so that gate never fired and an
# UNAPPLIED patch still had its eval.sh graded as a real run (false-positive
# "resolved"). Fix: write the official marker at the head of test_output.txt and run
# eval.sh ONLY when the patch actually applied. The verbose apply-loop noise stays in
# apply.log (out of the graded log), exactly like official, so get_logs_eval's
# bad-code scan is not tripped by apply chatter.
if [ "$APPLIED" = 1 ]; then
  echo '{APPLY_PATCH_PASS}' > /tmp/test_output.txt
  /bin/bash /eval.sh >> /tmp/test_output.txt 2>&1
  echo "EVAL_RC=$?" >> /tmp/apply.log
else
  echo '{APPLY_PATCH_FAIL}' > /tmp/test_output.txt
  echo "EVAL_RC=skipped_apply_failed" >> /tmp/apply.log
fi
mmptb_resource_diag after >> /tmp/preflight.log 2>&1
"""


def run_in_namespace(rootfs, logf, timeout=None):
    """chroot into rootfs and run /run_in_chroot.sh. Tries real-root unshare
    first (needs CAP_SYS_ADMIN), falls back to userns (--map-root-user).
    `timeout` (s) bounds each variant so a hung test suite can't stall the eval."""
    root = str(rootfs)
    # A fresh `mount -t proc` in the new PID ns is what makes /proc/{stat,cpuinfo}
    # visible inside the chroot; without it Node's os.cpus() returns undefined and
    # jest crashes in getMaxWorkers ("Cannot read property 'length' of undefined"),
    # which silently fails the whole eval. So: do NOT swallow the mount error, VERIFY
    # /proc/stat actually appears, and fall back to rbind'ing the host /proc if a
    # fresh proc mount didn't populate. Diagnostics land in {root}/tmp/ns_diag.log.
    inner = (
        f"set -u; D={root}/tmp/ns_diag.log; : >$D; "
        f"if mount -t proc proc {root}/proc 2>>$D; then echo 'proc: fresh mount ok' >>$D; "
        f"else echo 'proc: fresh mount FAILED, trying rbind' >>$D; "
        f"mount --rbind /proc {root}/proc 2>>$D && echo 'proc: rbind ok' >>$D; fi; "
        f"if [ ! -e {root}/proc/stat ]; then echo 'proc: /proc/stat MISSING after mount -> rbind host proc' >>$D; "
        f"mount --rbind /proc {root}/proc 2>>$D; fi; "
        f"echo -n 'proc/stat cpu lines: ' >>$D; grep -c '^cpu' {root}/proc/stat 2>>$D >>$D || echo 0 >>$D; "
        f"mount --rbind /dev {root}/dev 2>>$D || true; "
        f"mount --rbind /sys {root}/sys 2>>$D || true; "
        f"cp /etc/resolv.conf {root}/etc/resolv.conf 2>/dev/null || true; "
        f"chroot {root} /bin/bash /run_in_chroot.sh"
    )
    variants = [
        ["unshare", "--mount", "--uts", "--ipc", "--pid", "--fork", "--kill-child",
         "/bin/bash", "-c", inner],
        ["unshare", "--map-root-user", "--mount", "--uts", "--ipc", "--pid", "--fork",
         "--kill-child", "/bin/bash", "-c", inner],
    ]
    last = None
    for v in variants:
        logf.write(f"exec: {' '.join(v[:8])} ...\n"); logf.flush()
        try:
            r = subprocess.run(v, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            logf.write(f"  TIMEOUT after {timeout}s\n"); logf.flush()
            last = None
            continue
        logf.write(f"  rc={r.returncode}\n  stdout: {r.stdout[-800:]}\n  stderr: {r.stderr[-800:]}\n")
        logf.flush()
        # test_output.txt written == inner reached chroot; accept regardless of rc
        if (rootfs / "tmp" / "test_output.txt").exists():
            return r.returncode
        last = r
    logf.write("  !! no test_output.txt produced by any unshare variant\n")
    return last.returncode if last else 1


# ─────────────────────────── main ───────────────────────────
def _default_jobs():
    """Use one worker unless the operator explicitly configures parallel grading."""
    env = os.environ.get("MMSWE_GRADE_JOBS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    # Serial by default on portable hosts; operators may opt into parallel jobs.
    return 1


def load_asset_bytes(url):
    """An operator-frozen manifest eliminates network access during grading.

    Manifest entries bind the exact source URL to content SHA256 and size. A
    missing/corrupt entry is an infrastructure failure, never an online fallback.
    Without a manifest, keep the portable upstream network path.
    """
    manifest = os.environ.get("MMSWE_ASSET_MANIFEST")
    if not manifest:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if not data:
            raise ValueError("empty test asset")
        return data
    manifest_path = Path(manifest).resolve()
    raw = manifest_path.read_bytes()
    pinned = os.environ.get("MMSWE_ASSET_MANIFEST_SHA256")
    if pinned and hashlib.sha256(raw).hexdigest() != pinned:
        raise ValueError("asset manifest digest mismatch")
    table = json.loads(raw)
    if table.get("version") != 1:
        raise ValueError("unsupported asset manifest")
    entry = table["assets"][url]
    digest = entry["sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid asset digest")
    # Content-addressed relative files; entries cannot name arbitrary host paths.
    path = manifest_path.parent / "blobs" / digest
    if path.is_symlink():
        raise ValueError("asset blob must not be a symlink")
    data = path.read_bytes()
    if not data or len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("asset bytes differ from manifest")
    return data


def _stage_assets_into_rootfs(spec, rootfs, logf):
    """Real image_assets restore (Route B port of swebench._stage_image_assets).

    A text patch cannot carry binary files (e.g. expected.png rendering baselines),
    so the dataset lists them in image_assets with a raw.githubusercontent URL. We
    fetch each over the SAME public-host / system-CA path the screenshot pulls use,
    drop it into <rootfs>/image_assets/<flat>, and return `cp` restore commands to
    inject into eval.sh right before the test-output start marker -- so the assets
    land AFTER the test_patch git apply, exactly like upstream. Without this the
    11 dev / 54 test asset instances run against a MISSING baseline and fail
    environmentally (a false model failure). Returns (restore_cmds, unresolved_paths).
    """
    declared = spec.image_assets or {}
    assets = []
    for key in ("test_patch", "patch"):
        for entry in declared.get(key) or []:
            if entry.get("path"):
                assets.append(entry)
    if not assets:
        return [], []
    staging = rootfs / "image_assets"
    staging.mkdir(parents=True, exist_ok=True)
    restore, unresolved = [], []
    for a in assets:
        path, url = a["path"], a.get("url")
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("image asset path must stay within the test repository")
        data = None
        if url:
            for attempt in range(3):
                try:
                    data = load_asset_bytes(url)
                    break
                except Exception as e:
                    logf.write(f"asset fetch retry {attempt} for {path}: {e}\n")
                    if os.environ.get("MMSWE_ASSET_MANIFEST"):
                        break  # immutable local failures cannot improve on retry
        if data is None:
            unresolved.append(path)
            logf.write(f"WARNING: could not resolve asset {path} (url={url})\n")
            continue
        flat = hashlib.sha256(path.encode()).hexdigest()
        (staging / flat).write_bytes(data)
        parent = shlex.quote(str(Path(path).parent))
        restore.append(f"mkdir -p {parent} && cp /image_assets/{flat} {shlex.quote(path)} || exit 87")
    logf.write(f"staged {len(restore)}/{len(assets)} asset(s); unresolved={unresolved}\n")
    return restore, unresolved


def _inject_asset_restore(eval_script, restore_cmds):
    """Insert asset-restore cmds just before the test-output start marker (verbatim
    port of swebench.harness.run_evaluation._inject_asset_restore)."""
    if not restore_cmds:
        return eval_script
    lines = eval_script.split("\n")
    for idx, line in enumerate(lines):
        if START_TEST_OUTPUT in line:
            return "\n".join(lines[:idx] + restore_cmds + lines[idx:])
    raise ValueError("cannot restore image assets before tests: missing test start marker")


def prepare_grading_log(test_log):
    """Give upstream's strict text reader UTF-8 without altering raw evidence."""
    test_log = Path(test_log)
    raw = test_log.read_bytes()
    try:
        return test_log, raw.decode("utf-8"), None
    except UnicodeDecodeError as error:
        text = raw.decode("utf-8", errors="replace")
        parser_log = test_log.with_name(test_log.stem + ".utf8" + test_log.suffix)
        normalized = text.encode("utf-8")
        parser_log.write_bytes(normalized)
        decoding = {
            "encoding": "utf-8", "errors": "replace",
            "raw_log": test_log.name, "parser_log": parser_log.name,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "parser_sha256": hashlib.sha256(normalized).hexdigest(),
            "first_invalid_byte": error.start,
        }
        test_log.with_suffix(".decoding.json").write_text(
            json.dumps(decoding, indent=2), encoding="utf-8")
        return parser_log, text, decoding


def validate_grade_report(iid, report, apply_log, test_log, rc, unresolved_assets):
    """Reconcile runtime evidence with the parser; infra failures are never scores."""
    r = dict(report.get(iid, {}))
    applied = re.findall(r"^APPLIED=([01])$", apply_log, re.M)
    reasons = []
    if rc != 0:
        reasons.append("namespace runtime failed")
    if len(applied) != 1:
        reasons.append("missing or ambiguous apply status")
    if unresolved_assets:
        reasons.append("unavailable image assets")
    runtime_failures = ("cannot set groups", "unshare: unshare failed", "chroot: failed",
                        "mount: permission denied", "Operation not permitted")
    if any(p in test_log for p in runtime_failures):
        reasons.append("container permission failure")
    # Chrome can fail before tests start while the shell still writes the end
    # marker and exits zero. A parser's unresolved result is then an infra
    # failure, not a valid zero. Match Chrome's fatal source line, not incidental
    # prose about sandboxes; this diagnoses failure without disabling a sandbox.
    if re.search(r"(?m)^\[[^\r\n]*\bFATAL:(?:[\w./-]+/)?zygote_host_impl_linux\.cc"
                 r"(?:\(\d+\)|:\d+)\]\s*No usable sandbox\b", test_log):
        reasons.append("browser sandbox startup failure")
    # Karma can also exhaust Firefox/Chrome launch retries (e.g. a missing
    # Firefox snap) and still emit complete test markers. Only match terminal
    # launcher records, not a transient failed attempt or quoted application text.
    clean_log = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", test_log)
    browsers = re.findall(
        r"(?m)^\d{1,2} \d{1,2} \d{4} \d{2}:\d{2}:\d{2}\.\d+:ERROR "
        r"\[launcher\]:\s*(Firefox|Chrome) failed \d+ times \(cannot start\)\. Giving up\.\s*$",
        clean_log)
    for browser in sorted(set(browsers)):
        reasons.append("browser launch failure ("+browser+")")
    if applied == ["0"] and not reasons:
        # The model's patch did not apply: valid unresolved result, no tests run.
        if APPLY_PATCH_FAIL not in test_log:
            reasons.append("missing apply-failed marker")
        r.update(resolved=False, patch_successfully_applied=False)
    elif applied == ["1"]:
        if APPLY_PATCH_PASS not in test_log or START_TEST_OUTPUT not in test_log or ">>>>> End Test Output" not in test_log:
            reasons.append("incomplete test log")
        if r.get("patch_successfully_applied") is not True or type(r.get("resolved")) is not bool:
            reasons.append("grading parser disagrees with runtime evidence")
    if reasons:
        r.update(resolved=False, infra_failure=True, infra_reasons=reasons)
    else:
        r["infra_failure"] = False
    r["instance_id"] = iid
    return {iid: r}


def grade_one(iid, preds, by_id, out, scratch, args, make_test_spec, get_eval_report):
    """Grade a SINGLE instance end-to-end (pull -> unpack -> chroot -> eval report).
    Fully self-contained: its own output dir, log, rootfs and mount/PID namespace,
    so N of these run concurrently with no shared mutable state. Never raises — a
    failure is recorded as infra_failure and invalidates the scalar score.
    Returns (iid, rep_dict, resolved_bool) where rep_dict feeds report.update()."""
    idir = out / iid; idir.mkdir(parents=True, exist_ok=True)
    logf = open(idir / "pull.log", "w")
    try:
        if iid not in preds:
            raise RuntimeError("no prediction")
        if iid not in by_id:
            raise RuntimeError("instance not in dataset split")
        pred = preds[iid]
        inst = by_id[iid]
        spec = make_test_spec(inst)
        log(f"[{iid}] image={spec.image}")
        rootfs = scratch / iid / "rootfs"
        _rm(str(rootfs))
        t0 = time.time()
        cfg = pull_and_unpack(spec.image, rootfs, scratch / iid, logf,
                              deadline=time.time() + args.pull_budget)
        log(f"[{iid}] unpacked in {time.time()-t0:.0f}s")

        # stage patch + eval.sh + chroot driver into rootfs
        (rootfs / "tmp").mkdir(parents=True, exist_ok=True)
        (rootfs / "tmp" / "patch.diff").write_text(pred.get("model_patch") or "")
        # image_assets restore: fetch the declared binary baselines and inject cp
        # restore cmds into eval.sh before the test-output marker (upstream parity).
        # Empty for the gold probe / most instances -> eval_script == spec.eval_script.
        restore_cmds, unresolved_assets = _stage_assets_into_rootfs(spec, rootfs, logf)
        eval_script = _inject_asset_restore(spec.eval_script, restore_cmds)
        (rootfs / "eval.sh").write_text(eval_script)
        chroot_sh = build_chroot_script(cfg)
        (rootfs / "run_in_chroot.sh").write_text(chroot_sh)
        (idir / "eval.sh").write_text(eval_script)
        (idir / "run_in_chroot.sh").write_text(chroot_sh)
        (idir / "patch.diff").write_text(pred.get("model_patch") or "")

        t0 = time.time()
        rc = run_in_namespace(rootfs, logf, timeout=args.timeout)
        log(f"[{iid}] eval finished rc={rc} in {time.time()-t0:.0f}s")

        tout = rootfs / "tmp" / "test_output.txt"
        test_log = idir / "test_output.txt"
        if tout.exists():
            shutil.copy(str(tout), str(test_log))
        else:
            test_log.write_text("")
        for extra in ("apply.log", "preflight.log", "ns_diag.log"):
            src = rootfs / "tmp" / extra
            if src.exists():
                shutil.copy(str(src), str(idir / extra))

        apply_path = rootfs / "tmp" / "apply.log"
        apply_log = apply_path.read_text(errors="replace") if apply_path.exists() else ""
        parser_log, test_text, decoding = prepare_grading_log(test_log)
        # An unapplied patch must not be promoted by a permissive test parser.
        if re.findall(r"^APPLIED=([01])$", apply_log, re.M) == ["0"]:
            rep = {iid: {"resolved": False, "patch_successfully_applied": False}}
        else:
            rep = get_eval_report(test_spec=spec, prediction=pred,
                                  test_log_path=str(parser_log), include_tests_status=True)
        rep = validate_grade_report(iid, rep, apply_log, test_text, rc, unresolved_assets)
        if decoding is not None:
            rep[iid]["log_decoding"] = decoding
        resolved = bool(rep.get(iid, {}).get("resolved"))
        if unresolved_assets and not resolved:
            # asset(s) couldn't be fetched -> ENVIRONMENT failure, not a model
            # failure. The adapter invalidates the run rather than silently
            # dropping failed instances from the denominator.
            rep.setdefault(iid, {}).update(
                {"infra_failure": True, "unresolved_assets": unresolved_assets})
            log(f"[{iid}] INFRA: {len(unresolved_assets)} asset(s) unresolved -> infra_failure")
        log(f"[{iid}] resolved={rep.get(iid,{}).get('resolved')} "
            f"applied={rep.get(iid,{}).get('patch_successfully_applied')}")
        return iid, rep, resolved
    except Exception as e:
        import traceback
        logf.write("EXC: " + traceback.format_exc())
        log(f"[{iid}] ERROR {e}")
        return iid, {iid: {"instance_id": iid, "resolved": False,
                          "infra_failure": True, "error": str(e)}}, False
    finally:
        logf.close()
        if not args.keep_rootfs:
            _rm(str(scratch / iid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--instance-ids", nargs="*", default=None)
    ap.add_argument("--dataset", default="SWE-bench/SWE-bench_Multimodal")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--scratch", default="/dev/shm/mmswe")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="wall-clock bound (s) on the in-namespace test run")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel grading workers (threads). Default: env "
                         "MMSWE_GRADE_JOBS, else 1. Each "
                         "instance grades in its own rootfs + mount/PID namespace, "
                         "so they are independent; blob cache is content-addressed "
                         "and thread-safe. 1 = serial (old behavior).")
    ap.add_argument("--pull-budget", type=int,
                    default=int(os.environ.get("MMSWE_PULL_BUDGET", "3600")),
                    help="wall-clock bound (s) on a COLD image pull; cached "
                         "layers are seconds so this only bites on a stuck pull")
    ap.add_argument("--keep-rootfs", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    from datasets import load_dataset
    from swebench.harness.utils import make_test_spec
    from swebench.harness.grading import get_eval_report

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch); scratch.mkdir(parents=True, exist_ok=True)

    preds = {}
    with open(args.preds) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            iid = p["instance_id"]
            if not isinstance(iid, str) or not iid or "/" in iid or iid in (".", "..") or iid in preds:
                raise ValueError("invalid or duplicate prediction instance id")
            preds[p["instance_id"]] = p
    want = args.instance_ids or list(preds.keys())
    if not want or len(set(want)) != len(want) or any(iid not in preds for iid in want):
        raise ValueError("empty or invalid requested instance set")
    log(f"grading {len(want)} instance(s): {want}")

    frozen_file = os.environ.get('MMSWE_FROZEN_DATASET_FILE')
    if frozen_file:
        from datasets import Dataset
        if hashlib.sha256(Path(frozen_file).read_bytes()).hexdigest() != os.environ.get('MMSWE_FROZEN_DATASET_SHA256'):
            raise ValueError('frozen grading dataset bytes changed')
        ds = Dataset.from_file(frozen_file)
    else:
        ds = load_dataset(args.dataset, split=args.split)
    by_id = {r["instance_id"]: r for r in ds}

    jobs = args.jobs if args.jobs else _default_jobs()
    jobs = max(1, min(jobs, len(want)))
    log(f"grading with jobs={jobs} (cpu_count={os.cpu_count()})")

    # each instance is fully isolated (own rootfs + mount/PID ns + log); the only
    # shared state was report/resolved_ids, which we collect per-instance and merge
    # in preds order afterwards, so the summary is identical to the serial run.
    rep_by_id = {}
    if jobs == 1:
        for iid in want:
            _, rep, _ = grade_one(iid, preds, by_id, out, scratch, args,
                                  make_test_spec, get_eval_report)
            rep_by_id[iid] = rep
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(grade_one, iid, preds, by_id, out, scratch, args,
                              make_test_spec, get_eval_report): iid for iid in want}
            for fut in as_completed(futs):
                iid, rep, _ = fut.result()
                rep_by_id[iid] = rep

    report = {}
    resolved_ids = []
    for iid in want:  # merge in stable preds order (order-independent of completion)
        rep = rep_by_id.get(iid, {iid: {"instance_id": iid, "resolved": False,
                                         "error": "no result produced"}})
        report.update(rep)
        if rep.get(iid, {}).get("resolved"):
            resolved_ids.append(iid)

    (out / "report.json").write_text(json.dumps(report, indent=2))
    summary = {
        "dataset": args.dataset, "split": args.split,
        "n": len(want), "resolved": len(resolved_ids),
        "resolved_ids": resolved_ids,
        "instance_ids": want,
        "infra_failure_ids": [iid for iid in want if report[iid].get("infra_failure", True)],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE resolved {len(resolved_ids)}/{len(want)}: {resolved_ids}")
    (out / "GRADE_DONE").write_text(str(len(resolved_ids)))
    return 5 if summary["infra_failure_ids"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
