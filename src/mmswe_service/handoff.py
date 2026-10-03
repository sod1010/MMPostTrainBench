"""Operator-to-operator file transport, not a research-agent authentication API.

The registered request is private to the controller. A trusted generation worker
publishes predictions atomically. Intake freezes bytes before invoking a fixed
grader callback. Cross-node UID/mount isolation and GPU dispatch are external
deployment requirements; a shared-storage filename is never treated as authenticated UID.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import uuid

from .artifacts import Rejected, canonical, freeze, open_dir, discard_private_tree
from .backend import atomic, BackendUnhealthy
from .policy import identifier

MAX_BYTES = 128*1024*1024
TERMINAL = {'succeeded', 'infra_failed', 'rejected'}


def sha(body):
    return hashlib.sha256(body).hexdigest()


def publish(inbox, request, predictions):
    """Trusted producer helper. Never writes a partially complete ready directory."""
    identifier(request['job'])
    body = predictions if isinstance(predictions, bytes) else predictions.encode()
    if len(body) > MAX_BYTES:
        raise Rejected('prediction byte limit')
    inbox=Path(inbox)
    fd=open_dir(inbox);os.close(fd)
    tmp=inbox/('.publishing-'+uuid.uuid4().hex);tmp.mkdir(mode=0o700)
    try:
        ready={'version':1,'job':request['job'],'request_sha256':request['request_sha256'],
               'predictions_sha256':sha(body)}
        for name,data in [('predictions.jsonl',body),('ready.json',canonical(ready))]:
            with (tmp/name).open('xb') as f:
                f.write(data);f.flush();os.fsync(f.fileno())
        # A job directory is immutable once published. Never replace old work.
        os.rename(tmp,inbox/request['job'])
        fd=os.open(inbox,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    except BaseException:
        if tmp.exists():discard_private_tree(tmp)
        raise


class HandoffQueue:
    def __init__(self, root, policy):
        self.root=Path(root)
        self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
        fd=open_dir(self.root)
        try:
            s=os.fstat(fd)
            if s.st_uid!=os.geteuid() or stat.S_IMODE(s.st_mode)!=0o700:
                raise Rejected('handoff root must be private and controller-owned')
        finally:os.close(fd)
        if (set(policy)!={'version','runs','timeout_seconds','evaluation_contract_sha256'} or
                type(policy['version']) is not int or policy['version']!=1):
            raise Rejected('invalid handoff policy')
        if (not isinstance(policy['runs'],list) or not policy['runs'] or
                len(set(policy['runs']))!=len(policy['runs'])):
            raise Rejected('unique registered runs required')
        for run in policy['runs']:identifier(run)
        if type(policy['timeout_seconds']) is not int or not 0<policy['timeout_seconds']<=86400:
            raise Rejected('bounded handoff deadline required')
        digest=policy['evaluation_contract_sha256']
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise Rejected('evaluation contract digest required')
        self.policy=policy;self.policy_hash=sha(canonical(policy))
        for name in ('jobs','inbox','feedback'):(self.root/name).mkdir(mode=0o700,exist_ok=True)
        with self.lock():
            path=self.root/'policy.json'
            if path.exists():
                if json.loads(path.read_text())!={'sha256':self.policy_hash,'policy':policy}:
                    raise Rejected('handoff policy changed; use a new queue')
            else:atomic(path,{'sha256':self.policy_hash,'policy':policy})

    @contextmanager
    def lock(self):
        fd=os.open(self.root/'worker.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield
        finally:os.close(fd)

    def register(self, run, request_id, instance_ids, artifact_sha256):
        """Controller-only registration; client-supplied run names do not grant access."""
        identifier(run);identifier(request_id)
        if run not in self.policy['runs']:raise Rejected('unregistered run')
        if (not isinstance(instance_ids,list) or not instance_ids or len(instance_ids)>480 or
                any(not isinstance(i,str) or not i or len(i)>200 for i in instance_ids) or
                len(set(instance_ids))!=len(instance_ids)):
            raise Rejected('fixed ordered instance IDs required')
        if (not isinstance(artifact_sha256,str) or len(artifact_sha256)!=64 or
                any(c not in '0123456789abcdef' for c in artifact_sha256)):
            raise Rejected('frozen generation artifact identity required')
        request={'version':1,'run':run,'request_id':request_id,'split':'dev',
                 'instance_ids':instance_ids,'artifact_sha256':artifact_sha256,
                 'policy_sha256':self.policy_hash}
        job=sha(canonical({'run':run,'request_id':request_id}))[:32]
        request.update(job=job,request_sha256=sha(canonical(request)))
        with self.lock():
            directory=self.root/'jobs'/job
            if directory.exists():
                if json.loads((directory/'request.json').read_text())!=request:
                    raise Rejected('request ID reused with different inputs')
                return request
            directory.mkdir(mode=0o700)
            atomic(directory/'request.json',request)
            atomic(directory/'state.json',{'status':'queued','job':job,'created':time.time()})
        return request

    def public(self, request, state):
        result={k:request[k] for k in ('job','run','request_id','artifact_sha256','policy_sha256')}
        result['status']=state['status']
        for key in ('predictions_sha256','reason','score'):
            if key in state:result[key]=state[key]
        return result

    def process(self, job, grader):
        """Single-consumer step. Grader is fixed by operator code, never by files.

        Callback receives (frozen predictions path, private request, deadline,
        job directory) and returns exactly integer correct/n after independently
        validating the official report. Exceptions mean no numeric feedback.
        """
        identifier(job)
        with self.lock():
            directory=self.root/'jobs'/job
            request=json.loads((directory/'request.json').read_text())
            state=json.loads((directory/'state.json').read_text())
            if state['status'] in TERMINAL:
                # Repair a feedback publication interrupted after state commit.
                public=self.public(request,state)
                atomic(self.root/'feedback'/(job+'.json'),public)
                return public
            if (self.root/'blocked.json').exists():
                raise Rejected('handoff grading backend blocked pending operator cleanup')
            if state['status']=='running':
                state.update(status='infra_failed',reason='worker_restart')
            elif not (self.root/'inbox'/job).exists():
                return self.public(request,state)
            else:
                deadline=time.monotonic()+self.policy['timeout_seconds']
                state.update(status='running',started=time.time())
                atomic(directory/'state.json',state)
                try:
                    manifest=freeze(self.root/'inbox',job,directory/'frozen',MAX_BYTES+4096,2,deadline)
                    if {f['path'] for f in manifest['files']}!={'predictions.jsonl','ready.json'}:
                        raise Rejected('unexpected handoff files')
                    ready_path=directory/'frozen/ready.json'
                    if ready_path.stat().st_size>4096:raise Rejected('oversized ready metadata')
                    ready=json.loads(ready_path.read_text())
                    preds=directory/'frozen/predictions.jsonl';body=preds.read_bytes()
                    expected={'version':1,'job':job,'request_sha256':request['request_sha256'],
                              'predictions_sha256':sha(body)}
                    if ready!=expected:raise Rejected('handoff identity or digest mismatch')
                    rows=[json.loads(line) for line in body.splitlines() if line.strip()]
                    if ([row.get('instance_id') for row in rows]!=request['instance_ids'] or
                            any(not isinstance(row.get('model_patch'),str) for row in rows)):
                        raise Rejected('prediction coverage or patch schema mismatch')
                    state['predictions_sha256']=sha(body)
                    atomic(directory/'input_manifest.json',manifest)
                    score=grader(preds,request,deadline,directory)
                    if time.monotonic()>deadline:raise TimeoutError('grading deadline')
                    if (not isinstance(score,dict) or set(score)!={'correct','n'} or
                            type(score['correct']) is not int or type(score['n']) is not int or
                            score['n']!=len(rows) or not 0<=score['correct']<=score['n']):
                        raise RuntimeError('invalid grader coverage or score')
                    state.update(status='succeeded',score={**score,'accuracy':score['correct']/score['n']})
                except (Rejected,OSError,ValueError) as error:
                    # Error text and per-item details remain private.
                    state.update(status='rejected' if isinstance(error,Rejected) else 'infra_failed',reason='invalid_input' if isinstance(error,Rejected) else 'grading_failed')
                except BackendUnhealthy:
                    atomic(self.root/'blocked.json',{'reason':'cleanup_failed','job':job})
                    state.update(status='infra_failed',reason='cleanup_failed')
                except Exception:
                    state.update(status='infra_failed',reason='grading_failed')
            state['finished']=time.time()
            if state['status']!='succeeded':state.pop('score',None)
            atomic(directory/'state.json',state)
            public=self.public(request,state)
            atomic(self.root/'feedback'/(job+'.json'),public)
            return public
