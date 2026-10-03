"""Trusted transport invariants; these do not certify cross-node authorization."""
import hashlib,json,os
from pathlib import Path
import sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from mmswe_service.handoff import HandoffQueue,publish
from mmswe_service.artifacts import Rejected
from mmswe_service.backend import BackendUnhealthy


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()/'queue'
        self.policy={'version':1,'runs':['a','b'],'timeout_seconds':30,'evaluation_contract_sha256':'a'*64}
        self.queue=HandoffQueue(self.root,self.policy)
        self.body=b'{"instance_id":"repo-1","model_patch":"patch"}\n'
    def request(self,request_id='r1',run='a'):
        return self.queue.register(run,request_id,['repo-1'],'b'*64)
    def submit(self,request_id='r1'):
        request=self.request(request_id);publish(self.root/'inbox',request,self.body);return request
    def test_frozen_bytes_survive_producer_change_and_replay_does_not_grade_twice(self):
        request=self.submit();calls=[]
        def grader(preds,req,deadline,directory):
            calls.append(req['job'])
            (self.root/'inbox'/req['job']/'predictions.jsonl').write_bytes(b'changed')
            self.assertEqual(preds.read_bytes(),self.body)
            return {'correct':1,'n':1}
        result=self.queue.process(request['job'],grader)
        self.assertEqual(result['score'],{'correct':1,'n':1,'accuracy':1.0})
        self.assertNotIn('instance_ids',result)
        self.assertEqual(self.queue.process(request['job'],grader),result)
        self.assertEqual(len(calls),1)
        self.assertEqual(self.queue.register('a','r1',['repo-1'],'b'*64),request)
        with self.assertRaises(Rejected):self.queue.register('a','r1',['repo-2'],'b'*64)
    def test_unknown_run_changed_policy_and_incomplete_publish_are_refused(self):
        with self.assertRaises(Rejected):self.queue.register('unknown','r1',['repo-1'],'b'*64)
        with self.assertRaises(Rejected):HandoffQueue(self.root,{**self.policy,'timeout_seconds':40})
        request=self.request()
        self.assertEqual(self.queue.process(request['job'],lambda *args:self.fail('not published'))['status'],'queued')
        publish(self.root/'inbox',request,self.body)
        with self.assertRaises(OSError):publish(self.root/'inbox',request,self.body)
    def test_wrong_request_digest_coverage_and_symlink_do_not_reach_grader(self):
        for mode in ('digest','coverage','symlink'):
            request=self.submit(mode);base=self.root/'inbox'/request['job']
            if mode=='digest':
                ready=json.loads((base/'ready.json').read_text());ready['request_sha256']='0'*64
                (base/'ready.json').write_text(json.dumps(ready))
            if mode=='coverage':
                body=self.body.replace(b'repo-1',b'repo-2');(base/'predictions.jsonl').write_bytes(body)
                ready=json.loads((base/'ready.json').read_text());ready['predictions_sha256']=hashlib.sha256(body).hexdigest()
                (base/'ready.json').write_text(json.dumps(ready))
            if mode=='symlink':
                (base/'predictions.jsonl').unlink();(base/'predictions.jsonl').symlink_to('/etc/passwd')
            result=self.queue.process(request['job'],lambda *args:self.fail('invalid input executed'))
            self.assertEqual(result['status'],'rejected');self.assertNotIn('score',result)
    def test_valid_zero_is_distinct_from_infra_and_invalid_grader_coverage(self):
        for mode in ('zero','exception','coverage'):
            request=self.submit(mode)
            def grader(*args):
                if mode=='exception':raise RuntimeError('private test detail')
                return {'correct':0,'n':1 if mode=='zero' else 2}
            result=self.queue.process(request['job'],grader)
            if mode=='zero':self.assertEqual(result['score']['accuracy'],0.0)
            else:
                self.assertEqual(result['status'],'infra_failed');self.assertNotIn('score',result)
                self.assertNotIn('private test detail',json.dumps(result))
    def test_restart_does_not_replay_running_job_or_lose_terminal_feedback(self):
        request=self.submit();state=self.root/'jobs'/request['job']/'state.json'
        data=json.loads(state.read_text());data['status']='running';state.write_text(json.dumps(data))
        restarted=HandoffQueue(self.root,self.policy)
        result=restarted.process(request['job'],lambda *args:self.fail('restart replayed'))
        self.assertEqual(result['status'],'infra_failed');self.assertEqual(result['reason'],'worker_restart')
        feedback=self.root/'feedback'/(request['job']+'.json');feedback.unlink()
        self.assertEqual(restarted.process(request['job'],lambda *args:None),result)
        self.assertEqual(json.loads(feedback.read_text()),result)
    def test_cleanup_failure_blocks_later_execution(self):
        first=self.submit('first');second=self.submit('second')
        def broken(*args):raise BackendUnhealthy('private cleanup detail')
        result=self.queue.process(first['job'],broken)
        self.assertEqual(result['status'],'infra_failed');self.assertNotIn('score',result)
        with self.assertRaisesRegex(Rejected,'blocked'):
            self.queue.process(second['job'],lambda *args:self.fail('cleanup was ignored'))


if __name__=='__main__':unittest.main()
