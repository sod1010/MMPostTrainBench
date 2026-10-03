"""Real local TLS transport tests, not a claim of official grader acceptance."""
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from mmswe_service.offline_resources import load_snapshot, stage_snapshot, host_args, check_hosts
spec=importlib.util.spec_from_file_location('resource_freezer',ROOT/'scripts/freeze_mmswe_resources.py')
freezer=importlib.util.module_from_spec(spec);spec.loader.exec_module(freezer)
NODE=os.environ.get('MMSWE_TEST_NODE') or shutil.which('node')


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        (cls.root/'blobs').mkdir();cls.body=b'{"features":[{"properties":{"mag":1.0}}]}'
        digest=hashlib.sha256(cls.body).hexdigest();(cls.root/'blobs'/(digest+'.bin')).write_bytes(cls.body)
        freezer.tls_material(cls.root,{'resources.example.org'})
        data={'version':1,'mode':'loopback-https-snapshot',
              'resources':{'https://resources.example.org/data.json':{'sha256':digest,'size':len(cls.body),'content_type':'application/json'}},
              'tls_sha256':{n:hashlib.sha256((cls.root/n).read_bytes()).hexdigest() for n in ('ca.pem','server.pem','server.key')}}
        cls.manifest=cls.root/'manifest.json';cls.manifest.write_text(json.dumps(data))
        cls.pin={'path':str(cls.manifest),'sha256':hashlib.sha256(cls.manifest.read_bytes()).hexdigest()}
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def snapshot(self):
        with patch('mmswe_service.policy.private_file',side_effect=Path):return load_snapshot(self.pin)
    def test_verified_snapshot_stages_without_modifying_official_eval_script(self):
        snap=self.snapshot()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);original='  /bin/bash /eval.sh >> /tmp/test_output.txt 2>&1'
            (root/'run_in_chroot.sh').write_text(original);(root/'eval.sh').write_text('official command')
            stage_snapshot(root,snap)
            self.assertEqual((root/'eval.sh').read_text(),'official command')
            self.assertIn('/opt/mmptb-offline/offline_run.sh', (root/'run_in_chroot.sh').read_text())
            self.assertEqual(host_args(snap),['--add-host','resources.example.org:127.0.0.1'])
            check_hosts({'HostConfig':{'ExtraHosts':['resources.example.org:127.0.0.1']}},snap)
            with self.assertRaises(ValueError):check_hosts({'HostConfig':{'ExtraHosts':[]}},snap)
            checked=subprocess.run(['/bin/bash','-n',str(root/'opt/mmptb-offline/offline_run.sh')],capture_output=True)
            self.assertEqual(checked.returncode,0)
    def test_modified_blob_is_rejected_and_symlink_destination_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);clone=root/'copy';shutil.copytree(self.root,clone)
            blob=next((clone/'blobs').iterdir());blob.write_bytes(b'changed')
            with patch('mmswe_service.policy.private_file',side_effect=Path),self.assertRaisesRegex(ValueError,'bytes changed'):
                load_snapshot({**self.pin,'path':str(clone/'manifest.json')})
            fs=root/'rootfs';fs.mkdir();(fs/'opt').symlink_to(clone,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'unsafe'):stage_snapshot(fs,self.snapshot())
    def test_browser_trust_is_opt_in_pinned_and_cannot_replace_existing_database(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'snapshot';shutil.copytree(self.root,source)
            (source/'nssdb').mkdir();hashes={}
            for name in ('cert9.db','key4.db'):
                body=('test database bytes '+name).encode()
                (source/'nssdb'/name).write_bytes(body);hashes[name]=hashlib.sha256(body).hexdigest()
            data=json.loads((source/'manifest.json').read_text())
            data['browser_nss']={'homes':['/root'],'sha256':hashes}
            manifest=source/'manifest.json';manifest.write_text(json.dumps(data))
            pin={'path':str(manifest),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
            with patch('mmswe_service.policy.private_file',side_effect=Path):snap=load_snapshot(pin)
            for mode in ('normal','existing','symlink'):
                fs=root/mode;fs.mkdir();(fs/'run_in_chroot.sh').write_text('  /bin/bash /eval.sh >> /tmp/test_output.txt 2>&1')
                if mode=='existing':(fs/'root/.pki/nssdb').mkdir(parents=True)
                if mode=='symlink':(fs/'root').symlink_to(source,target_is_directory=True)
                if mode=='normal':
                    stage_snapshot(fs,snap)
                    self.assertEqual(hashlib.sha256((fs/'root/.pki/nssdb/cert9.db').read_bytes()).hexdigest(),hashes['cert9.db'])
                else:
                    with self.assertRaises((ValueError,FileExistsError)):stage_snapshot(fs,snap)
            (source/'nssdb/cert9.db').write_bytes(b'changed')
            with patch('mmswe_service.policy.private_file',side_effect=Path),self.assertRaisesRegex(ValueError,'trust bytes changed'):load_snapshot(pin)
    @unittest.skipUnless(NODE,'Node needed for real local TLS transport')
    def test_https_certificate_exact_resource_and_no_forwarding(self):
        with tempfile.TemporaryDirectory() as td:
            ready=Path(td)/'ready'
            p=subprocess.Popen([NODE,str(ROOT/'src/mmswe_service/offline_server.js'),str(self.root),'0',str(ready)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            self.addCleanup(p.stderr.close)
            try:
                for _ in range(100):
                    if ready.exists():break
                    if p.poll() is not None:self.fail(p.stderr.read().decode())
                    time.sleep(.05)
                self.assertTrue(ready.exists())
                port=int(ready.read_text());context=ssl.create_default_context(cafile=str(self.root/'ca.pem'))
                def request(target,host='resources.example.org',method='GET'):
                    with socket.create_connection(('127.0.0.1',port),timeout=3) as raw:
                        with context.wrap_socket(raw,server_hostname='resources.example.org') as tls:
                            tls.sendall((method+' '+target+' HTTP/1.1\r\nHost: '+host+'\r\nConnection: close\r\n\r\n').encode())
                            response=http.client.HTTPResponse(tls,method=method);response.begin();return response.status,response.read()
                self.assertEqual(request('/data.json'),(200,self.body))
                self.assertEqual(request('/data.json',method='HEAD'),(200,b''))
                self.assertEqual(request('/data.json?secret=x')[0],404)
                self.assertEqual(request('/data.json',host='unknown.example.org')[0],404)
                self.assertEqual(request('/data.json',method='POST')[0],405)
                self.assertEqual(request('https://unknown.example.org/data.json')[0],404)
            finally:
                p.terminate();p.wait(timeout=10)


if __name__=='__main__':unittest.main()
