import hashlib
import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from mmswe_service.contracts import load_contract, score_contract, load_baseline
from mmswe_service.artifacts import Rejected, canonical


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.cache=self.root/'cache';self.source=self.cache/'hf/snapshot'
        self.source.mkdir(parents=True);self.path=self.root/'contract.json'
        self.data={'version':1,'dataset':'SWE-bench/SWE-bench_Multimodal',
                   'source_snapshot':str(self.source),'source_files':{},'splits':{}}
        for name in ['dataset_info.json','swe-bench_multimodal-dev.arrow','swe-bench_multimodal-test.arrow']:
            body=name.encode();(self.source/name).write_bytes(body)
            self.data['source_files'][name]={'size':len(body),'sha256':hashlib.sha256(body).hexdigest()}
        for split,n in [('dev',100),('test',480)]:
            ids=[split+'-'+str(i) for i in range(n)]
            self.data['splits'][split]={'n':n,'instance_ids_in_order':ids,
                'ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()}
        self.trust=patch('mmswe_service.policy.private_file',side_effect=Path);self.trust.start();self.addCleanup(self.trust.stop)
        self.save()
    def save(self):
        self.path.write_text(json.dumps(self.data));self.pin={'path':str(self.path),'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()}
    def test_generation_and_grading_paths_bind_the_same_snapshot(self):
        r=score_contract({'evaluation_contract':self.pin,'cache':str(self.cache)},'dev',5)
        self.assertEqual(r['instance_ids'],['dev-'+str(i) for i in range(5)])
        self.assertEqual(r['container_dataset_file'],'/input/hf/snapshot/swe-bench_multimodal-dev.arrow')
        self.assertEqual(r['host_dataset_file'],str(self.source/'swe-bench_multimodal-dev.arrow'))
        self.assertEqual(r['contract_sha256'],self.pin['sha256'])
    def test_changed_bytes_and_same_count_changed_ids_are_rejected(self):
        (self.source/'dataset_info.json').write_text('changed')
        with self.assertRaisesRegex(Rejected,'bytes changed'):load_contract(self.pin,str(self.cache))
        (self.source/'dataset_info.json').write_bytes(b'dataset_info.json')
        self.data['splits']['dev']['instance_ids_in_order'][0]='different';self.save()
        with self.assertRaisesRegex(Rejected,'ID hash'):load_contract(self.pin,str(self.cache))
    def test_unmounted_dataset_and_path_escape_are_rejected(self):
        with self.assertRaisesRegex(Rejected,'inside'):load_contract(self.pin,str(self.root/'other'))
        self.data['source_files']['../escape']={'size':1,'sha256':'a'*64};self.save()
        with self.assertRaisesRegex(Rejected,'filename'):load_contract(self.pin,str(self.cache))
    def test_snapshot_pin_change_and_oversized_subset_are_rejected(self):
        self.path.write_text(self.path.read_text()+' ')
        with self.assertRaisesRegex(Rejected,'contract bytes'):load_contract(self.pin,str(self.cache))
        self.save()
        with self.assertRaisesRegex(Rejected,'size'):score_contract({'evaluation_contract':self.pin,'cache':str(self.cache)},'dev',101)


class BaselineTests(unittest.TestCase):
    def test_frozen_model_rejects_changed_added_and_redirected_files(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);model=root/'model';model.mkdir()
            (model/'config.json').write_text('{"model_type":"qwen3_omni_moe"}')
            (model/'tokenizer_config.json').write_text('{}')
            header=json.dumps({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
            (model/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'1234')
            records=[{'path':p.name,'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(model.iterdir())]
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'version':1,'path':str(model),'files':records,
                'bytes':sum(r['size'] for r in records),'sha256':hashlib.sha256(canonical(records)).hexdigest()}))
            pin={'path':str(manifest),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
            with patch('mmswe_service.policy.private_file',side_effect=Path):
                self.assertEqual(load_baseline(pin,model)['files'],records)
                with self.assertRaisesRegex(Rejected,'path mismatch'):load_baseline(pin,root/'other')
                (model/'new.json').write_text('{}')
                with self.assertRaisesRegex(Rejected,'file set'):load_baseline(pin,model)
                (model/'new.json').unlink()
                (model/'tokenizer_config.json').write_text('[]')
                with self.assertRaisesRegex(Rejected,'bytes changed'):load_baseline(pin,model)


if __name__=='__main__':unittest.main()
