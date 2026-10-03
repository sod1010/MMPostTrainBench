"""Deployment audit must not confuse fixtures/static success with a runnable loop."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('deployment_check',ROOT/'scripts/mmswe_deployment_check.py')
check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.contract=self.root/'contract.json'
        self.data={'version':1,'dataset':'SWE-bench/SWE-bench_Multimodal',
                   'source_snapshot':str(self.root),'source_files':{},'splits':{}}
        for name in ['dataset_info.json','swe-bench_multimodal-dev.arrow','swe-bench_multimodal-test.arrow']:
            (self.root/name).write_bytes(b'fixture')
            self.data['source_files'][name]={'size':7,'sha256':hashlib.sha256(b'fixture').hexdigest()}
        for split,n in [('dev',100),('test',480)]:
            ids=[split+'__repo-'+str(i) for i in range(n)]
            self.data['splits'][split]={'n':n,'instance_ids_in_order':ids,
                'ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()}
        self.save()
    def save(self):self.contract.write_text(json.dumps(self.data))
    def test_missing_production_policy_is_reported_without_docker_calls(self):
        with patch.object(check.subprocess,'run') as run:
            result=check.deployment_inventory(self.root/'missing.json')
        self.assertEqual(result['status'],'configuration_blocked')
        self.assertFalse(result['loop_ready']);run.assert_not_called()
    def test_reference_counts_and_bytes_are_checked(self):
        result=check.contract_inventory(self.contract)
        self.assertEqual(result['splits']['dev']['repository_counts'],{'dev__repo':100})
        (self.root/'swe-bench_multimodal-dev.arrow').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):check.contract_inventory(self.contract)
    def test_different_ids_same_denominator_fail_the_frozen_hash(self):
        self.data['splits']['dev']['instance_ids_in_order'][0]='other__repo-1';self.save()
        with self.assertRaisesRegex(ValueError,'hash mismatch'):check.contract_inventory(self.contract)
    def test_frozen_contract_rejects_missing_source_and_path_escape(self):
        self.data['source_files']['../escape']=self.data['source_files']['dataset_info.json'];self.save()
        with self.assertRaisesRegex(ValueError,'filename'):check.contract_inventory(self.contract)
        del self.data['source_files']['../escape'];del self.data['source_files']['dataset_info.json'];self.save()
        with self.assertRaisesRegex(ValueError,'missing'):check.contract_inventory(self.contract)


if __name__=='__main__':unittest.main()
