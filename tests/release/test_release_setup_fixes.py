import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
def load(name, p):
    s=importlib.util.spec_from_file_location(name, ROOT/p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
paths=load('runtime_guard','src/docker/runtime_paths.py')
ovb=load('public_ovb','eval_omni/runners/convert_omnivideobench.py')

class RuntimeTests(unittest.TestCase):
    def test_guard_protects_model_repo_and_symlink_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()/'runtime'
            env=dict(MMPTB_ROOT=str(root), REPO_ROOT=str(Path(tmp).resolve()/'repo'),MODEL_DIR=str(root/'models/base'),DATA_DIR=str(root/'data'),HF_CACHE_DIR=str(root/'cache'),WORKSPACE_HOST=str(root/'workspaces/run'),LOGS_HOST=str(root/'logs/run'))
            paths.validate(env)
            for bad in [str(root),str(root/'models/base'),str(root/'workspaces')]:
                with self.subTest(bad=bad),self.assertRaises(ValueError):paths.validate({**env,'WORKSPACE_HOST':bad})
            (root/'workspaces').mkdir(parents=True);(root/'workspaces/link').symlink_to(root/'models')
            with self.assertRaises(ValueError):paths.validate({**env,'WORKSPACE_HOST':str(root/'workspaces/link/run')})

    def test_empty_post_field_preserved(self):
        a=subprocess.run(['bash','-c',"IFS='|' read -r id repo glob post token <<< 'mmau|org/repo|*mmau*||0'; printf '%s,%s' \"$post\" \"$token\""],capture_output=True,text=True,check=True)
        self.assertEqual(a.stdout,',0')

    def test_selection_after_config_and_budget(self):
        s=(ROOT/'src/docker/run_agent.sh').read_text()
        start=s.index('if [ -n "${BENCH:');end=s.index('AGENT_ENGINE=',start)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fake=root/'runtime_paths.py';fake.write_text('')
            code='BENCH=mmar; REPO_ROOT=/checkout; HERE='+str(root)+'; '+s[start:end]+'printf "%s" "$TASK_DIR"'
            result=subprocess.run(['bash','-c',code],capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout,'/checkout/harbor_tasks/mmposttrainbench-mmar-qwen3-omni-30b')
        sys.path.insert(0,str(ROOT/'src/harbor_adapter'))
        from adapter import PostTrainBenchAdapter
        import inspect
        self.assertEqual(inspect.signature(PostTrainBenchAdapter.__init__).parameters['num_hours'].default,24)

class DownloaderTests(unittest.TestCase):
    def test_actual_download_script_handles_no_post_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp).resolve(); script=b/'prepare_data.sh'
            script.write_text((ROOT/'src/docker/prepare_data.sh').read_text())
            (b/'model').mkdir();(b/'model/config.json').write_text('{}');(b/'model/weights.safetensors').write_bytes(b'')
            manifest=b/'resources.json';manifest.write_text(json.dumps({'model':{'default_repo':'fixture/model'},'benches':[{'id':'mmau','repo':'fixture/data','cache_glob':'*fixture*'}]}))
            hf=b/'hf';hf.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$DOWNLOAD_CALLS"\n');hf.chmod(0o755)
            (b/'config.env').write_text(f'export MODEL_DIR="{b}/model" HF_CACHE_DIR="{b}/cache" DATA_DIR="{b}/data" HF_ENDPOINT=https://huggingface.co HF_BIN="{hf}" RESOURCES_JSON="{manifest}" REPO_ROOT="{ROOT}"\n')
            env={**os.environ,'DOWNLOAD_CALLS':str(b/'calls'),'PREPARE_PY':sys.executable}
            result=subprocess.run(['bash',str(script),'mmau'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('download fixture/data --repo-type dataset',(b/'calls').read_text())
            self.assertNotIn('--local-dir',(b/'calls').read_text())

class OVBTests(unittest.TestCase):
    def fixture(self, base):
        (base/'videos').mkdir();(base/'videos/v1.mp4').write_bytes(b'fixture')
        return [{'video':'v1','duration':'01:02:03','questions':[{'question':'q','options':['A.a','B.b'],'correct_option':'B'}]}]
    def test_official_grouped_schema_and_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp);rows=self.fixture(b);(b/'data.json').write_text(json.dumps(rows))
            out=b/'result/qa.json';n=ovb.convert(b,out,b/'staged')
            self.assertEqual(n,1);self.assertEqual(json.loads(out.read_text())[0]['duration'],'62:03')
            self.assertTrue((b/'staged/v1.mp4').is_file())
    def test_incomplete_annotations_never_emit_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp);rows=self.fixture(b);rows[0]['questions'][0].pop('correct_option');(b/'data.json').write_text(json.dumps(rows));out=b/'result.json'
            with self.assertRaises(ValueError):ovb.convert(b,out,b/'staged')
            self.assertFalse(out.exists())
            rows[0]['questions'][0]['correct_option']='B';rows[0]['video']='../escape';(b/'data.json').write_text(json.dumps(rows))
            with self.assertRaises(ValueError):ovb.convert(b,out,b/'staged')
    def test_verifier_translates_data_paths(self):
        s=(ROOT/'src/docker/run_verifier.sh').read_text()
        self.assertIn('-e DATA_DIR=/data',s)
        self.assertIn('JOINTAV_DATA=/data/evaluationbench/JointAVBench/jointavbench.json',s)
        self.assertIn('OVB_DATA=/data/evaluationbench/OmniVideoBench_local/data.json',s)
        self.assertNotIn('-e CODEX_API_KEY',s)

if __name__=='__main__':unittest.main()
