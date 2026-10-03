import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / 'src/eval'
sys.path.insert(0, str(EVAL))
from baseline_util import matched_baseline
from split_util import mmswe_config
from unittest.mock import patch


def mmswe_metrics(**changes):
    return dict(benchmark='mmswe', accuracy=11/480, n=480, eval_split='eval',
                dataset='SWE-bench/SWE-bench_Multimodal', dataset_split='test', **changes)


class PaperProtocolTests(unittest.TestCase):
    def test_published_sizes_and_reference(self):
        b = json.loads((EVAL/'baselines.json').read_text())
        self.assertEqual(b['n_dev']['mmswe'], 100)
        self.assertEqual(b['n_eval']['mmswe'], 480)
        self.assertEqual(b['scores']['mmswe'], .0229)
        self.assertEqual(b['n_eval']['mmmu_pro'], 1172)
        self.assertEqual(matched_baseline('mmswe', mmswe_metrics()), .0229)

    def test_wrong_split_count_or_task_is_not_comparable(self):
        for key,value in [('n',34),('n',100),('n',479),('eval_split','val'),
                          ('eval_split',''),('dataset_split','dev'),
                          ('benchmark','mmau'),('accuracy',float('nan'))]:
            with self.subTest(key=key,value=value):
                m=mmswe_metrics();m[key]=value
                with self.assertRaises(ValueError): matched_baseline('mmswe',m)

    def test_recipe_does_not_default_grader_to_dev(self):
        for split,want in [('val','dev'),('eval','test')]:
            with tempfile.TemporaryDirectory() as td:
                env=dict(os.environ, EVAL_SPLIT=split, OMNI_PY=sys.executable)
                env.pop('SWE_SPLIT',None);env.pop('MMSWE_SPLIT',None)
                command='source "$1"; emit_bench_cmd mmswe /model "$2" -1'
                r=subprocess.run(['bash','-c',command,'fixture',str(ROOT/'src/docker/bench_recipes.sh'),td],
                                 env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stderr)
                text=(Path(td)/'cmd.sh').read_text()
                self.assertNotIn('SWE_SPLIT=dev',text)
                with patch.dict(os.environ,env,clear=True):
                    self.assertEqual(mmswe_config(split)[1],want)

    def test_gate_matches_reference_and_never_falls_back_to_zero(self):
        source=(ROOT/'src/docker/cheat_gate.sh').read_text()
        code=source.split("<<'PY'\n")[-1].rsplit('\nPY',1)[0]
        for n,good in [(480,True),(34,False)]:
            with tempfile.TemporaryDirectory() as td:
                metrics=mmswe_metrics();metrics['n']=n
                mp=Path(td)/'metrics.json';mp.write_text(json.dumps(metrics))
                report=Path(td)/'report.json'
                r=subprocess.run([sys.executable,'-','mmswe',str(11/480),'True','false',
                    'false','',str(EVAL/'baselines.json'),str(report),'false',''],input=code,
                    text=True,capture_output=True,env=dict(os.environ,GATE_METRICS=str(mp)))
                data=json.loads(report.read_text())
                self.assertEqual(r.returncode==0,good,r.stderr)
                self.assertEqual(data['final_reward'],.0229 if good else None)

    def test_oracle_uses_full_raw_metrics_not_adjusted_reward(self):
        for count,score,good in [(480,11/480,True),(34,.0229,False),(480,0,False)]:
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'metrics.json';m=mmswe_metrics();m.update(n=count,accuracy=score)
                p.write_text(json.dumps(m))
                (Path(td)/'reward_final.txt').write_text('.0229')
                r=subprocess.run([sys.executable,str(EVAL/'baseline_util.py'),'--bench','mmswe',
                                  '--metrics',str(p)],capture_output=True,text=True)
                self.assertEqual(r.returncode==0,good,r.stdout)

    def test_explicit_denominator_not_resplit(self):
        spec=importlib.util.spec_from_file_location('core_metrics_paper',EVAL/'compute_core_metrics.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        b=json.loads((EVAL/'baselines.json').read_text())
        self.assertEqual(mod.derive_eval_n('mmswe',b),480)
        self.assertEqual(mod.derive_eval_n('omnivideobench',b),668)

    def test_task_dispatch_exports_selected_benchmark(self):
        source=(ROOT/'run_task.sh').read_text()
        dispatch=source[source.index('export BENCHMARK="$BENCH"'):source.index('BACKEND=')]
        r=subprocess.run(['bash','-c','BENCH=mmswe; AGENT_MODEL=; MODEL=;\n'+dispatch+
            '\n'+sys.executable+' -c \'import os; print(os.environ["BENCH"])\''],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(r.stdout.strip(),'mmswe')


if __name__=='__main__': unittest.main()
