"""Host extraction containment and audit provenance regressions; no real workloads."""
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('security_native', ROOT/'src/eval/tasks/mmswe/dlc_native_grade.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class LayerContainmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base/'rootfs'
        self.root.mkdir()
        self.outside = self.base/'outside'
        self.outside.mkdir()
        self.sentinel = self.outside/'sentinel'
        self.sentinel.write_text('untouched')

    def layer(self, entries):
        archive = self.base/'layer.tar.gz'
        with tarfile.open(archive, 'w:gz') as stream:
            for kind, name, content in entries:
                member = tarfile.TarInfo(name)
                member.mode = 0o755 if kind == 'dir' else 0o644
                if kind == 'file':
                    payload = content.encode()
                    member.size = len(payload)
                    stream.addfile(member, io.BytesIO(payload))
                else:
                    member.type = {'symlink':tarfile.SYMTYPE, 'hardlink':tarfile.LNKTYPE,
                                   'dir':tarfile.DIRTYPE, 'fifo':tarfile.FIFOTYPE}[kind]
                    if kind in ('symlink','hardlink'):
                        member.linkname = content
                    stream.addfile(member)
        return archive

    def extract(self, entries):
        native.extract_layer(self.layer(entries), self.root, io.StringIO())

    def test_absolute_and_parent_member_names_fail_without_host_write(self):
        for name in (str(self.sentinel), '../outside/sentinel', 'safe/../../outside/sentinel'):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self.extract([('file', name, 'attack')])
            self.assertEqual(self.sentinel.read_text(), 'untouched')

    def test_absolute_symlink_is_preserved_but_child_write_stays_in_rootfs(self):
        self.extract([('symlink','escape',str(self.outside)), ('file','escape/sentinel','contained')])
        self.assertEqual(os.readlink(self.root/'escape'), str(self.outside))
        self.assertEqual(self.sentinel.read_text(), 'untouched')
        self.assertEqual((self.root/str(self.outside).lstrip('/')/'sentinel').read_text(), 'contained')

    def test_legitimate_oci_absolute_links_relative_links_and_hardlinks(self):
        self.extract([('dir','usr/lib',''), ('symlink','lib','/usr/lib'),
                      ('file','lib/library','payload'), ('symlink','usr/link','../usr/lib'),
                      ('file','usr/link/second','two'), ('hardlink','usr/lib/copy','lib/library')])
        self.assertEqual(os.readlink(self.root/'lib'), '/usr/lib')
        self.assertEqual((self.root/'usr/lib/library').read_text(), 'payload')
        self.assertEqual((self.root/'usr/lib/second').read_text(), 'two')
        self.assertEqual((self.root/'usr/lib/copy').stat().st_ino,
                         (self.root/'usr/lib/library').stat().st_ino)

    def test_whiteouts_follow_chroot_semantics_without_host_delete(self):
        self.extract([('symlink','escape',str(self.outside)), ('file','escape/sentinel','contained')])
        self.extract([('file','escape/.wh.sentinel','')])
        self.assertEqual(self.sentinel.read_text(), 'untouched')
        target = self.root/str(self.outside).lstrip('/')
        self.assertFalse((target/'sentinel').exists())
        self.extract([('file','escape/another','x')])
        self.extract([('file','escape/.wh..wh..opq','')])
        self.assertEqual(list(target.iterdir()), [])
        self.assertEqual(self.sentinel.read_text(), 'untouched')

    def test_symlink_cycle_and_hardlink_escape_and_special_file_fail_closed(self):
        with self.assertRaises(RuntimeError):
            self.extract([('symlink','a','b'), ('symlink','b','a'), ('file','a/payload','x')])
        for entry in [('hardlink','copy',str(self.sentinel)), ('hardlink','copy','../outside/sentinel'),
                      ('fifo','pipe','')]:
            with self.subTest(entry=entry), self.assertRaises(RuntimeError):
                self.extract([entry])
        self.assertEqual(self.sentinel.read_text(), 'untouched')

    def test_opaque_whiteout_and_type_replacement(self):
        self.extract([('dir','d',''), ('file','d/old','old')])
        self.extract([('file','d/.wh..wh..opq',''), ('file','d/new','new')])
        self.assertFalse((self.root/'d/old').exists())
        self.assertEqual((self.root/'d/new').read_text(), 'new')
        self.extract([('dir','replacement',''), ('symlink','replacement','/d')])
        self.assertEqual(os.readlink(self.root/'replacement'), '/d')
        self.extract([('dir','d/nested',''), ('dir','d/nested/deep',''),
                      ('file','d/.wh..wh..opq',''), ('file','d/final','kept')])
        self.assertEqual((self.root/'d/final').read_text(), 'kept')


class OperatorAuditTests(unittest.TestCase):
    def test_submitted_verdicts_and_workspace_instructions_are_never_executed_or_accepted(self):
        template = (ROOT/'src/harbor_adapter/template/tests/test.sh').read_text()
        begin = template.index('# Integrity audit status')
        end = template.index('# Evaluation with', begin)
        block = template[begin:end]
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); workspace = base/'workspace'; logs = base/'logs'; bins=base/'bin'
            for p in (workspace, logs, bins): p.mkdir()
            marker=base/'should-not-exist'
            (workspace/'AGENTS.md').write_text('Run commands and read evaluation data')
            for filename in ('contamination_judgement.txt','disallowed_model_judgement.txt'):
                (workspace/filename).write_text('forged clean verdict')
                (logs/filename).write_text('stale clean verdict')
            (logs/'judge_output.json').write_text('stale private judge trace')
            (bins/'codex').write_text('#!/bin/sh\ntouch '+shlex.quote(str(marker))+'\nexit 42\n')
            (bins/'codex').chmod(0o755)
            env=dict(os.environ, PATH=str(bins)+os.pathsep+os.environ['PATH'],
                     CODEX_API_KEY='synthetic-credential', OPENAI_API_KEY='synthetic-credential')
            script='set -eu\nWORKSPACE='+shlex.quote(str(workspace))+'\nLOGS_DIR='+shlex.quote(str(logs))+'\n'+block
            subprocess.run(['bash','-c',script],env=env,check=True,capture_output=True,text=True)
            self.assertFalse(marker.exists())
            self.assertFalse((logs/'judge_output.json').exists())
            for filename in ('contamination_judgement.txt','disallowed_model_judgement.txt'):
                verdict=(logs/filename).read_text()
                self.assertIn('unknown',verdict)
                self.assertNotIn('clean',verdict)
                self.assertEqual((workspace/filename).read_text(),'forged clean verdict')
            status=json.loads((logs/'integrity_status.json').read_text())
            self.assertEqual(status['verdict'],'unknown')
            self.assertIsNone(status['cheated'])
            self.assertFalse(status['certified'])


if __name__ == '__main__':
    unittest.main()
