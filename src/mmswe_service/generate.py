"""Fixed generation entrypoint; has no Docker socket or grader assets."""
import json
import os
import subprocess
import sys
from pathlib import Path

request = json.loads(Path('/input/request/request.json').read_text())
if request['split'] not in ('dev', 'test') or request['eval_split'] != {'dev':'val','test':'eval'}[request['split']]:
    raise SystemExit('invalid split')
# Paths are image constants, not client request fields.
os.environ.update(HF_HOME='/input/hf', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  EVAL_SPLIT=request['eval_split'], MMSWE_SPLIT=request['split'], SWE_SPLIT=request['split'],
                  MMPTB_ROLE='verifier' if request['split']=='test' else 'agent',
                  MMSWE_FROZEN_DATASET_FILE=request['container_dataset_file'],
                  MMSWE_FROZEN_DATASET_SHA256=request['dataset_file_sha256'],
                  MMSWE_EXPECTED_IDS_JSON=json.dumps(request['instance_ids']),
                  MMSWE_EVALUATION_CONTRACT_SHA256=request['contract_sha256'],
                  MMSWE_IMAGE_MANIFEST='/input/images/manifest.json',
                  MMSWE_IMAGE_MANIFEST_SHA256=request['image_manifest_sha256'],
                  MMSWE_REQUIRE_FROZEN_IMAGES='1', MMPTB_SRC_EVAL='/opt/mmptb-service/eval')
raise SystemExit(subprocess.call([sys.executable, '/opt/eval/runners/run_mmswe_official.py',
                                  '--model-path', '/input/model', '--limit', str(request['limit']),
                                  '--out', '/output/predictions.jsonl']))
