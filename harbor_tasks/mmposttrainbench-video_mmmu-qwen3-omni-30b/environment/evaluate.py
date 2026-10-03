#!/usr/bin/env python3
"""Video-MMMU evaluate.py — omni seam adapter (lmms-eval class) for mmposttrainbench.

Thin shim: pins this bench's lmms-eval task config, then delegates to the shared
lmms_common adapter (src/eval/lmms_common/evaluate.py), which runs
`python -m lmms_eval --model qwen3_omni --tasks $LMMS_TASK ...` and normalizes the
result to the flat {"accuracy": <0-1 float>} contract that tests/test.sh reads.

Same verifier contract as the self-runner adapters (mmau/mmar): the verifier calls
    python3 evaluate.py --model-path ... --json-output-file ... --limit ...
and lmms_common parses those (extra flags tolerated). We locate lmms_common next to
us when bundled into the verifier image, else via the repo layout, $LMMS_COMMON, or
the baked omni-eval image path.
"""
import os
import runpy
import sys

# Per-bench lmms-eval config (mirrors src/docker/bench_recipes.sh video_mmmu recipe).
os.environ.setdefault("LMMS_TASK", "video_mmmu_comprehension")
os.environ.setdefault("LMMS_METRIC", "mmmu_acc")
os.environ.setdefault("LMMS_DIVISOR", "1")

_here = os.path.dirname(os.path.abspath(__file__))
_cands = [
    os.path.join(_here, "lmms_common", "evaluate.py"),                          # bundled beside us (verifier /tests)
    os.path.join(_here, os.pardir, os.pardir, "lmms_common", "evaluate.py"),    # repo: src/eval/tasks/<b>/ -> src/eval/lmms_common/
    os.environ.get("LMMS_COMMON", ""),
    "/opt/eval/lmms_common/evaluate.py",                                        # baked in omni-eval image (if present)
]
_common = next((p for p in _cands if p and os.path.isfile(p)), None)
if not _common:
    sys.stderr.write("[video_mmmu-adapter] ERROR: lmms_common/evaluate.py not found; searched: %s\n"
                     % [p for p in _cands if p])
    sys.exit(2)

# Delegate with the current argv (runpy preserves sys.argv); lmms_common's main()
# runs under __name__ == "__main__" and sys.exit()s with its own return code.
runpy.run_path(_common, run_name="__main__")
