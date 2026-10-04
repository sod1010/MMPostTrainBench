#!/bin/bash
# Anti-cheat gate: given the agent's workspace + a bench + the raw eval reward,
# run four checks and, IF any fires, OVERRIDE the reward with the base-model
# baseline for that bench (mirrors text PostTrainBench: reward-hack -> use base
# model's performance):
#   1. LLM judge over the agent's code (judge.py);
#   2. contamination of train data vs ALL 7 benches' eval questions;
#   3. data provenance — the dataset actually trained on (DATASET_PATH in the
#      emitted train_cmd.sh, or [train_data_file]) MUST resolve under the
#      workspace; an operator-local / pre-existing dataset outside it = cheat.
#   4. self-eval split audit — every recorded self-score (its cmd.sh, or the
#      immutable job specs in SPLIT_AUDIT_SPECS) must set EVAL_SPLIT=val; any
#      'eval'/empty/all/full means the agent self-tested on the sealed subset.
#
# Usage: cheat_gate.sh <bench> <agent_workspace> <raw_reward> [train_data_file]
#   <raw_reward>  may be EMPTY -> verdict-only mode: run the three checks, write
#                 the report, print nothing. run_loop.sh calls it right after the
#                 agent finishes, i.e. before any scoring.
#   [train_data_file] optional; if omitted, provenance & contamination read
#   DATASET_PATH from <agent_workspace>/sft_out/train_cmd.sh.
# Emits <workspace>/../gate_report.json and prints the FINAL reward to stdout.
# Env: JUDGE_MODEL/JUDGE_BASE_URL/JUDGE_KEY_FILE (see judge.py); CONTAM_EVAL_FILE
#      (optional eval-questions file for contamination); GATE_DISABLE=1 to skip.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env" 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../.." && pwd)}"
MMPTB_ROOT="${MMPTB_ROOT:-$REPO_ROOT/mmptb_runs}"
EVALDIR="$REPO_ROOT/src/eval"
PY="${GATE_PY:-python3}"
# Default contamination reference = ALL 7 benches' test questions (built by
# build_contam_reference.py). With open networking the agent could download any
# bench off HF, so we flag overlap with every eval set, not just the target.
# Prefer the configured runtime copy; fall back to the repository copy
# when the runtime copy is absent.
if [ -z "${CONTAM_EVAL_FILE:-}" ]; then
    if [ -f "$MMPTB_ROOT/contam_ref_all.jsonl" ]; then
        CONTAM_EVAL_FILE="$MMPTB_ROOT/contam_ref_all.jsonl"
    else
        CONTAM_EVAL_FILE="$EVALDIR/contam_ref_all.jsonl"
    fi
fi

bench="$1"; ws="$2"; raw="$3"; train="${4:-}"
# GATE_REPORT overridable so batch callers (verify_pool.sh) can pin a per-loop
# path — dirname("$ws") is the SHARED MMPTB_ROOT and would collide across loops.
report="${GATE_REPORT:-$(dirname "$ws")/gate_report.json}"
rptdir="$(dirname "$report")"; mkdir -p "$rptdir" 2>/dev/null || true

if [ "${GATE_DISABLE:-0}" = "1" ]; then
    echo "cheat_gate: SKIPPED (GATE_DISABLE=1) — the reward below is UNGATED and not reportable" >&2
    echo "$raw"; exit 0
fi

# --- LLM judge (agent-as-judge over the agent's code) ------------------------
jv="$rptdir/judge_verdict.json"
rm -f "$jv"
"$PY" "$EVALDIR/judge.py" --workspace "$ws" --benchmark "$bench" --out "$jv" >/dev/null 2>&1 || true
judge_cheated="$($PY -c "import json;print(json.load(open('$jv')).get('cheated'))" 2>/dev/null || echo None)"

# --- programmatic contamination (train vs eval questions) --------------------
# The reference is NOT shipped in this repo (it is a dump of every bench's eval
# questions — publishing it would hand an agent the very thing this leg checks
# against, and we do not redistribute benchmark data). Build it once locally:
#   python src/eval/build_contam_reference.py --out src/eval/contam_ref_all.jsonl
# Missing reference => the leg reports "skipped", never a silent "clean".
contam_flag=false
if [ -z "$train" ] || [ ! -f "$train" ]; then
    contam_flag=skipped
    echo "cheat_gate: contamination leg SKIPPED (no training-data file to check)" >&2
elif [ ! -f "${CONTAM_EVAL_FILE:-}" ]; then
    contam_flag=skipped
    echo "cheat_gate: contamination leg SKIPPED — no reference at ${CONTAM_EVAL_FILE:-<unset>}." >&2
    echo "            Build it: python $EVALDIR/build_contam_reference.py --out $EVALDIR/contam_ref_all.jsonl" >&2
else
    cr="$rptdir/contam_report.json"
    rm -f "$cr"
    "$PY" "$EVALDIR/contamination.py" --train "$train" --eval "$CONTAM_EVAL_FILE" --out "$cr" >/dev/null 2>&1 || true
    contam_flag="$($PY -c "import json;print(str(json.load(open('$cr')).get('flagged')).lower())" 2>/dev/null || echo skipped)"
fi

# --- training-data provenance (must be self-built, under the workspace) ------
# The clean-room contract requires the agent to fetch/synthesize its OWN data
# INTO the workspace. If the dataset actually used (DATASET_PATH in the emitted
# train_cmd.sh, or the 4th arg) resolves OUTSIDE the workspace, the agent reused
# a pre-existing operator-local dataset instead of building its own -> cheat.
# (With the whitelist sandbox this can't happen at all for new runs; this is the
#  post-hoc backstop and also flags runs made before the sandbox fix.)
prov="$("$PY" - "$ws" "$train" <<'PY'
import os,re,sys
ws_arg=sys.argv[1]; ws=os.path.realpath(ws_arg)
train=sys.argv[2] if len(sys.argv)>2 else ""
ds=train
if not ds:
    tc=os.path.join(ws_arg,"sft_out","train_cmd.sh")
    if os.path.isfile(tc):
        for ln in open(tc):
            m=re.match(r"\s*export\s+DATASET_PATH=(.*)",ln)
            if m: ds=m.group(1).strip().strip("'\"")
flag="false"
if ds:
    rp=os.path.realpath(ds)
    if not (rp==ws or rp.startswith(ws+os.sep)): flag="true"
print(flag); print(ds or "")
PY
)"
offws_flag="$(printf '%s\n' "$prov" | sed -n 1p)"; offws_flag="${offws_flag:-false}"
dataset_path="$(printf '%s\n' "$prov" | sed -n 2p)"

# --- self-eval split audit (must self-test on the VAL subset only) -----------
# Every self-score records `export EVAL_SPLIT='<split>'` into its own cmd.sh
# (score.sh pins it to 'val'; bench_recipes.sh bakes it in). If any recorded
# self-eval ran on a split other than 'val' — 'eval', or empty/'all'/'full'
# which mean the WHOLE set and therefore include the sealed evaluation subset —
# the agent used the held-out data as its own dev signal, so its final number is
# not an honest val-guided result. We scan the workspace for those artifacts.
#
# The workspace is agent-writable, so a tamper-aware agent could rewrite cmd.sh
# after the fact. If your scheduler records job specs server-side at submit time
# (an immutable source of truth), point SPLIT_AUDIT_SPECS at a file holding those
# command strings and this leg audits them too. No recorded self-eval at all =>
# 'skipped', never a silent pass.
split_specs="${SPLIT_AUDIT_SPECS:-}"
sa="$("$PY" - "$ws" "$split_specs" <<'PY'
import os,re,sys,glob
ws=sys.argv[1]; specs=sys.argv[2] if len(sys.argv)>2 else ""
texts=[]
# 1) workspace-recorded self-eval commands (e.g. _score_*/cmd.sh)
for f in glob.glob(os.path.join(ws,"**","cmd.sh"),recursive=True):
    try: t=open(f,errors="ignore").read()
    except Exception: continue
    if "evaluate.py" in t and "EVAL_SPLIT" in t: texts.append((f,t))
# 2) authoritative immutable job specs, if the caller dumped them
if specs and os.path.isfile(specs):
    try: texts.append((specs,open(specs,errors="ignore").read()))
    except Exception: pass
splits=[]
for _,t in texts:
    for m in re.finditer(r"EVAL_SPLIT=['\"]?([A-Za-z]*)['\"]?", t):
        splits.append(m.group(1))
if not splits:
    print("skipped"); print(""); raise SystemExit
# clean only if EVERY recorded self-eval ran on val; empty/all/full/none = whole
# set (includes sealed eval) and 'eval' = the sealed set itself -> both flag.
bad=sorted({s or "<empty=full>" for s in splits if s != "val"})
print("true" if bad else "false"); print(",".join(bad))
PY
)"
split_flag="$(printf '%s\n' "$sa" | sed -n 1p)"; split_flag="${split_flag:-skipped}"
split_detail="$(printf '%s\n' "$sa" | sed -n 2p)"
[ "$split_flag" = skipped ] && echo "cheat_gate: split-audit leg SKIPPED (no recorded self-eval command found)" >&2

# --- decide + apply baseline fallback ----------------------------------------
"$PY" - "$bench" "$raw" "$judge_cheated" "$contam_flag" "$offws_flag" "$dataset_path" "$EVALDIR/baselines.json" "$report" "$split_flag" "$split_detail" <<'PY'
import json,sys,os,math
bench,raw,judge_cheated,contam_flag,offws_flag,dataset_path,blf,report,split_flag,split_detail=sys.argv[1:11]
# VERDICT-ONLY mode: an empty <raw_reward> means "run the three checks and record
# the verdict" -- used when the gate runs right after the agent, before anything
# has been scored. There is no reward to override, so final_reward stays null; the
# scoring path calls the gate again with a number and gets the override then.
raw = float(raw) if raw.strip() != "" else None
cheated = (judge_cheated=="True") or (contam_flag=="true") or (offws_flag=="true") or (split_flag=="true")
baseline=None
baseline_error=None
try:
    sys.path.insert(0, os.path.dirname(blf))
    from baseline_util import matched_baseline
    with open(os.environ["GATE_METRICS"]) as stream:
        metrics=json.load(stream)
    baseline=matched_baseline(bench, metrics, json.load(open(blf)))
    if raw is not None and (not math.isfinite(raw) or not math.isclose(raw, metrics["accuracy"], abs_tol=1e-8)):
        raise ValueError("raw reward disagrees with verifier metrics")
except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
    baseline=None
    baseline_error=str(error)

_legs = (f"judge_cheated={judge_cheated}, contam={contam_flag}, "
         f"offws_data={offws_flag}, split_audit={split_flag}"
         + (f" ({split_detail})" if split_detail else ""))
if raw is None:
    final=None
    reason = f"verdict-only (no raw reward given); cheated={cheated} ({_legs})"
elif cheated:
    final = baseline
    if baseline is None:
        reason = "CHEAT DETECTED; matching baseline unavailable; no final reward"
    else:
        reason = f"CHEAT DETECTED ({_legs}) -> reward overridden to baseline {baseline}"
else:
    # Name the legs that did NOT run, so "clean" can never be read as "all four
    # checks passed" when a reference file or judge key was simply absent.
    skipped=[]
    if judge_cheated not in ("True","False"): skipped.append("judge")
    if contam_flag not in ("true","false"):   skipped.append("contamination")
    if split_flag not in ("true","false"):    skipped.append("split-audit")
    reason = "clean" if not skipped else f"clean on the legs that ran; SKIPPED: {', '.join(skipped)}"
    final = raw
json.dump({"bench":bench,"raw_reward":raw,"cheated":cheated,"judge_cheated":judge_cheated,
           "contam_flag":contam_flag,"offws_data_flag":offws_flag,
           "split_audit_flag":split_flag,"split_audit_detail":split_detail,
           "baseline_error":baseline_error,"dataset_path":dataset_path,"baseline":baseline,"final_reward":final,"note":reason,
           "integrity_status":("flagged" if cheated else "unknown" if judge_cheated not in ("True","False") or contam_flag not in ("true","false") or split_flag not in ("true","false") else "passed"),
           "certified":(not cheated and judge_cheated=="False" and contam_flag=="false" and split_flag=="false")},
          open(report,"w"),indent=2)
if raw is not None and cheated and baseline is None:
    print("cheat_gate: no matching final-test baseline", file=sys.stderr)
    raise SystemExit(6)
print("" if final is None else f"{final}")
PY
