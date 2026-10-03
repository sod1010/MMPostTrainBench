#!/usr/bin/env python3
"""Offline core-metric computation for the mmptb paper / GitHub release.

Reads, per model, the 7 per-bench agent-run workspaces (each already has a
verifier_score.json produced by assemble_verifier.py + ledgers + the researcher
agent's SDK transcript) and emits the paper's core metric list:

  1. MM-PTGain  -- BENCH-EQUAL, headroom-normalized final-capability gain
     (modality group means are reported as a breakdown, not as weights).
  2. Per-bench native scores + headroom gain g_b (audio / image / audio-video).
  3. Trajectory metrics from the visible per-checkpoint _score_<ts> series:
       Selection Regret, Post-Peak Regression, Frontier Refresh, AnytimeVal AUC.
  4. Integrity Violation Rate  (from each bench's gate_report.json, if present).
  5. Research Budget & Efficiency:
       T_wall (WALL-CLOCK HOURS, primary "how long it ran"),
       GPU-h (for compute-$ only), tokens in/out/cache, Cost$ (SDK + recomputed),
       ComputeEff, CostEff, Time-to-Best.

Feedback Gain (needs control arms) and Capability Retention (needs a frozen
anchor set) are NOT computable from a single run's artifacts and are reported
as N/A.

This is a pure read-only offline analysis. It never touches a running loop.

Usage:
  compute_core_metrics.py --model opus5=/path/agent_run_{bench}_opus5 \
                          --model opus48=/path/agent_run_{bench}_opusclean \
                          [--pricing src/eval/pricing.json] [--json out.json]
The {bench} token in a --model path template is expanded over the 7 benches.
"""
import argparse, glob, json, os, re, sys, math, random, statistics
from fractions import Fraction

# --------------------------------------------------------------------------
# Sampling uncertainty. Every sealed score is a proportion measured on a FINITE
# held-out set (n items), so it carries a binomial sampling error. Reporting it
# is what makes the leaderboard readable: a rank gap smaller than the error bar
# is not a result. Two flavours are needed and they are NOT interchangeable:
#
#   se_vs_base -- for "did this arm improve over the base model?". The delta is a
#                 difference of two proportions, so BOTH errors count.
#   se_arm     -- for "is arm A above arm B?". Every arm is compared against the
#                 SAME base-model measurement on the SAME sealed set, so the
#                 baseline error is common-mode and cancels in A-B. Using
#                 se_vs_base there would double-count it and overstate the bar.
# --------------------------------------------------------------------------
Z95 = 1.959963985394736

# Winner's-curse null for Selection Regret (see regret_null_mc). Fixed seed so
# the reported p-values are reproducible; the null is a property of (k, n, p),
# not of any random state the caller happens to be in.
REGRET_MC_TRIALS = 5000
REGRET_MC_SEED = 1234


def _se_prop(p, n):
    """Standard error of a single proportion p measured on n items."""
    if not isinstance(p, (int, float)) or not isinstance(n, int) or n <= 0:
        return None
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def derive_eval_n(bench, baselines):
    """Prefer the explicit final-test size; derive only for legacy metadata."""
    n_eval = ((baselines or {}).get("n_eval") or {}).get(bench)
    if type(n_eval) is int and n_eval > 0:
        return n_eval
    n_full = ((baselines or {}).get("n_full") or {}).get(bench)
    if not isinstance(n_full, int) or n_full <= 0:
        return None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import split_util
        return split_util.summarize(n_full)["eval"]
    except Exception:
        return None


def _wilson_ci(p, n, z=Z95):
    """95% Wilson score interval -- stays inside [0,1] near the tails, where the
    plain normal approximation does not."""
    if not isinstance(p, (int, float)) or not isinstance(n, int) or n <= 0:
        return None
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(max(p * (1.0 - p), 0.0) / n + z * z / (4.0 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]

# The 7 perception benches define the PRIMARY MM-PTGain (equal weight EACH).
# Keeping this set fixed preserves comparability with every number already
# reported for the perception arms.
PERCEPTION_BENCHES = ["mmau", "mmar", "jointavbench", "mmmu_pro",
                      "video_mmmu", "videomme_v2", "omnivideobench"]
# Code/SWE dimension (SWE-bench Multimodal). Scored and reported, but kept OUT
# of the primary MM-PTGain: mixing a code bench into a perception average would
# break comparability with every perception number already reported. It feeds
# MM_PTGain_with_code instead, so both readings are on the table.
CODE_BENCHES = ["mmswe"]
BENCHES = PERCEPTION_BENCHES + CODE_BENCHES

# Modality grouping. Used for the per-modality BREAKDOWN and the appendix
# modality-balanced variant -- NOT for the headline weights (those are 1/n_bench).
MODALITY = {
    "mmau": "Audio", "mmar": "Audio",
    "mmmu_pro": "Image",
    "jointavbench": "AudioVideo", "video_mmmu": "AudioVideo",
    "videomme_v2": "AudioVideo", "omnivideobench": "AudioVideo",
    "mmswe": "Code",
}
# Canonical model id used to look up token pricing. A name that is absent here
# (or absent from pricing.json) yields API$ = N/A rather than a silent 0.
MODEL_PRICING_ID = {"opus5": "claude-opus-5", "opus48": "claude-opus-4-8",
                    "gpt56": "gpt-5.6-sol", "gpt56terra": "gpt-5.6-terra"}


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# --------------------------------------------------------------------------
# per-checkpoint visible score trajectory from _score_<ts>/reward.txt
# (these are the agent's OWN visible eval calls during the run, timestamped by
#  the launch epoch embedded in the dir name -- NOT the sealed verifier, which
#  seals only the selected ckpt. Labelled "visible" throughout.)
# --------------------------------------------------------------------------
def _derive_visible_n(score):
    """Visible evals often do NOT log `n` (only the video benches' metrics.json
    omits it, but that is exactly where n is smallest). The logged accuracy is
    correct/n, so the reduced fraction's denominator is a lower bound on n and
    in practice recovers it: `--limit 64` filtered through the 30% val split
    leaves ~20 items, and 0.7727 -> 17/22 pins n=22. Denominators < 8 are
    discarded as degenerate (a score of exactly 0.5 reduces to 1/2)."""
    try:
        den = Fraction(float(score)).limit_denominator(4096).denominator
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return den if den >= 8 else None


def load_trajectory(ws):
    """-> [(ts, score, n_visible_or_None)] sorted by ts. n comes from the
    visible metrics.json when logged, else from the score's denominator."""
    pts = []
    for d in glob.glob(os.path.join(ws, "_score_*")):
        m = re.search(r"_score_(\d+)$", d)
        if not m:
            continue
        ts = int(m.group(1))
        r = _read(os.path.join(d, "reward.txt"))
        try:
            r = float(r)
        except (TypeError, ValueError):
            continue
        mj = _read_json(os.path.join(d, "metrics.json")) or {}
        n = mj.get("n")
        n_src = "recorded"
        if not (isinstance(n, int) and n > 0):
            n, n_src = _derive_visible_n(r), "denominator"
        pts.append((ts, r, n, n_src))
    pts.sort()
    return pts


def regret_null_mc(scores, n_visible, trials=REGRET_MC_TRIALS, seed=REGRET_MC_SEED):
    """Winner's-curse null for Selection Regret.

    The peak of a visible series is a MAXIMUM over k noisy draws, so E[max-last]
    is strictly positive even for a model whose true ability NEVER changes --
    and it grows with k and shrinks with n. Reporting raw regret therefore
    reports mostly sampling noise. This simulates the null (ability fixed at the
    series mean, k independent draws of n items each) and returns the null mean
    plus the one-sided p-value P(regret_null >= regret_observed).

    Returns None when n is unknown -- never silently substitute a guess."""
    k = len(scores)
    if k < 2 or not (isinstance(n_visible, int) and n_visible > 0):
        return None
    p = sum(scores) / k
    p = min(max(p, 0.0), 1.0)
    observed = max(scores) - scores[-1]
    rnd = random.Random(seed)
    binom = getattr(rnd, "binomialvariate", None)
    tot = 0.0
    hits = 0
    for _ in range(trials):
        if binom is not None:
            draws = [binom(n_visible, p) / n_visible for _ in range(k)]
        else:  # py<3.12
            draws = [sum(1 for _ in range(n_visible) if rnd.random() < p) / n_visible
                     for _ in range(k)]
        r0 = max(draws) - draws[-1]
        tot += r0
        if r0 >= observed - 1e-12:
            hits += 1
    null_mean = tot / trials
    return {
        "regret_observed": round(observed, 4),
        "regret_null_mean": round(null_mean, 4),
        "regret_excess": round(observed - null_mean, 4),
        "regret_p_value": round(hits / trials, 4),
        "regret_beats_null": bool(hits / trials < 0.05),
        "mc_n_visible": n_visible,
        "mc_k": k,
        "mc_trials": trials,
    }


def trajectory_metrics(pts):
    """Selection Regret / Post-Peak / Frontier Refresh / AnytimeVal, from the
    visible series. Regret & post-peak are measured against the LAST point in
    the series (the agent's final visible state) as the selection proxy.

    Raw regret is NOT reportable on its own -- see regret_null_mc()."""
    if not pts:
        return {"n_points": 0}
    scores = [s for _, s, _n, _src in pts]
    ts = [t for t, _s, _n, _src in pts]
    ns = [n for _t, _s, n, _src in pts if isinstance(n, int) and n > 0]
    n_med = int(statistics.median(ns)) if ns else None
    n_recorded = sum(1 for _t, _s, n, src in pts
                     if src == "recorded" and isinstance(n, int) and n > 0)
    peak = max(scores)
    peak_i = scores.index(peak)
    final = scores[-1]
    first_ts = ts[0]
    # best-so-far series -> AnytimeVal (mean best-so-far) + frontier refresh count
    #
    # TWO refresh counts, and only the second is the DESIGNED metric. The RAW
    # count asks "did best-so-far go up at all", which on a noisy self-eval
    # series counts luck: at n=22 visible items a single item flipping moves the
    # score .045, so a pure-noise series still "refreshes" O(log k) times. The
    # SIGNIFICANT count applies the noise threshold the metric was specified
    # with -- a refresh only counts if the new best beats the incumbent best by
    # more than the 95% bar on the DIFFERENCE of two independent proportions
    # measured on n_new / n_best items:
    #     delta = z * sqrt(p_new(1-p_new)/n_new + p_best(1-p_best)/n_best)
    # Points whose n could not be recovered cannot be tested; they are counted
    # in frontier_refresh_n_untested, never silently treated as passes.
    best = -1e9
    best_n = None
    bsf = []
    refresh = 0
    refresh_sig = 0
    n_untested = 0
    deltas = []
    for _t, s, n, _src in pts:
        if s > best + 1e-9:
            if best > -1e8:  # the first point establishes the frontier, not a refresh
                if isinstance(n, int) and n > 0 and isinstance(best_n, int) and best_n > 0:
                    d = Z95 * math.sqrt(max(s * (1 - s), 0.0) / n
                                        + max(best * (1 - best), 0.0) / best_n)
                    deltas.append(d)
                    if (s - best) > d:
                        refresh_sig += 1
                else:
                    n_untested += 1
            refresh += 1
            best, best_n = s, n
        bsf.append(best)
    anytime = sum(bsf) / len(bsf)
    n_steps = max(len(scores) - 1, 0)   # steps that COULD have produced a new best
    out = {
        "n_points": len(pts),
        "visible_n_median": n_med,
        "visible_n_recorded_points": n_recorded,
        "visible_n_all_recorded": bool(ns and n_recorded == len(ns)),
        "first_visible": round(scores[0], 4),
        "peak_visible": round(peak, 4),
        "final_visible": round(final, 4),
        "selection_regret": round(peak - final, 4),      # max_t s_t - s_selected
        "post_peak_regressed": bool(final < peak - 1e-6),  # RSIBench-style
        "post_peak_drop": round(peak - final, 4),
        # RAW count: any best-so-far improvement (reference only -- counts noise).
        "frontier_refresh_count": refresh,
        # DESIGNED metric: refreshes that exceed the per-pair 95% noise bar,
        # plus the rate over the steps that could have produced one.
        "frontier_refresh_sig_count": refresh_sig,
        "frontier_refresh_rate_sig": (round(refresh_sig / n_steps, 4) if n_steps else None),
        "frontier_refresh_rate_raw": (round(refresh / n_steps, 4) if n_steps else None),
        "frontier_refresh_n_untested": n_untested,
        "frontier_refresh_delta_median": (round(statistics.median(deltas), 4) if deltas else None),
        "anytime_val": round(anytime, 4),                  # AUC / iterations
        "time_to_best_h": round((ts[peak_i] - first_ts) / 3600.0, 3),
        "active_span_h": round((ts[-1] - first_ts) / 3600.0, 3),
        "first_ts": first_ts, "last_ts": ts[-1], "peak_ts": ts[peak_i],
    }
    out["regret_null"] = regret_null_mc(scores, n_med)
    return out


# --------------------------------------------------------------------------
# researcher token/cost totals from the Claude-Code SDK transcript
# (sum over all `result` records across every agent.resume*.jsonl)
# --------------------------------------------------------------------------
def token_cost_totals(ws):
    tot = {"sessions": 0, "input": 0, "output": 0, "cache_read": 0,
           "cache_write": 0, "thinking": 0, "sdk_cost_usd": 0.0}
    for jf in glob.glob(os.path.join(ws, "agent.resume*.jsonl")) + \
              glob.glob(os.path.join(ws, "agent.resume.jsonl")):
        seen = set()
        if jf in seen:
            continue
        for ln in open(jf, errors="ignore"):
            ln = ln.strip()
            if not ln or '"result"' not in ln and '"usage"' not in ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if not (isinstance(o, dict) and o.get("type") == "result"):
                continue
            u = o.get("usage") or {}
            tot["sessions"] += 1
            tot["input"] += u.get("input_tokens", 0)
            tot["output"] += u.get("output_tokens", 0)
            tot["cache_read"] += u.get("cache_read_input_tokens", 0)
            tot["cache_write"] += u.get("cache_creation_input_tokens", 0)
            det = (u.get("output_tokens_details") or {})
            tot["thinking"] += det.get("thinking_tokens", 0)
            c = o.get("total_cost_usd")
            if isinstance(c, (int, float)):
                tot["sdk_cost_usd"] += c
    return tot


def dedup_token_cost_totals(ws):
    """Sum SDK `result` records over ALL transcript files (agent.jsonl initial
    session + agent.resume*.jsonl resumes). `usage_available` is False when the
    transcript predates usage logging (older runs captured no token counts) --
    in that case token/API cost is genuinely unrecoverable and reported N/A,
    NOT zero."""
    files = sorted(set(glob.glob(os.path.join(ws, "agent*.jsonl"))))
    tot = {"sessions": 0, "input": 0, "output": 0, "cache_read": 0,
           "cache_write": 0, "thinking": 0, "sdk_cost_usd": 0.0,
           "usage_available": False}
    for jf in files:
        for ln in open(jf, errors="ignore"):
            ln = ln.strip()
            if not ln or '"result"' not in ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if not (isinstance(o, dict) and o.get("type") == "result"):
                continue
            u = o.get("usage") or {}
            tot["sessions"] += 1
            tot["input"] += u.get("input_tokens", 0)
            tot["output"] += u.get("output_tokens", 0)
            tot["cache_read"] += u.get("cache_read_input_tokens", 0)
            tot["cache_write"] += u.get("cache_creation_input_tokens", 0)
            det = (u.get("output_tokens_details") or {})
            tot["thinking"] += det.get("thinking_tokens", 0)
            c = o.get("total_cost_usd")
            if isinstance(c, (int, float)):
                tot["sdk_cost_usd"] += c
    tot["usage_available"] = tot["sessions"] > 0
    return tot


def recompute_token_cost(tot, rate):
    if not rate or not tot.get("usage_available"):
        return None
    return (tot["input"] / 1e6 * rate.get("input_usd_per_mtok", 0)
            + tot["output"] / 1e6 * rate.get("output_usd_per_mtok", 0)
            + tot["cache_read"] / 1e6 * rate.get("cache_read_usd_per_mtok", 0)
            + tot["cache_write"] / 1e6 * rate.get("cache_write_usd_per_mtok", 0))


# --------------------------------------------------------------------------
# integrity gate
# --------------------------------------------------------------------------
def integrity(ws):
    g = _read_json(os.path.join(ws, "gate_report.json"))
    if not g or not isinstance(g, dict):
        return {"present": False}
    # cheat_gate schema: a top-level `cheated` (OR of the legs) plus the four
    # per-leg flags. A leg reporting the string "skipped" (or None) did NOT run
    # -- it is neither a pass nor a violation, so we track it separately and do
    # NOT count it as clean. The violation is driven by the `cheated` verdict.
    LEG_KEYS = ["judge_cheated", "contam_flag", "offws_data_flag", "split_audit_flag"]

    def leg_state(v):
        # gate_report stores legs as STRINGS ("True"/"false"/"skipped") or bools.
        # A leg that produced no verdict is serialized as the literal string
        # "None" (json.dump of a stringified None), NOT as JSON null -- so it
        # must be matched textually. Treating it as "clean" (the old behaviour)
        # silently credited two opus48 cells with a judge verdict they never got.
        if v is None:
            return "skipped"
        s = str(v).strip().lower()
        if s in ("true", "1"):
            return "violated"
        if s in ("skipped", "none", "", "n/a", "na", "null"):
            return "skipped"
        return "clean"  # "false" / False / anything else that ran clean

    legs = {k: leg_state(g.get(k)) for k in LEG_KEYS}
    viol = bool(g.get("cheated")) or any(s == "violated" for s in legs.values())
    n_skipped = sum(1 for s in legs.values() if s == "skipped")
    return {
        "present": True,
        "violation": viol,
        "cheated": bool(g.get("cheated")),
        "legs": legs,
        "n_legs_skipped": n_skipped,
        "note": g.get("note"),
        "raw_reward": g.get("raw_reward"),
        "final_reward": g.get("final_reward"),  # gate-overridden to baseline if cheated
    }


# --------------------------------------------------------------------------
def compute_model(name, path_tmpl, pricing, baselines=None):
    rate = (pricing.get("models", {}) or {}).get(MODEL_PRICING_ID.get(name, name))
    r_gpu = pricing.get("gpu_usd_per_gpu_hour", 2.0)
    if baselines is None:
        baselines = _read_json(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "baselines.json")) or {}

    per_bench = {}
    for b in BENCHES:
        ws = path_tmpl.replace("{bench}", b)
        vs = _read_json(os.path.join(ws, "verifier_score.json")) or {}
        ev = vs.get("eval", {}) or {}
        ood = vs.get("ood", {}) or {}
        s_raw = ev.get("reward")           # verifier stores the RAW eval reward
        base = ev.get("baseline")
        integ = integrity(ws)
        # Cheat-adjusted score: if the integrity gate flagged this bench, the gate
        # ALREADY overrode the reward back to baseline (delta=0). Honour that here
        # so a cheated "gain" never counts toward MM-PTGain. Fall back to raw when
        # the gate is absent / clean.
        s = s_raw
        if integ.get("cheated") and isinstance(integ.get("final_reward"), (int, float)):
            s = integ["final_reward"]
        cheat_adjusted = (s != s_raw)

        def _g(x):
            if isinstance(x, (int, float)) and isinstance(base, (int, float)) and base < 1.0:
                return (x - base) / (1.0 - base)
            return None
        g_b = _g(s)          # cheat-adjusted headroom gain (primary)
        g_b_raw = _g(s_raw)  # raw headroom gain (reference)

        # ---- sampling uncertainty on this cell -------------------------------
        # n = size of the sealed eval split this cell was scored on. Prefer the
        # recorded value; fall back to the deterministic split derivation so the
        # three shim benches still get an error bar instead of a blank.
        n_eval = ev.get("n") if isinstance(ev.get("n"), int) else None
        n_src = "recorded" if n_eval else None
        if n_eval is None and isinstance(base, (int, float)):
            n_eval = derive_eval_n(b, baselines)
            n_src = "split_derived" if n_eval else None
        se_s = _se_prop(s_raw, n_eval)          # arm's own error
        se_base = _se_prop(base, n_eval)        # base model's error (same split)
        hr = (1.0 - base) if isinstance(base, (int, float)) and base < 1.0 else None
        if cheat_adjusted:
            # The reset-to-baseline is an accounting DECISION, not a measurement:
            # the adjusted delta is exactly 0 by construction, with no sampling
            # error of its own. Zeroing the error here keeps the CI honest about
            # what it describes, but it also means a cheat-heavy arm looks
            # artificially precise -- fmt_report flags that explicitly.
            se_d = se_arm_only = 0.0
        else:
            se_d = (math.sqrt(se_s ** 2 + se_base ** 2)
                    if (se_s is not None and se_base is not None) else None)
            se_arm_only = se_s
        se_g_vs_base = (se_d / hr) if (se_d is not None and hr) else None
        se_g_arm = (se_arm_only / hr) if (se_arm_only is not None and hr) else None
        delta_v = (s - base) if isinstance(s, (int, float)) and isinstance(base, (int, float)) else None

        # ---- held-out OOD probe (the retention PROXY) --------------------------
        # Every loop also scores one bench it was NOT optimizing (PROBE-A = mmar,
        # PROBE-B = mmau when the target IS mmar), on the same sealed split. That
        # is the only retention-like signal on disk, so give it a real error bar:
        # unpaired difference of two proportions on the probe's own n. Unpaired is
        # conservative -- a true paired test can only tighten it -- so a
        # significant regression here is not an artefact of the SE choice.
        ood_n = ood.get("n") if isinstance(ood.get("n"), int) else None
        ood_p, ood_pb, ood_d = ood.get("reward"), ood.get("baseline"), ood.get("delta")
        se_op, se_ob = _se_prop(ood_p, ood_n), _se_prop(ood_pb, ood_n)
        se_ood = (math.sqrt(se_op ** 2 + se_ob ** 2)
                  if (se_op is not None and se_ob is not None) else None)

        gpu_sec = _read(os.path.join(ws, ".compute_ledger"))
        gpu_h = (int(gpu_sec) / 3600.0) if (gpu_sec and gpu_sec.isdigit()) else None
        traj = trajectory_metrics(load_trajectory(ws))
        tok = dedup_token_cost_totals(ws)
        per_bench[b] = {
            "modality": MODALITY[b],
            "score": s, "score_raw": s_raw, "cheat_adjusted": cheat_adjusted,
            "baseline": base,
            "delta": round(delta_v, 4) if delta_v is not None else None,
            "g_headroom": round(g_b, 4) if g_b is not None else None,
            "g_headroom_raw": round(g_b_raw, 4) if g_b_raw is not None else None,
            # --- sampling uncertainty (95%) ---
            "n_eval": n_eval,
            "n_eval_source": n_src,
            "score_ci95": _wilson_ci(s_raw, n_eval),
            "se_score": round(se_s, 5) if se_s is not None else None,
            "se_delta": round(se_d, 5) if se_d is not None else None,
            "delta_ci95": ([round(delta_v - Z95 * se_d, 4), round(delta_v + Z95 * se_d, 4)]
                           if (delta_v is not None and se_d is not None) else None),
            "delta_significant": (bool(abs(delta_v) > Z95 * se_d)
                                  if (delta_v is not None and se_d) else None),
            "se_g_vs_base": round(se_g_vs_base, 5) if se_g_vs_base is not None else None,
            "se_g_arm": round(se_g_arm, 5) if se_g_arm is not None else None,
            "sealed_gap": vs.get("gap"),
            "ood_delta": ood.get("delta"),
            "ood_codename": ood.get("codename"),
            "ood_n": ood_n,
            "ood_se_delta": round(se_ood, 5) if se_ood is not None else None,
            "ood_significant": (bool(abs(ood_d) > Z95 * se_ood)
                                if (isinstance(ood_d, (int, float)) and se_ood) else None),
            "benchmax_flag": vs.get("benchmax_flag"),
            "gpu_h": round(gpu_h, 2) if gpu_h is not None else None,
            "trajectory": traj,
            "tokens": tok,
            "integrity": integ,
            "ws": ws,
        }

    # --- MM-PTGain: BENCH-EQUAL mean of g_b over the 7 perception benches -----
    # w_b = 1/n_benches. Bench-equal, NOT modality-equal: with the modality
    # grouping this suite actually has (Audio 2 / Image 1 / AudioVideo 4), an
    # equal-weight-per-modality average would hand the single Image bench 1/3 of
    # the headline while each AV bench got 1/12 -- one bench's sampling noise
    # would then drive the leaderboard. Modality group means are still reported,
    # as a BREAKDOWN of where the gain came from, plus a modality-balanced
    # variant for the appendix; neither is the headline.
    # Perception benches only. The Code group is reported separately and in
    # MM_PTGain_with_code, so adding mmswe never silently redefines the headline.
    per_g = {b: r["g_headroom"] for b, r in per_bench.items()
             if r["g_headroom"] is not None and b in PERCEPTION_BENCHES}
    mm_ptgain = (sum(per_g.values()) / len(per_g)) if per_g else None

    groups = {}
    for b, g in per_g.items():
        groups.setdefault(per_bench[b]["modality"], []).append(g)
    group_mean = {m: sum(v) / len(v) for m, v in groups.items()}
    mm_ptgain_modbal = (sum(group_mean.values()) / len(group_mean)) if group_mean else None

    # extended reading: bench-equal over perception + Code
    per_g_ext = {b: r["g_headroom"] for b, r in per_bench.items()
                 if r["g_headroom"] is not None}
    mm_ptgain_ext = ((sum(per_g_ext.values()) / len(per_g_ext))
                     if len(per_g_ext) > len(per_g) else None)
    groups_ext = {}
    for b, g in per_g_ext.items():
        groups_ext.setdefault(per_bench[b]["modality"], []).append(g)
    group_mean_ext = {m: sum(v) / len(v) for m, v in groups_ext.items()}
    # --- error bar on MM-PTGain ----------------------------------------------
    # MM-PTGain is a FIXED linear combination of the per-bench g_b, and the g_b
    # are measured on DISJOINT sealed sets, so their sampling errors are
    # independent and add in quadrature: se = sqrt(sum (w_b * se_b)^2).
    #   bench-equal (headline) : w_b = 1/n_benches
    #   modality-balanced (appx): w_b = 1/(n_groups * |benches in b's group|)
    def _ptgain_se(bench_set, mode="bench"):
        scored = [b for b, r in per_bench.items()
                  if r["g_headroom"] is not None and b in bench_set]
        if not scored:
            return None, None, 0
        cnt = {}
        for b in scored:
            m = per_bench[b]["modality"]
            cnt[m] = cnt.get(m, 0) + 1
        n_groups = len(cnt)
        v_base = v_arm = 0.0
        n_missing = 0
        for b in scored:
            r = per_bench[b]
            w = (1.0 / len(scored) if mode == "bench"
                 else 1.0 / (n_groups * cnt[r["modality"]]))
            if r["se_g_vs_base"] is None or r["se_g_arm"] is None:
                n_missing += 1          # cell has no n -> CI is a lower bound
                continue
            v_base += (w * r["se_g_vs_base"]) ** 2
            v_arm += (w * r["se_g_arm"]) ** 2
        return math.sqrt(v_base), math.sqrt(v_arm), n_missing

    se_pt_base, se_pt_arm, n_no_n = _ptgain_se(set(PERCEPTION_BENCHES), "bench")
    se_mb_base, se_mb_arm, _ = _ptgain_se(set(PERCEPTION_BENCHES), "modality")
    pt_ci = ([round(mm_ptgain - Z95 * se_pt_base, 4), round(mm_ptgain + Z95 * se_pt_base, 4)]
             if (mm_ptgain is not None and se_pt_base is not None) else None)

    # reference macro (simple, unweighted by modality)
    all_g = [r["g_headroom"] for r in per_bench.values() if r["g_headroom"] is not None]
    macro_g = sum(all_g) / len(all_g) if all_g else None
    all_d = [r["delta"] for r in per_bench.values() if r["delta"] is not None]
    macro_delta = sum(all_d) / len(all_d) if all_d else None

    # --- integrity violation rate --------------------------------------------
    integ_present = [r for r in per_bench.values() if r["integrity"].get("present")]
    integ_viol = [r for r in integ_present if r["integrity"].get("violation")]
    integ_rate = (len(integ_viol) / len(integ_present)) if integ_present else None

    # Per-LEG sub-category rates. The headline violation rate is an OR over four
    # independent gate legs, so on its own it cannot say WHICH failure mode the
    # suite actually catches, nor how much of the gate ran at all. Two rates per
    # leg, because they answer different questions:
    #   rate_of_ran = violations / cells where the leg RAN  -> how often that
    #                 failure mode fires when it is actually looked for;
    #   rate_of_all = violations / all gated cells          -> its contribution
    #                 to the headline rate.
    # coverage = ran / gated is what makes "the headline rate is a lower bound"
    # quantitative instead of a caveat: a leg with low coverage hides an unknown
    # number of violations, and the size of that blind spot is 1 - coverage.
    LEG_KEYS = ["judge_cheated", "contam_flag", "offws_data_flag", "split_audit_flag"]
    integ_legs = {}
    for k in LEG_KEYS:
        st = [r["integrity"].get("legs", {}).get(k) for r in integ_present]
        n_ran = sum(1 for s in st if s in ("violated", "clean"))
        n_v = sum(1 for s in st if s == "violated")
        n_sk = sum(1 for s in st if s != "violated" and s != "clean")
        integ_legs[k] = {
            "n_cells": len(st), "n_ran": n_ran, "n_skipped": n_sk, "n_violated": n_v,
            "rate_of_ran": (round(n_v / n_ran, 4) if n_ran else None),
            "rate_of_all": (round(n_v / len(st), 4) if st else None),
            "coverage": (round(n_ran / len(st), 4) if st else None),
            "violating_benches": [b for b, r in per_bench.items()
                                  if r["integrity"].get("legs", {}).get(k) == "violated"],
        }
    _covs = [v["coverage"] for v in integ_legs.values() if v["coverage"] is not None]
    integ_gate_coverage = (round(sum(_covs) / len(_covs), 4) if _covs else None)

    # --- Frontier Refresh Rate, arm level ------------------------------------
    # Pooled over benches: how often a visible evaluation step produced a new
    # best that the agent could actually TELL was a new best. sig/raw is the
    # headline pair -- a large raw with a near-zero sig means the agent kept
    # "improving" inside its own noise floor, which is the mechanism behind the
    # regret null and the reason none of these arms separates.
    _fr_sig = _fr_raw = _fr_steps = _fr_unt = 0
    _fr_d = []
    for r in per_bench.values():
        t = r["trajectory"]
        if not t.get("n_points"):
            continue
        _fr_sig += t.get("frontier_refresh_sig_count") or 0
        _fr_raw += t.get("frontier_refresh_count") or 0
        _fr_steps += max(t["n_points"] - 1, 0)
        _fr_unt += t.get("frontier_refresh_n_untested") or 0
        if t.get("frontier_refresh_delta_median") is not None:
            _fr_d.append(t["frontier_refresh_delta_median"])
    frontier_agg = ({
        "n_steps": _fr_steps,
        "refresh_sig": _fr_sig,
        "refresh_raw": _fr_raw,
        "rate_sig": round(_fr_sig / _fr_steps, 4) if _fr_steps else None,
        "rate_raw": round(_fr_raw / _fr_steps, 4) if _fr_steps else None,
        "n_untested": _fr_unt,
        "delta_median": round(statistics.median(_fr_d), 4) if _fr_d else None,
    } if _fr_steps else None)

    # --- budget & efficiency (aggregate over benches) ------------------------
    gpu_h_total = sum(r["gpu_h"] for r in per_bench.values() if r["gpu_h"]) or 0.0
    # WALL-CLOCK h: per bench = active_span_h from the visible-score trajectory;
    # total wall = sum across benches (they ran as separate sequential runs).
    wall_h_by_bench = {b: r["trajectory"].get("active_span_h", 0.0) or 0.0
                       for b, r in per_bench.items()}
    wall_h_total = sum(wall_h_by_bench.values())
    tok_tot = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
               "thinking": 0, "sessions": 0, "sdk_cost_usd": 0.0}
    for r in per_bench.values():
        for k in tok_tot:
            tok_tot[k] += r["tokens"].get(k, 0)
    usage_available = any(r["tokens"].get("usage_available") for r in per_bench.values())
    api_cost_recomp = recompute_token_cost({**tok_tot, "usage_available": usage_available}, rate)
    gpu_cost = gpu_h_total * r_gpu
    # total = GPU$ + API$; API$ N/A when the run predates usage logging.
    cost_total_recomp = (gpu_cost + api_cost_recomp) if api_cost_recomp is not None else None
    cost_total_sdk = (gpu_cost + tok_tot["sdk_cost_usd"]) if usage_available else None

    compute_eff = (mm_ptgain / gpu_h_total) if (mm_ptgain is not None and gpu_h_total) else None
    cost_eff = (mm_ptgain / cost_total_recomp) if (mm_ptgain is not None and cost_total_recomp) else None

    # --- arm-level held-out probe retention + sealed gap ----------------------
    # Retention PROXY, not general capability retention: the probe is a single
    # audio bench. The arm mean is an average of independent per-cell deltas
    # (disjoint targets, same probe split), so its error adds in quadrature.
    op = [(b, r) for b, r in per_bench.items()
          if b in PERCEPTION_BENCHES and isinstance(r.get("ood_delta"), (int, float))]
    if op:
        od = [r["ood_delta"] for _b, r in op]
        ses = [r["ood_se_delta"] for _b, r in op if r.get("ood_se_delta")]
        mean_od = sum(od) / len(od)
        se_mean = (math.sqrt(sum(s ** 2 for s in ses)) / len(od)) if len(ses) == len(od) else None
        ood_probe = {
            "mean_delta": round(mean_od, 4),
            "n_cells": len(od),
            "n_negative": sum(1 for x in od if x < 0),
            "se_mean": round(se_mean, 5) if se_mean is not None else None,
            "ci95": ([round(mean_od - Z95 * se_mean, 4), round(mean_od + Z95 * se_mean, 4)]
                     if se_mean else None),
            "z": round(mean_od / se_mean, 2) if se_mean else None,
            "significant": bool(abs(mean_od) > Z95 * se_mean) if se_mean else None,
            "significant_cells": sorted(b for b, r in op if r.get("ood_significant")),
            "probes": sorted({r.get("ood_codename") for _b, r in op if r.get("ood_codename")}),
            "note": ("held-out probe retention (one audio bench: PROBE-A=mmar, "
                     "PROBE-B=mmau when target==mmar). NOT general capability "
                     "retention, and never used by any gate -- diagnostic only."),
        }
    else:
        ood_probe = None

    gaps = [(b, r["sealed_gap"]) for b, r in per_bench.items()
            if b in PERCEPTION_BENCHES and isinstance(r.get("sealed_gap"), (int, float))]
    if gaps:
        gmax = max(gaps, key=lambda kv: kv[1])
        sealed_gap_agg = {"mean": round(sum(v for _b, v in gaps) / len(gaps), 4),
                          "max": round(gmax[1], 4), "max_bench": gmax[0], "n_cells": len(gaps)}
    else:
        sealed_gap_agg = None

    return {
        "model": name,
        "MM_PTGain": round(mm_ptgain, 4) if mm_ptgain is not None else None,
        # 95% error bar vs the base model (includes the baseline's own error).
        "MM_PTGain_ci95": pt_ci,
        "MM_PTGain_se_vs_base": round(se_pt_base, 5) if se_pt_base is not None else None,
        # Error to use for ARM-vs-ARM separation: the shared baseline cancels.
        "MM_PTGain_se_arm": round(se_pt_arm, 5) if se_pt_arm is not None else None,
        "MM_PTGain_ci_cells_without_n": n_no_n,
        "MM_PTGain_significant_vs_base": (bool(abs(mm_ptgain) > Z95 * se_pt_base)
                                          if (mm_ptgain is not None and se_pt_base) else None),
        "MM_PTGain_weighting": "bench-equal (w_b = 1/%d over the perception benches)"
                               % (len(per_g) or len(PERCEPTION_BENCHES)),
        "MM_PTGain_n_benches": len(per_g),
        # BREAKDOWN, not weights: which modality the gain came from.
        "MM_PTGain_by_modality": {m: round(v, 4) for m, v in group_mean.items()},
        # Appendix variant: equal weight per MODALITY GROUP instead of per bench.
        # Reported for transparency because it is what earlier drafts used; with
        # groups of 2/1/4 it over-weights the single Image bench 4x vs an AV one.
        "MM_PTGain_modality_balanced": round(mm_ptgain_modbal, 4) if mm_ptgain_modbal is not None else None,
        "MM_PTGain_modality_balanced_se_vs_base": round(se_mb_base, 5) if se_mb_base is not None else None,
        "MM_PTGain_modality_balanced_se_arm": round(se_mb_arm, 5) if se_mb_arm is not None else None,
        "MM_PTGain_with_code": round(mm_ptgain_ext, 4) if mm_ptgain_ext is not None else None,
        "MM_PTGain_with_code_by_modality": {m: round(v, 4) for m, v in group_mean_ext.items()},
        "macro_g_headroom": round(macro_g, 4) if macro_g is not None else None,
        "macro_delta": round(macro_delta, 4) if macro_delta is not None else None,
        "integrity_violation_rate": integ_rate,
        "integrity_n_gated": len(integ_present),
        # sub-category rates + how much of the gate actually ran (see above)
        "integrity_by_leg": integ_legs,
        "integrity_gate_coverage": integ_gate_coverage,
        "frontier_refresh": frontier_agg,
        "feedback_gain": "N/A (needs control arms: one-shot / no-feedback / static-retry)",
        "capability_retention": "N/A (needs frozen anchor set) -- see ood_probe_retention "
                                "for the narrow held-out proxy that IS on disk",
        "ood_probe_retention": ood_probe,
        "sealed_gap_agg": sealed_gap_agg,
        "budget": {
            "wall_h_total": round(wall_h_total, 2),
            "wall_h_by_bench": {k: round(v, 2) for k, v in wall_h_by_bench.items()},
            "gpu_h_total": round(gpu_h_total, 1),
            "tokens": tok_tot,
            "usage_available": usage_available,
            "cost_gpu_usd": round(gpu_cost, 2),
            "cost_api_usd_recomputed": round(api_cost_recomp, 2) if api_cost_recomp is not None else "N/A (no usage log)",
            "cost_api_usd_sdk_reported": round(tok_tot["sdk_cost_usd"], 2) if usage_available else "N/A (no usage log)",
            "cost_total_usd_recomputed": round(cost_total_recomp, 2) if cost_total_recomp is not None else "N/A (GPU=%.2f + API N/A)" % gpu_cost,
            "cost_total_usd_sdk": round(cost_total_sdk, 2) if cost_total_sdk is not None else "N/A",
        },
        "efficiency": {
            "compute_eff_ptgain_per_gpuh": round(compute_eff, 5) if compute_eff is not None else None,
            "cost_eff_ptgain_per_usd": round(cost_eff, 6) if cost_eff is not None else None,
        },
        "per_bench": per_bench,
    }


def fmt_report(res):
    L = []
    L.append(f"\n{'='*78}\nMODEL: {res['model']}\n{'='*78}")
    se_b, se_a = res.get("MM_PTGain_se_vs_base"), res.get("MM_PTGain_se_arm")
    bar = f" +/- {Z95*se_b:.4f}" if se_b else ""
    L.append(f"[1] MM-PTGain (BENCH-EQUAL, headroom-norm, PERCEPTION) = {res['MM_PTGain']}{bar}  (95%)")
    L.append(f"    weighting: {res.get('MM_PTGain_weighting')}")
    if se_b:
        sig = res.get("MM_PTGain_significant_vs_base")
        L.append(f"    95% CI vs base {res['MM_PTGain_ci95']}  -> gain {'IS' if sig else 'is NOT'} "
                 f"distinguishable from 0")
        L.append(f"    se_vs_base={se_b:.5f} (use vs base)   se_arm={se_a:.5f} (use for arm-vs-arm; "
                 f"shared baseline cancels)")
        if res.get("MM_PTGain_ci_cells_without_n"):
            L.append(f"    NOTE: {res['MM_PTGain_ci_cells_without_n']} cell(s) had no eval n -> "
                     f"error bar is a LOWER BOUND")
        n_c = sum(1 for r in res["per_bench"].values() if r.get("cheat_adjusted"))
        if n_c:
            L.append(f"    NOTE: {n_c} cheat-reset cell(s) contribute exactly 0 with zero variance "
                     f"(accounting decision, not a measurement) -> this arm's bar is narrower than "
                     f"a clean arm's for the same n")
    L.append(f"    by modality (BREAKDOWN, not weights): {res['MM_PTGain_by_modality']}")
    mb, mbse = res.get("MM_PTGain_modality_balanced"), res.get("MM_PTGain_modality_balanced_se_vs_base")
    if mb is not None:
        L.append(f"    appendix variant, equal weight per MODALITY GROUP = {mb:+.4f}"
                 + (f" +/-{Z95*mbse:.4f}" if mbse else "")
                 + "   (groups are 2/1/4 benches, so this weights mmmu_pro 4x an AV bench)")
    if res.get("MM_PTGain_with_code") is not None:
        L.append(f"    +Code reading:  MM-PTGain_with_code = {res['MM_PTGain_with_code']}  "
                 f"{res['MM_PTGain_with_code_by_modality']}")
    L.append(f"    (ref) macro g_headroom = {res['macro_g_headroom']}  |  macro delta = {res['macro_delta']}")
    L.append(f"\n[2] Per-bench native score / gain (score & gain are CHEAT-ADJUSTED:")
    L.append(f"    a gated-cheat bench is reset to baseline; 'flag' col: C=cheat-adj, B=benchmax):")
    L.append(f"    'n'=sealed eval items ('~'=derived from the deterministic split, not logged);")
    L.append(f"    '+/-d95'=95% half-width on delta; 'sig'=* when |delta| > that half-width")
    L.append(f"    {'bench':<16}{'modality':<12}{'n':>6}{'score':>8}{'base':>8}{'delta':>8}"
             f"{'+/-d95':>9}{'sig':>4}{'g_hd':>8}{'g_raw':>8}{'gap':>8}{'oodΔ':>8}{'flag':>6}")
    for b in BENCHES:
        r = res["per_bench"][b]
        def f(x, w=8, p=4):
            return (f"{x:.{p}f}".rjust(w)) if isinstance(x, (int, float)) else str(x).rjust(w)
        flag = ("C" if r.get("cheat_adjusted") else "") + ("B" if r.get("benchmax_flag") else "")
        se_d = r.get("se_delta")
        h95 = f(Z95 * se_d, 9) if isinstance(se_d, (int, float)) else "".rjust(9)
        sig = ("*" if r.get("delta_significant") else ("-" if r.get("delta_significant") is False else "")).rjust(4)
        # '~' marks an n recovered from the deterministic split rather than logged
        nn = ((str(r.get("n_eval")) + ("~" if r.get("n_eval_source") == "split_derived" else ""))
              if r.get("n_eval") else "-").rjust(6)
        L.append(f"    {b:<16}{r['modality']:<12}{nn}{f(r['score'])}{f(r['baseline'])}{f(r['delta'])}"
                 f"{h95}{sig}{f(r['g_headroom'])}{f(r.get('g_headroom_raw'))}{f(r['sealed_gap'])}"
                 f"{f(r['ood_delta'])}{flag:>6}")
    L.append(f"\n[3] Trajectory (visible per-ckpt _score_ series):")
    L.append(f"    RAW REGRET IS NOT REPORTABLE ALONE -- the peak is a max over k noisy draws, so")
    L.append(f"    'max-last' is > 0 even for a model that never changed. 'null'=that winner's-curse")
    L.append(f"    expectation (Monte Carlo, ability fixed); 'excs'=regret-null; 'p'=P(null >= obs);")
    L.append(f"    'nv'=visible items per eval ('~'=recovered from the score denominator, not logged).")
    L.append(f"    'refresh'=frontier refreshes as sig/raw: 'sig' counts only new bests that beat")
    L.append(f"    the incumbent by more than the 95% bar on the pair ('d'=median of those bars);")
    L.append(f"    'raw' counts any improvement, which on these n's is mostly noise.")
    L.append(f"    {'bench':<16}{'k':>3}{'nv':>6}{'peak':>8}{'final':>8}{'regret':>8}"
             f"{'null':>8}{'excs':>8}{'p':>7}{'sig':>4}{'refresh':>10}{'d':>7}"
             f"{'anytime':>9}{'t2best_h':>9}")
    for b in BENCHES:
        t = res["per_bench"][b]["trajectory"]
        if t.get("n_points"):
            rn = t.get("regret_null") or {}
            nv = t.get("visible_n_median")
            nvs = ((str(nv) + ("" if t.get("visible_n_all_recorded") else "~")) if nv else "-").rjust(6)
            def g(x, w, p=4):
                return (f"{x:.{p}f}".rjust(w)) if isinstance(x, (int, float)) else "".rjust(w)
            sig = ("*" if rn.get("regret_beats_null") else ("-" if rn else "")).rjust(4)
            L.append(f"    {b:<16}{t['n_points']:>3}{nvs}{t['peak_visible']:>8.4f}{t['final_visible']:>8.4f}"
                     f"{t['selection_regret']:>8.4f}{g(rn.get('regret_null_mean'),8)}"
                     f"{g(rn.get('regret_excess'),8)}{g(rn.get('regret_p_value'),7)}{sig}"
                     f"{(str(t.get('frontier_refresh_sig_count')) + '/' + str(t['frontier_refresh_count'])):>10}"
                     f"{g(t.get('frontier_refresh_delta_median'),7,3)}"
                     f"{t['anytime_val']:>9.4f}{t['time_to_best_h']:>9.2f}")
        else:
            L.append(f"    {b:<16}{'(no visible series)':>40}")
    _tr = [res["per_bench"][b]["trajectory"].get("regret_null") for b in BENCHES]
    _tr = [x for x in _tr if x]
    if _tr:
        n_beat = sum(1 for x in _tr if x.get("regret_beats_null"))
        m_obs = sum(x["regret_observed"] for x in _tr) / len(_tr)
        m_nul = sum(x["regret_null_mean"] for x in _tr) / len(_tr)
        L.append(f"    -> {len(_tr)} cell(s) with a usable null: mean observed regret {m_obs:.4f} vs "
                 f"mean null {m_nul:.4f}; {n_beat} beat the null at p<0.05")
        if not n_beat:
            L.append(f"       NO cell exceeds sampling noise -> do NOT claim 'the agent searched past "
                     f"its peak and regressed' from this series.")
    fr = res.get("frontier_refresh")
    if fr:
        L.append(f"\n[3b] Frontier Refresh Rate (pooled) = {fr['rate_sig']} significant "
                 f"({fr['refresh_sig']}/{fr['n_steps']} steps)  vs raw {fr['rate_raw']} "
                 f"({fr['refresh_raw']}/{fr['n_steps']})")
        L.append(f"     median noise threshold delta = {fr['delta_median']}"
                 + (f"; {fr['n_untested']} refresh(es) untestable (no n)" if fr["n_untested"] else ""))
        if fr["refresh_raw"] and not fr["refresh_sig"]:
            L.append(f"     -> EVERY apparent frontier refresh is inside its own noise bar: the")
            L.append(f"        agent could not have told a real improvement from a lucky draw.")

    L.append(f"\n[4] Integrity Violation Rate = {res['integrity_violation_rate']} "
             f"over {res['integrity_n_gated']} gated benches")
    legs = res.get("integrity_by_leg") or {}
    if legs:
        cov = res.get("integrity_gate_coverage")
        L.append(f"    per-leg sub-categories (gate coverage {cov}):")
        L.append(f"      {'leg':<18}{'ran':>5}{'skip':>6}{'viol':>6}{'r|ran':>8}{'r|all':>8}")
        for k, v in legs.items():
            def _r(x):
                return (f"{x:.3f}".rjust(8)) if isinstance(x, float) else "n/a".rjust(8)
            L.append(f"      {k:<18}{v['n_ran']:>5}{v['n_skipped']:>6}{v['n_violated']:>6}"
                     f"{_r(v['rate_of_ran'])}{_r(v['rate_of_all'])}"
                     + (f"  {v['violating_benches']}" if v["violating_benches"] else ""))
        blind = [k for k, v in legs.items()
                 if isinstance(v["coverage"], float) and v["coverage"] < 1.0]
        if blind:
            L.append(f"      -> NOT fully covered: {', '.join(blind)}. The headline rate is a")
            L.append(f"         lower bound and this is the size of the blind spot.")
    for b in BENCHES:
        ig = res["per_bench"][b]["integrity"]
        if not ig.get("present"):
            continue
        if ig.get("violation"):
            L.append(f"      VIOLATION {b:<15} legs={ig.get('legs')}  note={str(ig.get('note'))[:50]}")
        elif ig.get("n_legs_skipped"):
            L.append(f"      (clean*)  {b:<15} {ig.get('n_legs_skipped')} leg(s) SKIPPED -> lower-bound only")
    op = res.get("ood_probe_retention")
    if op:
        sig = "SIGNIFICANT" if op.get("significant") else "ns"
        ci = op.get("ci95")
        L.append(f"\n[4b] Held-out probe retention (proxy) = {op['mean_delta']:+.4f} "
                 f"+/-{Z95*op['se_mean']:.4f}  z={op.get('z')}  {sig}"
                 if op.get("se_mean") else
                 f"\n[4b] Held-out probe retention (proxy) = {op['mean_delta']:+.4f}")
        L.append(f"     {op['n_negative']}/{op['n_cells']} cells negative"
                 + (f"; 95% CI [{ci[0]:+.4f},{ci[1]:+.4f}]" if ci else "")
                 + f"; probes={op.get('probes')}")
        if op.get("significant_cells"):
            L.append(f"     individually significant cells: {', '.join(op['significant_cells'])}")
            for b in op["significant_cells"]:
                pb = res["per_bench"][b]
                tgt = pb.get("delta")
                arrow = ("  <-- target gain bought with probe loss"
                         if isinstance(tgt, (int, float)) and tgt > 0 else "")
                L.append(f"       {b:<15} oodD={pb['ood_delta']:+.4f} "
                         f"(target delta {tgt:+.4f}){arrow}")
        L.append(f"     CAVEAT: {op['note']}")
    sga = res.get("sealed_gap_agg")
    if sga:
        L.append(f"\n[4c] Sealed generalization gap (val - eval) mean = {sga['mean']:+.4f}, "
                 f"max {sga['max']:+.4f} ({sga['max_bench']}), over {sga['n_cells']} cells")
    b = res["budget"]
    L.append(f"\n[5] Research Budget & Efficiency:")
    L.append(f"    T_wall (WALL-CLOCK h, primary)   = {b['wall_h_total']} h   {b['wall_h_by_bench']}")
    L.append(f"    GPU-h (for compute-$ only)       = {b['gpu_h_total']} GPU-h")
    tk = b["tokens"]
    if b.get("usage_available"):
        L.append(f"    Researcher tokens                = in {tk['input']:,} | out {tk['output']:,} "
                 f"| cache_read {tk['cache_read']:,} | cache_write {tk['cache_write']:,} | think {tk['thinking']:,} "
                 f"| {tk['sessions']} sessions")
    else:
        L.append(f"    Researcher tokens                = N/A (transcript predates usage logging)")
    L.append(f"    Cost$  GPU={b['cost_gpu_usd']}  API(recomp)={b['cost_api_usd_recomputed']}  "
             f"API(SDK)={b['cost_api_usd_sdk_reported']}")
    L.append(f"           TOTAL(recomp)={b['cost_total_usd_recomputed']}  TOTAL(SDK-api)={b['cost_total_usd_sdk']}")
    e = res["efficiency"]
    L.append(f"    ComputeEff = {e['compute_eff_ptgain_per_gpuh']} PTGain/GPU-h   "
             f"CostEff = {e['cost_eff_ptgain_per_usd']} PTGain/$")
    L.append(f"\n[  ] Feedback Gain: {res['feedback_gain']}")
    L.append(f"[  ] Capability Retention: {res['capability_retention']}")
    return "\n".join(L)


def _pair_separated(ra, rb):
    """(gap, 95% bar, separated?) for two arms, using se_arm.

    Returns bar=None when either arm has no error bar, in which case the pair is
    NOT claimable as separated (unknown, not proven)."""
    d = ra["MM_PTGain"] - rb["MM_PTGain"]
    sa, sb = ra.get("MM_PTGain_se_arm"), rb.get("MM_PTGain_se_arm")
    if sa is None or sb is None:
        return d, None, False
    h = Z95 * math.sqrt(sa ** 2 + sb ** 2)
    return d, h, bool(abs(d) > h)


def fmt_leaderboard(out):
    """The paper's MAIN table: arms ranked by MM-PTGain with the four columns that
    decide how the rank may be read -- error bar, integrity violation rate, sealed
    (visible-vs-held-out) gap, and held-out probe delta -- plus wall time.

    A bare ranking of MM-PTGain would be misleading twice over: gaps smaller than
    the pairwise bar are not results, and an arm can buy target gain by spending
    held-out capability (that is what the oodD column exposes). So the table is
    rendered as TIERS derived from the pairwise test, never as 1..N."""
    rows = [(n, r) for n, r in out.items() if r.get("MM_PTGain") is not None]
    if not rows:
        return ""
    rows.sort(key=lambda kv: -kv[1]["MM_PTGain"])
    L = ["", "=" * 100, "LEADERBOARD -- MM-PTGain (BENCH-EQUAL w_b=1/7, headroom-normalized,",
         "                cheat-adjusted, 7 perception benches)", "=" * 100,
         f"  {'rank':<5}{'agent':<14}{'MM-PTGain':>11}{'+/-95%':>9}{'95% CI vs base':>22}"
         f"{'sig':>5}{'viol':>7}{'gap':>9}{'oodD':>10}{'T_wall(h)':>11}"]
    for i, (n, r) in enumerate(rows, 1):
        se_b = r.get("MM_PTGain_se_vs_base")
        bar = f"{Z95*se_b:.4f}".rjust(9) if se_b else "".rjust(9)
        ci = (f"[{r['MM_PTGain_ci95'][0]:+.4f},{r['MM_PTGain_ci95'][1]:+.4f}]".rjust(22)
              if r.get("MM_PTGain_ci95") else "".rjust(22))
        sig = ("*" if r.get("MM_PTGain_significant_vs_base") else "ns").rjust(5)
        vr = r.get("integrity_violation_rate")
        vrs = (f"{vr:.3f}" if isinstance(vr, (int, float)) else "n/a").rjust(7)
        sg = r.get("sealed_gap_agg") or {}
        sgs = (f"{sg['mean']:+.4f}" if isinstance(sg.get("mean"), float) else "n/a").rjust(9)
        op = r.get("ood_probe_retention") or {}
        if isinstance(op.get("mean_delta"), float):
            ods = f"{op['mean_delta']:+.4f}{'*' if op.get('significant') else ''}".rjust(10)
        else:
            ods = "n/a".rjust(10)
        L.append(f"  {i:<5}{n:<14}{r['MM_PTGain']:>+11.4f}{bar}{ci}{sig}{vrs}{sgs}{ods}"
                 f"{r['budget']['wall_h_total']:>11.1f}")
    L.append("  sig : * = gain distinguishable from 0 at 95%; ns = not distinguishable.")
    L.append("  viol: integrity violation rate (LOWER BOUND -- some gate legs get skipped).")
    L.append("  gap : mean sealed generalization gap, val - eval (+ = better on the split")
    L.append("        the agent could see).")
    L.append("  oodD: mean held-out probe delta, * = significant at 95%. This is the only")
    L.append("        retention-like signal on disk and it is ONE AUDIO BENCH wide -- call")
    L.append("        it held-out probe retention, never general capability retention.")

    L.append("")
    L.append("  Pairwise separation (row - col, +/- 95% bar; baseline cancels so this uses")
    L.append("  se_arm, the correct -- and smaller -- bar for comparing two arms):")
    names = [n for n, _ in rows]
    n_sep = 0
    for a in names:
        for b in names:
            if a >= b:
                continue
            d, h, sep = _pair_separated(out[a], out[b])
            if h is None:
                L.append(f"    {a:>12} - {b:<12} = {d:+.4f}   (no error bar available)")
                continue
            n_sep += bool(sep)
            L.append(f"    {a:>12} - {b:<12} = {d:+.4f} +/- {h:.4f}   "
                     f"{'SEPARATED' if sep else 'TIED (gap inside noise)'}")

    # --- tiers: a tier is a maximal run of consecutive arms that are pairwise
    # TIED with EVERY other member of that run. Consecutive-and-all-pairs (not
    # transitive closure) keeps the grouping from swallowing arms that ARE
    # separated from the top of the tier.
    tiers, cur = [], [names[0]]
    for nm in names[1:]:
        if all(not _pair_separated(out[m], out[nm])[2] for m in cur):
            cur.append(nm)
        else:
            tiers.append(cur)
            cur = [nm]
    tiers.append(cur)
    L.append("")
    L.append(f"  TIERS (the reportable form of this table -- {n_sep} of "
             f"{len(names)*(len(names)-1)//2} pairs separated):")
    for ti, t in enumerate(tiers, 1):
        span = [out[m]["MM_PTGain"] for m in t]
        detail = " = ".join(f"{m} ({out[m]['MM_PTGain']:+.4f})" for m in t)
        L.append(f"    tier {ti}: {detail}")
        if len(t) > 1:
            L.append(f"             within-tier spread {max(span)-min(span):.4f} -- order is "
                     f"NOT claimable, any re-run can permute it")
    if len(tiers) == 1:
        L.append("    -> ALL ARMS ARE ONE TIE GROUP. Do NOT report a winner; the nominal")
        L.append("       top arm's lead is smaller than the measurement noise.")
    L.append("")
    L.append("  READING GUIDE: a rank order is only claimable between arms marked SEPARATED.")
    L.append("  Report tiers, not ranks. Naming the tier-1 leader as 'the strongest agent'")
    L.append("  is a claim the data does not support unless its tier has one member.")
    return "\n".join(L)


def cross_arm_sensitivity(out):
    """Sens(b) / Perf(b): which benches actually RESPOND to post-training.

    Per-arm MM-PTGain answers "which agent is stronger"; this answers the
    orthogonal question "is this bench a usable target at all", pooling all
    arms x benches cells instead of one arm's seven. A bench where every arm
    lands inside the noise bar is not evidence that the agents failed -- it is
    a bench with no measurable headroom at this n, and a target-selection
    result rather than an agent result.

      Perf(b)  = base-model score (and its headroom 1 - base): how much room
                 there is to move at all.
      Sens(b)  = mean |g| over arms: how far post-training moves the score in
                 headroom-normalized units, regardless of direction.
      signed   = mean g: the net direction (Sens says "it moves", signed says
                 "which way").
      noise(b) = mean 95% arm bar over the same cells. Sens <= noise means the
                 movement is indistinguishable from sampling error.

    Verdicts: inert (Sens within noise) / responsive (moves, net up) /
    fragile (moves, net down) / mixed (moves, arms disagree in sign)."""
    arms = [n for n, r in out.items() if isinstance(r, dict) and r.get("per_bench")]
    rows = {}
    for b in BENCHES:
        gs, sigs, bars, bases = [], 0, [], []
        for a in arms:
            pb = out[a]["per_bench"].get(b) or {}
            if pb.get("g_headroom") is None:
                continue
            gs.append(pb["g_headroom"])
            sigs += bool(pb.get("delta_significant"))
            if pb.get("se_g_arm") is not None:
                bars.append(Z95 * pb["se_g_arm"])
            if isinstance(pb.get("baseline"), (int, float)):
                bases.append(pb["baseline"])
        if not gs:
            continue
        sens = sum(abs(g) for g in gs) / len(gs)
        signed = sum(gs) / len(gs)
        noise = (sum(bars) / len(bars)) if bars else None
        # A cheat-reset cell has g EXACTLY 0 by accounting decision, not by
        # measurement -- counting it as "degraded" would overstate degradation.
        EPS = 1e-9
        n_up = sum(1 for g in gs if g > EPS)
        n_dn = sum(1 for g in gs if g < -EPS)
        n_flat = len(gs) - n_up - n_dn
        if noise is not None and sens <= noise:
            verdict = "inert"
        elif n_up and n_up < len(gs) and min(gs) < 0 < max(gs) \
                and abs(signed) < (noise or 0):
            verdict = "mixed"
        elif signed > 0:
            verdict = "responsive"
        else:
            verdict = "fragile"
        rows[b] = {
            "modality": MODALITY[b],
            "n_arms": len(gs),
            "perf_base": round(sum(bases) / len(bases), 4) if bases else None,
            "headroom": round(1.0 - sum(bases) / len(bases), 4) if bases else None,
            "sens_mean_abs_g": round(sens, 4),
            "signed_mean_g": round(signed, 4),
            "g_min": round(min(gs), 4), "g_max": round(max(gs), 4),
            "g_range": round(max(gs) - min(gs), 4),
            "n_arms_improved": n_up, "n_arms_degraded": n_dn,
            "n_arms_flat": n_flat,   # g == 0 exactly = cheat-reset, not a measurement
            "n_cells_significant": sigs,
            "noise_bar_95": round(noise, 4) if noise is not None else None,
            "sens_exceeds_noise": (bool(sens > noise) if noise is not None else None),
            "verdict": verdict,
        }
    return rows


def fmt_sensitivity(rows):
    if not rows:
        return ""
    L = ["", "=" * 100,
         "CROSS-ARM BENCH SENSITIVITY -- Sens(b) / Perf(b) (all arms x benches pooled)",
         "=" * 100,
         f"  {'bench':<16}{'modality':<12}{'arms':>5}{'base':>8}{'hdrm':>7}{'Sens':>8}"
         f"{'signed':>9}{'g_min':>8}{'g_max':>8}{'up/0/dn':>9}{'sig':>5}{'noise':>8}{'verdict':>12}"]
    for b, r in sorted(rows.items(), key=lambda kv: -kv[1]["sens_mean_abs_g"]):
        def f(x, w, p=4):
            return (f"{x:.{p}f}".rjust(w)) if isinstance(x, (int, float)) else "n/a".rjust(w)
        L.append(f"  {b:<16}{r['modality']:<12}{r['n_arms']:>5}{f(r['perf_base'],8,3)}"
                 f"{f(r['headroom'],7,3)}{f(r['sens_mean_abs_g'],8)}{r['signed_mean_g']:>+9.4f}"
                 f"{r['g_min']:>+8.4f}{r['g_max']:>+8.4f}"
                 f"{('%d/%d/%d' % (r['n_arms_improved'], r.get('n_arms_flat', 0), r['n_arms_degraded'])):>9}"
                 f"{r['n_cells_significant']:>5}{f(r['noise_bar_95'],8)}{r['verdict']:>12}")
    inert = [b for b, r in rows.items() if r["verdict"] == "inert"]
    frag = [b for b, r in rows.items() if r["verdict"] == "fragile"]
    resp = [b for b, r in rows.items() if r["verdict"] == "responsive"]
    L.append("  Sens  : mean |g| across arms (headroom-normalized movement, any direction).")
    L.append("  signed: mean g -- the net direction of that movement.")
    L.append("  noise : mean 95% arm bar on the same cells; Sens <= noise => 'inert'.")
    L.append("  up/0/dn: arms that gained / were cheat-reset to exactly 0 / lost.")
    if inert:
        L.append(f"  -> INERT ({len(inert)}): {', '.join(inert)} -- no arm moved these beyond")
        L.append(f"     sampling noise. A flat result here is a TARGET property, not an agent")
        L.append(f"     failure; a suite of inert benches cannot separate agents at any n.")
    if frag:
        L.append(f"  -> FRAGILE ({len(frag)}): {', '.join(frag)} -- movement is real but net")
        L.append(f"     NEGATIVE, i.e. post-training reliably costs score here.")
    if resp:
        L.append(f"  -> RESPONSIVE ({len(resp)}): {', '.join(resp)} -- the only benches where")
        L.append(f"     post-training buys measurable headroom.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="name=path_template  (template contains {bench})")
    ap.add_argument("--pricing", default=os.path.join(os.path.dirname(__file__), "pricing.json"))
    ap.add_argument("--json", default=None, help="write full metrics JSON here")
    args = ap.parse_args()
    pricing = _read_json(args.pricing) or {}
    out = {}
    for spec in args.model:
        name, tmpl = spec.split("=", 1)
        res = compute_model(name, tmpl, pricing)
        out[name] = res
        print(fmt_report(res))
    if len(out) > 1:
        print(fmt_leaderboard(out))
        sens = cross_arm_sensitivity(out)
        print(fmt_sensitivity(sens))
        # reserved key, dunder-guarded so it can never collide with a model name
        out["__cross_arm_sensitivity__"] = sens
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[written] {args.json}")


if __name__ == "__main__":
    main()
