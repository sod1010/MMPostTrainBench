#!/usr/bin/env python3
"""MMAU 自包含 official-aligned runner(绕开 lmms-eval chat adapter 把 .wav 误当 video 的 bug)。

- 数据:datasets(lmms-lab-audio/mmau, test_mini,soundfile 解码为 array)。
- 推理:Qwen3-Omni-30B transformers + qwen_omni_utils.process_mm_info,**显式 audio 类型**。
- 打分:官方 MMAU evaluation.py 的 string_match(token 子集匹配),accuracy。
"""
from __future__ import annotations
import argparse, os, re, string as _string, sys

def string_match(pred: str, gold: str, choices) -> bool:
    """官方 MMAU string_match 精神:模型输出里出现正确选项文本/字母即算对(排除其它选项干扰)。"""
    p = pred.strip().lower()
    g = str(gold).strip().lower()
    # 1) 直接字母匹配 A/B/C/D
    letters = list(_string.ascii_uppercase[:len(choices)])
    gi = None
    for i, c in enumerate(choices):
        if str(c).strip().lower() == g:
            gi = i; break
    m = re.search(r"\b([a-h])\b", p)
    if m and gi is not None and m.group(1).upper() == letters[gi]:
        return True
    # 2) 正确选项文本子串命中,且不命中其它选项文本
    if g and g in p:
        others = [str(c).strip().lower() for i, c in enumerate(choices) if str(c).strip().lower() != g]
        if not any(o and o in p and len(o) > len(g) for o in others):
            return True
    return False

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out", default=os.environ.get("MMAU_OUT", "mmau_results/metrics.json"))
    args = ap.parse_args()

    import torch, json
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info
    from datasets import load_dataset

    d = load_dataset("lmms-lab-audio/mmau", split="test_mini")
    import os as _os, sys as _sys; _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "eval"))
    from split_util import keep as _keep, resolve_split as _resolve
    _SPLIT = _resolve(_os.environ.get("EVAL_SPLIT", ""))
    if _SPLIT:
        d = d.select([i for i in range(len(d)) if _keep(i, _SPLIT)])
        print(f"[mmau] EVAL_SPLIT={_SPLIT} -> {len(d)} items", flush=True)
    if args.limit > 0:
        d = d.select(range(min(args.limit, len(d))))
    print(f"[mmau] {len(d)} samples", flush=True)

    print("[mmau] loading model...", flush=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa").eval()
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    proc = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

    LETTERS = _string.ascii_uppercase
    correct = 0; n = 0; results = []
    for i, doc in enumerate(d):
        audio = doc["audio"]                       # {array, sampling_rate, path}
        q = doc["question"]; gold = doc["answer"]
        choices = doc["choices"]
        if isinstance(choices, str):               # mmau: choices 是 JSON 字符串
            import json as _json
            try: choices = _json.loads(choices)
            except Exception:
                import ast; choices = ast.literal_eval(choices)
        opt = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(choices))
        prompt = f"{q}\n{opt}\nAnswer with the option's letter from the given choices directly."
        conv = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "audio", "audio": audio["array"]},
                                             {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inputs = proc(text=text, audio=audios, images=images, videos=videos,
                      return_tensors="pt", padding=True, use_audio_in_video=False).to("cuda").to(model.dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                 use_audio_in_video=False, thinker_do_sample=False)
        if isinstance(out, tuple): out = out[0]
        full = proc.batch_decode(out, skip_special_tokens=True)[0]
        resp = full.split("assistant\n")[-1].strip()
        ok = string_match(resp, gold, choices)
        correct += int(ok); n += 1
        results.append({"question": q, "gold": gold, "resp": resp[:100], "ok": ok})
        print(f"  [{i+1}/{len(d)}] gold={gold!r} resp={resp[:40]!r} {'OK' if ok else 'x'}", flush=True)

    acc = correct / n if n else 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"task": "mmau_test_mini", "accuracy": acc, "correct": correct, "n": n, "results": results},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n[mmau] accuracy = {correct}/{n} = {acc:.4f} -> {args.out}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
