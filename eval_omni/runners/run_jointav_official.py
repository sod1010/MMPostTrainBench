#!/usr/bin/env python3
"""JointAVBench 自包含 official-aligned runner。

- 数据:官方 `jointavbench.json`(qid/question/options(JSON串)/answer_label(字母)/clip_path=videos/<qid>.mp4)。
- 模态:av —— 用已切好的 per-qid mp4 clip + `use_audio_in_video=True`(clip 自带音轨,无需 .m4a)。
- 推理:Qwen3-Omni-30B transformers + qwen_omni_utils。打分:字母精确匹配 answer_label(官方 MCQ accuracy)。
"""
from __future__ import annotations
import argparse, json, os, re, string as _string

def extract_letter(resp: str, n: int) -> str:
    """官方精神:从模型输出抽选项字母。"""
    t = resp.strip()
    for pat in (r"\\boxed\{([A-H])\}", r"(?:answer|the best answer)[:\s]*\(?([A-H])\)?",
                r"[（(\[]\s*([A-H])\s*[）)\]]", r"\b([A-H])\b"):
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            c = m.group(1).upper()
            if ord(c) - 65 < n:
                return c
    return t[:1].upper()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data", default=os.environ.get("JOINTAV_DATA", "JointAVBench/jointavbench.json"))
    ap.add_argument("--media-root", default=os.environ.get("JOINTAV_MEDIA_ROOT", "JointAVBench"))
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--out", default=os.environ.get("JOINTAV_OUT", "jointav_results/metrics.json"))
    args = ap.parse_args()

    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    data = json.load(open(args.data))
    # 只保留本地存在 clip 的样本
    def clip(rec):
        cp = rec.get("clip_path") or f"videos/{rec['qid']}.mp4"
        return os.path.join(args.media_root, cp)
    data = [r for r in data if os.path.isfile(clip(r))]
    import sys as _sys; _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "eval"))
    from split_util import keep as _keep, resolve_split as _resolve
    _SPLIT = _resolve(os.environ.get("EVAL_SPLIT", ""))
    if _SPLIT:
        data = [r for j, r in enumerate(data) if _keep(j, _SPLIT)]
        print(f"[jointav] EVAL_SPLIT={_SPLIT} -> {len(data)} items", flush=True)
    if args.limit > 0:
        data = data[: args.limit]
    print(f"[jointav] {len(data)} samples (local clip present)", flush=True)

    print("[jointav] loading model...", flush=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa").eval()
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    proc = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

    LETTERS = _string.ascii_uppercase
    correct = 0; n = 0; results = []
    for i, rec in enumerate(data):
        opts = rec["options"]
        if isinstance(opts, str):
            try: opts = json.loads(opts)
            except Exception:
                import ast; opts = ast.literal_eval(opts)
        gold = str(rec["answer_label"]).strip().upper()
        opt_block = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
        prompt = (f"Watch the video and listen to its audio, then answer the multiple-choice question.\n"
                  f"Question: {rec['question']}\n{opt_block}\n"
                  f"Answer with the option's letter from the given choices directly.")
        conv = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [
                    {"type": "video", "video": clip(rec), "use_audio_in_video": True, "max_frames": args.max_frames},
                    {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
        inputs = proc(text=text, audio=audios, images=images, videos=videos,
                      return_tensors="pt", padding=True, use_audio_in_video=True).to("cuda").to(model.dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                 use_audio_in_video=True, thinker_do_sample=False)
        if isinstance(out, tuple): out = out[0]
        full = proc.batch_decode(out, skip_special_tokens=True)[0]
        resp = full.split("assistant\n")[-1].strip()
        pred = extract_letter(resp, len(opts))
        ok = pred == gold
        correct += int(ok); n += 1
        results.append({"qid": rec["qid"], "idx": rec["qid"], "question": rec.get("question"),
                        "options": opts, "media": clip(rec), "gold": gold, "pred": pred,
                        "resp": resp[:80], "ok": ok})
        print(f"  [{i+1}/{len(data)}] gold={gold} pred={pred} {'OK' if ok else 'x'}", flush=True)

    acc = correct / n if n else 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"task": "jointavbench", "accuracy": acc, "correct": correct, "n": n, "results": results},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n[jointav] accuracy = {correct}/{n} = {acc:.4f} -> {args.out}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
