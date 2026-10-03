#!/usr/bin/env python3
"""MMAR self-contained official-aligned runner (mirrors run_mmau_official.py).

Why a self-runner (not lmms-eval): mmar's HF dataset schema uses the `List`
feature (datasets>=4 only), but under datasets 4 its `audio` column decodes to
a torchcodec `AudioDecoder` that lmms-eval's qwen3_omni wrapper mis-routes as a
"visual" -> crash; and datasets 3.6 (which avoids that) lacks the `List` type.
A self-runner sidesteps both: load with datasets 4.8.4 (List works), decode the
audio ourselves (dict OR AudioDecoder), feed it explicitly as audio, MCQ score.

- data: datasets(ngqtrung/mmar, split=test; needs HF token). choices already
  carry letters ["A. ...", ...]; answer is the letter.
- infer: Qwen3-Omni transformers + qwen_omni_utils, explicit audio, 16k mono.
- score: parse the option letter from the response, compare to the gold letter.
"""
from __future__ import annotations
import argparse, os, re, string as _string, sys


def get_audio_array(a):
    """Return (float32 mono array, sampling_rate) from a datasets Audio value,
    which may be a {'array','sampling_rate'} dict OR a torchcodec AudioDecoder."""
    import numpy as np
    if isinstance(a, dict) and "array" in a:
        return np.asarray(a["array"], dtype=np.float32), int(a.get("sampling_rate", 16000))
    if type(a).__name__ == "AudioDecoder":
        samples = a.get_all_samples()
        data = getattr(samples, "data", samples)
        arr = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
        sr = int(getattr(samples, "sample_rate", 16000))
        return arr.astype(np.float32), sr
    if isinstance(a, str):  # a path
        import soundfile as sf
        arr, sr = sf.read(a)
        return np.asarray(arr, dtype=np.float32), int(sr)
    raise ValueError(f"Unknown audio type: {type(a)}")


def to_16k_mono(arr, sr):
    import numpy as np
    if arr.ndim == 2:
        axis = 0 if arr.shape[0] <= arr.shape[1] else 1
        arr = arr.mean(axis=axis)
    arr = arr.astype(np.float32)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000).astype(np.float32)
    return arr


def parse_letter(resp: str, n_choices: int):
    letters = list(_string.ascii_uppercase[:n_choices])
    # "The best answer is: B" / "(B)" / "B." / standalone B
    m = re.search(r"\b([A-H])\b", resp.strip())
    if m and m.group(1) in letters:
        return m.group(1)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch, json
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info
    from datasets import load_dataset

    d = load_dataset("ngqtrung/mmar", split="test", token=True)
    import os as _os, sys as _sys; _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "eval"))
    from split_util import keep as _keep, resolve_split as _resolve
    _SPLIT = _resolve(_os.environ.get("EVAL_SPLIT", ""))
    if _SPLIT:
        d = d.select([i for i in range(len(d)) if _keep(i, _SPLIT)])
        print(f"[mmar] EVAL_SPLIT={_SPLIT} -> {len(d)} items", flush=True)
    if args.limit > 0:
        d = d.select(range(min(args.limit, len(d))))
    print(f"[mmar] {len(d)} samples", flush=True)

    print("[mmar] loading model...", flush=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa").eval()
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    proc = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

    instruction = ("Listen to the audio and answer the following multiple-choice question. "
                   "Respond with only the letter (A, B, C, or D) of the correct option.\n")
    correct = 0; n = 0; results = []
    for i, doc in enumerate(d):
        arr, sr = get_audio_array(doc["audio"])
        arr = to_16k_mono(arr, sr)
        q = doc["question"]; choices = doc["choices"]; gold = str(doc["answer"]).strip().upper()
        prompt = instruction + q + "\n" + "\n".join(choices) + "\nThe best answer is:"
        conv = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "audio", "audio": arr},
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
        pred = parse_letter(resp, len(choices))
        ok = (pred == gold)
        correct += int(ok); n += 1
        results.append({"question": q, "gold": gold, "pred": pred, "resp": resp[:100], "ok": ok})
        print(f"  [{i+1}/{len(d)}] gold={gold} pred={pred} {'OK' if ok else 'x'}", flush=True)

    acc = correct / n if n else 0.0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump({"task": "mmar", "accuracy": acc, "correct": correct, "n": n, "results": results},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n[mmar] accuracy = {correct}/{n} = {acc:.4f} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
