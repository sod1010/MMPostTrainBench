#!/usr/bin/env python3
"""benchmarkall 的扁平 omnivideobench.jsonl → OmniVideoBench 官方评测脚本期望的分组 json。

官方 dataloader(harness_repos/OmniVideoBench/dataloader.py)期望:
  [ {"video": "<无扩展名文件名>", "duration": "MM:SS",
     "questions": [ {"question":..., "options":[...], "correct_option":"A"} ] } ]
  视频路径 = <video_dir>/<video>.mp4；duration 仅用于 max_duration 过滤。

benchmarkall 每行:{task, prompt[[{video:oss://.../OmniEval/videos/video_N.mp4},{text}]], answer, question, options}
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("OVB_SRC_JSONL", "omnivideobench.jsonl"))
    ap.add_argument("--out", default=os.environ.get("OVB_DATA", "OmniVideoBench_local/data.json"))
    ap.add_argument("--duration", default="00:30", help="占位时长(< max_duration 即可;仅用于过滤,不影响真实抽帧)")
    args = ap.parse_args()

    vpat = re.compile(r'oss://\S+/([^/"]+)\.mp4')
    out = []
    n_skip = 0
    for line in open(args.src, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        m = vpat.search(json.dumps(r, ensure_ascii=False))
        if not m:
            n_skip += 1
            continue
        out.append({
            "video": m.group(1),                 # e.g. video_1  (dataloader 会补 .mp4)
            "duration": args.duration,
            "questions": [{
                "question": r.get("question", ""),
                "options": r.get("options", []),
                "correct_option": str(r.get("answer", "")).strip(),
            }],
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[convert] {len(out)} 条 -> {args.out} (skipped {n_skip})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
