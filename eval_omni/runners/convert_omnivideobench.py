#!/usr/bin/env python3
"""Normalize an authorized upstream OmniVideoBench snapshot for its dataloader.

Supports official grouped JSON/Parquet rows and explicit flat QA rows. Does not
invent annotations or durations, download gated files, or redistribute media.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


def load_rows(src):
    src = Path(src)
    if src.is_dir():
        paths = sorted(src.glob('data*.parquet')) or sorted(src.glob('data*.json'))
        if not paths:
            raise ValueError('Snapshot must contain data*.parquet or data*.json')
        return [row for p in paths for row in load_rows(p)]
    if src.suffix == '.parquet':
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError('Install pyarrow on the preparation host for upstream Parquet') from exc
        return pq.read_table(src).to_pylist()
    if src.suffix == '.jsonl':
        return [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    rows = json.loads(src.read_text())
    if not isinstance(rows, list):
        raise ValueError('Annotations must be a list of records')
    return rows


def duration_string(value, media):
    if isinstance(value, str) and value:
        parts = value.split(':')
        if len(parts) in (2, 3) and all(x.isdigit() for x in parts):
            seconds = sum(int(x) * 60**i for i, x in enumerate(reversed(parts)))
        else:
            raise ValueError('Invalid duration')
    elif isinstance(value, (float, int)) and math.isfinite(value) and value >= 0:
        seconds = math.ceil(value)
    else:
        # Measure missing durations from the actual authorized media.
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                                 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                                 str(media)], check=True, capture_output=True, text=True)
        seconds = math.ceil(float(result.stdout.strip()))
    return f'{seconds // 60:02d}:{seconds % 60:02d}'


def convert(src, out, video_dir):
    src = Path(src); out = Path(out); dest = Path(video_dir)
    media_root = src if src.is_dir() else src.parent
    rows = load_rows(src)
    if not rows:
        raise ValueError('Empty upstream annotations')
    normalized = []
    media_files = {}
    for row in rows:
        video = row.get('video') or row.get('video_id')
        if not isinstance(video, str) or not video or Path(video).name != video:
            raise ValueError('Missing or unsafe video identifier; unsupported annotation schema')
        video = video.removesuffix('.mp4')
        media = media_root / 'videos' / f'{video}.mp4'
        if not media.is_file():
            raise ValueError(f'Missing authorized media for {video}')
        questions = row.get('questions')
        if isinstance(questions, str):
            questions = json.loads(questions)
        if questions is None and 'question' in row:
            questions = [row]
        if not isinstance(questions, list) or not questions:
            raise ValueError('Missing questions; unsupported annotation schema')
        qas = []
        for q in questions:
            options = q.get('options')
            if isinstance(options, str):
                options = json.loads(options)
            answer = q.get('correct_option')
            if not isinstance(q.get('question'), str) or not isinstance(options, list) or len(options) < 2:
                raise ValueError('Invalid QA record')
            if not isinstance(answer, str) or answer not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(options)]:
                raise ValueError('Missing/invalid correct_option; refusing to guess answer')
            qas.append({'question': q['question'], 'options': options, 'correct_option': answer})
        normalized.append({'video': video, 'duration': duration_string(row.get('duration'), media), 'questions': qas})
        media_files[video] = media
    # Validate all records before writing any usable annotation output.
    dest.mkdir(parents=True, exist_ok=True)
    for video, media in media_files.items():
        target = dest / f'{video}.mp4'
        if target.exists() or target.is_symlink():
            if target.resolve() != media.resolve():
                raise ValueError(f'Conflicting media at {target}')
        else:
            target.symlink_to(os.path.relpath(media.resolve(), target.parent.resolve()))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.tmp')
    tmp.write_text(json.dumps(normalized, ensure_ascii=False, indent=2))
    tmp.replace(out)
    return sum(len(x['questions']) for x in normalized)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--video-dir', required=True)
    args = ap.parse_args()
    try:
        n = convert(args.src, args.out, args.video_dir)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        ap.exit(2, f'OmniVideoBench preparation failed: {exc}\n')
    print(f'Validated {n} QA pairs -> {args.out}')

if __name__ == '__main__':
    main()
