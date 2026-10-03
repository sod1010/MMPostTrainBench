#!/usr/bin/env python3
"""Freeze public test assets before grading. Does not load models or run tests.

Input JSON: {"urls": ["https://raw.githubusercontent.com/..."], ...provenance}.
Only publish manifest.json after every requested URL has been fetched. Source
URLs and content hashes are retained for reproducibility. An existing valid
manifest is checked and reused; invalid existing manifests are never trusted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request
import uuid


def atomic(path, data):
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def fetch(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or parsed.netloc != 'raw.githubusercontent.com':
        raise ValueError('test assets must use the declared official GitHub raw source')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(64 * 1024 * 1024 + 1)
            if not data or len(data) > 64 * 1024 * 1024:
                raise ValueError('empty or excessive asset')
            if parsed.path.lower().endswith('.png') and not data.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError('PNG URL did not return PNG data')
            return data
        except Exception:
            if attempt == 2:
                raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urls', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=4)
    args = p.parse_args()
    declared = json.loads(args.urls.read_text())
    urls = sorted(set(declared['urls']))
    if not urls:
        p.error('empty asset list')
    out = args.out
    (out / 'blobs').mkdir(parents=True, exist_ok=True)
    entries = {}
    if (out / 'manifest.json').exists():
        previous = json.loads((out / 'manifest.json').read_bytes())
        if previous.get('version') != 1:
            raise ValueError('invalid previous manifest version')
        for url, entry in previous['assets'].items():
            if url not in urls:
                continue
            digest = entry['sha256']
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('invalid previous digest')
            data = (out / 'blobs' / digest).read_bytes()
            if not data or hashlib.sha256(data).hexdigest() != digest or len(data) != entry['size']:
                raise ValueError('previous asset cache is corrupt')
            entries[url] = entry
    def one(url):
        if url in entries:
            return url, entries[url]
        data = fetch(url)
        digest = hashlib.sha256(data).hexdigest()
        atomic(out / 'blobs' / digest, data)
        return url, {'sha256': digest, 'size': len(data)}
    with ThreadPoolExecutor(max_workers=max(1, min(args.jobs, 8))) as pool:
        for url, entry in pool.map(one, urls):
            entries[url] = entry
    manifest = {'version': 1, 'provenance': {k: v for k, v in declared.items() if k != 'urls'},
                'assets': entries}
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    atomic(out / 'manifest.json', raw)
    print(json.dumps({'n_assets': len(entries), 'manifest_sha256': hashlib.sha256(raw).hexdigest()}))


if __name__ == '__main__':
    main()
