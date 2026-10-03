#!/usr/bin/env python3
"""Freeze an existing screenshot cache; never download or silently add inputs.

Input JSON has urls and dataset provenance. Unknown URLs (neither a cached
image nor a prior explicit .miss record) prevent publication. Missing images
are recorded, not represented as successful image reads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urls', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    declared = json.loads(args.urls.read_text())
    (args.out/'blobs').mkdir(parents=True, exist_ok=True)
    images, unknown = {}, []
    for url in sorted(set(declared['urls'])):
        key = hashlib.sha256(url.encode()).hexdigest()
        found = next((args.cache_dir/(key+ext) for ext in ('.png','.jpg','.jpeg','.gif','.webp','.bin')
                      if (args.cache_dir/(key+ext)).is_file()), None)
        if found:
            from PIL import Image
            with Image.open(found) as im:
                im.verify()
            data = found.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            # Decoder detects the actual format from bytes, like the old cache.
            target = args.out/'blobs'/(digest+'.png')
            tmp = target.with_name('.'+target.name+'.'+uuid.uuid4().hex)
            try:
                tmp.write_bytes(data)
                tmp.replace(target)
            finally:
                tmp.unlink(missing_ok=True)
            images[url] = {'status':'available','sha256':digest,'size':len(data)}
        elif (args.cache_dir/(key+'.miss')).is_file():
            images[url] = {'status':'unavailable'}
        else:
            unknown.append(url)
    if unknown:
        raise ValueError(f'{len(unknown)} image URLs are absent from the cache; prefetch before freezing')
    manifest = {'version':1,'provenance':{k:v for k,v in declared.items() if k!='urls'},'images':images}
    raw = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
    dest = args.out/'manifest.json'
    tmp = dest.with_name('.manifest.'+uuid.uuid4().hex)
    try:
        tmp.write_bytes(raw)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    print(json.dumps({'n_urls':len(images),'available':sum(x['status']=='available' for x in images.values()),
                      'unavailable':sum(x['status']=='unavailable' for x in images.values()),
                      'manifest_sha256':hashlib.sha256(raw).hexdigest()}))


if __name__ == '__main__':
    main()
