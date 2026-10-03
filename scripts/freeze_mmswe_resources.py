#!/usr/bin/env python3
"""Fetch explicit public HTTPS resources once; runtime never fetches upstream.

Creates a private, self-contained snapshot and a local-only TLS CA. It does not
install a CA on the host, change DNS, or run any grading/agent workload.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import urllib.request
from urllib.parse import urlsplit


def tls_material(out, hosts):
    with tempfile.TemporaryDirectory(prefix='mmptb-snapshot-ca-',dir=out) as td:
        temp=Path(td)
        extension=temp/'extensions.cnf'
        extension.write_text('basicConstraints=critical,CA:FALSE\nextendedKeyUsage=serverAuth\nsubjectAltName='+','.join('DNS:'+h for h in sorted(hosts))+'\n')
        commands=[
            ['req','-x509','-newkey','rsa:2048','-nodes','-sha256','-days','3650',
             '-subj','/CN=MMPTB isolated snapshot CA','-keyout',str(temp/'ca.key'),'-out',str(out/'ca.pem')],
            ['req','-newkey','rsa:2048','-nodes','-sha256','-subj','/CN='+sorted(hosts)[0],
             '-keyout',str(out/'server.key'),'-out',str(temp/'server.csr')],
            ['x509','-req','-in',str(temp/'server.csr'),'-CA',str(out/'ca.pem'),
             '-CAkey',str(temp/'ca.key'),'-set_serial','1','-days','3650','-sha256',
             '-extfile',str(extension),'-out',str(out/'server.pem')]]
        for command in commands:
            subprocess.run(['openssl',*command],check=True,capture_output=True,timeout=30)
    # The signing key is discarded; only a leaf key for this isolated snapshot remains.
    for name in ('ca.pem','server.pem','server.key'):(out/name).chmod(0o400)


def build(out, urls):
    os.umask(0o077)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'blobs').mkdir()
    entries,hosts={},set()
    for url in urls:
        u=urlsplit(url)
        if (u.scheme!='https' or not u.hostname or u.username or u.password or u.fragment
                or u.port is not None or not re.fullmatch(r'[a-z0-9]+(?:[.-][a-z0-9]+)*\.[a-z]{2,63}',u.hostname)
                or url!='https://'+u.hostname+(u.path or '/')+('?' + u.query if u.query else '')):
            raise ValueError('canonical HTTPS URL required')
        request=urllib.request.Request(url,headers={'User-Agent':'MMPTB-resource-freezer/1'})
        with urllib.request.urlopen(request,timeout=45) as response:
            if response.status!=200 or response.geturl()!=url:
                raise ValueError('resource fetch must return 200 without redirect')
            body=response.read(5*1024*1024+1)
            content_type=response.headers.get_content_type()
        if not 0<len(body)<=5*1024*1024:raise ValueError('resource exceeds size limit')
        digest=hashlib.sha256(body).hexdigest()
        (out/'blobs'/(digest+'.bin')).write_bytes(body)
        entries[url]={'sha256':digest,'size':len(body),'content_type':content_type}
        hosts.add(u.hostname)
    tls_material(out,hosts)
    manifest={'version':1,'mode':'loopback-https-snapshot','fetched_utc':datetime.now(timezone.utc).isoformat(),
              'resources':entries,'tls_sha256':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('ca.pem','server.pem','server.key')},
              'scope':'frozen external resources; official-test equivalence requires separate acceptance'}
    path=out/'manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',action='append',required=True)
    p.add_argument('--out',required=True,help='new private snapshot directory')
    a=p.parse_args()
    if not 1<=len(set(a.url))<=32:p.error('provide 1..32 unique URLs')
    print(json.dumps(build(a.out,list(dict.fromkeys(a.url)))))


if __name__=='__main__':main()
