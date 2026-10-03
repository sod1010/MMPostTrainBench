"""Submit one strict JSON request over a local UNIX socket; no shell forwarding."""
import argparse
import json
import socket
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket', required=True)
    p.add_argument('--request', required=True, help='JSON file, or - for stdin')
    args = p.parse_args()
    with (sys.stdin if args.request == '-' else open(args.request)) as f:
        request = json.load(f)
    message = json.dumps(request, allow_nan=False).encode()+b'\n'
    if len(message) > 16384:
        p.error('request too large')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(10)
        s.connect(args.socket)
        s.sendall(message)
        with s.makefile('rb') as f:
            result = json.loads(f.readline(16385))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get('ok') else 1


if __name__ == '__main__':
    raise SystemExit(main())
