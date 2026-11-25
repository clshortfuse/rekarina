"""Relay newline-delimited MCP stdio to a running Rekarina process on Windows."""

import argparse
import json
import os
import sys


PIPE_ROOT = '\\\\.\\pipe\\'
PIPE_PREFIX = 'rekarina-renderer-'


def discover_pid():
    pids = sorted(int(name[len(PIPE_PREFIX):]) for name in os.listdir(PIPE_ROOT)
                  if name.startswith(PIPE_PREFIX) and name[len(PIPE_PREFIX):].isascii()
                  and name[len(PIPE_PREFIX):].isdigit())
    if len(pids) != 1:
        raise ValueError(f'expected one telemetry-enabled game, found {len(pids)}; '
                         'launch with REKARINA_RENDERER_MCP=1 or pass the PID explicitly')
    return pids[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pid', type=int, nargs='?', help='PID (auto-detected when omitted)')
    args = parser.parse_args()
    try:
        pid = args.pid if args.pid is not None else discover_pid()
        if not 0 < pid <= 0xffffffff:
            parser.error('invalid Windows process ID')
        with open(f'{PIPE_ROOT}{PIPE_PREFIX}{pid}', 'r+b', buffering=0) as pipe:
            for line in sys.stdin.buffer:
                request = json.loads(line)
                pipe.write(json.dumps(request, separators=(',', ':')).encode('ascii') + b'\n')
                if 'id' not in request:
                    continue
                response = pipe.readline()
                if not response:
                    raise OSError('renderer disconnected')
                sys.stdout.buffer.write(response)
                sys.stdout.buffer.flush()
    except (OSError, ValueError) as exc:
        sys.exit(f'renderer MCP: {exc}')


if __name__ == '__main__':
    main()
