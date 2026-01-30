#!/usr/bin/env python3
"""OOB Coordinator Diagnostic Client.

Connects to a running OOB coordinator and queries its state.
Useful for debugging deadlocks in tests or production.

Usage:
    # Query status
    python scripts/oob_diag.py status

    # List all store keys
    python scripts/oob_diag.py keys

    # Dump all store contents
    python scripts/oob_diag.py dump

    # Get specific key
    python scripts/oob_diag.py get <key>

    # Connect to non-default host/port
    python scripts/oob_diag.py --host 192.168.1.10 --port 29401 status
"""

import argparse
import sys

import zmq


def connect(host: str, port: int) -> zmq.Socket:
    """Connect to the OOB coordinator's store socket."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 100)
    sock.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
    sock.setsockopt(zmq.SNDTIMEO, 5000)
    sock.connect(f"tcp://{host}:{port}")
    return sock


def cmd_status(sock: zmq.Socket) -> None:
    """Get coordinator status."""
    sock.send(b"STATUS")
    print(sock.recv().decode())


def cmd_keys(sock: zmq.Socket) -> None:
    """List all store keys."""
    sock.send(b"KEYS")
    response = sock.recv().decode()
    if response:
        for key in response.split("\n"):
            print(key)
    else:
        print("(no keys)")


def cmd_dump(sock: zmq.Socket) -> None:
    """Dump all store contents."""
    sock.send(b"DUMP")
    response = sock.recv().decode()
    if response:
        print(response)
    else:
        print("(store empty)")


def cmd_get(sock: zmq.Socket, key: str) -> None:
    """Get a specific key."""
    sock.send(f"GET:{key}".encode())
    response = sock.recv().decode()
    if response:
        print(response)
    else:
        print(f"(key '{key}' not found)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OOB Coordinator Diagnostic Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Coordinator host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=29401,  # Store port is base_port + 1
        help="Store socket port (default: 29401, which is base_port+1)",
    )
    parser.add_argument(
        "command",
        choices=["status", "keys", "dump", "get"],
        help="Command to run",
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Command arguments (e.g., key name for 'get')",
    )

    args = parser.parse_args()

    try:
        sock = connect(args.host, args.port)
    except zmq.ZMQError as e:
        print(f"Failed to connect to {args.host}:{args.port}: {e}", file=sys.stderr)
        return 1

    try:
        if args.command == "status":
            cmd_status(sock)
        elif args.command == "keys":
            cmd_keys(sock)
        elif args.command == "dump":
            cmd_dump(sock)
        elif args.command == "get":
            if not args.args:
                print("Error: 'get' requires a key argument", file=sys.stderr)
                return 1
            cmd_get(sock, args.args[0])
    except zmq.Again:
        print(f"Timeout waiting for response from {args.host}:{args.port}", file=sys.stderr)
        return 1
    except zmq.ZMQError as e:
        print(f"ZMQ error: {e}", file=sys.stderr)
        return 1
    finally:
        sock.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
