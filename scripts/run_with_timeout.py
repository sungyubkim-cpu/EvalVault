#!/usr/bin/env python3
"""Cross-platform timeout wrapper for shell commands.

Works on macOS, Linux, and Windows.
Usage: python run_with_timeout.py <timeout_seconds> <command> [args...]
"""

import subprocess
import sys
from typing import NoReturn


def run_with_timeout(timeout: int, command: list[str]) -> int:
    """Run command with timeout, return exit code."""
    try:
        result = subprocess.run(
            command,
            timeout=timeout,
            capture_output=False,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] Command exceeded {timeout} seconds", file=sys.stderr)
        return 124  # Same as GNU timeout
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Command cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        return 1


def main() -> NoReturn:
    if len(sys.argv) < 3:
        print("Usage: python run_with_timeout.py <timeout_seconds> <command> [args...]")
        print("Example: python run_with_timeout.py 60 uv run evalvault config")
        sys.exit(1)

    try:
        timeout = int(sys.argv[1])
    except ValueError:
        print(f"Error: Invalid timeout value: {sys.argv[1]}")
        sys.exit(1)

    command = sys.argv[2:]
    exit_code = run_with_timeout(timeout, command)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
