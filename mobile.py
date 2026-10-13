#!/usr/bin/env python
"""mobile.py - One-command launcher for Mobile AR Capture Gateway.

Usage:
    python mobile.py              # Launch HTTP (8137) and HTTPS (8138)
    python mobile.py --port 8137  # Custom port
"""
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"

# If run with a global Python, transparently hand off to .venv's python
if VENV_PY.exists() and Path(sys.executable).resolve() != VENV_PY.resolve():
    cmd = [str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]]
    try:
        sys.exit(subprocess.call(cmd))
    except KeyboardInterrupt:
        sys.exit(0)


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def main():
    port = 8137
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    elif "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1].isdigit():
            port = int(sys.argv[idx + 1])

    https_port = port + 1
    local_ip = get_local_ip()
    mobile_url = f"https://{local_ip}:{https_port}/viewer/capture.html"
    dashboard_url = f"http://localhost:3000"

    print("\n" + "=" * 70)
    print("  \033[1;32m🛰️  MOBILE AR CAPTURE & TELEMETRY GATEWAY\033[0m")
    print("=" * 70)
    print(f"\n  \033[1;36m📲 PHONE CAPTURE LINK (Open on your phone):\033[0m")
    print(f"     \033[1;33m{mobile_url}\033[0m\n")
    print(f"  \033[1;34m💻 DESKTOP GROUND CONTROL COCKPIT:\033[0m")
    print(f"     {dashboard_url}  (or http://localhost:{port}/viewer/pc.html)\n")
    print("-" * 70)
    print("  \033[1m📋 3-STEP SETUP ON YOUR PHONE:\033[0m")
    print("  1. Connect phone and PC to the SAME Wi-Fi network (or phone hotspot).")
    print("  2. Open the HTTPS link above on Chrome (Android) or Safari (iOS).")
    print("  3. Bypass the self-signed SSL warning: tap 'Advanced' -> 'Proceed'.")
    print("     (HTTPS is mandatory by mobile browsers to access Camera + Motion sensors)")
    print("  4. Tap 'Start camera capture' and walk in smooth arcs!")
    print("=" * 70 + "\n")
    print(f"[mobile] Starting dual-port server on HTTP :{port} and HTTPS :{https_port}...")

    # Invoke _serve.py with port and root directory
    serve_script = ROOT / "_serve.py"
    cmd = [sys.executable, str(serve_script), str(port), str(ROOT)]

    try:
        proc = subprocess.Popen(cmd)
        proc.wait()
    except KeyboardInterrupt:
        print("\n[mobile] Shutting down Mobile AR Gateway. Goodbye!")
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()
