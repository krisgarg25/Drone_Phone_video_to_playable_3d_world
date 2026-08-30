import functools
import http.server
import io
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

active_process = None
active_job_info = {"status": "idle", "scene": "", "step": "", "logs": []}
process_lock = threading.Lock()


def get_local_ip():
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


def safe_scene_name(name: str, fallback: str = "mobile_capture") -> str:
    """Reduce a user-supplied scene name to one safe path segment.

    The capture page sanitizes client-side too, but this is the boundary that
    matters: the result is joined onto videos/, so anything that could climb out
    of the tree (separators, .., drive letters) has to go.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = re.sub(r"_{2,}", "_", cleaned).strip("._-")[:64]
    return cleaned or fallback


def run_pipeline_thread(scene: str, preset: str, quality: str, extra_args: list):
    global active_process, active_job_info
    py_exe = str(ROOT / ".venv" / "Scripts" / "python.exe")
    if not Path(py_exe).exists():
        py_exe = sys.executable

    cmd = [py_exe, str(ROOT / "pipeline.py"), "run", scene, "--preset", preset, "--quality", quality] + extra_args
    with process_lock:
        active_job_info = {
            "status": "running",
            "scene": scene,
            "preset": preset,
            "quality": quality,
            "cmd": " ".join(cmd),
            "step": "initializing",
            "logs": []
        }

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )
        with process_lock:
            active_process = proc

        for line in proc.stdout:
            line_str = line.rstrip()
            with process_lock:
                active_job_info["logs"].append(line_str)
                if len(active_job_info["logs"]) > 300:
                    active_job_info["logs"].pop(0)

                # Track current stage
                if line_str.startswith("[") and "]" in line_str:
                    stage_match = re.search(r"\[([0-9a-zA-Z_\-/]+)\]", line_str)
                    if stage_match:
                        active_job_info["step"] = stage_match.group(1)

        proc.wait()
        with process_lock:
            active_job_info["status"] = "completed" if proc.returncode == 0 else f"failed (code {proc.returncode})"
            active_process = None
    except Exception as e:
        with process_lock:
            active_job_info["status"] = f"error: {e}"
            active_process = None


class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.mjs': 'text/javascript',
        '.wasm': 'application/wasm',
        '.jsonl': 'application/json',
        '.ply': 'application/octet-stream',
        '.splat': 'application/octet-stream'
    }

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/info"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            # List scenes in videos/ and work/
            v_dir = ROOT / "videos"
            w_dir = ROOT / "work"
            scenes = []
            if v_dir.exists():
                for p in v_dir.iterdir():
                    if p.is_dir() or p.suffix.lower() in (".mp4", ".mov", ".mkv"):
                        scenes.append(p.stem)
            if w_dir.exists():
                for p in w_dir.iterdir():
                    if p.is_dir() and p.name not in scenes:
                        scenes.append(p.name)

            data = {
                "local_ip": get_local_ip(),
                "port": port,
                "scenes": sorted(list(set(scenes))),
                "active_job": active_job_info
            }
            self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))
            return

        elif self.path.startswith("/api/status"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with process_lock:
                copy_info = dict(active_job_info)
            self.wfile.write(json.dumps(copy_info).encode("utf-8"))
            return

        elif self.path.startswith("/api/presets"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                from pipeline import PRESETS, QUALITY
                data = {"presets": PRESETS, "qualities": QUALITY}
            except Exception:
                data = {}
            self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))
            return

        return super().do_GET()

    def do_POST(self):
        global active_process
        if self.path.startswith("/api/run"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            req = json.loads(body.decode("utf-8")) if body else {}

            scene = safe_scene_name(req.get("scene", "room_w_jsonl"), "room_w_jsonl")
            preset = req.get("preset", "room")
            quality = req.get("quality", "high")
            extra_args = req.get("extra_args", [])

            with process_lock:
                if active_process and active_process.poll() is None:
                    self.send_response(409)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Job already running"}).encode("utf-8"))
                    return

            t = threading.Thread(target=run_pipeline_thread, args=(scene, preset, quality, extra_args), daemon=True)
            t.start()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started", "scene": scene, "preset": preset}).encode("utf-8"))
            return

        elif self.path.startswith("/api/kill"):
            with process_lock:
                if active_process:
                    try:
                        active_process.terminate()
                    except Exception:
                        pass
                    active_process = None
                    active_job_info["status"] = "cancelled"

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "killed"}).encode("utf-8"))
            return

        elif self.path.startswith("/api/upload"):
            # Simple multipart/form-data parser for mobile direct transfer
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Expected multipart/form-data")
                return

            boundary_match = re.search(
                r'(?:^|;)\s*boundary=(?:"([^"]+)"|([^;,\s]+))', content_type, re.I
            )
            if not boundary_match:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing multipart boundary")
                return
            boundary = (boundary_match.group(1) or boundary_match.group(2)).encode("ascii")
            content_length = int(self.headers.get("Content-Length", 0))
            raw_data = self.rfile.read(content_length)

            # Parse the simple browser FormData shape used by capture.html. The
            # CRLF immediately before a boundary is framing, not file data: drop
            # exactly that one pair, never rstrip() arbitrary file bytes.
            parts = []
            for raw_part in raw_data.split(b"--" + boundary):
                header_part, marker, data = raw_part.partition(b"\r\n\r\n")
                if not marker:
                    continue
                if data.endswith(b"\r\n"):
                    data = data[:-2]
                parts.append((header_part, data))

            # Query params or form field for scene name.
            scene_name = "mobile_capture"
            for header_part, data in parts:
                if b'name="scene"' in header_part:
                    scene_name = data.decode("utf-8", errors="replace").strip()
                    break

            # The name becomes a directory under videos/, so keep it to a single
            # safe path segment - no traversal, no separators, no drive letters.
            scene_name = safe_scene_name(scene_name)

            target_dir = ROOT / "videos" / scene_name
            target_dir.mkdir(parents=True, exist_ok=True)

            saved_files = []
            for header_part, file_data in parts:
                if b'filename="' in header_part:
                    fn_match = re.search(r'filename="([^"]+)"', header_part.decode("utf-8", errors="replace"))
                    if fn_match:
                        fname = Path(fn_match.group(1)).name
                        if not fname or fname in (".", ".."):
                            continue
                        out_path = target_dir / fname
                        out_path.write_bytes(file_data)
                        saved_files.append(fname)

            # Automatically remux WebM to standard MP4 with duration metadata & faststart seeking
            ff_exe = ROOT / "tools/ffmpeg/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe"
            webm_file = target_dir / "data.webm"
            mp4_file = target_dir / "data.mp4"
            if webm_file.exists() and ff_exe.exists():
                try:
                    cmd = [str(ff_exe), "-y", "-i", str(webm_file), "-c:v", "copy", "-movflags", "+faststart", str(mp4_file)]
                    subprocess.run(cmd, capture_output=True, timeout=15)
                    if mp4_file.exists() and mp4_file.stat().st_size > 0:
                        saved_files.append("data.mp4")
                except Exception as e:
                    print(f"[warn] ffmpeg remux error: {e}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "success",
                "scene": scene_name,
                "saved_files": saved_files,
                "target_dir": str(target_dir)
            }).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def ensure_certs(cert_file, key_file):
    if cert_file.exists() and key_file.exists():
        return
    import datetime
    import ipaddress
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.now(datetime.timezone.utc)
    ).not_valid_after(
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName('localhost'),
            x509.IPAddress(ipaddress.IPv4Address('127.0.0.1')),
            x509.IPAddress(ipaddress.IPv4Address(get_local_ip()))
        ]),
        critical=False,
    ).sign(key, hashes.SHA256())

    key_file.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()
    ))
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


port = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 8137
root = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else str(ROOT)
https_port = port + 1

# Generate SSL certs for HTTPS
cert_file = ROOT / '_cert.pem'
key_file = ROOT / '_key.pem'
try:
    ensure_certs(cert_file, key_file)
except Exception as e:
    print(f"[warn] Failed to generate SSL certs: {e}")

local_ip = get_local_ip()

print(f"===============================================================")
print(f" 🚀 Drone3D Studio Server running at:")
print(f"    - Local URL:      http://localhost:{port}/viewer/pc.html")
print(f"    - Phone HTTP:     http://{local_ip}:{port}/viewer/capture.html")
print(f"    - Phone HTTPS:    https://{local_ip}:{https_port}/viewer/capture.html  (CAMERA & SENSORS)")
print(f"    - Pipeline UI:    http://localhost:{port}/viewer/pipeline_gui.html")
print(f"===============================================================")

httpd = http.server.ThreadingHTTPServer(('0.0.0.0', port),
    functools.partial(H, directory=root))

def run_https():
    if cert_file.exists() and key_file.exists():
        try:
            httpd_ssl = http.server.ThreadingHTTPServer(('0.0.0.0', https_port),
                functools.partial(H, directory=root))
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
            httpd_ssl.socket = ctx.wrap_socket(httpd_ssl.socket, server_side=True)
            httpd_ssl.serve_forever()
        except Exception as e:
            print(f"[ssl server error] {e}")

t_https = threading.Thread(target=run_https, daemon=True)
t_https.start()

httpd.serve_forever()
