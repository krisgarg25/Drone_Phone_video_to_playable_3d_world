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
import time
import urllib.parse
import uuid
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

active_process = None
active_job_info = {"status": "idle", "scene": "", "step": "", "logs": []}
process_lock = threading.Lock()
port = 8137
https_port = 8138
# The server binds 0.0.0.0 so a phone can reach the capture page, which means a
# survey write has to prove it came from this machine: only these peers may write.
LOOPBACK_PEERS = ("127.0.0.1", "::1")


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


# ---------------------------------------------------------------------------
# What is on disk. The dashboard's scene list and its terminal both read the
# runner's own artifacts instead of this server's memory, so a run started from
# a plain terminal shows up exactly like one the dashboard started.
# ---------------------------------------------------------------------------
# The runner's own verdict, written as the last line of each attempt. `[exit N]`
# is the legacy form and still has to parse, because logs from older runs are on
# disk and the dashboard reads whatever is there, not what it remembers.
FOOTER_OK = re.compile(r"^\[ok(?: (\d+))?\]\s+([0-9.]+)s\s*$")
FOOTER_EXIT = re.compile(r"^\[exit (-?\d+)(?: (\S+))?\]\s+([0-9.]+)s\s*$")
RUNNING_WINDOW_S = 120.0


def _log_files(logs_dir: Path) -> list:
    if not logs_dir.is_dir():
        return []
    return sorted((p for p in logs_dir.iterdir()
                   if p.is_file() and p.suffix == ".log" and p.name[:2].isdigit()),
                  key=lambda p: p.name)


def _step_row(log: Path, now: float) -> dict:
    stem = log.stem
    row = {"i": int(stem[:2]), "name": stem.split("-", 1)[1] if "-" in stem else stem,
           "status": "pending", "secs": None, "exit": None,
           "attempts": None, "kind": None}
    try:
        with log.open("rb") as fh:
            fh.seek(0, io.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 256))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return row
    last = [ln for ln in tail.splitlines() if ln.strip()]
    line = last[-1] if last else ""
    m = FOOTER_OK.match(line)
    if m:
        row["exit"] = 0
        row["attempts"] = int(m.group(1) or 1)
        row["secs"] = float(m.group(2))
        # A step that only passed on its second or third rung is done, but the
        # rung is the interesting part - hiding it would report a clean pass on a
        # scene that took a fallback to survive.
        row["status"] = "recovered" if row["attempts"] > 1 else "done"
    else:
        m = FOOTER_EXIT.match(line)
        if m:
            row["exit"] = int(m.group(1))
            row["kind"] = m.group(2)
            row["secs"] = float(m.group(3))
            row["status"] = "done" if row["exit"] == 0 else "failed"
        elif size and now - log.stat().st_mtime < RUNNING_WINDOW_S:
            row["status"] = "running"
        elif size:
            row["status"] = "interrupted"
    return row


def scan_scenes(root: Path, now: float = None) -> list:
    now = time.time() if now is None else now
    work = Path(root) / "work"
    videos = Path(root) / "videos"
    work_names = {p.name for p in work.iterdir() if p.is_dir()} if work.is_dir() else set()
    video_names = set()
    if videos.is_dir():
        for path in videos.iterdir():
            if path.is_dir() and any(p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi") for p in path.iterdir()):
                video_names.add(path.name)
            elif path.is_file() and path.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"):
                video_names.add(path.stem)
    out = []
    for name in sorted(work_names | video_names):
        d = work / name
        logs = _log_files(d / "logs")
        steps = [_step_row(p, now) for p in logs]
        reg = None
        poses = d / "keyframes_poses.jsonl"
        if poses.is_file():
            def _lines(p):
                with p.open(encoding="utf-8", errors="replace") as fh:
                    return sum(1 for ln in fh if ln.strip())
            kf = d / "keyframes.jsonl"
            reg = [_lines(poses), _lines(kf) if kf.is_file() else None]
        mtimes = [p.stat().st_mtime for p in logs] or [d.stat().st_mtime if d.is_dir() else now]
        out.append({
            "name": d.name,
            "has_work": name in work_names,
            "has_video": name in video_names,
            "viewable": (d / "viewer_assets" / "scene.ply").is_file(),
            "trained": (d / "splat.ply").is_file(),
            "registered": reg,
            "running": any(s["status"] == "running" for s in steps),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(max(mtimes))),
            "steps": steps,
        })
    return out


def tail_run(root: Path, scene: str, cursor: str = "0", limit: int = 1 << 20) -> dict:
    """One merged stream over a run's logs, in step order, resumable by cursor.

    The cursor is "<logfile>:<bytes>". A poll finishes the file the cursor names
    and, only once that file is drained, continues into the next one from byte 0,
    so the client sees keyframes -> colmap -> train as a single terminal. A half
    written last line is held back rather than shown broken, and an offset past
    the end of a file (a --fresh run replaced it) restarts that file. The budget
    is a MiB per poll: enough for a reload mid-run to catch the live edge at once.
    """
    logs = _log_files(Path(root) / "work" / safe_scene_name(scene) / "logs")
    names = {p.name for p in logs}
    cur_file, cur_off = None, 0
    if cursor and cursor != "0":
        fname, _, off = cursor.partition(":")
        if fname in names:
            cur_file = fname
            cur_off = int(off) if off.isdigit() else 0
    lines, budget = [], limit
    pos_file, pos_off = None, 0
    started = cur_file is None
    for p in logs:
        if not started:
            if p.name != cur_file:
                continue
            started, off = True, cur_off
        else:
            off = 0
        size = p.stat().st_size
        if off > size:
            off = 0
        take = min(budget, size - off)
        if take <= 0:
            pos_file, pos_off = p.name, size
            continue
        with p.open("rb") as fh:
            fh.seek(off)
            chunk = fh.read(take)
        text = chunk.decode("utf-8", errors="replace")
        held = 0 if text.endswith("\n") else len(text.rsplit("\n", 1)[-1].encode("utf-8", "replace"))
        body = text if not held else text[:-len(text.rsplit("\n", 1)[-1])]
        # Windows children write CRLF, so the \r has to go or it lands in the terminal.
        lines.extend(ln.rstrip("\r") for ln in body.split("\n")[:-1])
        pos_file, pos_off = p.name, off + take - held
        budget -= take
        if pos_off < size:
            break                      # still growing: do not jump to the next step
    running = bool(logs) and _step_row(logs[-1], time.time())["status"] == "running"
    return {"cursor": f"{pos_file}:{pos_off}" if pos_file else "0",
            "current": pos_file, "lines": lines, "running": running}


def job_busy_locked():
    """Call while holding process_lock, including the pre-spawn reservation."""
    return (active_job_info.get("status") == "running"
            or active_process is not None and active_process.poll() is None)


def reserve_job_locked(scene, preset, quality, action="run", engine="pipeline"):
    global active_job_info
    token = uuid.uuid4().hex
    active_job_info = {"status": "running", "scene": scene, "preset": preset,
                       "quality": quality, "action": action, "engine": engine,
                       "step": "initializing", "logs": [], "job_id": token}
    return token


def spawn_pipeline_job(scene, preset, quality, extra_args, **options):
    try:
        threading.Thread(target=run_pipeline_thread,
                         args=(scene, preset, quality, extra_args),
                         kwargs=options, daemon=True).start()
    except Exception:
        with process_lock:
            if active_job_info.get("job_id") == options.get("reservation"):
                active_job_info["status"] = "error: Could not start job worker"
        raise


def terminate_process_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        # Target this PID and descendants, never an image name (other Python/
        # COLMAP jobs may belong to the operator). Windows has no killpg.
        result = subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=20, check=False)
        if result.returncode and proc.poll() is None:
            raise OSError("Could not terminate the reconstruction process tree.")
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def cancel_job(scene=None):
    with process_lock:
        if scene is not None and active_job_info.get("scene") != scene:
            return False
        if not job_busy_locked():
            return active_job_info.get("status") == "cancelled"
        if active_process is not None:
            terminate_process_tree(active_process)
        # Do not clear active_process: the reservation remains busy until the
        # worker reaps it, and its exit must not turn cancellation into failure.
        active_job_info["status"] = "cancelled"
        active_job_info["step"] = "cancelled"
        return True


def run_pipeline_thread(scene: str, preset: str, quality: str, extra_args: list,
                        *, root=None, action="run", engine="pipeline",
                        dense_profile="survey", reservation=None):
    global active_process
    import workspace_api as workspace
    root = Path(root or ROOT).resolve()
    py_exe = ROOT / ".venv" / "Scripts" / "python.exe"
    py_exe = str(py_exe) if py_exe.exists() else sys.executable
    if engine == "survey":
        cmd = [py_exe, str(root / "survey.py"), "reconstruct", scene,
               "--allow-gpu", "--dense-profile", dense_profile]
    else:
        cmd = [py_exe, str(root / "pipeline.py"), action, scene,
               "--preset", preset, "--quality", quality] + extra_args
    proc = None
    started = time.monotonic()
    try:
        with process_lock:
            if reservation is None:
                if job_busy_locked():
                    return
                reservation = reserve_job_locked(scene, preset, quality, action, engine)
            if active_job_info.get("job_id") != reservation or active_job_info.get("status") != "running":
                return
            log_dir = workspace.safe_path(root, "work/" + scene + "/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = workspace.safe_path(root, "work/" + scene + "/logs/00-workspace_" + ("survey" if engine == "survey" else action) + ".log")
            active_job_info["cmd"] = " ".join([Path(cmd[0]).name, Path(cmd[1]).name] + cmd[2:])
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n[workspace] starting " + engine + " " + action + "\n")
            log.flush()
            try:
                with process_lock:
                    if active_job_info.get("job_id") != reservation or active_job_info.get("status") != "running":
                        log.write("[exit -1 cancelled] 0.0s\n")
                        return
                    proc = subprocess.Popen(cmd, cwd=str(root), stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                            errors="replace", bufsize=1,
                                            **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                                               if os.name == "nt" else {"start_new_session": True}))
                    active_process = proc
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    with process_lock:
                        if active_job_info.get("job_id") != reservation:
                            continue
                        active_job_info["logs"].append(line.rstrip())
                        active_job_info["logs"] = active_job_info["logs"][-300:]
                        stage_match = re.search(r"\[\d+/\d+\]\s+([a-z_]+):|\[survey\]\s+([a-z_]+)", line)
                        if stage_match and active_job_info["status"] != "cancelled":
                            active_job_info["step"] = stage_match.group(1) or stage_match.group(2)
                proc.wait()
                with process_lock:
                    cancelled = active_job_info.get("job_id") == reservation and active_job_info.get("status") == "cancelled"
                    if active_job_info.get("job_id") == reservation and not cancelled:
                        active_job_info["status"] = "completed" if proc.returncode == 0 else f"failed (code {proc.returncode})"
                log.write(f"[exit {-1 if cancelled else proc.returncode}{' cancelled' if cancelled else ''}] {time.monotonic() - started:.3f}s\n")
            except Exception:
                log.write(f"[exit -1 error] {time.monotonic() - started:.3f}s\n")
                raise
    except Exception as error:
        print(f"[workspace worker] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        if proc is not None and proc.poll() is None:
            try:
                terminate_process_tree(proc)
                proc.wait(timeout=20)
            except (OSError, subprocess.SubprocessError):
                pass
        with process_lock:
            if active_job_info.get("job_id") == reservation and active_job_info.get("status") != "cancelled":
                active_job_info["status"] = "error: Reconstruction failed; inspect the server log"
    finally:
        if proc is not None and proc.stdout:
            proc.stdout.close()
        with process_lock:
            if active_job_info.get("job_id") == reservation and (proc is None or proc.poll() is not None):
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

    def end_headers(self):
        # Chrome on Android keeps an HTML file in disk cache unless the response
        # says otherwise, so a phone that once opened /viewer/capture.html keeps
        # replaying that byte-for-byte copy after a redeploy. no-store forces
        # every reload to hit this server. Only added if the response has not
        # already declared its own cache policy, so an API that wants to opt out
        # can still call self.send_header("Cache-Control", ...) itself.
        buf = getattr(self, "_headers_buffer", None) or []
        if not any(b"Cache-Control" in line for line in buf):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    _scenes_cache = (0.0, None)

    def _json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _scenes(self):
        now = time.time()
        ts, data = H._scenes_cache
        if data is None or now - ts > 2.0:
            data = scan_scenes(Path(self.directory), now)
            H._scenes_cache = (now, data)
        return {"scenes": data}

    def _survey_json(self, data, status=200):
        payload = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _survey_refused(self):
        """Why this survey write must be refused, or None when it may proceed.

        The previous guard read `if origin and origin != Host`, so a client that is
        not a browser - which sends no Origin at all - walked straight past it, and
        Host is whatever the caller typed. Both halves are required now: the TCP
        peer has to be this machine, and a present Origin has to name it.
        """
        peer = self.client_address[0] if self.client_address else ""
        if peer not in LOOPBACK_PEERS:
            return "Survey writes are accepted only from this machine."
        origin = self.headers.get("Origin")
        if not origin:
            return "Survey writes require a browser Origin header."
        parsed = urllib.parse.urlparse(origin)
        scheme = "https" if isinstance(self.connection, ssl.SSLSocket) else "http"
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.scheme != scheme:
            return "Survey writes require a trusted local origin."
        if parsed.netloc.lower() != self.headers.get("Host", "").lower():
            return "Cross-origin survey writes are not allowed."
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username:
            return "Invalid survey Origin header."
        return None

    def _consume_request_body(self, cap=4 << 20):
        """Read the declared body once, so no reply races still-unread bytes.

        Writing a response while the body is in flight makes Winsock reset the
        connection and the client never sees the refusal. The cap exists so a
        forged Content-Length cannot make the handler hold gigabytes: a request
        past it is refused as oversized.
        """
        try:
            remaining = min(max(int(self.headers.get("Content-Length") or 0), 0), cap)
        except ValueError:
            remaining = 0
        chunks = []
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 1 << 16))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    _ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|(?<![\w.])/(?:\w+/?)+)")

    def _survey_failure(self, status, error):
        """Client sees an actionable reason with no local paths; log keeps detail.

        ValueError text is authored by the survey workflow and names relative
        artifacts ("keyframes_poses.jsonl is missing"), which the operator needs
        to fix the request. Anything else (OSError from a stat, a stray KeyError)
        can carry an absolute path, so it collapses to the generic reason.
        """
        print(f"[survey] {status} {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        if status == 409:
            message = "Survey evidence already exists for this scene."
        elif isinstance(error, ValueError):
            message = str(error)[:300]
            if self._ABSOLUTE_PATH.search(message):
                message = "Survey request failed; the server log holds the detail."
        else:
            message = "Survey request failed; the server log holds the detail."
        self._survey_json({"error": message, "error_class": type(error).__name__}, status)

    def _survey(self, action=None):
        sys.path.insert(0, str(ROOT / "scripts"))
        import survey_workflow as survey
        try:
            if action is None:
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                result = survey.scene_status(Path(self.directory), query.get("scene", [""])[0])
            else:
                raw = self._consume_request_body()
                refused = self._survey_refused()
                if refused:
                    self._survey_json({"error": refused}, 403)
                    return
                actions = {"inputs": survey.save_inputs, "prepare": survey.prepare_scene,
                           "align": survey.align_scene, "evaluate": survey.evaluate_scene}
                if action not in actions:
                    self._survey_json({"error": "Unknown survey action. GPU execution is available only through the explicit CLI gate."}, 404)
                    return
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected application/json.")
                if not raw or len(raw) > survey.MAX_INPUT_BYTES:
                    raise ValueError("Supply a JSON request below 2 MiB.")
                request = json.loads(raw.decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object.")
                scene = request.get("scene", "")
                arguments = [Path(self.directory), scene]
                if action == "inputs":
                    arguments += [request.get("telemetry_csv"), request.get("metadata")]
                with process_lock:
                    if job_busy_locked():
                        self._survey_json({"error": "Wait for the running reconstruction before changing survey evidence."}, 409)
                        return
                    result = actions[action](*arguments)
            self._survey_json(result)
        except FileExistsError as error:
            self._survey_failure(409, error)
        except (ValueError, KeyError, TypeError, OSError) as error:
            self._survey_failure(400, error)

    def _workspace(self):
        import workspace_api
        workspace_api.handle(self, sys.modules[__name__])

    def do_HEAD(self):
        if urllib.parse.urlparse(self.path).path.startswith("/api/workspace/"):
            self._workspace()
            return
        return super().do_HEAD()

    def do_GET(self):
        if urllib.parse.urlparse(self.path).path.startswith("/api/workspace/"):
            self._workspace()
            return
        if urllib.parse.urlparse(self.path).path == "/api/survey":
            # Read-only status stays LAN-reachable because the dashboard iframe and
            # the phone viewer both render it. Every write route passes the guard.
            self._survey()
            return
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
                "https_port": https_port,
                "dashboard": "/viewer/pipeline_gui.html",
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

        elif self.path.startswith("/api/scenes"):
            self._json(self._scenes())
            return

        elif self.path.startswith("/api/tail"):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            self._json(tail_run(Path(self.directory), qs.get("scene", [""])[0],
                                qs.get("cursor", ["0"])[0]))
            return

        return super().do_GET()

    def do_POST(self):
        global active_process
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/api/workspace/"):
            self._workspace()
            return
        if path.startswith("/api/survey/"):
            self._survey(path.removeprefix("/api/survey/"))
            return
        if path == "/api/run":
            import workspace_api
            from pipeline import PRESETS, QUALITY
            try:
                req = workspace_api.json_body(self)
                preset = req.get("preset", "room")
                quality = req.get("quality", "high")
                extra_args = req.get("extra_args", [])
                if (not isinstance(req.get("scene", "room_w_jsonl"), str)
                        or not isinstance(preset, str) or preset not in PRESETS
                        or not isinstance(quality, str) or quality not in QUALITY
                        or not isinstance(extra_args, list) or len(extra_args) > 128
                        or any(not isinstance(arg, str) or len(arg) > 4096 for arg in extra_args)):
                    raise workspace_api.Error(400, "Invalid reconstruction options.")
                scene = safe_scene_name(req.get("scene", "room_w_jsonl"), "room_w_jsonl")
            except workspace_api.Error as error:
                self.close_connection = True
                self._survey_json({"error": str(error)}, error.status)
                return

            with process_lock:
                if job_busy_locked():
                    self.send_response(409)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Job already running"}).encode("utf-8"))
                    return
                token = reserve_job_locked(scene, preset, quality)

            spawn_pipeline_job(scene, preset, quality, extra_args, reservation=token)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started", "scene": scene, "preset": preset}).encode("utf-8"))
            return

        elif self.path.startswith("/api/kill"):
            try:
                cancel_job()
            except (OSError, subprocess.SubprocessError):
                pass

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


def main() -> None:
    global port, https_port
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
    print(f"    - Dashboard:      http://localhost:{port}/viewer/pipeline_gui.html")
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


if __name__ == "__main__":
    main()
