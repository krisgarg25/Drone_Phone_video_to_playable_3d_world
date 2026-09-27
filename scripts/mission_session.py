"""Multi-user rehearsal sessions with an instructor (RH-9), held in memory on the local server.

An instructor opens a session for a mission from this machine and gets a random join token.
Players' viewers - on this machine or on the LAN (a Quest browser, another laptop) - join
with that token, then post their pose about ten times a second and receive everyone else's
pose plus the instructor's commands since they last asked:

* ``move_bot`` {bot, to} - every client moves that enemy post (its local simulation of it);
* ``message`` {text} - shown on every player's HUD;
* ``end`` - every client ends its rehearsal.

What it is and is not: poses are shared; the enemy bots are simulated by each client
independently (a shot one player sees land is not replayed on another), so it is a shared
exercise, not a lock-step simulation. Nothing a LAN client sends is written to disk: the
session lives in memory, is bounded (players, commands, message length), and expires after
an hour idle. The token is the only thing that lets a non-local device post, and only to
``join`` and ``state``.
"""
import secrets
import threading
import time

MAX_PLAYERS = 16
MAX_COMMANDS = 500
IDLE_EXPIRY_S = 3600
LOST_AFTER_S = 5.0

_LOCK = threading.Lock()
_SESSIONS = {}


class SessionError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _now():
    return time.monotonic()


def _expire():
    cutoff = _now() - IDLE_EXPIRY_S
    for key in [k for k, s in _SESSIONS.items() if s["touched"] < cutoff]:
        del _SESSIONS[key]


def open_session(scene, mission, *, name="Exercise"):
    with _LOCK:
        _expire()
        sid = "s-" + secrets.token_hex(4)
        token = secrets.token_urlsafe(9)
        _SESSIONS[sid] = {"id": sid, "scene": scene, "mission": mission, "name": str(name)[:60], "token": token,
                          "players": {}, "commands": [], "seq": 0, "created": _now(), "touched": _now(), "bots": []}
        return {"session": sid, "token": token}


def _get(sid, token=None):
    s = _SESSIONS.get(sid)
    if s is None:
        raise SessionError(404, "No such session (it may have expired).")
    if token is not None and not secrets.compare_digest(str(token), s["token"]):
        raise SessionError(403, "Wrong session token.")
    return s


def token_ok(data):
    """True when a request carries a valid token for an open session (the LAN exception)."""
    try:
        with _LOCK:
            _get(str(data.get("session", "")), data.get("token"))
        return True
    except SessionError:
        return False


def join(sid, token, name, role="player"):
    with _LOCK:
        s = _get(sid, token)
        if role not in ("player", "observer"):
            raise SessionError(400, "role is player or observer")
        live = [p for p in s["players"].values() if _now() - p["seen"] < LOST_AFTER_S * 6]
        if len(live) >= MAX_PLAYERS:
            raise SessionError(409, f"The session is full ({MAX_PLAYERS} players).")
        pid = "p" + secrets.token_hex(3)
        s["players"][pid] = {"id": pid, "name": (str(name or "Player")[:24]).strip() or "Player", "role": role,
                             "pose": None, "health": None, "alive": True, "seen": _now(), "joined": _now()}
        s["touched"] = _now()
        return {"player": pid, "seq": s["seq"], "mission": s["mission"], "scene": s["scene"], "name": s["name"]}


def _finite(values, n):
    if not isinstance(values, list) or len(values) != n:
        return None
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    return out if all(abs(v) < 1e7 for v in out) else None


def report(sid, token, pid, *, pose=None, health=None, alive=True, since=0, bots=None):
    """Store one player's pose; return the others and the commands after ``since``."""
    with _LOCK:
        s = _get(sid, token)
        me = s["players"].get(pid)
        if me is None:
            raise SessionError(404, "Join the session first.")
        clean = _finite(pose, 4)
        if clean is not None:
            me["pose"] = [round(v, 3) for v in clean]
        if isinstance(health, (int, float)):
            me["health"] = max(0.0, min(100.0, float(health)))
        me["alive"] = bool(alive)
        me["seen"] = _now()
        s["touched"] = _now()
        if isinstance(bots, list) and len(bots) <= 64:
            rows = [_finite(b, 4) for b in bots]
            if all(r is not None for r in rows):
                s["bots"] = [[round(v, 2) for v in r] for r in rows]
                s["bots_from"] = pid
        others = [{"id": p["id"], "name": p["name"], "pose": p["pose"], "alive": p["alive"], "health": p["health"],
                   "lost": _now() - p["seen"] > LOST_AFTER_S}
                  for p in s["players"].values() if p["id"] != pid and p["role"] == "player" and p["pose"] is not None]
        commands = [c for c in s["commands"] if c["seq"] > int(since or 0)]
        return {"players": others, "commands": commands, "seq": s["seq"]}


def command(sid, cmd):
    """Instructor command (local machine only - enforced by the HTTP guard, not here)."""
    if not isinstance(cmd, dict):
        raise SessionError(400, "command must be an object")
    kind = cmd.get("type")
    if kind == "move_bot":
        to = _finite(cmd.get("to"), 3)
        if to is None or not isinstance(cmd.get("bot"), int) or not 0 <= cmd["bot"] < 64:
            raise SessionError(400, "move_bot needs bot (index) and to [x, y, z]")
        clean = {"type": kind, "bot": cmd["bot"], "to": [round(v, 3) for v in to]}
    elif kind == "message":
        text = str(cmd.get("text", "")).strip()[:140]
        if not text:
            raise SessionError(400, "message needs text")
        clean = {"type": kind, "text": text}
    elif kind == "end":
        clean = {"type": kind}
    else:
        raise SessionError(400, "command type is move_bot, message or end")
    with _LOCK:
        s = _get(sid)
        s["seq"] += 1
        clean["seq"] = s["seq"]
        clean["at"] = round(_now() - s["created"], 2)
        s["commands"].append(clean)
        s["commands"] = s["commands"][-MAX_COMMANDS:]
        s["touched"] = _now()
        return clean


def snapshot(sid):
    with _LOCK:
        s = _get(sid)
        return {"session": s["id"], "name": s["name"], "scene": s["scene"], "mission": s["mission"], "seq": s["seq"],
                "players": [{"id": p["id"], "name": p["name"], "role": p["role"], "pose": p["pose"], "alive": p["alive"],
                             "health": p["health"], "lost": _now() - p["seen"] > LOST_AFTER_S,
                             "seconds_since": round(_now() - p["seen"], 1)} for p in s["players"].values()],
                "bots": s["bots"], "bots_from": s.get("bots_from"),
                "commands": s["commands"][-20:],
                "note": "Poses are shared; each client simulates the enemy posts itself, so hits are not replayed "
                        "across clients."}


def sessions_for(scene, mission):
    with _LOCK:
        _expire()
        return [{"session": s["id"], "name": s["name"], "players": len(s["players"])}
                for s in _SESSIONS.values() if s["scene"] == scene and s["mission"] == mission]
