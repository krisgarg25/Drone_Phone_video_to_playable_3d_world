"""Multi-user rehearsal sessions (RH-9): tokens, poses, instructor commands, bounds."""
import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.s = importlib.import_module("mission_session")
        self.opened = self.s.open_session("site", "p-0123456789", name="Night exercise")

    def test_two_players_see_each_other_and_the_instructor(self):
        sid, tok = self.opened["session"], self.opened["token"]
        a = self.s.join(sid, tok, "Alpha")["player"]
        b = self.s.join(sid, tok, "Bravo")["player"]
        self.s.report(sid, tok, a, pose=[1, 2, 3, 0.5])
        out = self.s.report(sid, tok, b, pose=[10, 2, 3, 1.0], bots=[[5, 0, 5, 0]])
        self.assertEqual([p["name"] for p in out["players"]], ["Alpha"])
        self.assertEqual(out["players"][0]["pose"], [1, 2, 3, 0.5])
        cmd = self.s.command(sid, {"type": "move_bot", "bot": 0, "to": [7, 0, 7]})
        self.s.command(sid, {"type": "message", "text": "Hold at PL BLUE"})
        got = self.s.report(sid, tok, a, since=0)["commands"]
        self.assertEqual([c["type"] for c in got], ["move_bot", "message"])
        self.assertEqual(self.s.report(sid, tok, a, since=got[-1]["seq"])["commands"], [])
        snap = self.s.snapshot(sid)
        self.assertEqual(len(snap["players"]), 2)
        self.assertEqual(snap["bots"], [[5.0, 0.0, 5.0, 0.0]])
        self.assertEqual(cmd["seq"], 1)

    def test_token_and_input_guards(self):
        sid, tok = self.opened["session"], self.opened["token"]
        self.assertTrue(self.s.token_ok({"session": sid, "token": tok}))
        self.assertFalse(self.s.token_ok({"session": sid, "token": "nope"}))
        self.assertFalse(self.s.token_ok({"session": "s-deadbeef", "token": tok}))
        with self.assertRaises(self.s.SessionError):
            self.s.join(sid, "wrong", "Eve")
        with self.assertRaises(self.s.SessionError):
            self.s.report(sid, tok, "p-unknown", pose=[0, 0, 0, 0])
        with self.assertRaises(self.s.SessionError):
            self.s.command(sid, {"type": "launch_missiles"})
        pid = self.s.join(sid, tok, "X" * 100)["player"]
        self.assertEqual(len(self.s.snapshot(sid)["players"][-1]["name"]), 24)
        self.s.report(sid, tok, pid, pose=["a", 0, 0, 0])            # junk pose ignored, not stored
        self.assertIsNone(self.s.snapshot(sid)["players"][-1]["pose"])

    def test_session_is_full_at_the_cap(self):
        sid, tok = self.opened["session"], self.opened["token"]
        for k in range(self.s.MAX_PLAYERS):
            self.s.join(sid, tok, f"P{k}")
        with self.assertRaises(self.s.SessionError):
            self.s.join(sid, tok, "one too many")


if __name__ == "__main__":
    unittest.main()
