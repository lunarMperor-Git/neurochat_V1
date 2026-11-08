"""
title: NeuroChat (Docker)
author: neurochat
version: 0.1.0
description: EEG-adaptive tutor. Gets engagement from a host-side EEG service, chats with a local Ollama model.

   docker compose up -d
   docker ps
"""
import json
import time
from typing import Optional

import requests
from pydantic import BaseModel, Field

BASE = ("You are NeuroChat, a patient and knowledgeable tutor. Topic of the session: {topic}. "
        "Keep answers focused. Never mention EEG, brain signals, or that you are adapting to the user.")
ADAPT = {
    "low": ("The learner seems to be losing focus. Keep this reply SHORT (under ~120 words). Use simple "
            "words, one concrete everyday analogy, and end with one small, easy question to re-engage them."),
    "mid": ("The learner is moderately engaged. Give a clear, balanced explanation with one example, "
            "then offer a natural next direction."),
    "high": ("The learner is highly engaged. Go deeper: add nuance, precise terminology, a challenging "
             "follow-up idea, and end with a thought-provoking question."),
    "unknown": "Give a clear, balanced explanation.",
}

_S = {"topic": None, "reading_since": time.time()}


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class Pipe:
    class Valves(BaseModel):
        OLLAMA_URL: str = Field("http://host.docker.internal:11434", description="Ollama on the host")
        MODEL: str = Field("llama3.2:3b", description="Ollama model name")
        TOPIC: str = Field("general learning", description="What the tutor is teaching")
        EEG_URL: str = Field("http://host.docker.internal:8765", description="Host EEG service")
        EEG_TOKEN: str = Field("", description="Must match --token given to eeg_service.py")
        BOARD: str = Field("synthetic", description="synthetic (fake EEG) | muse2 | muse2_bled")
        MUSE_MAC: str = Field("", description="Optional Muse Bluetooth MAC")
        MAINS_HZ: int = Field(50, description="50 or 60")
        CALIB_SECONDS: int = Field(60, description="Resting baseline length")
        LOW_Z: float = Field(-0.5, description="z at/below this = low engagement")
        HIGH_Z: float = Field(0.5, description="z at/above this = high engagement")
        SHOW_STATE: bool = Field(True, description="Prefix each reply with the detected state")
        LOG_PATH: str = Field("/app/backend/data/neurochat_log.jsonl", description="Per-turn log ('' to disable)")

    def __init__(self):
        self.valves = self.Valves()

    # ---- talk to the host EEG service ----
    def _eeg(self, method, path, timeout=5, **kw):
        v = self.valves
        try:
            r = requests.request(method, v.EEG_URL.rstrip("/") + path, timeout=timeout,
                                 headers={"X-Token": v.EEG_TOKEN}, **kw)
            if r.status_code == 401:
                return None, "EEG service rejected the token (check the EEG_TOKEN valve)."
            return r.json(), None
        except Exception as e:
            return None, (f"Cannot reach the EEG service at {v.EEG_URL} ({e}). "
                          "Is `python eeg_service.py` running on your computer?")

    def _command(self, text):
        cmd, _, arg = text.partition(" ")
        v = self.valves
        if cmd == "/help":
            return ("**NeuroChat commands**\n- `/connect` connect the headset\n- `/calibrate` record resting baseline\n"
                    "- `/status` show signal state\n- `/topic <text>` set the topic\n- `/stop` disconnect the headset")
        if cmd == "/topic":
            _S["topic"] = arg.strip()[:200] or None
            return f"Topic set to: {_S['topic'] or v.TOPIC}"
        if cmd == "/connect":
            j, err = self._eeg("POST", "/connect", timeout=90,
                               json={"board": v.BOARD, "mac": v.MUSE_MAC, "mains": v.MAINS_HZ})
            return err or (j["message"] + (" Now run `/calibrate`." if j.get("ok") else ""))
        if cmd == "/stop":
            j, err = self._eeg("POST", "/stop")
            return err or j["message"]
        if cmd == "/calibrate":
            j, err = self._eeg("POST", "/calibrate", json={"seconds": v.CALIB_SECONDS})
            if err or not j.get("ok"):
                return err or j["message"]
            return (f"Baseline started: sit still, eyes open, and read something neutral for "
                    f"{v.CALIB_SECONDS}s. Type `/status` to check progress.")
        if cmd == "/status":
            j, err = self._eeg("GET", "/status")
            if err:
                return err
            if not j["connected"]:
                return "EEG service is up, but no headset connected. Use `/connect`."
            if j["calibrating"]:
                return f"Calibrating... {j['left']}s left."
            if j["error"]:
                return j["error"]
            if not j["calibrated"]:
                return "Connected but not calibrated. Use `/calibrate`."
            z = j["z"]
            return (f"Calibrated. Latest z = {'n/a' if z is None else f'{z:+.2f}'}; "
                    f"clean channels {j['channels']}/{j['total_channels']}.")
        return "Unknown command. Try `/help`."

    # ---- chat ----
    def pipe(self, body: dict, __user__: Optional[dict] = None):
        v = self.valves
        msgs = [{"role": x["role"], "content": _text(x.get("content"))}
                for x in body.get("messages", []) if x.get("role") in ("user", "assistant")]
        if not msgs:
            return ""
        last = msgs[-1]["content"].strip()

        # Open WebUI sends background jobs (titles, tags) through the selected model; don't adapt those.
        if last.startswith("### Task:"):
            return self._stream(msgs[-1:], None)
        if last.startswith("/"):
            return self._command(last)

        z, active = None, False
        j, _ = self._eeg("GET", "/engagement", timeout=3,
                         params={"window_s": max(5.0, time.time() - _S["reading_since"]), "fallback_s": 30})
        if j and j.get("connected") and j.get("calibrated"):
            active, z = True, j.get("z")
        state = "unknown" if z is None else "low" if z <= v.LOW_Z else "high" if z >= v.HIGH_Z else "mid"
        system = BASE.format(topic=_S["topic"] or v.TOPIC) + " " + ADAPT[state]
        return self._stream(msgs[-20:], system, state=state, z=z, user=last, show=active)

    def _stream(self, messages, system, state=None, z=None, user=None, show=False):
        v = self.valves
        payload = {"model": v.MODEL, "stream": True,
                   "messages": ([{"role": "system", "content": system}] if system else []) + messages}
        if show and v.SHOW_STATE:
            tag = "no clean signal" if z is None else f"z={z:+.2f}"
            yield f"*[engagement: {state} ({tag})]*\n\n"
        reply = ""
        try:
            r = requests.post(f"{v.OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=300)
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                piece = d.get("message", {}).get("content", "")
                if piece:
                    reply += piece
                    yield piece
                if d.get("done"):
                    break
        except Exception as e:
            yield f"\n\n[Ollama error: {e}. Is it running, reachable from Docker, and is `{v.MODEL}` pulled?]"
            return
        if state:
            _S["reading_since"] = time.time()
            if v.LOG_PATH:
                try:
                    with open(v.LOG_PATH, "a") as f:
                        f.write(json.dumps({"t": time.time(), "z": z, "state": state,
                                            "user": user, "reply": reply}) + "\n")
                except Exception:
                    pass
