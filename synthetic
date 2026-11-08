"""
title: NeuroChat (Synthetic EEG)
author: neurochat
version: 0.3.0
description: NeuroChat-style closed loop using SYNTHETIC EEG. Starts each chat with the paper's two-task calibration, then opens every reply with the engagement score and what the model will do because of it.
"""
import asyncio
import json
import re
import time
from typing import Optional

import httpx
import numpy as np
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paper pipeline (Baradari et al., CUI '25). Muse 2: 4 channels at 256 Hz.
#  - band-pass 1-30 Hz, 60 Hz notch
#  - 1 s epochs every 250 ms, FFT power spectrum, band power
#  - E = beta / (alpha + theta), theta 4-7, alpha 7-11, beta 11-20 Hz
#  - average E over a 15 s sliding window (10 s during calibration)
#  - E_norm = (E - E_min) / (E_max - E_min), E_min/E_max from a 2 min relaxation
#    task and a 2 min word-association task
# Assumptions where the paper is silent: FFT-mask filtering, Hann window per epoch,
# band power = summed PSD over bins in [lo, hi), averaged across the 4 channels,
# E_norm clipped to [0, 1].
# ---------------------------------------------------------------------------
FS = 256
N_CH = 4
MAINS_HZ = 60.0
BANDS = {"theta": (4.0, 7.0), "alpha": (7.0, 11.0), "beta": (11.0, 20.0)}
E0, E1 = 0.2, 2.0          # rough shape of E vs raw level (exact mapping is measured at calibration)
BG_AT, BG_B = 10.6, 6.3    # background power in alpha+theta and in beta


def preprocess(x, fs=FS):
    """x: (channels, samples) -> zero-phase band-pass 1-30 Hz plus 60 Hz notch."""
    X = np.fft.rfft(x, axis=1)
    f = np.fft.rfftfreq(x.shape[1], 1.0 / fs)
    keep = (f >= 1.0) & (f <= 30.0)
    keep &= ~((f >= MAINS_HZ - 1) & (f <= MAINS_HZ + 1))  # redundant here, kept to mirror the paper
    return np.fft.irfft(X * keep, n=x.shape[1], axis=1)


def epoch_engagement(x, fs=FS, epoch_s=1.0, hop_s=0.25):
    """Preprocessed (ch, samples) -> (epoch end times in s, E per epoch)."""
    n, hop = int(epoch_s * fs), int(hop_s * fs)
    starts = np.arange(0, x.shape[1] - n + 1, hop)
    segs = np.stack([x[:, s:s + n] for s in starts]) * np.hanning(n)  # (epochs, ch, n)
    psd = np.abs(np.fft.rfft(segs, axis=2)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    bp = {k: psd[:, :, (f >= lo) & (f < hi)].sum(axis=2).mean(axis=1) for k, (lo, hi) in BANDS.items()}
    return (starts + n) / fs, bp["beta"] / (bp["alpha"] + bp["theta"])


def window_mean_E(x_raw, fs=FS):
    """Raw window -> E averaged over all epochs in the window."""
    return float(epoch_engagement(preprocess(x_raw, fs), fs)[1].mean())


def sliding_mean_E(x_raw, window_s, step_s=1.0, fs=FS):
    """Sliding-window average of E, one value per step (used for calibration)."""
    ends, E = epoch_engagement(preprocess(x_raw, fs), fs)
    total, out, t = x_raw.shape[1] / fs, [], window_s
    while t <= total + 1e-9:
        m = (ends > t - window_s) & (ends <= t + 1e-9)
        if m.any():
            out.append(E[m].mean())
        t += step_s
    return np.array(out)


# ------------------------------------------------------- synthetic "Muse 2" ----
def _shaped_noise(n, rng, lo=None, hi=None, pink=False, fs=FS):
    F = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    if pink:
        F[1:] /= np.sqrt(f[1:])
        F[0] = 0
    else:
        F = F * ((f >= lo) & (f < hi))
    x = np.fft.irfft(F, n=n)
    return x / (x.std() + 1e-12)


class SyntheticMuse:
    """Fake 4-channel 256 Hz EEG. A hidden TARGET level in [0,1] decides the band amplitudes (higher ->
    more beta, a little less alpha/theta). After calibration the target is calibrated so that the paper
    pipeline's normalized score lands near it; the score itself always comes from the full pipeline."""

    def __init__(self, seed=7):
        self.seed, self.mode, self.fixed = seed, "auto", 0.5
        self.curve = None  # (grid, E table) measured at calibration
        self.cal = None    # (E_min, E_max)

    def effective(self, target):
        """Map a TARGET normalized level (0-1) to the raw band-amplitude level that the pipeline will turn
        into about that score, using the E-vs-level curve measured at calibration."""
        if self.curve is None:
            return target
        grid, Et = self.curve
        lo, hi = self.cal
        return np.interp(lo + np.asarray(target) * (hi - lo), Et, grid)

    @staticmethod
    def _amps(L):
        """Band amplitudes (uV rms) for raw level L. Alpha/theta power fall a little as L rises; beta power
        rises so E climbs roughly linearly from E0 to E1. BG_* is the power the 1/f background adds."""
        a_p, t_p = 70 - 20 * L, 45 - 10 * L
        b_p = np.maximum((E0 + (E1 - E0) * L) * (a_p + t_p + BG_AT) - BG_B, 1.0)
        return np.sqrt(t_p), np.sqrt(a_p), np.sqrt(b_p)

    def level(self, t):
        """Latent engagement level at absolute times t (seconds): smooth drift, or fixed."""
        if self.mode == "fixed":
            return np.full(t.shape, self.fixed)
        k0, k1 = int(np.floor(t.min() / 20)), int(np.floor(t.max() / 20)) + 1
        va = np.array([np.random.default_rng([self.seed, k]).uniform(0.05, 0.95) for k in range(k0, k1 + 1)])
        kk = np.floor(t / 20).astype(int)
        w = (1 - np.cos(np.pi * (t / 20 - kk))) / 2
        return va[kk - k0] * (1 - w) + va[kk - k0 + 1] * w

    def signal(self, t0, dur, level=None):
        n = int(round(dur * FS))
        t = t0 + np.arange(n) / FS
        L = self.effective(self.level(t)) if level is None else level
        th_a, al_a, be_a = self._amps(L)
        rng = np.random.default_rng()
        out = np.empty((N_CH, n))
        for c in range(N_CH):
            x = 6.0 * _shaped_noise(n, rng, pink=True)              # 1/f background
            x += th_a * _shaped_noise(n, rng, 4, 7)                 # theta
            x += al_a * _shaped_noise(n, rng, 7, 11)                # alpha
            x += be_a * _shaped_noise(n, rng, 11, 20)               # beta
            x += 2.0 * np.sin(2 * np.pi * MAINS_HZ * t + rng.uniform(0, 6.28))  # mains hum
            out[c] = rng.uniform(0.8, 1.2) * x
        return out


def calibration_signals(muse, secs):
    """Synthetic version of the paper's two calibration tasks (relaxation, then word association),
    scored with a 10 s sliding window. Returns the two score arrays."""
    t = np.arange(secs * FS) / FS
    relax = muse.signal(0, secs, level=0.10 + 0.05 * np.sin(2 * np.pi * t / 17))
    words = muse.signal(0, secs, level=0.90 + 0.05 * np.sin(2 * np.pi * t / 13))
    return sliding_mean_E(relax, 10), sliding_mean_E(words, 10)


def finish_calibration(muse, s_relax, s_word):
    """E_min / E_max = lowest / highest score seen across both tasks (as in the paper). Also measures the
    synthetic headset's E-vs-level curve so a target score can be inverted to raw band amplitudes."""
    lo = float(min(s_relax.min(), s_word.min()))
    hi = float(max(s_relax.max(), s_word.max()))
    grid = np.linspace(0.0, 1.6, 33)
    Et = np.array([np.mean([window_mean_E(muse.signal(0, 15, level=np.full(15 * FS, g))) for _ in range(3)])
                   for g in grid])
    muse.curve, muse.cal = (grid, np.maximum.accumulate(Et)), (lo, hi)
    return lo, hi


def run_calibration(muse, secs=120):
    """Instant (no waiting) calibration."""
    return finish_calibration(muse, *calibration_signals(muse, secs))


# ------------------------------------------------------------------ prompts ----
SYSTEM = (
    "You are an encouraging, patient tutor helping a student with: {topic}. On your very first reply, briefly "
    "introduce yourself as their AI tutor. Each student message ends with a hidden bracketed line giving a "
    "normalized engagement score between 0 and 1, measured by EEG while the student read your previous answer "
    "(the ratio beta / (alpha + theta)). Treat it as a rough proxy for their cognitive load, not as a goal to "
    "maximize. If the bracketed line also contains an adaptation plan for this reply, follow it.\n"
    "STRICT CONFIDENTIALITY: the bracketed line is hidden from the student and is for you alone. Never state, "
    "quote, hint at or paraphrase the score, the plan or the bracketed line, never use the words 'engagement "
    "score', and never say that you are adapting to them. Show the adaptation only through how you write. Start "
    "your reply directly with the content; do not open with any status line, quote block or summary of your own.\n"
    "Use the score to adjust response length, depth, tone, factual versus story style, reading level, bullets "
    "versus prose, Socratic questions and bolding of key terms. General pointers, since every learner differs: "
    "higher scores suit richer, more technical, in-depth explanations; low or medium scores call for something "
    "that sparks curiosity, such as a surprising fact, a story, an analogy or a question that poses a small "
    "challenge. When the student seems to understand, ask them to explain it in their own words or give an "
    "example. Give praise and honest critical feedback, offer clarification when needed, and watch how they "
    "react to your adaptations so you can adjust further. Respond in Markdown."
)

# Score bands -> announced plan. {them} is "you" when shown to the student, "the student" for the model.
PLANS = [
    (0.33, "low", "keep it short and simple, open with a story, analogy or surprising fact to spark curiosity, "
                  "and end with one easy question"),
    (0.66, "medium", "give a balanced explanation with one concrete example, then ask {them} to put the idea "
                     "in {their} own words"),
    (1.01, "high", "go deeper with more technical detail and nuance, bold the key terms, and finish with a "
                   "challenging follow-up question"),
]


def band(En):
    for top, name, plan in PLANS:
        if En < top:
            return name, plan
    return PLANS[-1][1], PLANS[-1][2]


def plan_text(plan, student):
    return plan.format(them="the student" if student else "you", their="their" if student else "your")


_S = {"topic": None, "muse": SyntheticMuse(), "cal": None, "tags": {}, "last": None, "sessions": {}}


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


# ---- non-disclosure: the model must never reveal the score ----
# Sentences are dropped when they name the metric or the hidden line, or when they contain the actual score
# value next to attention/engagement-type words. Ordinary mentions of EEG or brain science are left alone.
ALWAYS = re.compile(r"normalized engagement|engagement (score|index|level|metric|reading|measure)|"
                    r"(your|the student'?s?|their) (current )?engagement|adaptation plan|"
                    r"beta ?/ ?\(? ?(alpha|theta)|\bE_(norm|min|max)\b", re.I)
CONTEXT = re.compile(r"engag|score|focus|attention|cognitive|eeg|brain", re.I)


def _drain(buf, final=False):
    """Split buf into complete sentences/lines plus an unfinished remainder."""
    parts = re.split(r"([.!?][\]\)\"'\u201d\u2019*_]*\s+|\n+)", buf)   # also ends after .] .) ." etc.
    rest = "" if final else parts.pop()
    return ["".join(parts[i:i + 2]) for i in range(0, len(parts), 2)], rest


class LeakGuard:
    def __init__(self, score=None):
        self.buf, self.removed, self.active = "", 0, score is not None
        if self.active:
            pct = int(round(score * 100))
            forms = dict.fromkeys([f"{score:.2f}", f".{pct:02d}" if pct < 100 else "1.00",
                                   f"{score:.1f}", f"{pct}%", f"{pct} percent"])
            self.nums = re.compile(r"(?<![\d.])(?:" + "|".join(re.escape(f) for f in forms) + r")(?!\d)", re.I)

    def _emit(self, segs):
        out = []
        for seg in segs:
            if ALWAYS.search(seg) or (self.nums.search(seg) and CONTEXT.search(seg)):
                self.removed += 1
                out.append("\n" if seg.endswith("\n") else "")
            else:
                out.append(seg)
        return out

    def feed(self, piece):
        if not self.active:
            return [piece]
        self.buf += piece
        segs, self.buf = _drain(self.buf)
        return self._emit(segs)

    def flush(self):
        if not self.active:
            return []
        segs, _ = _drain(self.buf, final=True)
        self.buf = ""
        return self._emit(segs)


def _clean_history(content):
    """Strip what THIS app added to earlier replies (calibration text, announcement block, reasoning), so the
    model never sees them in its history and never imitates them."""
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.S)
    content = re.sub(r"<details[^>]*>.*?</details>\s*", "", content, flags=re.S)
    content = re.sub(r"\A### Calibration.*?\n---\n\n", "", content, flags=re.S)
    content = re.sub(r"\A(?:> .*\n)+\n", "", content)
    return content.strip()


class Pipe:
    class Valves(BaseModel):
        OLLAMA_URL: str = Field("http://host.docker.internal:11434", description="Ollama on the host")
        MODEL: str = Field("qwen3.5:9b", description="Ollama model name (must match `ollama list`)")
        THINKING: str = Field("off", description="off = answer directly (recommended for Qwen3.5), on = show its reasoning first (also leak-filtered), auto = model default")
        TOPIC: str = Field("general learning", description="What the tutor teaches")
        SKIP_CALIBRATION: bool = Field(True, description="True = no calibration test at the start of a chat (an instant synthetic baseline is used silently). False = run the paper's two-task test first")
        CALIB_TASK_SECONDS: int = Field(120, description="Length of EACH calibration task. 120 = the paper's 2 min. Minimum 20 (use 20 for a quick test)")
        EXPLICIT_PLAN: bool = Field(True, description="Announce the plan at the top of each reply AND give it to the model. Off = paper-style: the model gets only the score")
        SHOW_DEBUG: bool = Field(True, description="Add a small line with raw E, calibration range and the synthetic target")
        SYNTH_SEED: int = Field(7, description="Seed for the synthetic engagement drift")
        LOG_PATH: str = Field("/app/backend/data/neurochat_synth_log.jsonl", description="Per-turn log ('' to disable)")

    def __init__(self):
        self.valves = self.Valves()

    # ---- synthetic EEG -> engagement (CPU work; callers run it in a thread) ----
    def _calibrate(self):
        _S["muse"].seed = self.valves.SYNTH_SEED
        _S["cal"] = run_calibration(_S["muse"])
        _S["last"] = None
        return _S["cal"]

    def _score(self):
        muse = _S["muse"]
        muse.seed = self.valves.SYNTH_SEED
        if _S["cal"] is None:
            self._calibrate()
        lo, hi = _S["cal"]
        now = time.time()
        E = window_mean_E(muse.signal(now - 15.0, 15.0))          # previous 15 s
        En = float(np.clip((E - lo) / (hi - lo), 0.0, 1.0))
        latent = float(muse.level(np.array([now - 7.5]))[0])
        _S["last"] = (E, En, latent)
        return E, En, latent, lo, hi

    @staticmethod
    async def _status(emit, text, done=False):
        if emit:
            try:
                await emit({"type": "status", "data": {"description": text, "done": done}})
            except Exception:
                pass

    # ---- the paper's calibration: 2 tasks, lowest/highest score become E_min/E_max ----
    async def _calibration_protocol(self, emit):
        v, muse = self.valves, _S["muse"]
        secs = max(20, int(v.CALIB_TASK_SECONDS))
        muse.seed = v.SYNTH_SEED
        dur = f"{secs // 60} min" if secs % 60 == 0 else f"{secs} s"
        yield ("### Calibration\n"
               "As in the NeuroChat paper, two short tasks set your personal minimum and maximum engagement "
               "before we start. The EEG here is synthetic, so the waiting is only for realism "
               f"(each task is {dur}; shorten it with the CALIB_TASK_SECONDS valve).\n\n")
        tasks = [
            ("relaxation", "Sit still, relax, and keep your mind as quiet as you can. This sets your *minimum*."),
            ("word association", "Silently think of a word, then a word that starts with the last letter of the "
                                 "previous one (elephant, then tiger, then ...), and keep going without repeating. "
                                 "This sets your *maximum*."),
        ]
        step = 30 if emit else 15
        for i, (name, text) in enumerate(tasks, 1):
            yield f"**Task {i} of 2: {name} ({dur}).** {text}\n\n"
            for left in range(secs, 0, -1):
                await self._status(emit, f"Calibration {i}/2 - {name} - {left // 60}:{left % 60:02d} left")
                if left != secs and left % step == 0:
                    yield f"{left} s left...\n\n"
                await asyncio.sleep(1)
            yield "Done.\n\n"
        await self._status(emit, "Calibration: working out your range")
        s_relax, s_word = await asyncio.to_thread(calibration_signals, muse, secs)
        lo, hi = await asyncio.to_thread(finish_calibration, muse, s_relax, s_word)
        _S["cal"], _S["last"] = (lo, hi), None
        await self._status(emit, "Calibration complete", done=True)
        yield (f"Relaxation: E averaged {s_relax.mean():.2f} (lowest {s_relax.min():.2f}). "
               f"Word association: E averaged {s_word.mean():.2f} (highest {s_word.max():.2f}).\n\n"
               f"**E_min = {lo:.2f}, E_max = {hi:.2f}.** From now on every reply opens with your engagement "
               "score and what I will do because of it.\n\n---\n\n")

    async def _calibrate_stream(self, emit, quick, sid):
        if quick:
            lo, hi = await asyncio.to_thread(self._calibrate)
            _S["sessions"][sid] = True
            yield f"Quick calibration on synthetic data (no waiting): E_min = {lo:.2f}, E_max = {hi:.2f}."
            return
        async for piece in self._calibration_protocol(emit):
            yield piece
        _S["sessions"][sid] = True

    # ---- commands ----
    async def _command(self, text):
        cmd, _, arg = text.partition(" ")
        muse, arg = _S["muse"], arg.strip().lower()
        if cmd == "/help":
            return ("**NeuroChat (synthetic EEG) commands**\n- `/calibrate` run the two-task calibration again "
                    "(`/calibrate quick` skips the waiting)\n"
                    "- `/synth low|mid|high|auto|0.0-1.0` set the fake headset's hidden target engagement\n"
                    "- `/status` show calibration and the latest scores\n- `/topic <text>` set the topic")
        if cmd == "/topic":
            _S["topic"] = text.partition(" ")[2].strip()[:200] or None
            return f"Topic set to: {_S['topic'] or self.valves.TOPIC}"
        if cmd == "/synth":
            if arg in ("", "status"):
                return f"Synthetic level mode: {muse.mode}" + (f" ({muse.fixed:.2f})" if muse.mode == "fixed" else " (slow random drift)")
            if arg == "auto":
                muse.mode = "auto"
                return "Synthetic engagement now drifts on its own."
            val = {"low": 0.1, "mid": 0.5, "high": 0.9}.get(arg)
            try:
                val = float(arg) if val is None else val
            except ValueError:
                return "Use `/synth low`, `mid`, `high`, `auto`, or a number from 0 to 1."
            muse.mode, muse.fixed = "fixed", float(np.clip(val, 0.0, 1.0))
            return f"Synthetic target fixed at {muse.fixed:.2f}. The score comes from the last 15 s, so the next prompt already reflects it (expect some noise around the target)."
        if cmd == "/status":
            cal, last = _S["cal"], _S["last"]
            return (f"Calibration: {'not done yet' if cal is None else f'E_min={cal[0]:.2f}, E_max={cal[1]:.2f}'}\n"
                    f"Last score: {'none yet' if last is None else f'E={last[0]:.3f}, normalized={last[1]:.2f}, target={last[2]:.2f}'}\n"
                    f"Mode: {muse.mode}")
        return "Unknown command. Try `/help`."

    # ---- chat ----
    async def pipe(self, body: dict, __user__: Optional[dict] = None,
                   __event_emitter__=None, __metadata__: Optional[dict] = None):
        msgs = [{"role": x["role"], "content": (_clean_history(_text(x.get("content")))
                                                if x["role"] == "assistant" else _text(x.get("content")))}
                for x in body.get("messages", []) if x.get("role") in ("user", "assistant")]
        if not msgs:
            return ""
        last = msgs[-1]["content"].strip()
        if last.startswith("### Task:"):  # Open WebUI background jobs (titles, tags): pass through untouched
            return self._stream(msgs[-1:], None)
        chat_id = (__metadata__ or {}).get("chat_id")
        sid = chat_id or "global"
        if last.startswith("/calibrate"):
            return self._calibrate_stream(__event_emitter__, "quick" in last.lower(), sid)
        if last.startswith("/"):
            return await self._command(last)
        n_user = sum(1 for m in msgs if m["role"] == "user")
        # A new chat runs the calibration first. Mid-chat with no calibration on record (e.g. after a restart)
        # does NOT start a multi-minute test; it falls back to an instant calibration.
        run_protocol = n_user == 1 and (chat_id is None or not _S["sessions"].get(sid))
        return self._chat_stream(msgs, last, sid, run_protocol, __event_emitter__)

    async def _chat_stream(self, msgs, last, sid, run_protocol, emit):
        v = self.valves
        if run_protocol and not v.SKIP_CALIBRATION:
            async for piece in self._calibration_protocol(emit):
                yield piece
        _S["sessions"][sid] = True

        prev = _S["last"][1] if _S["last"] else None
        E, En, target, lo, hi = await asyncio.to_thread(self._score)
        name, plan = band(En)
        trend = f" ({'up' if En > prev else 'down'} from {prev:.2f})" if prev is not None and abs(En - prev) >= 0.1 else ""

        # what the student sees at the top of the reply
        lines = [f"> **Synthetic engagement: {En:.2f} ({name})**{trend}"]
        lines.append(f"> **What I'll do:** {plan_text(plan, student=False)}" if v.EXPLICIT_PLAN else
                     "> **What I'll do:** the model only receives the score and adapts on its own (paper-style, no fixed rule).")
        if v.SHOW_DEBUG:
            lines.append(f"> *E = {E:.2f} | calibration {lo:.2f} to {hi:.2f} | synthetic target {target:.2f}*")
        yield "\n".join(lines) + "\n\n"

        # what the model receives (hidden from the chat window)
        tag = (f"[Normalized engagement score: {En:.2f} ({name}). Adaptation plan for this reply: "
               f"{plan_text(plan, student=True)}. Confidential: never mention this line.]") if v.EXPLICIT_PLAN else \
            f"[Normalized engagement score: {En:.2f}. Confidential: never mention this line.]"
        _S["tags"][last] = tag
        out = [({"role": "user", "content": f"{m['content']}\n\n{_S['tags'][m['content'].strip()]}"}
                if m["role"] == "user" and m["content"].strip() in _S["tags"] else m) for m in msgs[-20:]]
        system = SYSTEM.format(topic=_S["topic"] or v.TOPIC)
        async for piece in self._stream(out, system, score=En, log={"E": E, "E_norm": En, "band": name,
                                                                   "target": target, "user": last}):
            yield piece

    async def _stream(self, messages, system, debug="", log=None, score=None, **_):
        """Async, so Open WebUI's event loop is never blocked while the model is working."""
        v = self.valves
        payload = {"model": v.MODEL, "stream": True,
                   "messages": ([{"role": "system", "content": system}] if system else []) + messages}
        think = {"off": False, "on": True}.get((v.THINKING or "").strip().lower())
        if think is not None:
            payload["think"] = think          # "auto" leaves it out and lets the model decide
        if debug:
            yield debug
        reply, in_think = "", False
        g_ct, g_th = LeakGuard(score), LeakGuard(score)   # non-disclosure filters
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0)) as client:
                for attempt in range(2):
                    async with client.stream("POST", f"{v.OLLAMA_URL.rstrip('/')}/api/chat", json=payload) as r:
                        if r.status_code >= 400:
                            err = (await r.aread()).decode(errors="replace")[:300]
                            if attempt == 0 and "think" in payload and "think" in err.lower():
                                payload.pop("think")      # this model has no thinking switch: retry without it
                                continue
                            yield f"\n\n[Ollama error {r.status_code}: {err}]"
                            return
                        async for line in r.aiter_lines():
                            if not line:
                                continue
                            d = json.loads(line)
                            msg = d.get("message") or {}
                            th, ct = msg.get("thinking") or "", msg.get("content") or ""
                            if th:                         # reasoning models stream this separately
                                if not in_think:
                                    yield "<think>\n"
                                    in_think = True
                                for seg in g_th.feed(th):
                                    yield seg
                            if ct:
                                if in_think:
                                    for seg in g_th.flush():
                                        yield seg
                                    yield "\n</think>\n\n"
                                    in_think = False
                                for seg in g_ct.feed(ct):
                                    reply += seg
                                    yield seg
                            if d.get("done"):
                                break
                    break
            if in_think:
                for seg in g_th.flush():
                    yield seg
                yield "\n</think>\n\n"
            for seg in g_ct.flush():
                reply += seg
                yield seg
            if not reply.strip():
                yield ("\n\n[The model returned no answer text. If this repeats, set the THINKING valve to `on` "
                       "or `auto`, or try another model.]")
        except httpx.ConnectError:
            yield (f"\n\n[Cannot reach Ollama at {v.OLLAMA_URL}. Is the Ollama app running? "
                   "From Docker the address should be http://host.docker.internal:11434]")
            return
        except Exception as e:
            yield f"\n\n[Ollama error: {type(e).__name__}: {e}]"
            return
        if log and v.LOG_PATH:
            try:
                with open(v.LOG_PATH, "a") as f:
                    f.write(json.dumps(log | {"t": time.time(), "reply": reply,
                                              "leaks_removed": g_ct.removed + g_th.removed}) + "\n")
            except Exception:
                pass