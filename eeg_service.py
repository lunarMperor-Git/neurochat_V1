"""NeuroChat EEG service: runs ON YOUR COMPUTER (not in Docker) so it can reach the Muse's Bluetooth.
The Open WebUI container asks it for your engagement z-score over HTTP.

  python eeg_service.py --token mysecret                  # Docker Desktop (mac/Windows)
  python eeg_service.py --host 0.0.0.0 --token mysecret   # Linux (see README)
"""
import argparse
import os
import threading
import time
from collections import deque

import numpy as np
from flask import Flask, jsonify, request
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes, NoiseTypes

class EEGMonitor:
    def __init__(self, board, mac, mains_hz, window_s=4.0, step_s=1.0, artifact_uv=150.0):
        ids = {"muse2": BoardIds.MUSE_2_BOARD, "muse2_bled": BoardIds.MUSE_2_BLED_BOARD,
               "synthetic": BoardIds.SYNTHETIC_BOARD}
        self.board_id = ids[board].value
        params = BrainFlowInputParams()
        if mac:
            params.mac_address = mac
        params.timeout = 20
        BoardShim.disable_board_logger()
        self.board = BoardShim(self.board_id, params)
        self.fs = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_ch = BoardShim.get_eeg_channels(self.board_id)
        self.mains = NoiseTypes.FIFTY.value if mains_hz == 50 else NoiseTypes.SIXTY.value
        self.window_n = int(window_s * self.fs)
        self.step_s, self.artifact_uv = step_s, artifact_uv
        self.samples = deque(maxlen=7200)  # (t, raw_engagement, z)
        self.mu = self.sd = None
        self.good_channels = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        self.board.prepare_session()
        self.board.start_stream(45000)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            self.board.stop_stream()
            self.board.release_session()
        except Exception:
            pass

    def _compute(self):
        data = self.board.get_current_board_data(self.window_n)
        if data.shape[1] < self.window_n:
            return None
        rows = []
        for ch in self.eeg_ch:
            x = np.ascontiguousarray(data[ch], dtype=np.float64)
            DataFilter.detrend(x, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(x, self.fs, 1.0, 45.0, 4, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.remove_environmental_noise(x, self.fs, self.mains)
            if np.ptp(x[int(self.fs):]) < self.artifact_uv:
                rows.append(x)
        self.good_channels = len(rows)
        if not rows:
            return None
        arr = np.vstack(rows)
        avg, _ = DataFilter.get_avg_band_powers(arr, list(range(arr.shape[0])), self.fs, False)
        theta, alpha, beta = avg[1], avg[2], avg[3]
        return float(beta / (alpha + theta)) if alpha + theta > 0 else None

    def _loop(self):
        while not self._stop.wait(self.step_s):
            try:
                e = self._compute()
            except Exception:
                continue
            if e is None or e <= 0:
                continue
            with self._lock:
                z = None if self.mu is None else (np.log(e) - self.mu) / self.sd
                self.samples.append([time.time(), e, z])

    def calibrate(self, seconds, progress=None):
        with self._lock:
            self.mu = None
        t0 = time.time()
        while time.time() - t0 < seconds:
            if progress:
                progress(int(seconds - (time.time() - t0)))
            time.sleep(1)
        with self._lock:
            vals = [np.log(s[1]) for s in self.samples if s[0] >= t0]
            if len(vals) < 10:
                raise RuntimeError(f"Only {len(vals)} clean EEG windows during baseline. "
                                   "Check headset fit (forehead + behind ears) and retry.")
            self.mu = float(np.mean(vals))
            self.sd = float(max(np.std(vals), 0.05))
            for s in self.samples:
                s[2] = (np.log(s[1]) - self.mu) / self.sd

    @property
    def calibrated(self):
        return self.mu is not None

    def z_since(self, t0, min_samples=5):
        with self._lock:
            vals = [s[2] for s in self.samples if s[0] >= t0 and s[2] is not None]
        return float(np.mean(vals)) if len(vals) >= min_samples else None

    def latest_z(self):
        with self._lock:
            return self.samples[-1][2] if self.samples else None


app = Flask(__name__)
ST = {"monitor": None, "board": None, "token": "",
      "calib": {"running": False, "left": 0, "error": None}}


@app.before_request
def auth():
    if ST["token"] and request.headers.get("X-Token") != ST["token"]:
        return jsonify(ok=False, message="bad token"), 401


@app.get("/status")
def status():
    m, c = ST["monitor"], ST["calib"]
    return jsonify(connected=bool(m), calibrated=bool(m and m.calibrated), calibrating=c["running"],
                   left=c["left"], error=c["error"], board=ST["board"],
                   z=(m.latest_z() if m else None), channels=(m.good_channels if m else 0),
                   total_channels=(len(m.eeg_ch) if m else 0), fs=(m.fs if m else None))


@app.post("/connect")
def connect():
    if ST["monitor"]:
        return jsonify(ok=True, message="Already connected.")
    j = request.get_json(silent=True) or {}
    board = j.get("board", "synthetic")
    try:
        m = EEGMonitor(board, j.get("mac") or None, int(j.get("mains", 50)))
        m.start()
    except Exception as e:
        return jsonify(ok=False, message=f"Could not connect ({board}): {e}")
    ST["monitor"], ST["board"] = m, board
    return jsonify(ok=True, message=f"Connected ({board}, {m.fs} Hz, {len(m.eeg_ch)} channels).")


@app.post("/calibrate")
def calibrate():
    m, c = ST["monitor"], ST["calib"]
    if not m:
        return jsonify(ok=False, message="Not connected.")
    if c["running"]:
        return jsonify(ok=False, message=f"Calibration already running ({c['left']}s left).")
    secs = int((request.get_json(silent=True) or {}).get("seconds", 60))

    def run():
        c.update(running=True, error=None, left=secs)
        try:
            m.calibrate(secs, progress=lambda s: c.update(left=s))
        except RuntimeError as e:
            c["error"] = str(e)
        c["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify(ok=True, message="Calibration started.")


@app.post("/stop")
def stop():
    m = ST["monitor"]
    if m:
        m.stop()
        ST["monitor"] = None
    return jsonify(ok=True, message="Headset disconnected.")


@app.get("/engagement")
def engagement():
    m = ST["monitor"]
    if not m:
        return jsonify(connected=False, calibrated=False, z=None)
    window = float(request.args.get("window_s", 30))
    fallback = float(request.args.get("fallback_s", 30))
    z = None
    if m.calibrated:
        z = m.z_since(time.time() - window)
        if z is None:
            z = m.z_since(time.time() - fallback)
    return jsonify(connected=True, calibrated=m.calibrated, z=z)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", default=os.environ.get("NEUROCHAT_TOKEN", ""))
    a = ap.parse_args()
    if a.host not in ("127.0.0.1", "localhost") and not a.token:
        raise SystemExit("Refusing to listen beyond localhost without --token (your EEG data would be open to the network).")
    ST["token"] = a.token
    BoardShim.disable_board_logger()
    print(f"EEG service on http://{a.host}:{a.port}")
    try:
        app.run(host=a.host, port=a.port, threaded=True, debug=False)
    finally:
        if ST["monitor"]:
            ST["monitor"].stop()
