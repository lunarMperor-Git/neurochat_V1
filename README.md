# NeuroChat: Open WebUI in Docker + EEG service on the host

Why split: Docker can't reach your Muse's Bluetooth, so the EEG part runs on your computer and
the container asks it for your engagement score over HTTP.

    Muse 2 --BLE--> eeg_service.py (host, :8765) <--HTTP-- Open WebUI (Docker, :3000) --> Ollama (host, :11434)

## 1. Host: Ollama
    ollama pull llama3.2:3b
Linux only: Ollama must listen beyond localhost so the container can reach it (set OLLAMA_HOST=0.0.0.0
for the service, e.g. `sudo systemctl edit ollama`, then restart) and firewall port 11434.

## 2. Host: EEG service
    pip install -r requirements.txt
    python eeg_service.py --token mysecret                  # macOS / Windows (Docker Desktop)
    python eeg_service.py --host 0.0.0.0 --token mysecret   # Linux (firewall port 8765 from outside)

## 3. Docker: Open WebUI
    docker compose up -d
Open http://localhost:3000, create the local admin account.

## 4. Add the function
Admin Panel -> Functions -> + -> paste `neurochat_docker_pipe.py` -> save -> enable.
Gear icon (valves): set EEG_TOKEN to your token; BOARD=synthetic first, then muse2; MAINS_HZ 50 or 60.

## 5. Use it
New chat -> pick NeuroChat -> `/connect`, `/calibrate`, then chat. `/help` lists commands.

## Troubleshooting
- "Cannot reach the EEG service": service not running, wrong port, or (Linux) not started with --host 0.0.0.0.
- "rejected the token": EEG_TOKEN valve differs from --token.
- Ollama error: model not pulled, or (Linux) Ollama not listening on 0.0.0.0.
- Test from the host: `curl -H "X-Token: mysecret" http://127.0.0.1:8765/status`
