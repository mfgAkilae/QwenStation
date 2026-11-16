# Qwen Station

One exe that installs and runs Qwen3.8 27B on a single RTX 5090, serves an OpenAI-compatible API and shows
what the model is doing.

- **Model:** Qwen3.8 27B as NVFP4 (Gittensor build for the RTX 5090), FP8 KV cache, DFlash2 drafter for
  speculative decoding
- **Engine:** vLLM 0.32.0 in its own WSL distribution, with five small patches and verified kernel tuning
- **API:** `http://127.0.0.1:8000/v1`, model `qwen38-gittensor-dflash`, no API key
- **Speed on the test machine:** 296 to 323 tok/s on short prompts, 280 to 288 tok/s at 33K context,
  204 to 260 tok/s at 129K context; up to 262,144 tokens of context

![The dashboard while the model answers a request](docs/dashboard.png)

## Requirements

- Windows 11, or Windows 10 from version 21H2
- NVIDIA RTX 5090 (32 GB) with a current driver. The model file and the kernels are built for this card.
- 45 GB of free space on a built-in drive, ideally an SSD
- Internet for the setup: about 30 GB of downloads
- WSL 2 from version 2.4.4. The setup installs or updates WSL if needed (see below).

## Install

1. Download `Qwen-Station-<version>-windows-x64.zip` from the releases and unpack it.
2. Start `Qwen-Station.exe`. The file is not signed, so Windows SmartScreen warns: "More info", then
   "Run anyway".
3. The window lists what the machine needs. Choose a folder for engine and model and click **Set up**.

![The setup view](docs/setup.png)

The setup then runs without further questions:

| Step | What happens |
|---|---|
| App | Copies itself to `%LOCALAPPDATA%\Programs\Qwen Station`, adds Start menu and desktop shortcuts and an entry under Installed apps. No administrator rights needed. |
| WSL | Checks WSL. If it is missing, runs `wsl --install --no-distribution` (Windows asks for administrator rights). If Windows needs a restart, the app offers it and continues by itself after the next sign-in. An old WSL is updated. |
| Linux environment | Creates a WSL distribution named `QwenStation` in the chosen folder. Other distributions are not touched. |
| Engine | Installs a compiler, uv, Python 3.12, vLLM with PyTorch, the CUDA compiler, the patches and prebuilt GPU kernels. |
| Model | Downloads the model (17.9 GB) and the drafter (2.3 GB) from Hugging Face, pinned to fixed revisions. |

On a fast connection (about 35 MB/s) the whole setup took 7.5 minutes. Every step can be repeated: a cancelled or
failed setup continues where it stopped.

After the setup the model starts by itself. The first start takes two to three minutes, later starts 45 to 80 seconds.

## Use

- **OpenCode and Pi:** the app registers the model at every start (`~/.config/opencode/opencode.json` and
  `~/.pi/agent/models.json`, provider `vllm`). Before its first change it writes a backup next to the file.
- **T3 Code:** choose the provider OpenCode or Pi and the model `vllm/qwen38-gittensor-dflash`. If Pi does
  not list it, add it as a custom model with exactly this slug.
- **Other clients:** base URL `http://127.0.0.1:8000/v1`, any API key (for example `local`).
- **Reasoning effort:** `low`, `medium` or `xhigh`. This checkpoint knows no other levels.
- vLLM rejects `min_p` and `logit_bias` with HTTP 400 while speculative decoding is active.

Closing the window stops the model and frees the VRAM. A minimized window keeps it running. Starting the
exe a second time only shows the window again.

### What the window shows

- Decode speed live (tok/s, averaged over 3 seconds) with a history of 5, 15 or 30 minutes
- GPU: VRAM, utilization, power, temperature
- Time to first token, KV cache fill level, prefix cache hit rate, tokens per draft round, preemptions
- The last requests: prompt tokens, cached tokens, output tokens, TTFT, speed, finish reason
- Draft acceptance per position of the DFlash2 drafter
- The server log, with a filter for warnings and errors

All values come from the `/metrics` endpoint of the vLLM server. Nothing sits in the request path: the API
is the unmodified vLLM endpoint.

### Free VRAM decides the context size

The card has 31.8 GiB; with the full context of 262,144 tokens the model takes about 30 GiB.

| Free at start | Behavior |
|---|---|
| 29.3 GiB or more | Full context |
| less | The context shrinks in steps of 32,768 tokens until the model fits. The window says so, and OpenCode and Pi get the smaller limit. |
| too little for 65,536 tokens | No automatic start. The window lists the programs that hold the most GPU memory. |

The reason: if the model does not fit, Windows pages GPU memory and the speed drops to a quarter (measured:
prefill 1,825 instead of 8,800 tok/s). Close other GPU programs before starting to get the full context.

## Settings

Optional file `%LOCALAPPDATA%\QwenStation\qwen-station.json` (the setup creates it). All keys are optional:

```json
{
  "preset": "262k",
  "close_action": "stop",
  "autostart": true,
  "lan": false,
  "fit_context": true,
  "api_port": 8000,
  "station_port": 8010,
  "extra_args": "",
  "extra_env": ""
}
```

- `preset`: `262k` (default), `237k` (about 0.8 GB more VRAM headroom), `parallel-96k` (two slots,
  experimental) or `unsloth-82k` (Unsloth's NVFP4 build with the same drafter; needs
  `provision.sh model-unsloth` inside the distribution first).
- `close_action`: `stop` ends the model when the window closes, `keep` leaves it running until **Quit**.
- `lan`: `true` binds the API to `0.0.0.0` and sets up the Windows port forwarding and firewall rule for the
  private network (asks for administrator rights once). The API has no authentication, so the default is local only.
- `fit_context`: `false` switches the automatic context reduction off.
- `extra_args`: appended to `vllm serve`. `extra_env`: `KEY=VALUE` pairs for the start script, for example
  `XQA_DECODE=0`.

The same values work as command line options: `Qwen-Station.exe --preset 237k --keep --no-autostart`.
`--check` prints the state of the installation as JSON and exits.

## Remove

- **App:** Windows Settings, Apps, Installed apps, uninstall "Qwen Station".
- **Engine and model** (about 32 GB): `wsl --unregister QwenStation` in PowerShell.
- **Settings and logs:** delete `%LOCALAPPDATA%\QwenStation`.

## How it is built

Plain Python (3.10 or newer), standard library only. The exe is a PyInstaller one-file build.

| Path | Purpose |
|---|---|
| `qwen_station/app.py` | HTTP API for the dashboard, polling, window handling, entry point |
| `qwen_station/setup.py` | First-run setup: requirements, WSL, the distribution, progress of the provisioning |
| `qwen_station/installer.py` | Installs and removes the app itself (per user, no administrator rights) |
| `qwen_station/supervisor.py` | Preflight, start inside WSL, startup phases from the log, stop |
| `qwen_station/metrics.py` | Parses `/metrics`; speed and request rows from counter differences |
| `qwen_station/gpu.py` | `nvidia-smi` readings, GPU memory per Windows process |
| `qwen_station/clients.py` | Entries for OpenCode and Pi |
| `qwen_station/dashboard.html` | The dashboard: one file, no external resources |
| `qwen_station/engine/provision.sh` | Installs the engine below one prefix. Idempotent, also runs on native Linux. |
| `qwen_station/engine/serve.sh` | Starts `vllm serve` with the tested settings |
| `qwen_station/engine/patches/` | Five idempotent patches for vLLM; each file explains its reason |
| `qwen_station/engine/autotune/` | Verified FlashInfer kernel tuning |
| `qwen_station/engine/engine.env` | Version pins: vLLM, Python, uv, CUDA packages, model revisions |

The reasons behind the engine settings, with the measurements, are in [docs/engine-notes.md](docs/engine-notes.md).

The dashboard listens on `127.0.0.1` only. Start, stop, setup and quit need a token that exists only in the
served page, so other websites cannot control the app.

### Develop

```powershell
python run-station.py --mock            # fake server instead of the model: the dashboard without a GPU
python run-station.py --check           # state of the installation as JSON
python -m unittest discover -s tests    # 51 tests; they start the mock server as a real subprocess
.\build-exe.ps1 -Release                # dist\Qwen-Station.exe and the release zip with its SHA256 file
```

`build-exe.ps1` creates its own venv in `.build\` and installs PyInstaller there.

### Linux

The engine bundle does not depend on Windows: `provision.sh` and `serve.sh` find everything relative to their
install prefix and run on a native Linux with an NVIDIA driver. The app around them (setup, window,
installer) is Windows-only for now.

## Tested, and not tested

Tested on 2026-11-17 on one machine (Windows 11, RTX 5090, WSL 2.7.3) with the release exe: uninstall,
fresh setup into a new distribution, first start, benchmark, stop.

Not tested on a real machine: the path where WSL is missing or outdated (covered by unit tests with a faked
WSL only), Windows 10, the `lan` option of the installed app, the preset `parallel-96k` on the GPU.
