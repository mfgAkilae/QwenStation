Qwen Station
============

Starts Qwen3.8 27B (NVFP4, FP8 KV cache, DFlash2 drafter) on vLLM, shows live monitoring and serves an
OpenAI-compatible API: http://127.0.0.1:8000/v1, model qwen38-gittensor-dflash.

Requirements
------------
- Windows 11, or Windows 10 from version 21H2
- NVIDIA RTX 5090 (32 GB) with a current driver
- 45 GB of free space on a built-in drive, ideally an SSD
- Internet for the setup (about 30 GB of downloads)
- WSL 2 from version 2.4.4. If WSL is missing or too old, the setup installs or updates it. Windows then
  asks for administrator rights and may need one restart; Qwen Station continues by itself afterwards.

Installation
------------
1. Start Qwen-Station.exe. Windows SmartScreen reports an unknown app because the file is not signed:
   click "More info", then "Run anyway".
2. The window lists what the machine needs. Choose the folder for engine and model, then click "Set up".
3. Wait. The app installs itself (Start menu, desktop), creates its own Linux environment "QwenStation"
   in WSL and downloads vLLM and the model. You can cancel; the setup continues where it stopped.
   On a fast connection the whole setup takes about 8 minutes.
4. The model then starts by itself. The first start takes two to three minutes, later ones about one.

You can delete the downloaded file afterwards. The app lives in %LOCALAPPDATA%\Programs\Qwen Station
and starts from the Start menu.

Use
---
- The app registers the model in OpenCode and Pi at every start (provider vllm, model
  qwen38-gittensor-dflash). In T3 Code choose the provider OpenCode or Pi and the model
  vllm/qwen38-gittensor-dflash.
- Other clients: base URL http://127.0.0.1:8000/v1, any API key (for example "local").
- Closing the window stops the model and frees the VRAM.

Remove
------
- App: Windows Settings, Apps, Installed apps, uninstall "Qwen Station".
- Engine and model (about 32 GB): in PowerShell run "wsl --unregister QwenStation".
- Settings and logs: delete the folder %LOCALAPPDATA%\QwenStation.

Licenses: vLLM and the models are Apache 2.0 and are downloaded during the setup from PyPI and
Hugging Face (gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090, TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm).
