# Voice and optional models

## Enable the microphone

Mira transcribes speech on your PC using Parakeet and Silero VAD. Short commands such as `click`, `next`, and `scroll down` are interpreted directly. Try them in **Practice** first.

1. Download the [Parakeet v3 INT8 archive](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2) (487 MB) and [Silero VAD](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx) from Sherpa-ONNX. Upstream licenses are [CC BY 4.0 for Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and [MIT for Silero VAD](https://github.com/snakers4/silero-vad/blob/master/LICENSE).
2. Extract the archive with `tar -xf sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2`, or get the four Parakeet files individually from the [pinned Sherpa-ONNX mirror](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tree/876ff91b4ab4b89c328afdb2b27ff879d3e42f87). Copy the files into Mira's data directory (`.local` for a source checkout, `%LOCALAPPDATA%\Mira` for a package install).

   ```text
   models/
     parakeet/
       encoder.int8.onnx
       decoder.int8.onnx
       joiner.int8.onnx
       tokens.txt
     silero_vad.onnx
   ```

3. Allow Mira to start local models. See [Local model access](#local-model-access).

4. In **Overview → More controls**, choose **Turn on microphone**. Mira uses the system microphone by default. To select a different input, use **Settings → Advanced → Voice and recognition**, save, and turn the microphone on again.

## Local model access

Mira starts a local model (speech recognition, the webcam runtime, target
reranking or screen recognition) or sends a command to a model server on this
PC only when it is allowed to. Mira does not start that server. Close Mira and set the opt-in in PowerShell, then launch Mira from
the same window.

```powershell
$env:MIRA_ENABLE_STANDALONE_MODELS = '1'
```

Launch a package install with `.\.venv\Scripts\pythonw.exe -m context_operator.app`
from the install folder. From a source checkout, run
`.\.venv\Scripts\pythonw.exe launch.pyw` from the repository root.

The opt-in applies only when `MIRA_LOCAL_MODELS_ROOT` is unset and there is no
`LocalModels` folder next to the Mira folder. LocalModels is a separate model
manager that is not part of this repository. When it is present, Mira asks its
start policy before loading a model and reads the reranking and screen
recognition models from that installation. Speech models are always read from
Mira's data directory.

## Interpret complex commands

Under **Settings → Advanced → Voice and recognition**, set **Model location**, **Model name**, and **Model address**, then save. Use the base URL ending in `/v1` if your server exposes its API there. Mira appends `/chat/completions` and asks for an action using structured JSON output. Choose **Local model** for a loopback server or **Remote API (HTTPS)** for a trusted remote endpoint. The endpoint must accept requests without an `Authorization` header.

The command request contains your text, up to 100 elements with labels and screen positions, the title of the desktop map's main window, the current focus, and up to four recent command records. If **Recognize on-screen text and icons** is enabled and capture succeeds, Mira also sends an image of the visible part of the current window. Turn that setting off to use text-only requests. Audio stays on your PC.

Local target reranking and screen recognition are off by default. Each needs its own Python runtime and weights, and this repository has no setup guide for them. The pinned upstream assets are listed in the [reranker manifest](../context_operator/foundations/ranking/model.json) and [screen recognition manifest](../context_operator/foundations/perception/perception-models.json). Windows accessibility controls and short commands work without them.
