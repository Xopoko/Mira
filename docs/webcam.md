# Webcam fallback

Mira can use a calibrated webcam when the iPhone temporarily loses gaze. A
fresh iPhone estimate always takes priority. The webcam needs at least three
distinct supported frames spanning at least 120 ms to take over. Observations
aged 200 ms or more are rejected. Mira switches back when the iPhone recovers.

## Runtime

Use a separate Python environment with NumPy and MediaPipe, and obtain a
Face Landmarker `.task` model. Set the interpreter and model paths before
launching Mira.

```powershell
$env:MIRA_WEBCAM_PYTHON = "$PWD\.local\webcam-venv\Scripts\python.exe"
$env:MIRA_WEBCAM_MODEL = "$PWD\.local\face_landmarker.task"
$env:MIRA_ENABLE_STANDALONE_MODELS = '1'
.\.venv\Scripts\python.exe launch.pyw
```

`MIRA_ENABLE_STANDALONE_MODELS` is explained in
[Voice and models](models.md#local-model-access). See the
[MediaPipe Face Landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
for the Python runtime and model.

## Setup

1. Under **Settings → Advanced → Webcam · second angle**, select **Find
   cameras**, choose the camera and select **Set up webcam**.
2. In the setup window, select **Connect webcam**, check the preview and
   select **Start setup**. Look at the center of each ring until it moves.
   Setup learns from nine targets and checks five new targets. Use **Space**
   to pause or **Check camera** to review the preview.
3. Select **Use webcam profile** if offered. To start the fallback, choose
   **Connect webcam** in Settings. Connect the iPhone separately in
   **Overview**.

Set up again after moving the webcam, changing monitor geometry or changing
your viewing position. A failed attempt preserves the previous accepted
profile. **Overview** reports which source is active. Switching sources clears
pending actions. Click learning and the [game gaze feed](game-gaze.md) still
use iPhone measurements.

## Capture and profiles

Camera capture starts only when requested. Qt keeps a preview in memory during
setup and passes the latest RGB frame to a separate inference process.
MediaPipe returns numeric face and iris features. The frames stay in local
memory. Ambiguous detections, blinks, unsupported head poses and
stale samples are discarded.

Mira fits the webcam mapping from the nine training targets and selects its
model using held-out training targets. The five check targets are reserved for
evaluating the selected model. The saved profile is tied to the camera ID and
display bounds. The accepted profile, its previous backup and recent attempts
are stored as `webcam-profile.json`, `webcam-profile.bak.json` and
`webcam-attempts/` in [Mira's data directory](development.md#local-data). The
attempts contain numeric observations and results.

Run the webcam unit tests from the repository root.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_webcam_calibration tests.test_webcam_features tests.test_webcam_integration tests.test_webcam_capture
```
