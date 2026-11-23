# Webcam fallback

Mira can use a calibrated webcam when the iPhone temporarily loses gaze. The
iPhone remains the primary source while it provides a fresh supported
estimate. The webcam needs at least three distinct supported frames spanning
at least 120 ms to take over. Observations aged 200 ms or more are rejected. Mira
switches back when the iPhone recovers.

## Runtime

Use an isolated Python environment with NumPy and MediaPipe, and obtain a
compatible Face Landmarker `.task` model. Set the interpreter and model paths
before launching Mira. This example assumes the environments and model are in
the specified locations.

```powershell
$env:MIRA_WEBCAM_PYTHON = "$PWD\.local\webcam-venv\Scripts\python.exe"
$env:MIRA_WEBCAM_MODEL = "$PWD\.local\face_landmarker.task"
$env:MIRA_ENABLE_STANDALONE_MODELS = '1'
.\.venv\Scripts\python.exe launch.pyw
```

An available LocalModels installation controls permission to start inference.
The standalone setting applies when LocalModels is absent. See the
[MediaPipe Face Landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
for the Python runtime and model.

## Setup

1. Under **Settings → Advanced → Webcam · second angle**, select **Find
   cameras**, choose the camera and select **Set up webcam**.
2. Check the preview and start the guided task. Look at the center of each
   ring until it moves. Setup learns from nine targets and checks five new
   targets. Use **Space** to pause or **Check camera** to review the preview.
3. After accepting the profile, choose **Connect webcam** in Settings.
   Connect the iPhone separately in **Overview**.

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
display bounds. The accepted profile and its previous backup are stored in
`.local/webcam-profile.json` and `.local/webcam-profile.bak.json`. The recent
attempts in `.local/webcam-attempts/` contain numeric observations and results.

Run the webcam unit tests from the repository root.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_webcam_calibration tests.test_webcam_features tests.test_webcam_integration tests.test_webcam_capture
```
