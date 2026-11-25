# Development

Start with the [Windows setup](../README.md#build-from-source) and the
[architecture overview](architecture.md). The application code is in
`context_operator/`, the Swift companion is in `phone/`, and the Windows tests
are in `tests/`.

## Run and test

Run these commands from the repository root.

```powershell
.\.venv\Scripts\python.exe launch.pyw
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe scripts\check_desktop_ui.py
```

The first command keeps the console attached for debugging. Use `pythonw.exe`
to launch without a console. Mira runs as one instance. Close the existing
window before restarting to load code changes.

The Qt check runs offscreen with synthetic sources and writes screenshots under
`.local/evidence/desktop-ui/`. CI runs the unit tests and this check. The
other `scripts/check_*.py` are manual checks. Read a script before running it,
since some open hardware, run models or use a native window. Unit and
offscreen tests cover behavior under controlled inputs. Camera accuracy and
real Windows input delivery need a device. See the
[iPhone build and tests](../phone/README.md#checks) for Swift changes and
[Contributing](../CONTRIBUTING.md) for pull requests.

## Regenerate README images

```powershell
.\.venv\Scripts\python.exe scripts\render_readme_media.py
```

The script renders Mira offscreen with simulated gaze and writes
`mira-overview.png`, `mira-gaze.gif`, and `mira-social.png` to `docs/images/`.
The 3D illustration (`mira-gaze-explainer.gif`) and the iOS Simulator screenshot
(`mira-iphone.png`) are produced separately.

## Local data

A source checkout uses `.local/`. A normal Python package installation uses
`%LOCALAPPDATA%\Mira`. The source launcher writes startup errors to
`.local/startup-error.log`. The running app writes uncaught exceptions to
`crash.log` in its data directory.

| File or directory | Contents |
| --- | --- |
| `settings.json` | Source preferences, model endpoint, and control settings |
| `gaze-personal.npz` and `personal-profile.json` | Accepted model and its validation report |
| `gaze-personal.bak.npz` and `personal-profile.bak.json` | Previous accepted pair |
| `personal-starter-session.json`, `starter-candidates/`, `starter-attempts/` | Setup observations, candidates, and attempt history |
| `starter-last-result.json` | Report of the latest setup attempt |
| `gaze-adaptation-personal-*.json` | Click-based gaze corrections, one file per accepted profile |
| `webcam-profile.json`, `webcam-profile.bak.json`, `webcam-attempts/` | Webcam fallback profile, its backup, and recent attempts |
| `intent-prediction.json` | Next-target prediction state |
| `models/` | Optional local model assets |

Use `ProfileStore` to load or adopt a profile. It checks display geometry,
compatibility, and the saved model hash. Keep the model and its report together
when restoring a backup. The latest attempt can fail while the previous profile
continues to be usable.

When investigating a setup failure, match the saved observations and display
bounds with the candidate identified by `candidate_sha256`. Read the raw
predictions before clipping or smoothing. Coverage measures samples whose poses
are supported and whose predictions are on screen. Inspect coverage, error, and
jitter for each held-out target. Keep the five check targets out of training and
parameter selection.

To judge a change to the gaze model, replay the saved attempts before running a
new setup:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_gaze_setup.py
```

It fits each attempt's setup points with the current model and prints the error
on that attempt's check points next to the error recorded at the time. Pass a
folder to replay attempts kept elsewhere. The `now` and `p90` columns are over
frames. `point` is how far whole fixations landed from their targets, which is
what remains once jitter is smoothed away. Five check points are few, so
`--leave-one-out` also holds out every fixation in turn.

## Feature guides

- [iPhone Wi-Fi](phone-wifi.md) and [webcam fallback](webcam.md)
- [Voice and models](models.md) and [desktop context](desktop-context.md)
- [Click learning](click-learning.md) and [next-target prediction](anticipation.md)
- [Game gaze feed](game-gaze.md)
- [Overlay performance](overlay-performance.md)
