# Architecture

```text
iPhone TrueDepth: eye and head measurements (numbers, timestamped)
        │  pulled by the PC over Wi-Fi or USB
        ▼
gaze setup: 9 training targets → model → 5 separate check targets → accepted profile
        │
        ▼
estimated screen point and search radius  ←  optional corrections learned from clicks
        │
        ▼
visible controls near the point → highlight overlay
        │  optional: voice or typed command, dwell
        ▼
target checked again → Windows input
```

## Code layout

| Path | Role |
| --- | --- |
| `core/` | Decisions without I/O: data contracts, focus selection, gaze-source arbitration, click matching, next-target prediction, command parsing, desktop map composition. Imports no Qt, network or hardware. |
| `foundations/` | The phone reader, gaze smoothing, and the child-process runtimes for speech, target reranking and screen recognition. |
| `personalization/` | Gaze setup, the profile model and its store, webcam calibration. |
| `providers/` | Adapters that own hardware, COM and model processes. |
| `ui/` | Qt windows and the overlay. |
| `controller.py` | One `Controller` connects the providers and runs a 25 ms tick. |

## Coordinates, time and staleness

All coordinates are physical pixels on the Windows virtual desktop, whose
origin can be negative. Timestamps use host `time.monotonic()`. For a phone
frame the reader subtracts the age reported by the phone and the full request
round trip, so the estimate errs towards older.

Nothing is used past its age limit. A gaze sample lasts 200 ms, a window's
element map 1.25 s and a scene 3 s. Duplicate, out-of-order and invalid frames
do not count as new data, and a missing sample clears the live gaze point.

`Controller.attention` is the point used to select an element. It is the
active gaze source, or the mouse pointer while **Try with mouse** is on or
Shift is held.

Pausing raises the controller's generation and action epoch before it stops
the workers, so results that arrive late from the previous run are dropped. It
also closes gaze setup and forgets the focus and any voice command in
progress. A Windows action that was already dispatched cannot be retracted.
Its outcome is still reported, and no other action starts until it returns.
Child processes are started with `CREATE_NO_WINDOW`.

## Gaze setup and the profile

`personalization/session.py` collects nine labeled fixations for training and
five at other screen positions for checking, all in one seated position. A
sample is 22 numbers from ARKit: eye directions, head pose and 3D eye origins.
It needs no camera image. A target counts after a steady fixation of at least
12 fresh frames over 0.65 s, confirmed with Space. Stale, duplicate and blink
frames do not advance collection.

The model in `personalization/model.py` intersects each eye ray with the
camera's `z=0` plane, averages the two intersections and maps the result to
desktop pixels with an affine transform. The five check targets, which the fit
never sees, give the median and 90th-percentile error, jitter and coverage. A
setup within the limits is `usable` and can be adopted. `passed` stays false
for every setup: five points show that it is usable, not that it is accurate.
The errors are in pixels. They say nothing about angular accuracy, which would
need the physical size and position of the screen.

While tracking, an observation is supported only if its head position, head
rotation and eye direction all agree with the same training fixation, within
fixed tolerances. Otherwise Mira shows no gaze point and reports that the
position is outside the setup range.

`personalization/store.py` saves the accepted model and its report as a pair
and keeps the previous pair as a backup. It reads back what it wrote and
restores the old pair if anything fails. The report records the model's hash
and the display bounds, and a profile is not loaded on a different display
layout. The files hold numeric arrays and JSON, never pickle.

## Corrections, focus and the overlay

[Click learning](click-learning.md) nudges the estimate using manual clicks.
[Next-target prediction](anticipation.md) ranks likely targets from mouse
motion and gaze dwell. It only draws a preview and never becomes a gaze label.

`core/fusion.py` chooses the focus among visible elements from gaze distance,
element size and, for a spoken or typed request, how well the label matches. A
new best candidate has to hold for the stability time, 110 ms by default,
before it replaces the current one. When a spoken command starts, the
controller saves the focus and gaze of that moment and applies the transcript
to that snapshot, so looking away during recognition does not redirect the
command.

`highlight_mode` selects how the focus is shown.

- **Follow gaze**, the default, draws the gaze dot and a radial light over
  nearby elements. The action target has its own contour.
- **Magnetic focus** outlines only the focused element. The element keeps the
  focus until gaze moves 18–60 physical pixels, scaled from the search radius,
  towards another element and stays there for 180 ms across distinct frames.
  A clearly better candidate switches sooner. A removed or changed element, or
  a stale source, drops the lock. The lock is never fed back into gaze
  measurements or click labels.

`quiet_mode` hides the overlay and the live preview in **Overview**. It does
not pause tracking, click learning or the [game gaze feed](game-gaze.md), and
gaze setup still shows its targets. See
[Overlay rendering](overlay-performance.md) for how the overlay is painted and
erased.

## Webcam fallback

A calibrated webcam can supply gaze while the iPhone has none. A fresh iPhone
estimate always wins, and the two are never averaged. Switching the active
source cancels the saved voice target, dwell and pending decisions. See
[Webcam fallback](webcam.md).

## Desktop map, commands and output

`providers/desktop.py` builds the scene from per-window UI Automation scans,
optionally supplemented by screen recognition. See
[Desktop map](desktop-context.md).

A command is first parsed directly in `core/commands.py`, then tried by the
small local classifier in `providers/fast_intent.py`, and only then sent to
the configured model. The model must answer with one JSON action, which the
client validates. See [Voice and models](models.md).

Before acting, `providers/windows.py` checks that the scene and the window's
map are still fresh, that the window has the same process and bounds, that
the target is still in the scene, and that nothing covers it. For synthetic
input it also re-reads the control under the target point, and for a
recognized element it compares the pixels. Mira prefers a UI Automation call
and uses synthetic mouse and keyboard input when the control offers none. Once
either has been sent there is no retry and no fallback, and a short
`SendInput` count is reported as an error. The scene is discarded and rebuilt
after every action. Gaze alone activates a control only when dwell is enabled.
