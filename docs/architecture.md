# Architecture

```text
iPhone TrueDepth: eye and head measurements (numbers, timestamped)
        │  pulled by the PC over Wi-Fi or USB
        ▼
gaze setup: 13 training targets → model → 5 separate check targets → accepted profile
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

`personalization/session.py` collects thirteen labeled fixations for training
and five at other screen positions for checking, all from one seat. A
sample is 22 numbers from ARKit: eye directions, head pose and 3D eye origins.
It needs no camera image. A target counts after a steady fixation of at least
12 fresh frames over 0.65 s, confirmed with Space. Stale, duplicate and blink
frames do not advance collection.

The model in `personalization/geometry.py` and `model.py` treats the screen as
a plane in the iPhone camera's space. A reading becomes a gaze ray that starts
between the eyes and points where the head points, turned by the eye rotation.
ARKit reports larger eye rotations than the eye makes, so the eye part is
scaled down first: by about 0.65 sideways and 0.85 up and down, and setup may
move these a little. The fit then finds the position and tilt of the plane
that sends the thirteen rays to their targets. The pixel size comes from the
physical size the monitor reports to Windows, and the fit estimates it when
Windows does not know. Nothing asks where the iPhone stands.

Head and eyes are kept apart, so a target reached by turning the head and the
same target reached with the eyes alone map to the same pixel. When the screen
size is known the mapping also holds as the head moves. One seat cannot tell a
large far screen from a small near one, so a fitted size is trusted only within
8 cm of the setup positions. Several monitors are treated as one flat surface
with one pixel size, which is less accurate where they meet at an angle.

What the plane leaves over is not random. The error drifts slowly across the
screen and repeats from one setup to the next, so the profile adds a smooth
field through the leftover error at the thirteen setup points. It reaches
about half a screen height from each point and fades to nothing beyond.

The five check targets, which the fit never sees, give the median and
90th-percentile error, jitter and coverage. A setup within the limits is
`usable` and can be adopted. `passed` stays false for every setup: five points
show that it is usable, not that it is accurate. Seven recorded nine-point
setups from one 49-inch 5120×1440 screen, replayed with
`scripts/evaluate_gaze_setup.py`, give a typical check error of 120–250 px,
170 px on average, where the affine map Mira used before gave 370 px. The
first thirteen-point setup on that screen replays at 117 px, against 189 px
for the plane alone. Published measurements of ARKit gaze report about 3°,
which is roughly 190 px there.

### The iris the iPhone measures itself

Within one fixation the ARKit reading moves by 25–40 px on that screen, yet
the fixation as a whole lands 100–200 px away. Averaging over time cannot
remove that, so the iPhone companion also measures the eye on its own. In each
camera frame it looks for the edge of each iris in the brightness plane and
adds what it finds to its reply. The frame stays on the phone.

ARKit's own eye rotation says where the iris should be to within a few
pixels, and the search covers only a small window around that place. It looks
for the dark-to-bright step along the two sides of the iris, where lids
rarely cover it, then centres itself on four brightness profiles: the way out
of the iris towards the left, the right, down and up, each averaged along an
arc. A search over the whole eye opening, which the first version used,
locked onto lids and shadows on a real phone.

```json
"iris": {"image": [1920, 1080], "camera": [a, b, c, d, x0, y0],
         "left": [x, y, radius, score], "right": [x, y, radius, score],
         "profile": {"step": 0.05, "left": [[…], […], […], […]], "right": […]}}
```

`x`, `y` and `radius` are frame pixels, and `score` is how large the
brightness step across the edge is on the two sides, from 0 to 1. A PC treats
a score below 0.3 as not found. `camera` maps camera space to the frame: a
point `(X, Y, Z)` appears at `(x0, y0) + [[a, b], [c, d]] · (X, Y) / −Z`.

`profile` holds the four profiles behind each fit, towards −x, +x, −y and +y
of the face. Each has 25 samples spaced `step` iris radii apart; sample 12
lies on the fitted edge and lower ones inside the iris. They show how sharp
the edge was and whether the fit sits on it, so the fit can be judged and
redone on the PC from recorded numbers. They are averages along arcs, a
hundred numbers per eye, and not pixels of the frame. An eye whose window
leaves the frame is `null`. The `iris` member is optional. A PC that does not
know it ignores it, and a reply without it comes from an older phone.

`personalization/iris.py` turns the reading into an eye direction. The ray
through the iris pixel meets a sphere around the eye centre at the iris, and
the line from the centre through that point is where the eye looks. The
screen fit is the one above with this direction in place of ARKit's eye
rotation. A phone below the screen sees the eyes from well below, where a
centre a few millimetres off bends the whole mapping, so the fit also moves
the centre the eye turns about away from the one ARKit tracks. Gaze setup
stores the reading and the profiles with every fixation, and
`scripts/evaluate_gaze_setup.py` scores such a setup with both eye signals.

This is an experiment, and tracking still uses ARKit's eye rotation. One
degree of eye rotation moves the iris by about 0.18 mm, which at 60 cm is
0.4 of a frame pixel, so the fit has to hold to a small part of a pixel to
beat ARKit. On rendered eyes the centre typically lands within 0.2 px of the
true one, and within 0.45 px in hard scenes such as an iris in the corner of
the eye or a dim room. Whether a phone on a desk reaches that is what
recorded setups have to show.

While tracking, an observation is supported when the head is within 20 cm of a
head position seen in setup, 8 cm when the screen size was fitted, and turned
about as far as it was there. Otherwise Mira shows no gaze point and reports
that the position is outside the setup range.

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
