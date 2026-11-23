# Architecture

```text
TrueDepth eyes + head + 3D origins
             ↓ timestamped numeric observations
personal setup → training → disjoint held-out check → accepted profile
             ↓                        ↓
personal gaze + search radius    local numerical receipt
             ↓
conservative click residuals + optional personal regression update
             ↓
operational point → scene fusion → concrete action target
             ↓             ↓
 Qt gaze/radial light   voice / USB / optional dwell
             ↓             ↓
       target highlight → fresh-target validation → Windows output
```

## Source, coordinates and lifecycle

The optional webcam path has independent capture, calibration and validation
([webcam.md](webcam.md)). `core/webcam.py` keeps a fresh supported iPhone
prediction authoritative and admits a calibrated webcam fallback only after
distinct fresh observations. It never averages uncalibrated angles or feeds
webcam predictions into the iPhone learner. A missing sample clears the live
gaze point. Switching the active camera or explicitly disconnecting it revokes
the saved voice target, dwell and pending decisions. Qt selects the camera ID
and captures RGB frames in memory. A hidden isolated MediaPipe child returns
numeric observations.
Pause stops both paths; startup never opens the webcam automatically.

`core` imports no Qt, network or hardware. Its immutable contracts carry physical
virtual-desktop pixels, including negative monitor origins. Providers own hardware,
COM and models; Qt owns presentation. Input timestamps use host `time.monotonic()`.
The phone adapter conservatively combines phone-reported age and the full request
round trip. Duplicate, old or invalid observations cannot become current evidence.

`Controller.attention` is the point used to select an element. It comes from
the active gaze source or the mouse during an explicit override.

The controller independently owns screen, vision, voice, action and phone work.
Pausing increments generations and action epochs before worker shutdown, closes
the setup and revokes pending sources. It cannot retract a
native action already dispatched; such an outcome remains visible and blocks
overlapping output. Background child processes use `CREATE_NO_WINDOW`.

## One personal setup

`personalization/session.py` builds a short starter from nine labeled fixations and
checks it on five different screen targets during one comfortable, stationary pose.
The target sets are disjoint. Each sample contains the existing numeric ARKit eye,
head and 3D eye-origin fields; it needs no RGB frame or previous screen predictor.
After a stable fixation, Space confirms the displayed target. Missing, stale or
duplicate observations do not advance collection.

The starter averages the intersections of each eye ray with the camera's
`z=0` plane. An affine transform maps that average to physical desktop pixels.
The five held-out targets measure median and P90 error, jitter and coverage. A
usable starter can be adopted for gradual refinement. Its `passed` status stays
false: the short check does not establish the former full three-pose accuracy
contract. Completing collection alone never means that usability passed.
The starter's versioned `joint_training_poses_v2` support guard compares each
observation with recorded training neighborhoods: head translation, head rotation,
and eye direction must agree with the same training anchor. Tolerances remain
fixed and are not estimated from the check targets. This allows head turns already
present during training, instead of rejecting them against one median head pose.
Legacy saved models retain their original support policy. Per-target diagnostics
separate unsupported observations from predictions outside the display; these
are independent of prediction error and jitter.
The measurement is in pixels and cannot be called angular accuracy without verified
screen dimensions and eye-to-screen extrinsics.

`personalization/store.py` commits the accepted profile and its receipt together,
retaining the prior accepted profile for rollback. Profile identity includes the
display and engine geometry. Older accepted profiles remain readable; old numerical
setup archives remain available for recovery and are not mixed into the new
geometry. Persistent measurements use numeric formats without pickle. One setup
entry point is shared by Overview and Profile.

## Adaptation and inference

`core/click_learning.py` matches a manual click to distinct supported observations
from the 450 ms before press, requires a stable fixation, and rejects missing-eye
barriers, opposite movement and clicks beyond the combined source-radius, screen-size
and 650 px limits. This is weak behavioral evidence, not gaze ground truth.
`providers/learning.py` applies only a bounded local correction per accepted click;
the global regressor receives a row only after three agreeing observations in a
similar eye/head feature neighborhood. A cross-validation winner remains pending
until 12 future corroborated clicks across at least three screen cells.
Promotion requires at least 6% lower median error and P90 no more than 3% above the
baseline. On load, older unguarded physical-click rows and networks trained from
them are set aside in a backup. Previously saved manual pose data is still loaded.

`core/fusion.py` uses gaze position and recent focus to choose among visible
elements. For a spoken request it can shortlist candidates by position and label.
With reranking enabled, the first complex command with a fresh scene starts
the CPU reranker in the background.
The interpreter can handle that command immediately and uses reranker scores
on subsequent requests once the model is ready. Pausing the application or
disabling the reranker closes its process.
Qt renders the current gaze point and radial light over the exposed elements; fresh
action focus receives its own contour. Stale observations and targets are omitted.
Native geometry caches store drawn shapes, not screen captures.

`highlight_mode` selects the default radial presentation or opt-in magnetic focus.
Magnetic selection remains inside `Fusion`: the displayed yellow element is the
actual `Focus.target`, including for voice/USB activation. Its separate hysteresis
retains a full element identity and the acquisition gaze point. A directional
shift of 18–60 physical pixels (scaled from source radius) must persist across
distinct gaze timestamps for 180 ms; a strong spatial advantage switches sooner.
Naming a target selects that exact element for three seconds, even without gaze.
Cycling selects a neighbour while gaze remains fresh. Removed or changed
elements and stale sources cannot retain a magnetic lock. Neither the lock nor
its displayed center is fed back into gaze measurements or click labels.

Qt draws only that complete element, with a blue outer glow and yellow contour;
the raw gaze dot, radial field and violet forecast are hidden in this mode.
Switches brighten the new contour briefly without interpolating a shape across
the desktop. Once settled, the layer repaints only for changed scene geometry or
source expiry. Both old and new damage regions are erased, including on mode
switch, pause or source loss. The original mode and its settings remain available.

`quiet_mode` is an independent persisted presentation preference. `Window.refresh`
passes a disabled presentation to every overlay, reusing the existing stale-ink
cleanup for both radial and magnetic modes. The overview focus panel is hidden,
and `SceneCanvas` independently skips gaze drawing. The main window remains usable.
This does not pause the controller, clear action focus, change highlight selection,
stop click learning or disable the opt-in game gaze publisher. Explicit profile
setup still presents its targets; quiet mode does not hide an initiated task.

`core/anticipation.py` and `providers/anticipation.py` own two distinct predictors:
one ranks targets from mouse motion, the other from stable supported gaze dwell.
Each has its own ranker and weights; the mouse ranker can additionally use fresh,
supported gaze proximity as a feature. Manual click outcomes can train intent
ranking, but a click does not become a gaze coordinate label. The gaze channel only
learns clicks near the recorded fixation; the mouse channel can learn a mapped click
away from gaze. Predictions are captured before the outcome and remain
presentation-only. Both channels evaluate predictions made at least 150 ms
before the click. Mouse statistics also require the pointer to be outside the
clicked element at prediction time when that element can be identified.
Gaze statistics include disagreements with clicks too far away to train on.

## Scene, commands and output

`providers/desktop.py` composes independently aged WindowMaps from bounded COM-owned
UIA readers. `core/desktop.py` computes exposed rectangles and removes moved,
replaced or stale maps. Screen topology is refreshed separately from slow queries.
`providers/visual_cache.py` permits bounded OCR/model inference with independent
pixel verification. A changed screenshot, scene or action epoch rejects late output.
See [desktop-context.md](desktop-context.md).

Short commands use direct parsing or the small local classifier. For other
commands, the configured model is asked for a JSON action and the client validates
its reply. [Local inference settings](models.md) control access to local models.
The controller saves the current focus and gaze when it receives the speech-start
event. The transcript uses that snapshot, so moving the gaze during recognition
cannot redirect a short command to the new focus.

Before action, output checks the fresh element identity, process, geometry,
ownership and visual digest where applicable. Semantic UIA runs on its own worker;
an error after dispatch does not trigger a coordinate fallback or retry. USB-button
presses use the target displayed at the native press timestamp and the pointer route
chosen before dispatch. No gaze or inferred attention alone triggers an action unless
the user explicitly enables dwell. Details: [usb-button.md](usb-button.md).
SendInput reports how many events Windows accepted. A successful UIA call reports
that the provider returned without an error. The application refreshes the scene
after dispatch.

USB activation observes keyboard Raw Input through one process-wide message-only
window. Binding includes an opaque device identity, VK, scan code and extended
flags; enrollment first excludes the user's ordinary keyboard. No keyboard hook
or legacy-input suppression is installed. Old VK-only settings stay disabled.

`Controller.register_source(name, source)` accepts adapters with `start(emit)`
and `stop()` methods. The USB button emits an activation event. Programmatic
sources can supply gaze, text commands or a stop event. The controller rejects
events from earlier generations after a source stops or restarts.
