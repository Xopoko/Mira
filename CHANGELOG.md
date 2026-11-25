# Changelog

## Unreleased

- Make gaze setup about twice as accurate. Mira now works out where the screen
  is in space from the setup points and treats head and eye movement
  separately, so looking at a point with your eyes and turning your head
  toward it give the same result, and shifting in your seat matters less. On
  seven saved setups from one 5120×1440 screen, replayed through the new model,
  the typical error fell from 370 px to 170 px. Setup uses thirteen points in
  place of nine, reaches the edges of the screen, and reads the monitor size
  from Windows. The error that remains drifts slowly across the screen, and
  Mira now smooths it out between the setup points: the first thirteen-point
  setup on that screen replays at 117 px. Earlier gaze setups need to be run
  again.
- Measure the iris on the iPhone. The companion finds the edge of each iris
  in its camera frame, in a small window around the place ARKit's eye
  rotation predicts, and sends where it is with every reading, together with
  the brightness across the edge on four sides, averaged along arcs. These
  are numbers, and the frame stays on the phone. Gaze setup records them with
  each point, and `scripts/evaluate_gaze_setup.py` scores a recorded setup
  with ARKit's eye rotation and with the measured iris side by side. This is
  an experiment to find out whether Mira's own measurement beats ARKit's.
  Tracking does not use it yet. The iPhone companion is at 0.4.2.
- Redesign the Windows app as a dark companion app. **Overview** now leads
  with a live preview of your screen beside the iPhone, accuracy and quick
  settings. **Profile** is now **Calibration**, the highlight settings moved
  out of **Advanced**, and the screen map no longer needs a setting. Click
  learning, next-target prediction and typed commands sit behind **Advanced**
  and **More practice tools**.
- Gaze setup shows its result, what it means and the next button in one
  place, and no longer says it kept a profile when there was none.
- Redesign the iPhone companion to match. It shows a monitor with the gaze
  light following your eyes in place of the top-down plan, as soon as it sees
  you and before the PC connects. The connection settings moved to their own
  page, and the app is dark only. The iPhone companion is at 0.4.0.
- Bring gaze tracking and screen contours to the front of the Windows setup.
- Make local voice commands optional during setup and ordinary tracking.
- Remove USB confirmation buttons. The iPhone USB connection remains available.
- Redesign Practice around the live gaze contours.
- Add a Windows overlay demo and an illustration of the TrueDepth setup.
- Replace the app icon. `scripts/render_readme_media.py` regenerates the README
  screenshots and the overlay demo.
- Give the iPhone companion the same icon. The iPhone companion is at 0.3.5.
- Keep the iPhone Wi-Fi port usable after the app returns from the background.
  A port that fails to open is retried up to 30 times, one second apart. The
  iPhone companion is at 0.3.4.
- Remove the RGB face-image mode from the iPhone companion. It now sends
  numeric measurements only and no longer claims that the PC follows your
  gaze as soon as something connects. The iPhone companion is at 0.3.6.
- Show errors that used to be hidden: screen recognition and the game gaze
  feed failing to start, a screen map that cannot be read, a failed iPhone
  search, and a command model that cannot be reached. A skipped click now
  gives its actual reason.
- Screen recognition and local reranking are off by default. Both need
  separately installed runtimes. Saved settings keep their values.
- **Ctrl+Alt+F12** also stops iPhone gaze setup. The setup ring fills only
  when the fixation is ready to confirm.
- Saving a webcam profile no longer replaces the previous one when its backup
  cannot be written.

## 1.3.1

First public release of the Windows app. The iPhone companion is at 0.3.3.

- Select Windows controls with iPhone gaze and see the target before acting.
- Confirm with local voice recognition, a bound USB button, or optional dwell.
- Try typed commands and mouse focus in Practice without a phone.
- Connect the iPhone over Wi-Fi and check gaze setup on separate targets.
- Calibrate a webcam for fallback and pause control with Ctrl+Alt+F12.
