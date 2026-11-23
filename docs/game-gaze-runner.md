# Lightweight gaze runner

Use Python 3.11 on Windows and set up an iPhone gaze profile in Mira first.
Then launch the standalone viewer from the repository root.

```powershell
.\.venv\Scripts\python.exe scripts\run_game_gaze.pyw
```

Select **Connect gaze** to start the saved iPhone transport and the
[local shared-memory feed](game-gaze.md). The window starts disconnected and
only runs the phone receiver and the numeric publisher. The main app and the
runner share a feed. Disconnect the main app's publisher before starting the
runner.

**Gaze live** indicates that fresh, supported measurements are being published.
An unsupported, missing or stale estimate produces an inactive packet. Select
**Disconnect gaze**, close the window, or press **Ctrl+Alt+F12** to stop.
The runner reads the saved profile and corrections without updating either.
