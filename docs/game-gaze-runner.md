# Standalone gaze runner

Use Python 3.11 on Windows and set up an iPhone gaze profile in Mira first.
Then launch the runner from the repository root.

```powershell
.\.venv\Scripts\python.exe scriptsun_game_gaze.pyw
```

Select **Connect gaze** to start the saved iPhone transport and the
[local shared-memory feed](game-gaze.md). The window starts disconnected and
only runs the phone receiver and the numeric publisher. The main app and the
runner write to the same feed, and only one can hold it. Pause or close the
main app before connecting gaze in the runner.

**Gaze live · receiving fresh data** means the last sample was written.
**Gaze live · output busy, update skipped** means a reader held the feed and
the write was skipped. A missing, stale or out-of-range estimate produces an
inactive packet. To stop, select **Disconnect gaze**, close the window, or
press **Ctrl+Alt+F12** when the window says the shortcut is available. The
runner reads the saved profile and corrections without updating either.
