# Context Operator

Mira is a Windows app with an iPhone companion. Start with the [README](README.md),
[architecture](docs/architecture.md), and [development guide](docs/development.md).
Keep user-facing text and documentation in English.

- Keep `context_operator/core` independent of Qt, devices, and the network.
  Provider coordinates are physical virtual-desktop pixels.
- Revoke action targets when gaze, scene data, or the source becomes stale.
  Background subprocesses on Windows use `CREATE_NO_WINDOW`.
- Keep local data, device pairing files, downloaded weights, and screen or audio captures
  out of Git and out of automatic capture logs.
- Keep training and check targets separate. A short setup can intentionally have
  `usable=True` and `passed=False`. Read accepted profiles through `ProfileStore`
  and preserve the model, its report, and the backup pair.
- Run `.venv/Scripts/python.exe -m unittest discover -s tests` for desktop changes.
  Use targeted UI checks as well when layout or interaction changes.
