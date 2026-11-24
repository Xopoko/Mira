# Local gaze feed for games

Mira can publish the current gaze point to other programs on the same PC.
Either start the [standalone gaze runner](game-gaze-runner.md), or set
`game_gaze_enabled` to `true` in `settings.json` in
[Mira's data directory](development.md#local-data) and connect the iPhone in
the desktop app. The setting has no control in the app. Quiet mode hides the
desktop gaze cues while the publisher keeps running.

## Shared-memory format

Open the Windows named mapping `Local\ContextOperator.Gaze.v1` with
`OpenFileMappingW(FILE_MAP_READ, ...)`. Take the named mutex
`Local\ContextOperator.Gaze.v1.Lock` with a zero-timeout wait, copy all
128 bytes, and release the mutex. If the mutex is busy, retry on the next poll.
The publisher holds `Local\ContextOperator.Gaze.v1.Writer` to enforce a single
writer.

The buffer uses the little-endian layout `<8sIIQ9d32s`.

| Field | Representation |
| --- | --- |
| Magic | 8 bytes, `COGAZE1\0` |
| Version | Unsigned 32-bit integer, `1` |
| Flags | Unsigned 32-bit integer, with bit 0 set for active samples |
| Sequence | Unsigned 64-bit integer, incremented after each successful write |
| Timestamps | Two doubles, captured and published time in `GetTickCount64` uptime seconds |
| Gaze position | Two doubles, absolute physical pixel x and y |
| Desktop bounds | Four doubles, physical pixel x, y, width and height |
| Radius | Double, gaze search radius in physical pixels |
| Reserved | 32 zero bytes |

Check the magic, version, active flag and capture timestamp before using a
sample. The publisher marks a sample active only if it is under 200 ms old,
the head and eye position is within the range covered by gaze setup and the
point is inside the current desktop bounds. It preserves the original frame
timestamp when the same frame is published again. Pause, setup, disconnection,
pointer control and stale or invalid data produce inactive packets with zeroed
gaze and geometry. A write is skipped while a reader holds the mutex, so the
previous packet can stay in place for a moment. Always check the capture age.
