# Local gaze feed for games

On Windows with Python 3.11, enable the `game_gaze_enabled` setting on the
desktop app or start the [standalone gaze runner](game-gaze-runner.md). Connect
a calibrated iPhone in TrueDepth mode. Quiet mode can hide the desktop gaze
cues while the enabled publisher continues running.

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
| Radius | Double, estimated uncertainty in physical pixels |
| Reserved | 32 zero bytes |

Check the magic, version, active flag and capture timestamp before using a
sample. The publisher requires an age under 200 ms, a supported non-motion
calibration and a point inside the current desktop bounds. It preserves the
original frame timestamp when the same frame is published again. Pause, setup,
disconnection, pointer control and stale or invalid data produce inactive
packets with zeroed gaze and geometry. This format carries only timestamps,
screen geometry and gaze data.
