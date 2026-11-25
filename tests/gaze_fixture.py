"""Synthetic TrueDepth readings for a known screen, seat and eye response."""

import numpy as np

from context_operator.personalization import geometry

PIXEL_MM = .25
EYE_OFFSETS = np.array([[.032, .028, .026], [-.032, .028, .026]])
SCREEN_CENTER = np.array([0., .12, .05])
# Camera space: the screen faces -z, its rows run toward -y.
SCREEN_FRAME = np.array([[1., 0, 0], [0, -1., 0], [0, 0, -1.]])


def reading(target, bounds, *, head=(0., 0., -.7), turn=0., gains=geometry.EYE_GAIN, pitch_mm=PIXEL_MM):
    """Features and eye origins for a look at `target`.

    `turn` is the share of the horizontal angle taken by the head; the eyes do
    the rest, and ARKit over-reports their rotation by 1/gains.
    """
    head = np.asarray(head, dtype=float)
    display = np.asarray(bounds.as_list(), dtype=float)
    offset = (np.asarray(target, dtype=float)-display[:2]-display[2:]/2)*pitch_mm/1000
    point = SCREEN_CENTER+offset[0]*SCREEN_FRAME[:, 0]+offset[1]*SCREEN_FRAME[:, 1]
    yaw = turn*np.arctan2(point[0]-head[0], point[2]-head[2])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, 0, s], [0, 1., 0], [-s, 0, c]])
    origins = head+EYE_OFFSETS@rotation.T
    local = rotation.T@(point-origins.mean(axis=0))
    local /= np.linalg.norm(local)
    side = np.arctan2(local[0], local[2])/gains[0]
    lift = np.arcsin(local[1])/gains[1]
    direction = rotation@np.array([np.sin(side)*np.cos(lift), np.sin(lift), np.cos(side)*np.cos(lift)])
    row = np.zeros(22)
    row[0:4] = np.concatenate([origin[:2]-origin[2]*direction[:2]/direction[2] for origin in origins])
    row[4:7] = head
    row[7:13] = rotation[:, :2].ravel()
    row[13:19] = np.r_[direction, direction]
    row[19:22] = direction
    return row, origins


def turned(row, origins, camera):
    """The same reading seen by a camera whose axes are rotated by `camera`."""
    row = row.copy()
    row[7:13] = (camera@row[7:13].reshape(3, 2)).ravel()
    for part in (slice(4, 7), slice(13, 16), slice(16, 19), slice(19, 22)):
        row[part] = camera@row[part]
    return row, origins@camera.T


def burst(target, bounds, *, camera=None, **seat):
    row, origins = reading(target, bounds, **seat)
    if camera is not None:
        row, origins = turned(row, origins, camera)
    rows = np.tile(row, (12, 1))
    rows[:, 13] += np.linspace(-.0004, .0004, len(rows))
    return rows, np.tile(origins, (12, 1, 1))
