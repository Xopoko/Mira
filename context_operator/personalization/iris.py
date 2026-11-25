"""Gaze from where the iPhone finds each iris in its own camera frame.

The phone reports the centre of each limbus in frame pixels and how camera
space maps to those pixels. The ray through that pixel meets the eyeball, a
sphere around the centre the eye turns about, at the iris, and the line from
that centre through the iris is where the eye points. ARKit's eye rotation is
not used. Experimental: `scripts/evaluate_gaze_setup.py` compares the two
signals on recorded setups, and nothing else reads this yet.
"""

import numpy as np

from . import geometry

LIMBUS_M = .0105  # From the eye's centre of rotation to the centre of the limbus.
MIN_SCORE = .3


def found(readings):
    """Frames in which the phone found both irises."""
    readings = np.asarray(readings, dtype=float)
    determinant = readings[:, 2]*readings[:, 5]-readings[:, 3]*readings[:, 4]
    return (readings[:, 11] >= MIN_SCORE) & (readings[:, 15] >= MIN_SCORE) & (np.abs(determinant) > 1e-9)


class IrisEyes:
    """The measured iris position as the eye signal for `geometry.fit_screen`.

    Setup fits five numbers. Two gains scale the iris offset from the projected
    eye centre, across the face and up it, before the ray meets the eyeball:
    they stand for the size of this person's eye and for eyelids hiding part
    of the limbus. Three more move the centre the eye turns about away from
    the eye centre ARKit tracks, in millimetres along the face's axes. A phone
    below the screen sees the eyes from well below, and there a centre that is
    a few millimetres too shallow bends the whole mapping.
    """

    start = (1., 1., 0., 0., 0.)
    sigma = np.array([.15, .15, 2., 2., 3.])
    lower = np.array([.5, .5, -8., -8., -10.])
    upper = np.array([1.8, 1.8, 8., 8., 10.])

    def __init__(self, features, origins, readings):
        self.features = np.asarray(features, dtype=float)
        self.origins = np.asarray(origins, dtype=float)
        readings = np.asarray(readings, dtype=float)
        # frame pixel = shift + camera @ (x, y)/-z for a point in camera space.
        self.camera = readings[:, 2:6].reshape(-1, 2, 2)
        self.shift = readings[:, 6:8]
        self.iris = readings[:, [8, 9, 12, 13]].reshape(-1, 2, 2)
        self.head = geometry.head_rotations(self.features)

    def _pixels(self, points):
        flat = points[..., :2]/-points[..., 2:]
        return np.einsum('nij,nej->nei', self.camera, flat)+self.shift[:, None]

    def directions(self, values=start):
        """Where each eye points, in camera space."""
        values = np.asarray(values, dtype=float)
        centre = self.origins+(self.head@(values[2:5]/1000))[:, None]
        # How a step across the face and up it, taken at the eye, shows in frame pixels.
        span = .01
        eye = self._pixels(centre)
        face = np.stack([self._pixels(centre+span*self.head[:, None, :, axis])-eye for axis in range(2)], axis=3)/span
        offset = np.linalg.solve(face, (self.iris-eye)[..., None])[..., 0]*values[:2]
        pixel = eye+np.einsum('neij,nej->nei', face, offset)
        flat = np.linalg.solve(self.camera[:, None], (pixel-self.shift[:, None])[..., None])[..., 0]
        ray = np.concatenate([flat, -np.ones_like(flat[..., :1])], axis=2)
        ray /= np.linalg.norm(ray, axis=2, keepdims=True)
        along = np.einsum('nei,nei->ne', ray, centre)
        # A ray that passes outside the eyeball takes the point nearest to it.
        inside = np.maximum(along**2-np.einsum('nei,nei->ne', centre, centre)+LIMBUS_M**2, 0)
        direction = (along-np.sqrt(inside))[..., None]*ray-centre
        return direction/np.linalg.norm(direction, axis=2, keepdims=True)

    def rays(self, values):
        direction = self.directions(values).sum(axis=1)
        return self.origins.mean(axis=1), direction/np.linalg.norm(direction, axis=1, keepdims=True)
