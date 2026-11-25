"""Where the screen is, worked out from the fixations of one setup.

TrueDepth reports the head pose and both eye rotations in camera space. The
setup fits the screen as a plane in that same space, so a gaze ray lands on a
pixel without asking where the iPhone stands or how large the monitor is.
"""

import numpy as np
from scipy.optimize import least_squares

# ARKit reports larger eye rotations than the eye makes. These are the typical
# corrections; each setup may move them a little.
EYE_GAIN = (.65, .85)
GAIN_SIGMA = .03
SIZE_SIGMA = .35
FACING_SIGMA = np.radians(8)
NOISE_RAD = np.radians(1.5)
PIXEL_RANGE_MM = (.05, 1.2)
VIEWING_RANGE_M = (.15, 3.)
MAX_RAY_M = 10.
MIN_APPROACH = .05
DEFAULT_PIXEL_MM = .25


def head_rotations(features):
    columns = features[:, 7:13].reshape(-1, 3, 2)
    first, second = columns[:, :, 0], columns[:, :, 1]
    return np.stack([first, second, np.cross(first, second)], axis=2)


def eye_offsets(features, origins):
    """Eye centres in the face frame. They stay put for one person."""
    features = np.asarray(features, dtype=float)
    origins = np.asarray(origins, dtype=float)
    local = np.einsum('nji,nkj->nki', head_rotations(features), origins-features[:, None, 4:7])
    return np.median(local, axis=0)


def rays(features, offsets, gains):
    """Corrected gaze rays: the midpoint between the eyes and one direction."""
    rotations = head_rotations(features)
    direction = features[:, 13:16]+features[:, 16:19]
    direction = direction/np.linalg.norm(direction, axis=1, keepdims=True)
    local = np.einsum('nji,nj->ni', rotations, direction)
    side = gains[0]*np.arctan2(local[:, 0], local[:, 2])
    lift = gains[1]*np.arcsin(np.clip(local[:, 1], -1, 1))
    corrected = np.stack([np.sin(side)*np.cos(lift), np.sin(lift), np.cos(side)*np.cos(lift)], axis=1)
    origin = features[:, 4:7]+np.einsum('nij,j->ni', rotations, np.mean(offsets, axis=0))
    return origin, np.einsum('nij,nj->ni', rotations, corrected)


class ArkitEyes:
    """ARKit's eye rotation as the eye signal; the fit scales it by two gains."""

    lower, upper = .3, 1.3

    def __init__(self, features, origins):
        self.features = features
        self.offsets = eye_offsets(features, origins)
        self.start, self.sigma = EYE_GAIN, GAIN_SIGMA

    def rays(self, gains):
        return rays(self.features, self.offsets, gains)


def _rotation(vector):
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    k = vector/angle
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*cross@cross


def _candidate_frames(features):
    """Screens that face the user: four turns in the plane, both mirror senses."""
    forward = head_rotations(features)[:, :, 2].mean(axis=0)
    normal = -forward/np.linalg.norm(forward)
    seed = np.array([1., 0, 0]) if abs(normal[0]) < .9 else np.array([0, 1., 0])
    first = seed-normal*(seed@normal)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    frames = []
    for mirror in (1., -1.):
        for turn in range(4):
            right = np.cos(turn*np.pi/2)*first+np.sin(turn*np.pi/2)*second
            frames.append(np.stack([right, mirror*np.cross(normal, right), normal], axis=1))
    return frames


def _center(display):
    return display[:2]+display[2:]/2


def project(features, display, frame, center, pixel_m, offsets, gains):
    """Screen pixels for ARKit readings; NaN when the ray misses the plane."""
    features = np.atleast_2d(np.asarray(features, dtype=float))
    return project_rays(*rays(features, offsets, gains), display, frame, center, pixel_m)


def project_rays(origin, direction, display, frame, center, pixel_m):
    """Screen pixels hit by each gaze ray; NaN when the ray misses the plane."""
    relative = origin-center
    approach = direction@frame[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        distance = -(relative@frame[:, 2])/approach
        hit = relative+distance[:, None]*direction
        pixels = np.c_[hit@frame[:, 0], hit@frame[:, 1]]/pixel_m+_center(display)
    # A ray that only grazes the plane lands anywhere; it is a miss, not an edge.
    hits = np.isfinite(distance) & (distance > 0) & (distance < MAX_RAY_M) & (np.abs(approach) > MIN_APPROACH)
    pixels[~hits] = np.nan
    return pixels


def fit(bursts, origins, targets, display, pixel_mm=None):
    """Fit the screen plane and the eye gains to the setup fixations."""
    features = np.concatenate([np.asarray(burst, dtype=float) for burst in bursts])
    eyes = ArkitEyes(features, np.concatenate([np.asarray(burst, dtype=float) for burst in origins]))
    screen = fit_screen(eyes, [len(burst) for burst in bursts], targets, display, pixel_mm)
    return dict(screen, gains=screen['eye'], eye_offsets=eyes.offsets)


def fit_screen(eyes, counts, targets, display, pixel_mm=None):
    """Fit the plane and the few numbers an eye signal leaves open.

    `eyes` holds the frames of every fixation in order: `features`, a
    `rays(values)` method, and for its own values the typical ones in `start`
    with their `sigma`, `lower` and `upper`.
    """
    display = np.asarray(display, dtype=float)
    features = eyes.features
    pixels = np.repeat(np.asarray(targets, dtype=float), counts, axis=0)
    # Every fixation counts once, however many frames it kept.
    weights = np.repeat(1/np.sqrt(counts), counts)/NOISE_RAD
    known_size = pixel_mm is not None and PIXEL_RANGE_MM[0] <= pixel_mm <= PIXEL_RANGE_MM[1]
    base_pixel_m = (pixel_mm if known_size else DEFAULT_PIXEL_MM)/1000
    relative = pixels-_center(display)
    eye_center = features[:, 4:7].mean(axis=0)
    typical = np.asarray(eyes.start, dtype=float)
    size_at = 6+len(typical)

    def unpack(values):
        return (values[:3], values[3:6], values[6:size_at],
                base_pixel_m if known_size else base_pixel_m*np.exp(values[size_at]))

    def residuals(values, base):
        center, turn, eye, pixel_m = unpack(values)
        frame = base@_rotation(turn)
        origin, direction = eyes.rays(eye)
        toward = center+(relative[:, :1]*frame[:, 0]+relative[:, 1:]*frame[:, 1])*pixel_m-origin
        toward /= np.linalg.norm(toward, axis=1, keepdims=True)
        # People sit in front of their screen: its normal points at them.
        sight = eye_center-center
        facing = np.array([sight@frame[:, 0], sight@frame[:, 1]])/(np.linalg.norm(sight)*FACING_SIGMA)
        prior = np.r_[(eye-typical)/eyes.sigma, facing]
        if not known_size:
            # One seat cannot tell a large far screen from a small near one.
            prior = np.r_[prior, values[size_at]/SIZE_SIGMA]
        return np.r_[((direction-toward)*weights[:, None]).ravel(), prior]

    def solution(values, base):
        """The fitted screen, or None when most rays of some fixation miss it."""
        center, turn, eye, pixel_m = unpack(values)
        frame = base@_rotation(turn)
        viewing = np.linalg.norm(eye_center-center)
        if not VIEWING_RANGE_M[0] <= viewing <= VIEWING_RANGE_M[1]:
            return None
        predicted = project_rays(*eyes.rays(eye), display, frame, center, pixel_m)
        errors = []
        for first, count, target in zip(np.r_[0, np.cumsum(counts)[:-1]], counts, np.asarray(targets, dtype=float)):
            landed = predicted[first:first+count]
            landed = landed[np.isfinite(landed).all(axis=1)]
            if len(landed)*2 < count:
                return None
            errors.append(np.linalg.norm(np.median(landed, axis=0)-target))
        return {'frame': frame, 'center': center, 'pixel_m': float(pixel_m), 'eye': eye,
                'known_size': bool(known_size), 'viewing_m': float(viewing),
                'fit_error_px': float(np.median(errors))}

    lower = np.r_[[-3.]*3, [-1.3]*3, np.broadcast_to(eyes.lower, typical.shape), -1.4]
    upper = np.r_[[3.]*3, [1.3]*3, np.broadcast_to(eyes.upper, typical.shape), 1.4]
    if known_size:
        lower, upper = lower[:size_at], upper[:size_at]
    best = None
    for base in _candidate_frames(features):
        for distance in (.5, .8):
            start = np.r_[eye_center-base[:, 2]*distance, 0, 0, 0, typical, 0][:len(lower)]
            try:
                # Cauchy loss: a frame caught mid-blink or mid-saccade must not move the screen.
                result = least_squares(residuals, start, args=(base,), bounds=(lower, upper),
                                       loss='cauchy', f_scale=1.5, max_nfev=200)
            except (ValueError, FloatingPointError):
                continue
            if best is not None and result.cost >= best[0]:
                continue
            screen = solution(result.x, base)
            if screen is not None:
                best = result.cost, screen
    if best is None:
        raise ValueError('Mira couldn’t work out where the screen is. Keep the iPhone still and try setup again.')
    return best[1]
