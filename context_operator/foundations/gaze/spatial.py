"""Effective metric screen plane fitted from ARKit gaze, with a rigid reanchor.

The saved plane and eye offsets live in the original camera frame.  A reanchor
only changes the current-camera -> original-camera rigid transform; it cannot
rescale or warp the user's personal calibration.  A fitted head-local angular
correction separates eye-direction bias from the effective screen plane; neither
is a measurement of anatomical optical axes.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .profile import Profile


ENGINE = 'arkit-truedepth-spatial-v1'


class SpatialFitError(ValueError):
    """A collection failure with JSON-safe measurements for the session receipt."""

    def __init__(self, code, message, diagnostics=None):
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics or {}


def _pose(features):
    x = np.asarray(features, dtype=float)
    if x.shape != (22,) or not np.isfinite(x).all():
        raise ValueError('Invalid TrueDepth data')
    a = x[7:13].reshape(3, 2)
    r = np.column_stack((a, np.cross(a[:, 0], a[:, 1])))
    if not np.allclose(r.T @ r, np.eye(3), atol=.035) or np.linalg.det(r) < .97:
        raise ValueError('Invalid head position')
    directions = x[13:19].reshape(2, 3)
    lengths = np.linalg.norm(directions, axis=1)
    if np.any((lengths < .8) | (lengths > 1.2)):
        raise ValueError('Invalid gaze directions')
    return x[4:7], r, directions / lengths[:, None]


def _records(records, bounds, minimum_targets, require_motion):
    b = np.asarray(bounds, dtype=float)
    if b.shape != (4,) or not np.isfinite(b).all() or np.any(b[2:] <= 0):
        raise SpatialFitError('invalid_display', 'Invalid screen bounds')
    if not isinstance(records, (list, tuple)) or len(records) < minimum_targets:
        raise SpatialFitError('insufficient_targets', 'Too few points on the screen',
                              {'required_targets': minimum_targets,
                               'collected_records': len(records) if isinstance(records, (list, tuple)) else 0})
    targets, features, origins = [], [], []
    for record in records:
        if not isinstance(record, dict) or not {'target', 'features', 'eye_origins'} <= record.keys():
            raise SpatialFitError('invalid_record', 'Gaze measurements are incomplete')
        target = np.asarray(record['target'], dtype=float)
        x = np.asarray(record['features'], dtype=float)
        o = np.asarray(record['eye_origins'], dtype=float)
        if (target.shape != (2,) or x.ndim != 2 or x.shape[1:] != (22,) or
                len(x) < 5 or o.shape != (len(x), 2, 3) or
                not np.isfinite(target).all() or not np.isfinite(x).all() or
                not np.isfinite(o).all()):
            raise SpatialFitError('invalid_record', 'Gaze measurements are incomplete')
        if np.any(target < b[:2]) or np.any(target > b[:2] + b[2:]):
            raise SpatialFitError('target_outside_display', 'Point is outside the screen')
        for row, eye_origins in zip(x, o):
            _, _, directions = _pose(row)
            if np.any(directions[:, 2] <= .1):
                raise SpatialFitError('reverse_eye_axis', 'Gaze is directed away from the screen')
            intersections = eye_origins[:, :2] - eye_origins[:, 2:3] * directions[:, :2] / directions[:, 2:3]
            if np.max(np.linalg.norm(intersections - row[:4].reshape(2, 2), axis=1)) > .015:
                raise SpatialFitError('inconsistent_eye_origins',
                                      'Eye positions don’t match the TrueDepth data')
        targets.extend([target] * len(x))
        features.extend(x)
        origins.extend(o)
    y = np.asarray(targets)
    x = np.asarray(features)
    o = np.asarray(origins)
    unique = np.unique(y, axis=0)
    uv = (unique - b[:2]) / b[2:]
    quadrants = [np.any((uv[:, 0] < .3 if col == 0 else uv[:, 0] > .7) &
                        (uv[:, 1] < .3 if row == 0 else uv[:, 1] > .7))
                 for row in range(2) for col in range(2)]
    if (len(unique) < minimum_targets or np.ptp(uv[:, 0]) < .55 or
            np.ptp(uv[:, 1]) < .55 or not all(quadrants) or
            np.linalg.svd(uv - uv.mean(axis=0), compute_uv=False)[-1] < .25):
        raise SpatialFitError('insufficient_target_coverage', 'Points don’t cover the screen',
                              {'distinct_targets': int(len(unique)),
                               'horizontal_span': float(np.ptp(uv[:, 0])),
                               'vertical_span': float(np.ptp(uv[:, 1]))})
    if require_motion:
        # Time or pose rotation alone is insufficient to identify metric depth.
        centers = x[:, 4:7]
        spans = np.percentile(centers, 95, axis=0) - np.percentile(centers, 5, axis=0)
        if np.linalg.norm(spans) < .08 or np.count_nonzero(spans > .025) < 2:
            raise SpatialFitError('insufficient_head_translation',
                                  'Move your head to several distinct positions',
                                  {'head_span_m': spans.tolist()})
    return x, y, o


def _linear_design(raw_slopes):
    n = len(raw_slopes)
    return np.column_stack((np.tile(np.eye(2), (n, 1)), raw_slopes.reshape(-1, 2)))


def _rays(features, origins, gain=None, bias=None, linear=None):
    directions = features[:, 13:19].reshape(-1, 2, 3)
    directions = directions / np.linalg.norm(directions, axis=2)[:, :, None]
    if gain is not None:
        frames = np.array([_pose(row)[1] for row in features])
        local = np.einsum('nji,nkj->nki', frames, directions)
        if np.any(local[:, :, 2] < .15):
            raise SpatialFitError('invalid_eye_axis', 'Your eyes turned too far away from the screen')
        raw_slopes = local[:, :, :2] / local[:, :, 2:]
        slopes = (raw_slopes - bias) / gain
        if linear is not None:
            center, scale, coefficient = linear
            delta = ((_linear_design(raw_slopes) - center) / scale) @ coefficient
            slopes += delta.reshape(-1, 2, 2)
        corrected = np.concatenate((slopes, np.ones((*slopes.shape[:2], 1))), axis=2)
        corrected /= np.linalg.norm(corrected, axis=2, keepdims=True)
        directions = np.einsum('nij,nkj->nki', frames, corrected)
    return origins.reshape(-1, 3), directions.reshape(-1, 3)


def _plane_points(y, bounds, center, horizontal, vertical):
    uv = (y - bounds[:2]) / bounds[2:] - .5
    return center + uv[:, :1] * horizontal + uv[:, 1:] * vertical


def _cylinder_points(y, bounds, center, horizontal, vertical, radius):
    uv = (y - bounds[:2]) / bounds[2:] - .5
    width, height = np.linalg.norm(horizontal), np.linalg.norm(vertical)
    right, down = horizontal / width, vertical / height
    normal = np.cross(right, down)
    theta = uv[:, 0] * width / radius
    return (center + radius * np.sin(theta)[:, None] * right +
            radius * (1-np.cos(theta))[:, None] * normal +
            uv[:, 1:2] * vertical)


def _screen_descriptor(value):
    if (not isinstance(value, dict) or set(value) !=
            {'kind', 'width_m', 'height_m', 'radius_m'} or value.get('kind') != 'cylinder'):
        raise ValueError('Invalid curved screen dimensions')
    try:
        width, height, radius = [float(value[name]) for name in ('width_m', 'height_m', 'radius_m')]
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid curved screen dimensions') from exc
    if (not np.isfinite([width, height, radius]).all() or
            not .35 <= width <= 1.8 or not .15 <= height <= 1.0 or
            not .3 <= radius <= 5.0 or width / (2*radius) >= 1.3):
        raise ValueError('The curved screen size or radius is out of range')
    return {'kind': 'cylinder', 'width_m': width, 'height_m': height, 'radius_m': radius}


def _ray_errors(points, origins, directions):
    displacement = points - origins
    along = np.sum(displacement * directions, axis=1)
    perpendicular = displacement - along[:, None] * directions
    return np.linalg.norm(perpendicular, axis=1), along


def _fit_plane(x, y, o, bounds):
    ray_o, ray_d = _rays(x, o)
    uv = (y - bounds[:2]) / bounds[2:] - .5
    uv = np.repeat(uv, 2, axis=0)
    projectors = np.eye(3)[None] - ray_d[:, :, None] * ray_d[:, None, :]
    design = np.concatenate((projectors, uv[:, :1, None] * projectors,
                             uv[:, 1:, None] * projectors), axis=2)
    system = design.reshape(-1, 9)
    singular = np.linalg.svd(system, compute_uv=False)
    if singular[-1] < 1e-8 or singular[0] / singular[-1] > 1e4:
        raise SpatialFitError('underdetermined_plane',
                              'Move your head to more positions so Mira can locate the screen',
                              {'plane_condition': float(singular[0] / max(singular[-1], 1e-12))})
    initial = np.linalg.lstsq(system,
                              np.einsum('nij,nj->ni', projectors, ray_o).ravel(), rcond=None)[0]
    def residual(beta):
        points = beta[:3] + uv[:, :1] * beta[3:6] + uv[:, 1:] * beta[6:9]
        return np.einsum('nij,nj->ni', projectors, points - ray_o).ravel()
    fit = least_squares(residual, initial, loss='soft_l1', f_scale=.025, max_nfev=300)
    c, u, v = fit.x[:3], fit.x[3:6], fit.x[6:9]
    return c, u, v


def _fit_calibrated_plane(x, y, o, bounds):
    """Jointly fit a rigid pixel plane and an identifiable head-local eye model.

    Raw ARKit eye +Z is a measured axis, not an exact optical gaze ray.  Each
    eye has a constant angular offset; both eyes share horizontal/vertical
    gain.  Head translation separates angular gain from metric screen size.
    """
    raw_c, raw_u, raw_v = _fit_plane(x, y, o, bounds)
    raw_points = np.repeat(_plane_points(y, bounds, raw_c, raw_u, raw_v), 2, axis=0)
    raw_o, raw_d = _rays(x, o)
    raw_errors, _ = _ray_errors(raw_points, raw_o, raw_d)
    if np.max(raw_errors) < 1e-7:
        # Exact physical rays need no personal angular correction.  Besides
        # preserving precision, this avoids moving a noiseless solution under
        # weak regularization merely to make gains numerically closer to one.
        try:
            _check_geometry(raw_c, raw_u, raw_v, x, y, o, bounds)
        except SpatialFitError:
            pass
        else:
            return (raw_c, raw_u, raw_v, np.ones(2), np.zeros((2, 2)),
                    {'raw_plane_width_m': float(np.linalg.norm(raw_u)),
                     'raw_plane_height_m': float(np.linalg.norm(raw_v)),
                     'fitted_width_m': float(np.linalg.norm(raw_u)),
                     'fitted_height_m': float(np.linalg.norm(raw_v)),
                     'angular_gain': [1., 1.],
                     'angular_bias': [[0., 0.], [0., 0.]],
                     'angular_residual_median': 0., 'fit_mode': 'exact_raw_rays'})
    uv = (y - bounds[:2]) / bounds[2:] - .5
    frames = np.array([_pose(row)[1] for row in x])
    directions = x[:, 13:19].reshape(-1, 2, 3)
    local = np.einsum('nji,nkj->nki', frames, directions)
    if np.any(local[:, :, 2] < .15):
        raise SpatialFitError('invalid_eye_axis', 'Your eyes turned too far away from the screen')
    raw_slopes = local[:, :, :2] / local[:, :, 2:]
    aspect = float(bounds[2] / bounds[3])
    right = raw_u / max(np.linalg.norm(raw_u), 1e-8)
    down = raw_v - np.dot(raw_v, right) * right
    if np.linalg.norm(down) < .02:
        raise SpatialFitError('underdetermined_orientation',
                              'Mira couldn’t determine the screen angle from your gaze',
                              {'raw_plane_width_m': float(np.linalg.norm(raw_u)),
                               'raw_plane_height_m': float(np.linalg.norm(raw_v))})
    down /= np.linalg.norm(down)
    rotation = Rotation.from_matrix(np.column_stack((right, down, np.cross(right, down)))).as_rotvec()

    def unpack(parameters):
        center = parameters[:3]
        axes = Rotation.from_rotvec(parameters[3:6]).as_matrix()
        width, ratio, skew = parameters[6:9]
        horizontal = axes[:, 0] * width
        vertical = (axes[:, 1] + skew * axes[:, 0]) * (width / (aspect * ratio)) / np.sqrt(1 + skew*skew)
        gain = parameters[9:11]
        bias = parameters[11:15].reshape(2, 2)
        return center, horizontal, vertical, gain, bias

    def residual(parameters):
        center, horizontal, vertical, gain, bias = unpack(parameters)
        points = center + uv[:, :1] * horizontal + uv[:, 1:] * vertical
        towards = np.einsum('nji,nkj->nki', frames, points[:, None, :] - o)
        # A point behind the head-local eye plane is infeasible.  Clipping keeps
        # the optimizer finite while the final forward-ray gate rejects it.
        depth = np.maximum(towards[:, :, 2:], .08)
        expected = gain * towards[:, :, :2] / depth + bias
        data = (expected - raw_slopes).ravel()
        regularization = np.r_[.012 * (gain - 1.), .008 * bias.ravel(),
                               .025 * (parameters[7] - 1.), .008 * parameters[8]]
        return np.r_[data, regularization]

    lower = np.r_[[-1.5, -1.5, -1., -3.2, -3.2, -3.2, .35, .65, -.21],
                  [.20, .20], np.full(4, -.50)]
    upper = np.r_[[1.5, 1.5, 1., 3.2, 3.2, 3.2, 1.8, 1.55, .21],
                  [2.5, 2.5], np.full(4, .50)]
    fits = []
    for width in (np.clip(np.linalg.norm(raw_u), .36, 1.75),
                  np.clip(max(.8, aspect * .27), .36, 1.75)):
        raw_height = np.linalg.norm(raw_v)
        gain = np.clip([np.linalg.norm(raw_u) / width,
                        raw_height / (width / aspect)], .21, 2.49)
        center = np.clip(raw_c, lower[:3] + 1e-4, upper[:3] - 1e-4)
        skew = np.clip(np.dot(raw_v, right) / max(np.linalg.norm(raw_v), 1e-8), -.2, .2)
        seed = np.r_[center, rotation, width, 1., skew, gain, np.zeros(4)]
        fit = least_squares(residual, seed, bounds=(lower, upper), loss='soft_l1',
                            f_scale=.04, max_nfev=220)
        fits.append(fit)
    fit = min(fits, key=lambda result: np.linalg.norm(result.fun))
    c, u, v, gain, bias = unpack(fit.x)
    singular = np.linalg.svd(fit.jac[:len(x)*4], compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))
    raw_width, raw_height = np.linalg.norm(raw_u), np.linalg.norm(raw_v)
    diagnostics = {'raw_plane_width_m': float(raw_width),
                   'raw_plane_height_m': float(raw_height),
                   'fitted_width_m': float(np.linalg.norm(u)),
                   'fitted_height_m': float(np.linalg.norm(v)),
                   'angular_gain': gain.tolist(),
                   'angular_bias': bias.tolist(),
                   'fit_condition': condition,
                   'angular_residual_median': float(np.median(abs(fit.fun[:len(x)*4])))}
    if not fit.success or singular[-1] < 1e-7 or condition > 1e5:
        raise SpatialFitError('underdetermined_angular_fit',
                              'More head movement is needed to estimate screen size and correct gaze angle',
                              diagnostics)
    if (np.any(gain <= .205) or np.any(gain >= 2.49) or
            np.any(abs(bias) >= .49) or fit.x[6] <= .355 or fit.x[6] >= 1.795 or
            abs(fit.x[8]) >= .209):
        raise SpatialFitError('angular_fit_at_limit',
                              'The estimated screen dimensions are out of range',
                              diagnostics)
    return c, u, v, gain, bias, diagnostics


def _fit_cylinder(x, y, o, bounds, geometry):
    width, height, radius = (geometry[key] for key in ('width_m', 'height_m', 'radius_m'))
    raw_c, raw_u, raw_v = _fit_plane(x, y, o, bounds)
    try:
        c0, u0, v0, gain0, bias0, _ = _fit_calibrated_plane(x, y, o, bounds)
    except SpatialFitError:
        c0, u0, v0 = raw_c, raw_u, raw_v
        gain0 = np.clip([np.linalg.norm(raw_u)/width,
                         np.linalg.norm(raw_v)/height], .21, 2.49)
        bias0 = np.zeros((2, 2))
    right = u0 / np.linalg.norm(u0)
    down = v0 - np.dot(v0, right)*right
    if np.linalg.norm(down) < .02:
        raise SpatialFitError('underdetermined_orientation',
                              'Mira couldn’t determine the curved screen’s orientation from your gaze')
    down /= np.linalg.norm(down)
    axes0 = np.column_stack((right, down, np.cross(right, down)))
    initial_rotation = Rotation.from_matrix(axes0).as_rotvec()
    poses = np.array([_pose(row)[1] for row in x])
    directions = x[:, 13:19].reshape(-1, 2, 3)
    local = np.einsum('nji,nkj->nki', poses, directions)
    if np.any(local[:, :, 2] < .15):
        raise SpatialFitError('invalid_eye_axis', 'Your eyes turned too far away from the screen')
    observed = local[:, :, :2]/local[:, :, 2:]

    def residual(parameters):
        center = parameters[:3]
        axes = Rotation.from_rotvec(parameters[3:6]).as_matrix()
        gain = parameters[6:8]
        bias = parameters[8:12].reshape(2, 2)
        points = _cylinder_points(y, bounds, center, width*axes[:, 0],
                                  height*axes[:, 1], radius)
        toward = np.einsum('nji,nkj->nki', poses, points[:, None, :] - o)
        depth = np.maximum(toward[:, :, 2:], .08)
        expected = gain * toward[:, :, :2]/depth + bias
        return np.r_[(expected-observed).ravel(), .012*(gain-1), .008*bias.ravel()]

    lower = np.r_[[-1.5, -1.5, -1., -3.2, -3.2, -3.2, .2, .2], np.full(4, -.5)]
    upper = np.r_[[1.5, 1.5, 1., 3.2, 3.2, 3.2, 2.5, 2.5], np.full(4, .5)]
    candidates = []
    for displacement in (0., -.10, .10):
        center = np.clip(c0 + displacement*axes0[:, 2], lower[:3]+1e-4, upper[:3]-1e-4)
        seed = np.r_[center, initial_rotation, gain0, bias0.ravel()]
        candidates.append(least_squares(residual, seed, bounds=(lower, upper),
                                        loss='soft_l1', f_scale=.04, max_nfev=300))
    fit = min(candidates, key=lambda value: np.linalg.norm(value.fun))
    axes = Rotation.from_rotvec(fit.x[3:6]).as_matrix()
    gain, bias = fit.x[6:8], fit.x[8:12].reshape(2, 2)
    singular = np.linalg.svd(fit.jac[:len(x)*4], compute_uv=False)
    condition = float(singular[0]/max(singular[-1], 1e-12))
    diagnostics = {'geometry_kind': 'cylinder', 'screen_width_m': width,
                   'screen_height_m': height, 'screen_radius_m': radius,
                   'raw_plane_width_m': float(np.linalg.norm(raw_u)),
                   'raw_plane_height_m': float(np.linalg.norm(raw_v)),
                   'angular_gain': gain.tolist(), 'angular_bias': bias.tolist(),
                   'fit_condition': condition,
                   'angular_residual_median': float(np.median(abs(fit.fun[:len(x)*4])))}
    if not fit.success or singular[-1] < 1e-7 or condition > 1e5:
        raise SpatialFitError('underdetermined_cylinder_fit',
                              'Mira couldn’t locate the curved screen from these measurements', diagnostics)
    if np.any(gain <= .205) or np.any(gain >= 2.49) or np.any(abs(bias) >= .49):
        raise SpatialFitError('cylinder_fit_at_limit',
                              'The estimated screen dimensions are out of range', diagnostics)
    return fit.x[:3], width*axes[:, 0], height*axes[:, 1], gain, bias, diagnostics


def _fit_linear_correction(x, y, o, bounds, geometry, center, horizontal, vertical, gain, bias):
    poses = np.array([_pose(row)[1] for row in x])
    directions = x[:, 13:19].reshape(-1, 2, 3)
    local = np.einsum('nji,nkj->nki', poses, directions)
    raw = local[:, :, :2]/local[:, :, 2:]
    base = (raw-bias)/gain
    points = _cylinder_points(y, bounds, center, horizontal, vertical, geometry['radius_m'])
    toward = np.einsum('nji,nkj->nki', poses, points[:, None, :] - o)
    truth = toward[:, :, :2]/toward[:, :, 2:]
    design = _linear_design(raw)
    target = (truth-base).reshape(-1, 2)
    mean = design.mean(axis=0)
    mean[:2] = 0.
    scale = np.maximum(design.std(axis=0), .03)
    scale[:2] = 1.
    z = (design-mean)/scale
    penalty = np.diag([0., 0., 10., 10.])
    coef = np.linalg.solve(z.T@z + penalty + np.eye(4)*1e-8, z.T@target)
    correction = (z@coef).reshape(-1, 2, 2)
    if np.percentile(np.linalg.norm(correction, axis=2), 99) > .25:
        raise SpatialFitError('excessive_angular_correction',
                              'The angle correction is too large for reliable tracking')
    return mean, scale, coef


def _check_cylinder(c, u, v, x, y, o, bounds, geometry, gain, bias, linear, diagnostics):
    points = np.repeat(_cylinder_points(y, bounds, c, u, v, geometry['radius_m']), 2, axis=0)
    ray_o, ray_d = _rays(x, o, gain, bias, linear)
    errors, along = _ray_errors(points, ray_o, ray_d)
    details = dict(diagnostics, ray_error_median_m=float(np.median(errors)),
                   ray_error_p90_m=float(np.percentile(errors, 90)),
                   forward_distance_min_m=float(np.min(along)),
                   forward_distance_max_m=float(np.max(along)))
    if (np.any(along < .10) or np.any(along > 3.) or
            np.median(errors) > .075 or np.percentile(errors, 90) > .135):
        raise SpatialFitError('cylinder_ray_mismatch',
                              'Gaze measurements don’t converge on the curved screen', details)
    return float(np.median(errors))


def _check_geometry(c, u, v, x, y, o, bounds, gain=None, bias=None, diagnostics=None,
                    error_limit=.075):
    width, height = np.linalg.norm(u), np.linalg.norm(v)
    skew = float(abs(np.dot(u, v) / max(width * height, 1e-12)))
    details = dict(diagnostics or {}, fitted_width_m=float(width),
                   fitted_height_m=float(height), fitted_skew=skew,
                   fitted_pixel_aspect_ratio=float((width / max(height, 1e-12)) / (bounds[2] / bounds[3])))
    if (not .35 <= width <= 1.8 or not .15 <= height <= 1.0 or
            skew > .22 or
            not .65 <= details['fitted_pixel_aspect_ratio'] <= 1.55):
        raise SpatialFitError('implausible_screen_geometry',
                              'The estimated screen size or angle is out of range', details)
    points = np.repeat(_plane_points(y, bounds, c, u, v), 2, axis=0)
    ray_o, ray_d = _rays(x, o, gain, bias)
    errors, along = _ray_errors(points, ray_o, ray_d)
    normal = np.cross(u, v) / (width * height)
    details.update(ray_error_median_m=float(np.median(errors)),
                   ray_error_p90_m=float(np.percentile(errors, 90)),
                   forward_distance_min_m=float(np.min(along)),
                   forward_distance_max_m=float(np.max(along)))
    if (np.any(along < .10) or np.any(along > 3.) or
            np.median(errors) > error_limit or
            np.percentile(errors, 90) > error_limit * 1.8 or
            np.median(abs(ray_d @ normal)) < .15):
        raise SpatialFitError('ray_plane_mismatch',
                              'Gaze measurements don’t converge on the screen', details)
    return float(np.median(errors))


class SpatialProfile(Profile):
    def __init__(self, arrays, metadata, settings):
        curved = isinstance(settings, dict) and settings.get('kind') == 'spatial-v3'
        required={'mean','scale','target_scale','training_x','training_y','support',
                  'plane_center','plane_u','plane_v','eye_offsets','camera_rotation',
                  'camera_translation','angular_gain','angular_bias'}
        if curved:
            required |= {'angular_linear_center', 'angular_linear_scale', 'angular_linear_coef'}
        if isinstance(arrays, dict) and ('angular_gain' not in arrays or 'angular_bias' not in arrays):
            raise ValueError('Run setup again to update your older spatial gaze profile')
        if (not isinstance(arrays,dict) or not required<=arrays.keys() or
                not isinstance(metadata,dict) or not isinstance(settings,dict) or
                any(not isinstance(a,np.ndarray) or a.dtype.kind not in 'fiu' for a in arrays.values())):
            raise ValueError('Saved spatial gaze measurements are incomplete')
        super().__init__(arrays, metadata, settings)
        if (settings.get('version') != 2 or settings.get('kind') not in ('spatial-v2', 'spatial-v3') or
                metadata.get('engine') != ENGINE or
                np.asarray(metadata.get('display')).shape != (4,) or
                self.training_x.shape[1:] != (22,) or
                arrays['mean'].shape != (22,) or arrays['scale'].shape != (22,)):
            raise ValueError('Invalid spatial gaze profile')
        bounds = np.asarray(metadata['display'], dtype=float)
        if not np.isfinite(bounds).all() or np.any(bounds[2:] <= 0):
            raise ValueError('Invalid screen bounds')
        geometry = _screen_descriptor(metadata.get('screen_geometry')) if curved else None
        if not curved and 'screen_geometry' in metadata:
            raise ValueError('Invalid screen layout in the spatial gaze profile')
        shapes = {'plane_center': (3,), 'plane_u': (3,), 'plane_v': (3,),
                  'eye_offsets': (2, 3), 'camera_rotation': (3, 3),
                  'camera_translation': (3,), 'angular_gain': (2,),
                  'angular_bias': (2, 2)}
        if any(arrays.get(key, np.empty(0)).shape != shape for key, shape in shapes.items()):
            raise ValueError('Invalid screen layout')
        r = arrays['camera_rotation']
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1., atol=1e-5):
            raise ValueError('Camera realignment failed')
        if (np.linalg.norm(arrays['camera_translation']) > 1.0 or
                Rotation.from_matrix(r).magnitude() > 1.2):
            raise ValueError('The camera moved too far for realignment')
        offsets = arrays['eye_offsets']
        ipd = np.linalg.norm(offsets[0] - offsets[1])
        if not .04 <= ipd <= .09 or np.max(np.linalg.norm(offsets, axis=1)) > .12:
            raise ValueError('The estimated distance between eyes is out of range')
        if (np.any(arrays['angular_gain'] < .2) or np.any(arrays['angular_gain'] > 2.5) or
                np.any(abs(arrays['angular_bias']) > .5)):
            raise ValueError('Invalid gaze angle correction')
        if curved:
            if (arrays['angular_linear_center'].shape != (4,) or
                    arrays['angular_linear_scale'].shape != (4,) or
                    arrays['angular_linear_coef'].shape != (4, 2) or
                    np.any(arrays['angular_linear_scale'] <= 0)):
                raise ValueError('Invalid gaze position correction')
        if (not isinstance(settings.get('personal_fingerprint'), str) or
                settings['personal_fingerprint'] != self._personal_digest(arrays, metadata)):
            raise ValueError('Your gaze profile is damaged')
        c, u, v = (arrays[key] for key in ('plane_center', 'plane_u', 'plane_v'))
        if (not .35 <= np.linalg.norm(u) <= 1.8 or not .15 <= np.linalg.norm(v) <= 1.0 or
                abs(np.dot(u, v)) > .22 * np.linalg.norm(u) * np.linalg.norm(v) or
                not .65 <= (np.linalg.norm(u) / np.linalg.norm(v)) /
                (bounds[2] / bounds[3]) <= 1.55):
            raise ValueError('Invalid screen plane')
        if curved and (abs(np.linalg.norm(u)-geometry['width_m']) > 1e-4 or
                       abs(np.linalg.norm(v)-geometry['height_m']) > 1e-4 or
                       abs(np.dot(u, v)) > 1e-4*np.linalg.norm(u)*np.linalg.norm(v)):
            raise ValueError('The saved measurements don’t match the curved screen')

    @classmethod
    def fit_records(cls, records, bounds, screen_geometry=None):
        x, y, o = _records(records, bounds, 9, True)
        b = np.asarray(bounds, dtype=float)
        offsets = np.array([(origins - face) @ r for row, origins in zip(x, o)
                            for face, r, _ in [_pose(row)]])
        eye_offsets = np.median(offsets, axis=0)
        if np.percentile(np.linalg.norm(offsets - eye_offsets, axis=2), 90) > .025:
            raise SpatialFitError('unstable_eye_origins',
                                  'Eye position readings are unstable',
                                  {'eye_offset_p90_m': float(np.percentile(
                                      np.linalg.norm(offsets - eye_offsets, axis=2), 90))})
        geometry = _screen_descriptor(screen_geometry) if screen_geometry is not None else None
        if geometry:
            c, u, v, gain, bias, diagnostics = _fit_cylinder(x, y, o, b, geometry)
            linear = _fit_linear_correction(x, y, o, b, geometry, c, u, v, gain, bias)
            fit_error = _check_cylinder(c, u, v, x, y, o, b, geometry, gain, bias, linear, diagnostics)
        else:
            c, u, v, gain, bias, diagnostics = _fit_calibrated_plane(x, y, o, b)
            fit_error = _check_geometry(c, u, v, x, y, o, b, gain, bias, diagnostics)
        # Keep canonical training samples for compatibility with personal models.
        arrays = dict(mean=x.mean(axis=0), scale=np.maximum(x.std(axis=0), .001),
                      target_scale=np.ones(2), training_x=x, training_y=y,
                      support=np.empty((0, 22)), plane_center=c, plane_u=u, plane_v=v,
                      eye_offsets=eye_offsets, angular_gain=gain, angular_bias=bias,
                      camera_rotation=np.eye(3),
                      camera_translation=np.zeros(3))
        metadata = {'display': b.tolist(), 'engine': ENGINE}
        if geometry:
            metadata['screen_geometry'] = geometry
            arrays.update(angular_linear_center=linear[0], angular_linear_scale=linear[1],
                          angular_linear_coef=linear[2])
        settings = {'version': 2, 'kind': 'spatial-v3' if geometry else 'spatial-v2',
                    'personal_fingerprint': cls._personal_digest(arrays, metadata),
                    'fit_median_ray_error_m': fit_error,
                    'fit_diagnostics': diagnostics}
        return cls(arrays, metadata, settings)

    @staticmethod
    def _personal_digest(arrays, metadata):
        digest = hashlib.sha256()
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        for key in ('plane_center', 'plane_u', 'plane_v', 'eye_offsets',
                    'angular_gain', 'angular_bias', 'training_x', 'training_y'):
            value = np.asarray(arrays[key], dtype='<f8')
            digest.update(key.encode())
            digest.update(str(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        for key in ('angular_linear_center', 'angular_linear_scale', 'angular_linear_coef'):
            if key in arrays:
                value = np.asarray(arrays[key], dtype='<f8')
                digest.update(key.encode())
                digest.update(str(value.shape).encode())
                digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    def fingerprint(self):
        return self.settings['personal_fingerprint']

    def canonicalize(self, features):
        raw = np.asarray(features, dtype=float)
        if raw.ndim == 2:
            return np.array([self.canonicalize(row) for row in raw])
        face, face_r, directions = _pose(raw)
        transform = self.arrays['camera_rotation']
        translation = self.arrays['camera_translation']
        origins = face + (face_r @ self.arrays['eye_offsets'].T).T
        new_face = transform @ face + translation
        new_rotation = transform @ face_r
        new_origins = (transform @ origins.T).T + translation
        new_directions = (transform @ directions.T).T
        if np.any(new_directions[:, 2] <= .1):
            raise ValueError('Gaze is directed away from the setup camera plane')
        intersections = new_origins[:, :2] - new_origins[:, 2:3] * new_directions[:, :2] / new_directions[:, 2:3]
        result = raw.copy()
        result[:4] = intersections.ravel()
        result[4:7] = new_face
        result[7:13] = new_rotation[:, :2].ravel()
        result[13:19] = new_directions.ravel()
        result[19:22] = transform @ raw[19:22]
        return result

    def _intersections(self, features):
        x = np.atleast_2d(np.asarray(features, dtype=float))
        c, u, v = (self.arrays[key] for key in ('plane_center', 'plane_u', 'plane_v'))
        geometry = self.metadata.get('screen_geometry')
        linear = (self.arrays['angular_linear_center'], self.arrays['angular_linear_scale'],
                  self.arrays['angular_linear_coef']) if geometry else None
        normal = np.cross(u, v)
        normal /= np.linalg.norm(normal)
        points, differences = [], []
        for row in x:
            face, r, directions = _pose(row)
            origins = face + (r @ self.arrays['eye_offsets'].T).T
            _, directions = _rays(row[None], origins[None], self.arrays['angular_gain'],
                                  self.arrays['angular_bias'], linear)
            directions = directions.reshape(2, 3)
            if geometry:
                radius = geometry['radius_m']
                right, down = u/np.linalg.norm(u), v/np.linalg.norm(v)
                axis = c + radius*normal
                q = origins-axis
                q -= (q@down)[:, None]*down
                d = directions-(directions@down)[:, None]*down
                a = np.sum(d*d, axis=1)
                b = 2*np.sum(q*d, axis=1)
                discriminant = b*b-4*a*(np.sum(q*q, axis=1)-radius*radius)
                if np.any(discriminant < 0) or np.any(a < 1e-8):
                    raise ValueError('Gaze misses the curved screen')
                root = np.sqrt(discriminant)
                first, second = (-b-root)/(2*a), (-b+root)/(2*a)
                along = np.minimum(np.where(first > .10, first, np.inf),
                                   np.where(second > .10, second, np.inf))
                if np.any(along > 3.):
                    raise ValueError('Gaze misses the screen in front of your eyes')
                hit = origins + along[:, None]*directions
                radial = hit-axis
                theta = np.arctan2(radial@right, -radial@normal)
                if np.any(radial@normal >= 0):
                    raise ValueError('Gaze reached the back of the screen plane')
                uv = np.array([.5+radius*np.mean(theta)/geometry['width_m'],
                               .5+np.mean((hit-c)@down)/geometry['height_m']])
            else:
                denominator = directions @ normal
                if np.any(abs(denominator) < .15):
                    raise ValueError('Gaze is nearly parallel to the screen')
                along = ((c - origins) @ normal) / denominator
                if np.any((along < .10) | (along > 3.)):
                    raise ValueError('Gaze misses the screen in front of your eyes')
                hit = origins + along[:, None] * directions
                point = hit.mean(axis=0) - c
                gram = np.array([[u @ u, u @ v], [u @ v, v @ v]])
                uv = np.linalg.solve(gram, [point @ u, point @ v]) + .5
            differences.append(float(np.linalg.norm(hit[0] - hit[1])))
            points.append(np.asarray(self.metadata['display'][:2]) + uv * self.metadata['display'][2:])
        return np.asarray(points), np.asarray(differences)

    def predict_base(self, features):
        points, _ = self._intersections(features)
        return points[0] if np.asarray(features).ndim == 1 else points

    def supports(self, features):
        try:
            points, disparity = self._intersections(features)
            if len(points) != 1 or disparity[0] > .15:
                return False
            b = np.asarray(self.metadata['display'])
            return bool(np.all(points[0] >= b[:2] - .2*b[2:]) and
                        np.all(points[0] <= b[:2] + 1.2*b[2:]))
        except (ValueError, FloatingPointError):
            return False

    def reanchor(self, records):
        b = np.asarray(self.metadata['display'], dtype=float)
        x, y, o = _records(records, b, 5, False)
        # PnP with metric screen targets.  Every origin and direction is measured
        # in the new camera frame; the target points stay in the original frame.
        geometry = self.metadata.get('screen_geometry')
        screen_points = (_cylinder_points(y, b, self.arrays['plane_center'],
                                          self.arrays['plane_u'], self.arrays['plane_v'],
                                          geometry['radius_m']) if geometry else
                         _plane_points(y, b, self.arrays['plane_center'],
                                       self.arrays['plane_u'], self.arrays['plane_v']))
        targets = np.repeat(screen_points, 2, axis=0)
        linear = (self.arrays['angular_linear_center'], self.arrays['angular_linear_scale'],
                  self.arrays['angular_linear_coef']) if geometry else None
        ray_o, ray_d = _rays(x, o, self.arrays['angular_gain'], self.arrays['angular_bias'], linear)
        def residual(params):
            r = Rotation.from_rotvec(params[:3]).as_matrix()
            transformed_o = ray_o @ r.T + params[3:]
            transformed_d = ray_d @ r.T
            delta = targets - transformed_o
            return (delta - np.sum(delta * transformed_d, axis=1)[:, None] * transformed_d).ravel()
        current = self.arrays['camera_rotation']
        seed = np.r_[Rotation.from_matrix(current).as_rotvec(), self.arrays['camera_translation']]
        fit = least_squares(residual, seed, loss='soft_l1', f_scale=.025, max_nfev=400)
        rotation = Rotation.from_rotvec(fit.x[:3]).as_matrix()
        translation = fit.x[3:]
        transformed_o = ray_o @ rotation.T + translation
        transformed_d = ray_d @ rotation.T
        errors, along = _ray_errors(targets, transformed_o, transformed_d)
        target_errors = [np.median(errors.reshape(-1, 2)[(y == target).all(axis=1)])
                         for target in np.unique(y, axis=0)]
        singular = np.linalg.svd(fit.jac, compute_uv=False)
        if (not fit.success or singular[-1] < 1e-8 or singular[0] / singular[-1] > 1e4 or
                np.median(errors) > .05 or np.percentile(errors, 90) > .10 or
                max(target_errors) > .07 or
                np.any((along < .10) | (along > 3.)) or np.linalg.norm(translation) > 1.0 or
                Rotation.from_matrix(rotation).magnitude() > 1.2):
            raise SpatialFitError('reanchor_mismatch',
                                  'Camera realignment doesn’t match the saved screen',
                                  {'reanchor_ray_error_median_m': float(np.median(errors)),
                                   'reanchor_ray_error_p90_m': float(np.percentile(errors, 90)),
                                   'reanchor_condition': float(singular[0] / max(singular[-1], 1e-12))})
        offsets = np.array([(origins - face) @ r for row, origins in zip(x, o)
                            for face, r, _ in [_pose(row)]])
        if np.percentile(np.linalg.norm(offsets - self.arrays['eye_offsets'], axis=2), 90) > .025:
            raise SpatialFitError('reanchor_eye_geometry_changed',
                                  'Eye geometry changed during camera realignment')
        arrays = {key: np.array(value, copy=True) for key, value in self.arrays.items()}
        arrays['camera_rotation'] = rotation
        arrays['camera_translation'] = translation
        settings = dict(self.settings, reanchor_median_ray_error_m=float(np.median(errors)))
        return type(self)(arrays, dict(self.metadata), settings)

    @classmethod
    def load(cls, path: Path, metadata):
        try:
            with np.load(path, allow_pickle=False) as saved:
                actual = json.loads(str(saved['metadata']))
                settings = json.loads(str(saved['settings']))
                if actual != metadata:
                    raise ValueError('The camera or screen changed. Realign the camera or run setup again.')
                arrays = {key: saved[key] for key in saved.files if key not in ('metadata', 'settings')}
            return cls(arrays, actual, settings)
        except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            raise ValueError('The spatial gaze profile is damaged') from exc
