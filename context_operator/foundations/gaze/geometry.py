"""ARKit camera coordinates stay unchanged for calibration.

Portrait presentation and human-readable head angles share this proper rotation;
mirroring is applied only at the final pixel projection, never to eye identities.
"""
import math
import numpy as np

PORTRAIT = np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])


def head_angles(face):
    r=PORTRAIT@face[:3,:3]
    return (math.atan2(r[0,2],r[2,2]),
            math.atan2(-r[1,2],math.hypot(r[0,2],r[2,2])),
            math.atan2(r[1,0],r[0,0]))


def eye_ray_points(eye):
    origin=eye[:3,3]
    direction=eye[:3,2]
    length=.18
    if direction[2]>0:
        length=min(length,max(0.,(-.02-origin[2])/direction[2]))
    return np.array([origin,origin+direction*length])


def lens_angle(eye):
    """Angular separation from the camera optical centre, not gaze accuracy."""
    toward_camera=-eye[:3,3]
    length=np.linalg.norm(toward_camera)
    if length<1e-6:return None
    return math.degrees(math.acos(float(np.clip(np.dot(eye[:3,2],toward_camera/length),-1,1))))
