import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

class CorrectionMap:
    """Regularized local correction, fading to the base away from known anchors.

    Inputs include base screen prediction and head pose. A correction measured
    at one head orientation must not act as a universal screen offset.
    """
    def __init__(self, features, base_predictions, targets, bounds):
        self.bounds = np.asarray(bounds,dtype=float)
        self.support = self.coordinates(features,base_predictions)
        residual = np.asarray(targets)-np.asarray(base_predictions)
        self.coefficients = np.linalg.solve(rbf_kernel(self.support,gamma=.5)+np.eye(len(residual))*.25,residual)
        self.limit = min(600.,float(np.max(np.linalg.norm(residual,axis=1)))*1.15)

    def coordinates(self, features, predictions):
        features = np.atleast_2d(features)
        # Local scale adapts to screen size; rotation/center/scale carry head context.
        screen = (np.atleast_2d(predictions)-self.bounds[:2])/(self.bounds[2:]*[.22,.35])
        if features.shape[1] != 267:
            return screen
        head = features[:,258:267]/np.array([.7]*6+[.12,.12,.3])
        return np.c_[screen,head]

    def adjust(self, features, base_predictions):
        points = self.coordinates(features,base_predictions)
        displacement = rbf_kernel(points,self.support,gamma=.5)@self.coefficients
        length = np.linalg.norm(displacement,axis=1)
        displacement *= np.minimum(1.,self.limit/np.maximum(length,1e-9))[:,None]
        return np.atleast_2d(base_predictions)+displacement
