import numpy as np
import scipy as sp
from stonesoup.base import Property
from stonesoup.models.base import LinearModel, GaussianModel, TimeVariantModel
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition.linear import LinearGaussianTransitionModel
from stonesoup.types.array import CovarianceMatrix


class SDFMessmodell(MeasurementModel, LinearModel, GaussianModel):
    noise_covar = Property(CovarianceMatrix, doc="Noise covariance")

    @property
    def ndim_meas(self):
        return 2

    def covar(self):
        # return np.array((1, 0, 0, 0), (0, 0, 1, 0))
        return self.noise_covar

    def rvs(self):
        return 0.5

    def pdf(self):
        pass

    def matrix(self, **kwargs):
        model_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        return model_matrix


class PCWAModel(LinearGaussianTransitionModel, TimeVariantModel):

    def matrix(self, time_interval, **kwargs):
        delta_t = int(time_interval.total_seconds())
        return sp.array([[1, delta_t], [0, 1]])

    def covar(self, time_interval, **kwargs):
        time_interval_sec = int(time_interval.total_seconds())
        Sigma = 5.0

        covar = sp.array([[sp.power(time_interval_sec, 4) / 4,
                           sp.power(time_interval_sec, 3) / 2],
                          [sp.power(time_interval_sec, 3) / 2,
                           sp.power(time_interval_sec, 2)]]) * sp.power(Sigma, 2)

        return CovarianceMatrix(covar)
