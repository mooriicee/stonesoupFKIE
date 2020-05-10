from functools import lru_cache

import numpy as np
import scipy as sp
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from stonesoup.models.base import LinearModel, GaussianModel, TimeVariantModel
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition.linear import LinearGaussianTransitionModel
from stonesoup.types.array import CovarianceMatrix
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import GaussianMeasurementPrediction
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.updater import Updater
from stonesoup.predictor import Predictor
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.state import GaussianState


class SDFMessmodell(MeasurementModel, LinearModel, GaussianModel):

    @property
    def ndim_meas(self):
        return 2

    def matrix(self, **kwargs):
        model_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        return model_matrix

    def covar(self):
        sigma = 50
        cov = CovarianceMatrix([[np.power(sigma, 2), 0], [0, np.power(sigma, 2)]])
        return cov

    def rvs(self):
        noise = multivariate_normal.rvs(np.zeros(self.ndim_meas), self.covar(), 1)
        return noise.reshape((-1, 1))

    def pdf(self):
        pass

class PCWAModel(LinearGaussianTransitionModel, TimeVariantModel):

    def matrix(self, timedelta=5, **kwargs):
        delta_t = timedelta
        F = np.array([[1, delta_t], [0, 1]])
        return block_diag(F, F)


    def covar(self, timedelta=5, **kwargs):
        delta_t = timedelta
        Sigma = 5.0

        covar = np.array([[np.power(delta_t, 4) / 4,
                           np.power(delta_t, 3) / 2],
                          [np.power(delta_t, 3) / 2,
                           np.power(delta_t, 2)]]) * np.power(Sigma, 2)

        covar = block_diag(covar, covar)

        return CovarianceMatrix(covar)


class SdfKalmanPredictor(Predictor):
    @lru_cache()
    def predict(self, prior, timestamp=0, **kwargs):
        delta_t = timestamp - prior.timestamp
        # Transition model parameters
        transition_matrix = self.transition_model.matrix(timedelta=delta_t, **kwargs)
        transition_noise_covar = self.transition_model.covar(timedelta=delta_t, **kwargs)

        # Perform prediction
        prediction_mean = transition_matrix @ prior.mean
        prediction_covar = transition_matrix @ prior.covar @ transition_matrix + transition_noise_covar

        return GaussianStatePrediction(prediction_mean, prediction_covar)


class SDFUpdater(Updater):
    def get_measurement_prediction(self, state_prediction, measurement_model=None, **kwargs):
        pass

    def update(self, hypothesis, measurementmodel, **kwargs):
        measurement_matrix = measurementmodel.matrix()  # H
        measurement_noise_covar = measurementmodel.covar()  # R
        prediction_covar = hypothesis.prediction.covar  # P
        messprediction = measurement_matrix @ hypothesis.prediction.mean  # H @ x

        S = measurement_matrix @ prediction_covar @ measurement_matrix.T + measurement_noise_covar  # S
        W = prediction_covar @ measurement_matrix.T @ np.linalg.pinv(S)  # W
        Innovation = hypothesis.measurement.state_vector - (measurement_matrix @ hypothesis.prediction.mean)  # v

        x_post = hypothesis.prediction.mean + W @ Innovation  # x + W @ v
        P_post = prediction_covar - (W @ S @ W.T)  # P - ( W @ S @ W.T )

        posterior_mean = x_post
        posterior_covar = P_post
        meas_pred_mean = messprediction
        meas_pred_covar = S

        # Augment hypothesis with measurement prediction
        hypothesis = SingleHypothesis(hypothesis.prediction,
                                      hypothesis.measurement,
                                      GaussianMeasurementPrediction(
                                          meas_pred_mean, meas_pred_covar,
                                          hypothesis.prediction.timestamp)
                                      )

        return GaussianStateUpdate(posterior_mean,
                                   posterior_covar,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)


def retrodict(current_state, prior_state, transition_model):
    delta_t = current_state.timestamp - prior_state.timestamp
    F = transition_model.matrix(timedelta = delta_t)
    D = transition_model.covar(timedelta = delta_t)

    x_ll = prior_state.mean     # Vorzustand (der verbessert wird)
    P_ll = prior_state.covar

    x_l1k = current_state.mean  # momentaner Zustand
    P_l1k = current_state.covar

    x_l1l = F @ x_ll    # predizierter Zustand
    P_l1l = F @ P_ll @ np.transpose(F) + 25*D

    W_l1l = P_ll @ np.transpose(F) @ np.linalg.pinv(P_l1l)   # Gewichtsmatrix

    x_lk = x_ll + W_l1l @ (x_l1k - x_l1l)   # verbesserter Zustand
    P_lk = P_ll + W_l1l @ (P_l1k - P_l1l) @ np.transpose(W_l1l)

    return GaussianState(x_lk, P_lk, timestamp=prior_state.timestamp)
