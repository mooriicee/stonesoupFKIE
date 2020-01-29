from functools import lru_cache

import numpy as np
import scipy as sp
from scipy.linalg import block_diag
from stonesoup.base import Property
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
from stonesoup.types.state import State, GaussianState


class SDFMessmodell(MeasurementModel, LinearModel, GaussianModel):
    noise_covar = Property(CovarianceMatrix, doc="Noise covariance")

    @property
    def ndim_meas(self):
        return 2

    def covar(self):
        return self.noise_covar

    def rvs(self):
        return 0.5

    def pdf(self):
        pass

    def matrix(self, **kwargs):
        model_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        return model_matrix


class PCWAModel(LinearGaussianTransitionModel, TimeVariantModel):

    def matrix(self, timedelta=5, **kwargs):
        delta_t = timedelta
        return sp.array([[1, delta_t, 0, 0], [0, 1, 0, 0], [0, 0, 1, delta_t], [0, 0, 0, 1]])

    def covar(self, timedelta=5, **kwargs):
        delta_t = timedelta
        Sigma = 5.0

        covar = sp.array([[sp.power(delta_t, 4) / 4,
                           sp.power(delta_t, 3) / 2],
                          [sp.power(delta_t, 3) / 2,
                           sp.power(delta_t, 2)]]) * sp.power(Sigma, 2)

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
    messprediction = None  # mess-mean
    S = None  # messkovarianz
    Pxy = None

    @lru_cache()
    def get_measurement_prediction(self, state_prediction, measurement_model=SDFMessmodell(4, (0, 2), np.array([[0.75, 0], [0, 0.75]])), **kwargs):
        measurement_matrix = measurement_model.matrix()
        measurement_noise_covar = measurement_model.covar()
        state_prediction_mean = state_prediction.mean
        state_prediction_covar = state_prediction.covar

        self.messprediction = measurement_matrix @ state_prediction_mean
        self.S = measurement_matrix @ state_prediction_covar @ measurement_matrix.T + measurement_noise_covar
        self.Pxy = state_prediction_covar @ measurement_matrix.T

        return GaussianMeasurementPrediction(self.messprediction, self.S,
                                             state_prediction.timestamp,
                                             self.Pxy)

    def update(self, hypothesis, measurementmodel, **kwargs):
        test = self.get_measurement_prediction(hypothesis.prediction, measurementmodel)  # damit messprediction, kamalngain etc berechnet werden
        W = self.Pxy @ np.linalg.pinv(self.S)
        x_post = hypothesis.prediction.mean + W @ (hypothesis.measurement.state_vector - self.messprediction)
        P_post = hypothesis.prediction.covar - (W @ self.S @ W.T)  # Dimensionen passen nicht
        # P_post = (P_post + P_post.T) / 2

        posterior_mean = x_post
        posterior_covar = P_post
        meas_pred_mean = self.messprediction
        meas_pred_covar = self.S
        cross_covar = self.Pxy
        _ = W

        # Augment hypothesis with measurement prediction
        hypothesis = SingleHypothesis(hypothesis.prediction,
                                      hypothesis.measurement,
                                      GaussianMeasurementPrediction(
                                          meas_pred_mean, meas_pred_covar,
                                          hypothesis.prediction.timestamp,
                                          cross_covar)
                                      )

        return GaussianStateUpdate(posterior_mean,
                                   posterior_covar,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)


def retrodict(state, prior_state, transition_model):
    F = transition_model.matrix()
    D = transition_model.covar()

    x_ll = prior_state.mean
    P_ll = prior_state.covar

    x_l1k = state.mean
    P_l1k = state.covar

    x_l1l = F @ x_ll
    P_l1l = F @ P_ll @ F.T + D

    W_l1l = P_ll @ F.T @ np.linalg.pinv(P_l1l)

    x_lk = x_ll + W_l1l @ (x_l1k - x_l1l)
    P_lk = P_ll + W_l1l @ (P_l1k - P_l1l) @ W_l1l.T

    return GaussianState(x_lk, P_lk, timestamp=state.timestamp)
