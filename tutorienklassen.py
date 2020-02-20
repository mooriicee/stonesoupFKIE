from functools import lru_cache
from wave import Wave_write

import numpy as np
import scipy as sp
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
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
        noise = multivariate_normal.rvs(sp.zeros(self.ndim_meas), self.covar(), 1)
        return noise.reshape((-1, 1))

    def pdf(self):
        pass

class PCWAModel(LinearGaussianTransitionModel, TimeVariantModel):

    def matrix(self, timedelta=5, **kwargs):
        delta_t = timedelta
        F = sp.array([[1, delta_t], [0, 1]])
        return block_diag(F, F)


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


'''class SDFUpdater(Updater):
    messprediction = None  # mess-mean
    S = None  # messkovarianz
    Pxy = None

    @lru_cache()
    def get_measurement_prediction(self, state_prediction, measurement_model=SDFMessmodell(4, (0, 2)), **kwargs):
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
        test = self.get_measurement_prediction(hypothesis.prediction, measurementmodel)  # damit messprediction, kalmangain etc berechnet werden
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
                                   hypothesis.measurement.timestamp)'''


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
        # P_post = (P_post + P_post.T) / 2

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
    P_l1l = F @ P_ll @ np.transpose(F) + 10*D

    W_l1l = P_ll @ np.transpose(F) @ np.linalg.pinv(P_l1l)   # Gewichtsmatrix

    x_lk = x_ll + W_l1l @ (x_l1k - x_l1l)   # verbesserter Zustand
    P_lk = P_ll + W_l1l @ (P_l1k - P_l1l) @ np.transpose(W_l1l)

    return GaussianState(x_lk, P_lk, timestamp=prior_state.timestamp)
