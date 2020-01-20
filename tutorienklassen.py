from functools import lru_cache

import numpy as np
import scipy as sp
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
from stonesoup.types.state import State



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


class SdfKalmanPredictor(Predictor):
    @lru_cache()
    def predict(self, prior, control_input=None, timestamp=None, **kwargs):

        # Compute time_interval
        try:
            time_interval = timestamp - prior.timestamp
        except TypeError:
            # TypeError: (timestamp or prior.timestamp) is None
            time_interval = None

        # Transition model parameters
        transition_matrix = self.transition_model.matrix(
            timestamp=timestamp,
            time_interval=time_interval,
            **kwargs)
        transition_noise_covar = self.transition_model.covar(
            timestamp=timestamp,
            time_interval=time_interval,
            **kwargs)

        # Control model parameters
        if self.control_model is None:
            control_matrix = np.zeros(prior.covar.shape)
            contol_noise_covar = np.zeros(prior.covar.shape)
            control_input = State(np.zeros(prior.state_vector.shape))
        else:
            # Extract control matrix
            control_matrix = self.control_model.matrix(
                timestamp=timestamp,
                time_interval=time_interval,
                **kwargs)
            # Extract control noise covariance
            try:
                # covar() is implemented for control_model
                control_noise_covar = self.control_model.covar(
                    timestamp=timestamp,
                    time_interval=time_interval,
                    **kwargs)
            except AttributeError:
                # covar() is NOT implemented for control_model
                contol_noise_covar = np.zeros(self.control_model.ndim_ctrl)
            if control_input is None:
                control_input = np.zeros((self.control_model.ndim_ctrl, 1))

        # Perform prediction
        prediction_mean, prediction_covar = self.predict_lowlevel(
            prior.mean, prior.covar, transition_matrix,
            transition_noise_covar, control_input.state_vector,
            control_matrix, contol_noise_covar)

        return GaussianStatePrediction(prediction_mean,
                                       prediction_covar,
                                       timestamp)

class SDFUpdater(Updater):
    messprediction = None  # mess-mean
    kalman_gain = None  # messkovarianz
    Pxy = None

    @lru_cache()
    def get_measurement_prediction(self, state_prediction, measurement_model=None, **kwargs):
        measurement_matrix = measurement_model.matrix()
        measurement_noise_covar = measurement_model.covar()
        state_prediction_mean = state_prediction.mean
        state_prediction_covar = state_prediction.covar

        self.messprediction = measurement_matrix @ state_prediction_mean
        self.kalman_gain = measurement_matrix @ state_prediction_covar @ measurement_matrix.T + measurement_noise_covar
        self.Pxy = state_prediction_covar @ measurement_matrix.T

        return GaussianMeasurementPrediction(self.messprediction, self.kalman_gain,
                                             state_prediction.timestamp,
                                             self.Pxy)

    def update(self, hypothesis, measurementmodel, **kwargs):
        test = self.get_measurement_prediction(hypothesis.prediction, measurementmodel)     # damit messprediction, kamalngain etc berechnet werden
        K = self.Pxy @ np.linalg.pinv(self.kalman_gain)
        x_post = hypothesis.prediction.mean + K @ (hypothesis.measurement.state_vector - self.messprediction)   # K @ (hypo..- ..) Dimensionen passen nicht
        P_post = self.kalman_gain - K @ self.Pxy.T  # Dimensionen passen nicht
        P_post = (P_post + P_post.T) / 2

        posterior_mean = x_post
        posterior_covar = P_post
        meas_pred_mean = self.messprediction
        meas_pred_covar = self.kalman_gain
        cross_covar = self.Pxy
        _ = K

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
