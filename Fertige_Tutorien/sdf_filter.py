from functools import lru_cache

import math
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from stonesoup.models.base import LinearModel, GaussianModel, TimeVariantModel
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition.linear import LinearGaussianTransitionModel
from stonesoup.types.array import CovarianceMatrix
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import GaussianMeasurementPrediction
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.updater import Updater
from stonesoup.predictor import Predictor
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.state import GaussianState
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState


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
        x_pre = state_prediction
        Messmatrix = measurement_model.matrix()
        messprediction = Messmatrix @ x_pre
        return messprediction

    def update(self, hypothesis, measurementmodel, **kwargs):
        measurement_matrix = measurementmodel.matrix()  # H
        measurement_noise_covar = measurementmodel.covar()  # R
        prediction_covar = hypothesis.prediction.covar  # P
        messprediction = self.get_measurement_prediction(hypothesis.prediction.mean, measurementmodel)

        S = measurement_matrix @ prediction_covar @ measurement_matrix.T + measurement_noise_covar  # S
        W = prediction_covar @ measurement_matrix.T @ np.linalg.pinv(S)  # W
        Innovation = hypothesis.measurement.state_vector - (measurement_matrix @ hypothesis.prediction.mean)  # v

        x_post = hypothesis.prediction.mean + W @ Innovation  # x + W @ v
        P_post = prediction_covar - (W @ S @ W.T)  # P - ( W @ S @ W.T )


        hypothesis = SingleHypothesis(hypothesis.prediction,
                                      hypothesis.measurement,
                                      GaussianMeasurementPrediction(
                                          messprediction, S,
                                          hypothesis.prediction.timestamp)
                                      )

        return GaussianStateUpdate(x_post,
                                   P_post,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)


"""Figur zum Plotten"""
figure = plt.figure(figsize=(16, 9))
ax = figure.add_subplot(1, 1, 1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

"""Erstellen der Groundtruth"""
velocity = 300.0
acceleration = 9.0
omega = acceleration / (2 * velocity)
A = (velocity ** 2) / acceleration

truth = GroundTruthPath()

for t in range(math.ceil((2 * math.pi) / omega)):
    x = A * np.sin(omega * t)
    y = A * np.sin(2 * omega * t)

    truth.append(GroundTruthState(np.array([[x], [y]]), timestamp=t))

# Plot
ax.plot([state.state_vector[0, 0] for state in truth],
        [state.state_vector[1, 0] for state in truth],
        linestyle="--", color="grey")


"""Erstellen der Messungen"""
measurements = []
for state in truth:
    if state.timestamp % 5 == 0:
        x = state.state_vector[0, 0]
        y = state.state_vector[1, 0]

        mean = [x, y]
        cov = [[2500, 0], [0, 2500]]
        xDet, yDet = np.random.multivariate_normal(mean, cov)

        measurements.append(Detection(np.array([[xDet], [yDet]]), timestamp=state.timestamp))


# Plot
ax.scatter([state.state_vector[0, 0] for state in measurements],
           [state.state_vector[1, 0] for state in measurements],
           color='black', s=10)


"""Komponenten initiieren"""
transition_model = PCWAModel()
predictor = SdfKalmanPredictor(transition_model)

measurement_model = SDFMessmodell(
    4,  # Dimensionen (Position and Geschwindigkeit in 2D)
    (0, 2),  # Mapping
)
updater = SDFUpdater(measurement_model)


"""Erstellen eines Anfangszustandes"""
prior = GaussianState([[0.0], [0.0], [0.0], [0.0]], np.diag([0.0, 0.0, 0.0, 0.0]), timestamp=0)


"""Erstellen einer Trajektorie, sodass das Filter arbeiten kann"""
track = Track()

for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)

    hypothesis = SingleHypothesis(prediction, measurement)

    post = updater.update(hypothesis, measurement_model)

    track.append(post)

    prior = track[-1]


# Plot
ax.plot([state.state_vector[0] for state in track],
        [state.state_vector[2] for state in track],
        marker=".", color="yellow")


"""Darstellen der Kovarianz durch Ellipsen"""
for state in track:
    w, v = np.linalg.eig(measurement_model.matrix() @ state.covar @ measurement_model.matrix().T)
    max_ind = np.argmax(v[0, :])
    orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
    ellipse = Ellipse(xy=state.state_vector[(0, 2), 0],
                      width=np.sqrt(w[0]) * 2, height=np.sqrt(w[1]) * 2,
                      angle=np.rad2deg(orient),
                      alpha=0.2)
    ax.add_artist(ellipse)


plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)

plt.show()
