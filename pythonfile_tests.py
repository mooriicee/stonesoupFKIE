import math

import numpy as np
from numpy.distutils.system_info import xft_info

velocity = 300.0
acceleration = 9.0
omega = acceleration / (2 * velocity)
A = (velocity ** 2) / acceleration

# Figure to plot truth (and future data)
from matplotlib import pyplot as plt

figure = plt.figure(figsize=(16, 9))
ax = figure.add_subplot(1, 1, 1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

truth = GroundTruthPath()

for t in range(math.ceil((2 * math.pi) / omega)):
    x = A * np.sin(omega * t)
    y = A * np.sin(2 * omega * t)

    truth.append(GroundTruthState(np.array([[x], [y]]), timestamp=t))

ax.plot([state.state_vector[0, 0] for state in truth],
        [state.state_vector[1, 0] for state in truth],
        linestyle="--", color="grey")

from stonesoup.types.detection import Detection

'''
sensorAnzahl = 3
from tutorienklassen import SDFMessmodell

measurement_model = SDFMessmodell(
    4,  # Number of state dimensions (position and velocity in 2D)
    (0, 2),  # Mapping measurement vector index to state index
'''

x_Offsets = []

measurements = []
for state in truth:
    if state.timestamp % 5 == 0:
        '''
        mean = 0
        sensormessungen = []
        for i in range(sensorAnzahl):
            xOffset = 50 * np.random.normal(-1, 1, 1)
            yOffset = 50 * np.random.normal(-1, 1, 1)
            x = state.state_vector[0, 0]
            y = state.state_vector[1, 0]
            sensormessungen.append((x, y))

        for j in range(len(sensormessungen)):
            temp = sensormessungen[i]
            cov = np.linalg.inv(measurement_model.covar())
            mean += cov @ state.state_vector
        '''
        # xOffset = 50 * np.random.normal(-1, 1, 1)
        # yOffset = 50 * np.random.normal(-1, 1, 1)
        x = state.state_vector[0, 0]
        y = state.state_vector[1, 0]

        mean = [x, y]
        cov = [[2500, 0], [0, 2500]]
        xDet, yDet = np.random.multivariate_normal(mean, cov)

        x_Offsets.append(xDet - x)

        measurements.append(Detection(np.array([[xDet], [yDet]]), timestamp=state.timestamp))

        # measurements.append(Detection(
        #    np.array([[x] + xOffset, [y] + yOffset]), timestamp=state.timestamp))

# Plot the result
ax.scatter([state.state_vector[0, 0] for state in measurements],
           [state.state_vector[1, 0] for state in measurements],
           color='black', s=10)

from tutorienklassen import PCWAModel

transition_model = PCWAModel()

transition_model.matrix()
transition_model.covar()

from tutorienklassen import SdfKalmanPredictor

predictor = SdfKalmanPredictor(transition_model)

from tutorienklassen import SDFMessmodell

measurement_model = SDFMessmodell(
    4,  # Number of state dimensions (position and velocity in 2D)
    (0, 2),  # Mapping measurement vector index to state index
)

measurement_model.matrix()
measurement_model.covar()

from tutorienklassen import SDFUpdater

updater = SDFUpdater(measurement_model)

from stonesoup.types.state import GaussianState

prior = GaussianState([[0.0], [50.0], [0.0], [50.0]], np.diag([0.0, 0.0, 0.0, 0.0]), timestamp=0)

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)

    hypothesis = SingleHypothesis(prediction, measurement)  # Used to group a prediction and measurement together

    post = updater.update(hypothesis, measurement_model)

    track.append(post)

    prior = track[-1]

# Plot the resulting track
ax.plot([state.state_vector[0, 0] for state in track],
        [state.state_vector[2, 0] for state in track],
        marker=".", color="yellow")

# %%

from matplotlib.patches import Ellipse

for state in track:
    w, v = np.linalg.eig(measurement_model.matrix() @ state.covar @ measurement_model.matrix().T)
    max_ind = np.argmax(v[0, :])
    orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
    ellipse = Ellipse(xy=state.state_vector[(0, 2), 0],
                      width=np.sqrt(w[0]) * 2, height=np.sqrt(w[1]) * 2,
                      angle=np.rad2deg(orient),
                      alpha=0.2)
    ax.add_artist(ellipse)

from tutorienklassen import retrodict

retrodiction_track = Track()
for i in range(len(track) - 1):
    if i == 0:
        retrodiction_track.append(retrodict(track[-i - 1], track[-i - 2], transition_model))
    else:
        retrodiction_track.append(retrodict(retrodiction_track[i - 1], track[-i - 2], transition_model))

ax.plot([state.state_vector[0, 0] for state in retrodiction_track],
        [state.state_vector[2, 0] for state in retrodiction_track],
        marker=".", color="brown")

for state in retrodiction_track:
    w, v = np.linalg.eig(measurement_model.matrix() @ state.covar @ measurement_model.matrix().T)
    max_ind = np.argmax(v[0, :])
    orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
    ellipse = Ellipse(xy=state.state_vector[(0, 2), 0],
                      width=np.sqrt(w[0]) * 2, height=np.sqrt(w[1]) * 2,
                      angle=np.rad2deg(orient),
                      alpha=0.2)
    ax.add_artist(ellipse)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

plt.show()
