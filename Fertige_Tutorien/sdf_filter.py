import math



"""Hier die Klassen einfügen"""

import numpy as np
from matplotlib import pyplot as plt
"""Figur zum Plotten"""
figure = plt.figure(figsize=(16, 9))
ax = figure.add_subplot(1, 1, 1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection

"""Hier die Erstellung der Groundtruth und der Messungen"""


transition_model = PCWAModel()

transition_model.matrix()
transition_model.covar()



predictor = SdfKalmanPredictor(transition_model)



measurement_model = SDFMessmodell(
    4,  # Number of state dimensions (position and velocity in 2D)
    (0, 2),  # Mapping measurement vector index to state index
)


measurement_model.matrix()
measurement_model.covar()

updater = SDFUpdater(measurement_model)

from stonesoup.types.state import GaussianState
"""Erstellen eines Anfangszustandes"""
prior = GaussianState([[0.0], [0.0], [0.0], [0.0]], np.diag([0.0, 0.0, 0.0, 0.0]), timestamp=0)


from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()

for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)

    hypothesis = SingleHypothesis(prediction, measurement)

    post = updater.update(hypothesis, measurement_model)

    track.append(post)

    prior = track[-1]

# Plot the resulting track
ax.plot([state.state_vector[0] for state in track],
        [state.state_vector[2] for state in track],
        marker=".", color="yellow")


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

"""
for state in retrodiction_track:
    w, v = np.linalg.eig(measurement_model.matrix() @ state.covar @ measurement_model.matrix().T)
    max_ind = np.argmax(v[0, :])
    orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
    ellipse = Ellipse(xy=state.state_vector[(0, 2), 0],
                      width=np.sqrt(w[0]) * 2, height=np.sqrt(w[1]) * 2,
                      angle=np.rad2deg(orient),
                      alpha=0.2)
    ax.add_artist(ellipse)"""

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

plt.show()
