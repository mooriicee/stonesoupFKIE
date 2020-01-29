import numpy as np

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

data = []
temp = 0
for i in range(85):
    data.append(temp)
    if (i == 83):
        temp += 3.879
    else:
        temp += 5

for t in data:
    x = A * np.sin(omega * t)
    y = A * np.sin(2 * omega * t)
    truth.append(GroundTruthState(np.array([[x], [y]]), timestamp=t))

ax.plot([state.state_vector[0, 0] for state in truth],
        [state.state_vector[1, 0] for state in truth],
        linestyle="--", color="grey")

from stonesoup.types.detection import Detection

measurements = []
for state in truth:
    xOffset = 50 * np.random.normal(0, 1, 1)
    yOffset = 50 * np.random.normal(0, 1, 1)
    x = state.state_vector[0, 0]
    y = state.state_vector[1, 0]
    measurements.append(Detection(
        np.array([[x] + xOffset, [y] + yOffset]), timestamp=state.timestamp))

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
    np.array([[np.power(50, 2), 0],  # Covariance matrix for Gaussian PDF
              [0, np.power(50, 2)]])
)

measurement_model.matrix()
measurement_model.covar()

from tutorienklassen import SDFUpdater

updater = SDFUpdater(measurement_model)

from stonesoup.types.state import GaussianState

prior = GaussianState([[0], [1], [0], [1]], np.diag([10.5, 0.5, 10.5, 0.5]), timestamp=0)

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

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

plt.show()