import numpy as np
import adskalman.adskalman as adskalman
from filterpy.kalman import KalmanFilter

def column(arr):
    """convert 1D array-like to a 2D vertical array

    >>> column((1,2,3))

    array([[1],
           [2],
           [3]])
    """
    arr = np.array(arr)
    assert arr.ndim == 1
    a2 = arr[:, np.newaxis]
    return a2


class TrajSimulator1D:
    def __init__(self, dt=0.01, duration=0.5, initial_state=column([0.0]), q_var=1000.0, r_var=10.0):
        self.dt = dt
        self.duration = duration
        self.t = np.arange(0.0, duration, dt)
        self.initial_state = initial_state
        self.q_var = q_var
        self.r_var = r_var

        # State space model.
        self.F = np.array([[1.0]])
        self.Q = self.q_var * np.array([[self.dt]])

        # Observation model.
        self.H = np.array([[1.0]])
        self.R = np.array([[self.r_var]])

    def simulate(self):
        # Simulate trajectory.
        current_state = self.initial_state
        state = []
        for _ in self.t:
            state.append(current_state[:, 0])
            noise_sample = adskalman.rand_mvn(np.zeros(1), self.Q, 1).T
            current_state = np.dot(self.F, current_state) + noise_sample

        # Simulate observations.
        observation = []
        for current_state in state:
            noise_sample = adskalman.rand_mvn(np.zeros(1), self.R, 1).T
            current_observation = np.dot(self.H, column(current_state)) + noise_sample
            observation.append(current_observation[:, 0])

        return np.array(state), np.array(observation)


class KalmanFilterTuningModel:
    def __init__(
            self,
            dim_x, dim_z,
            F, H, initial_state, P,
            observation
    ):
        self.kalman_filter = KalmanFilter(dim_x, dim_z)
        self.kalman_filter.F = F
        self.kalman_filter.H = H
        self.initial_state = initial_state
        self.P = P
        self.observation = observation

    def get_filter_nis_loss(self, Q, R):
        self.kalman_filter.x = self.initial_state
        self.kalman_filter.P = self.P
        self.kalman_filter.Q = Q
        self.kalman_filter.R = R
        nis_list = []
        for i in range(len(self.observation)):
            z = self.observation[i]
            self.kalman_filter.predict()
            self.kalman_filter.update(z)
            nis_list.append(self.kalman_filter.y.T @ np.linalg.inv(self.kalman_filter.S) @ self.kalman_filter.y)

        return np.abs(np.log(np.mean(nis_list) / 2))

