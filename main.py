import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils import KalmanFilterTuningModel, build_4d_Q_matrix
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from filterpy.kalman import KalmanFilter
from filterpy.stats import plot_covariance
import argparse

np.random.seed(10015)

parser = argparse.ArgumentParser()
parser.add_argument('--min_q_var', type=float,
                    default=0,
                    help='Lower bound of q_var searching space.')
parser.add_argument('--max_q_var', type=float,
                    default=5000,
                    help='Upper bound of q_var searching space.')
parser.add_argument('--min_r_var', type=float,
                    default=0,
                    help='Lower bound of r_var searching space.')
parser.add_argument('--max_r_var', type=float,
                    default=1,
                    help='Upper bound of r_var searching space.')
parser.add_argument('--epoch', type=int,
                    default=100,
                    help='Number of iterations of optimization.')
args = parser.parse_args()


if __name__ == '__main__':
    # Load trajectories.
    # First 5 trajectories are used to find the optimal parameters for the Kalman filter.
    # The last trajectory is used to test the performance of Kalman filter with optimal parameters.
    file_list = os.listdir('./data')
    obs_traj = []
    initial_state = []
    dt = 0
    for f in file_list[:-1]:
        df = pd.read_csv(os.path.join('./data', f), comment='#')
        obs_traj.append(np.array([df['x_px'], df['y_px']]).T)
        initial_state.append(np.array([[df.loc[0, 'x_px'], 0, df.loc[0, 'y_px'], 0]]).T)
        dt += np.mean(np.diff(df['time_microseconds'])) * 1e-6
    dt /= len(obs_traj)

    # ------ Tuning ------
    # Build the Kalman filter tuning model.
    F = np.array([
        [1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1]
    ])

    H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])

    P = 1000.0 * np.eye(4)

    tuning_model = KalmanFilterTuningModel(dim_x=4, dim_z=2,
                                           F=F, H=H,
                                           initial_state=initial_state,
                                           P=P,
                                           observation=obs_traj)

    # Build the optimizer.
    # Set Objective function.
    def J(optim_args):
        q_var, r_var = optim_args
        Q = build_4d_Q_matrix(dt, q_var)
        R = np.eye(2) * r_var
        nis_loss = tuning_model.get_filter_nis_loss(Q, R)
        return {'loss': nis_loss, 'status': STATUS_OK}

    # Set parameter space for searching.
    param_space = [
        hp.uniform('q_var', args.min_q_var, args.max_q_var),
        hp.uniform('r_var', args.min_r_var, args.max_r_var)
    ]

    # Run optimizer
    trials = Trials()
    optimal_param = fmin(J, param_space, algo=tpe.suggest, max_evals=args.epoch, trials=trials)

    print('Optimal parameters:\n'
          '\tq_var: {}\n\tr_var: {}\n'
          'Minimum objective function value: {}'.format(optimal_param['q_var'], optimal_param['r_var'],
                                                        np.min(trials.losses())))
    log_file = './output/log.txt'
    with open(log_file, 'w') as log_f:
        print('Optimal parameters:\n'
              '\tq_var: {}\n\tr_var: {}\n'
              'Minimum objective function value: {}'.format(optimal_param['q_var'], optimal_param['r_var'],
                                                            np.min(trials.losses())),
              file=log_f)

    # ------ Testing ------
    # Load test trajectory.
    test_df = pd.read_csv(os.path.join('./data', file_list[-1]), comment='#')
    test_initial_state = [np.array([test_df.loc[0, 'x_px'], 0, test_df.loc[0, 'y_px'], 0]).T]
    test_obs_traj = [np.array([test_df['x_px'], test_df['y_px']]).T]

    # Perform a Kalman filter using optimal parameters (Q, R) on a test trajectory.
    kalman_filter = KalmanFilter(dim_x=4, dim_z=2)
    kalman_filter.F = F
    kalman_filter.H = H
    kalman_filter.x = test_initial_state[0]
    kalman_filter.P = P

    best_q_var = optimal_param['q_var']
    best_r_var = optimal_param['r_var']
    kalman_filter.Q = build_4d_Q_matrix(dt, best_q_var)
    kalman_filter.R = np.eye(2) * best_r_var

    estimate_traj_mu, estimate_traj_cov, _, _ = kalman_filter.batch_filter(test_obs_traj[0])

    # Draw the estimated trajectory with optimal filter parameters.
    fig = plt.figure(dpi=600)
    plt.plot(test_df['x_px'], test_df['y_px'], linestyle='-', label='Observation')
    plt.plot(estimate_traj_mu.reshape(-1, 4)[:, 0], estimate_traj_mu.reshape(-1, 4)[:, 2],
             linestyle='-', label='Estimation')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')

    for i in range(0, estimate_traj_cov.shape[0], 100):
        plot_covariance((estimate_traj_mu.reshape(-1, 4)[i, 0], estimate_traj_mu.reshape(-1, 4)[i, 2]),
                        estimate_traj_cov[i, :2, :2])
    plt.savefig('./output/test_performance.png', dpi=600)

