# Kalman-Filter-Auto-Tuning
In this repository we try to build a Kalman filter auto-tuning process.
Object function is based on *normalized innovation error squared (NIS)*.
Bayesian optimization method is used to find the optimal parameters
for the filter.

## Background

### Normalized Innovation Error Squared (NIS)

Given a Kalman filter

**Predict**

Predicted (a priori) state estimate	
$\hat{x}_{k|k-1} = F_k \hat{x}_{k|k-1} + B_k u_k$

Predicted (a priori) estimate covariance
$P_{k|k-1} = F_k P_{k|k-1} F_k^T + Q_k$

**Update**

Innovation or measurement pre-fit residual
$e_{z, k} = z_k - H_k \hat{x}_{k|k-1}$

Innovation (or pre-fit residual) covariance
$S_k = H_k P_{k|k-1} H_k^T + R_k$

Optimal Kalman gain	
$K_k = P_{k|k-1}H_k^TS_k^{-1}$

Updated (a posteriori) state estimate
$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k e_{z, k}$

Updated (a posteriori) estimate covariance
$P_{k|k} = (I - K_k H_k)P_{k|k-1}$

NIS $\epsilon_{z, k}$ is computed from 

$$
\epsilon_{z, k} = e_{z, k}^T S_{k|k-1}^{-1} e_{z, k}
$$




### Reference



## Requirements

### Troubleshooting (June 19, 2022)

A bug has been reported when using the latest release of BayesianOptimization 
package (bayesian-optimization) on pypi with scipy-1.8.0 (or higher)
(https://github.com/fmfn/BayesianOptimization/issues/300).
Up to June 19, 2022, the fix has been merged in the BayesianOptimization package, 
but the new maintainer is unable to push a release to pypi 
(https://github.com/fmfn/BayesianOptimization/issues/300#issuecomment-1146903850).

So you could either:

* roll back to scipy 1.7.0.
* install directly from the master repo on GitHub:
`pip install git+https://github.com/fmfn/BayesianOptimization`

