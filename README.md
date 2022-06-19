# Kalman-Filter-Auto-Tuning
In this repository we try to build a Kalman filter auto-tuning process.
Object function is based on *normalized innovation error squared (NIS)*.
Bayesian optimization method is used to find the optimal parameters
for the filter.

## Reference



## Requirements

### Troubleshooting (June 19, 2022)

A bug has been reported when using the latest release of BayesianOptimization 
package (bayesian-optimization) on pypi with scipy-1.8.0 (or higher).

https://github.com/fmfn/BayesianOptimization/issues/300

Up to June 19, 2022, the fix has been merged in the BayesianOptimization package, 
but the new maintainer is unable to push a release to pypi.

https://github.com/fmfn/BayesianOptimization/issues/300#issuecomment-1146903850

So you could either:

* roll back to scipy 1.7.0.
* install directly from the master repo on GitHub:
`pip install git+https://github.com/fmfn/BayesianOptimization`

