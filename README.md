# Kalman-Filter-Auto-Tuning
In this repository we try to build a Kalman filter auto-tuning process for filter 
parameters: state noise covariance matrix (*Q*) and observation noise covariance matrix (*R*).
Objective function is based on Normalized Innovation Error Squared (NIS).
Tree of Parzen Estimators (TPE) method is used to find the minimum value of the objective function.

## Reference
### Objective function
* Chen, Z., N. Ahmed, S. Julier and C. Heckman (2019). 
"Kalman filter tuning with Bayesian optimization." 
arXiv preprint arXiv:1912.08601.
(https://arxiv.org/pdf/1912.08601v1.pdf)
* Chen, Z., C. Heckman, S. Julier and N. Ahmed (2018). 
Weak in the NEES?: Auto-tuning Kalman filters with Bayesian optimization. 
2018 21st International Conference on Information Fusion (FUSION), IEEE.
(https://arxiv.org/pdf/1807.08855v1.pdf)

### Optimization
* Bergstra, J., R. Bardenet, Y. Bengio and B. Kégl (2011). 
"Algorithms for hyper-parameter optimization." 
Advances in neural information processing systems 24.
(https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
* [Hyperopt library](https://github.com/hyperopt/hyperopt)
* Bergstra, J., Yamins, D., Cox, D. D. (2013) 
Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. 
TProc. of the 30th International Conference on Machine Learning (ICML 2013), 
June 2013, pp. I-115 to I-23.
(http://proceedings.mlr.press/v28/bergstra13.pdf)

## Requirements
```
adskalman==0.3.11
filterpy==1.4.5
numpy==1.22.4
matplotlib==3.5.1
hyperopt==0.2.7
pandas==1.4.2
```
To install the requirements:
```
pip install -r requirements.txt
```

## Example
In this example, we need to find the optimal Kalman filter parameters for smoothing the recorded *Drosophila* 
trajectory in an arena.
We already have several recorded trajectories (observations) which are consists of the *x* and *y* position 
(in pixel unit) of the fly in the arena.
The basic idea is that we can first find the optimal filter parameters based on some recorded trajectories 
through the auto-tuning procedure described above, and then test the filter performance on a test trajectory.
Then the optimal parameters found can be used in the smoothing of other *Drosophila* trajectories.

More details see also [`Kalman-Filter-Auto-Tuning.pdf`](https://github.com/HaoZhu10015/Kalman-Filter-Auto-Tuning/blob/78a431b57432dccee71c014fe1b36cfc67761a66/pdf/Kalman-Filter-Auto-Tuning.pdf).

To run the example code, run [`main.py`](https://github.com/HaoZhu10015/Kalman-Filter-Auto-Tuning/blob/b0a764f4451a35be123db7cd38203114184b29ab/main.py):
```
python main.py --min_q_var=0 --max_q_var=5000 --min_r_var=0 --max_r_var=1 --epoch=100
```
Results can be found in the `output` folder.

## Thanks
* *Drosophila* trajectory was recorded with [Strand Camera](https://strawlab.org/strand-cam/)
  (GitHub repository: https://github.com/strawlab/strand-braid.git)

