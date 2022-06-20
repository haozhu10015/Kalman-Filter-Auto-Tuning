# Kalman-Filter-Auto-Tuning
In this repository we try to build a Kalman filter auto-tuning process.
Object function is based on *normalized innovation error squared (NIS)*.
Tree of Parzen Estimators (TPE) method is used to find the optimal parameters
for the filter.

## Reference
### Object function
* https://arxiv.org/pdf/1912.08601v1.pdf
* https://arxiv.org/pdf/1807.08855v1.pdf

### Optimization
* https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
* https://github.com/hyperopt/hyperopt
* http://proceedings.mlr.press/v28/bergstra13.pdf

## Requirements
```
adskalman==0.3.11
filterpy==1.4.5
numpy==1.22.4
matplotlib==3.5.1
hyperopt==0.2.7
```
To install the requirements:
```
pip install -r requirements.txt
```


