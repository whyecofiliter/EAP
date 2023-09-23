### adjust

This module consists of several common adjustment method in factor analysis.



##### def ols(y, x, constant=*True*):

The function is for OLS regression, which is equal to the OLS module in package *statsmodels*. 

**input :**

*y (ndarray):* The dependent variable.

*x (ndarray):* The explanatory variable.

*constant (boolean):* add constant or not in OLS model. The default is *True*.

**output :**

*result (OLSRegressResult):* The result of the regression.

```python
import numpy as np
from statsmodels.base.model import Model
from statsmodels.tools.tools import add_constant
from EAP.adjust import newey_west, newey_west_t, ols, white_t
from EAP.adjust import white

X = np.random.normal(loc=0.0, scale=1.0, size=(2000,10))
b = np.random.uniform(low=0.1, high=1.1, size=(10,1))
e = np.random.normal(loc=0.0, scale=1.0, size=(2000,1))
y = X.dot(b) + e + 1.0

re = ols(y, X, constant=True)
print('\nTrue b : ', b)
print('\nEstimated b : ', re.params)
print('\nresidue : ', re.resid.shape)
====================================================
True b :  [[0.59169091]
 [0.91342353]
 [0.19599503]
 [0.9112773 ]
 [0.70647024]
 [0.41873624]
 [0.64871071]
 [0.20685505]
 [0.13172035]
 [0.82358063]]

Estimated b :  [1.00206596 0.57602159 0.91832825 0.2104454  0.935377   0.71526534
 0.39181771 0.65432445 0.22666925 0.13173488 0.83208811]

residue :  (2000,)
```



##### def white(y, X, **kwargs):

This function is for the White test. White estimate of variance:

X(r,c): r is sample number, c is variable number, S0 is covariance matrix of residue.

The variance estimate is 
$$
V_{ols}=T(X^`X)^{-1}S_0(X^`X)^{-1}.
$$
The white estimation of S0 is 
$$
S_0=\frac 1T X^`(XR).
$$
**input :**

*y (ndarray):* The dependent variable.

*X (ndarray):* The explanatory variable.

**output :**

*V_ols (ndarray):* The white variance estimate

**Example**

```python
# continue the previous code
import numpy as np
from statsmodels.api import add_constant

X = np.random.normal(loc=0.0, scale=1.0, size=(2000,10))
b = np.random.uniform(low=0.1, high=1.1, size=(10,1))
e = np.random.normal(loc=0.0, scale=1.0, size=(2000,1))
y = X.dot(b) + e + 1.0

re = white(y, X, constant=True)
np.set_printoptions(formatter={'float':'{:0.3f}'.format})
print(re*100)
X = add_constant(X)
r, c = np.shape(X)
print('\n', 1/r*X.T.dot(X).dot(re).dot(X.T.dot(X)))
============================================================================
[[0.052 0.004 0.002 0.002 0.000 -0.002 0.002 -0.004 -0.001 0.000 0.003]
 [0.004 0.056 0.002 -0.002 -0.003 -0.004 -0.001 -0.001 -0.003 0.004 0.001]
 [0.002 0.002 0.049 -0.003 -0.003 0.003 0.000 -0.002 -0.002 -0.003 0.002]
 [0.002 -0.002 -0.003 0.051 -0.002 -0.002 0.004 -0.000 -0.000 -0.001 0.002]
 [0.000 -0.003 -0.003 -0.002 0.054 0.004 0.000 -0.001 0.000 -0.001 0.003]
 [-0.002 -0.004 0.003 -0.002 0.004 0.055 -0.001 0.000 0.000 -0.002 0.001]
 [0.002 -0.001 0.000 0.004 0.000 -0.001 0.053 -0.003 -0.005 -0.002 0.004]
 [-0.004 -0.001 -0.002 -0.000 -0.001 0.000 -0.003 0.057 -0.001 -0.001 -0.002]
 [-0.001 -0.003 -0.002 -0.000 0.000 0.000 -0.005 -0.001 0.051 0.003 0.000]
 [0.000 0.004 -0.003 -0.001 -0.001 -0.002 -0.002 -0.001 0.003 0.049 0.002]
 [0.003 0.001 0.002 0.002 0.003 0.001 0.004 -0.002 0.000 0.002 0.052]]

 [[1.036 0.029 -0.028 0.031 0.007 -0.067 -0.004 0.003 0.029 -0.022 -0.041]
 [0.029 1.001 0.020 -0.056 -0.066 -0.023 0.000 -0.010 -0.070 0.069 0.023]
 [-0.028 0.020 0.972 -0.057 -0.019 0.058 -0.048 -0.002 0.084 -0.005 0.045]
 [0.031 -0.056 -0.057 1.004 -0.004 0.017 0.017 -0.072 -0.008 -0.042 0.041]
 [0.007 -0.066 -0.019 -0.004 1.024 0.045 -0.035 0.017 0.003 0.006 0.029]
 [-0.067 -0.023 0.058 0.017 0.045 1.125 0.010 -0.014 0.037 0.053 -0.017]
 [-0.004 0.000 -0.048 0.017 -0.035 0.010 1.053 -0.022 -0.047 0.047 0.035]
 [0.003 -0.010 -0.002 -0.072 0.017 -0.014 -0.022 0.955 -0.007 -0.002 -0.019]
 [0.029 -0.070 0.084 -0.008 0.003 0.037 -0.047 -0.007 1.008 -0.025 0.025]
 [-0.022 0.069 -0.005 -0.042 0.006 0.053 0.047 -0.002 -0.025 1.010 0.026]
 [-0.041 0.023 0.045 0.041 0.029 -0.017 0.035 -0.019 0.025 0.026 1.058]]
```



##### def newey_west(y, X, J=*None*):

This function is for Newey-West adjustment. Newey-West estimate of variance:

X(r,c): r is sample number, c is variable number, and S0 is covariance matrix of residue.

The estimate variance is
$$
V_{ols}=T(X^`X)^{-1}S_0(X^`X)^{-1}.
$$
The Newey-West estimate variance of S0 is
$$
S_0= \frac1T X^`(XR)+\frac1T \sum_j\sum_tw_je_te_{t-j}(X_tX^`_{t-j}+X_{t-j}X^`_t).
$$
**input :**

*y (ndarray):* The dependent variable.

*X (ndarray):* The explanatory variable.

*J (int):* The lag.

**output :**

*V_ols (ndarray):* The Newey-West variance estimate.

**Example**

```python
# continue the previous code
import numpy as np
from statsmodels.stats.sandwich_covariance import cov_hac

X = np.random.normal(loc=0.0, scale=1.0, size=(10000,10))
b = np.random.uniform(low=0.1, high=1.1, size=(10,1))
e = np.random.normal(loc=0.0, scale=1.0, size=(10000,1))
y = X.dot(b) + e + 1.0

re = newey_west(y, X, constant=False)
np.set_printoptions(formatter={'float':'{:0.2f}'.format})
print(re)
# X = add_constant(X)
r, c = np.shape(X) 
print('\n', 1/r*X.T.dot(X).dot(re).dot(X.T.dot(X)))
result = ols(y, X, constant=False)
print('\n', cov_hac(result))
print('\n', 1/r*X.T.dot(X).dot(cov_hac(result)).dot(X.T.dot(X)))
========================================================================
[[0.00 -0.00 -0.00 0.00 0.00 0.00 -0.00 0.00 -0.00 -0.00]
 [-0.00 0.00 -0.00 -0.00 -0.00 0.00 -0.00 -0.00 0.00 -0.00]
 [-0.00 -0.00 0.00 -0.00 -0.00 -0.00 -0.00 -0.00 -0.00 0.00]
 [0.00 -0.00 -0.00 0.00 -0.00 -0.00 -0.00 0.00 -0.00 -0.00]
 [0.00 -0.00 -0.00 -0.00 0.00 0.00 -0.00 -0.00 0.00 0.00]
 [0.00 0.00 -0.00 -0.00 0.00 0.00 -0.00 -0.00 -0.00 0.00]
 [-0.00 -0.00 -0.00 -0.00 -0.00 -0.00 0.00 0.00 0.00 0.00]
 [0.00 -0.00 -0.00 0.00 -0.00 -0.00 0.00 0.00 -0.00 -0.00]
 [-0.00 0.00 -0.00 -0.00 0.00 -0.00 0.00 -0.00 0.00 0.00]
 [-0.00 -0.00 0.00 -0.00 0.00 0.00 0.00 -0.00 0.00 0.00]]

 [[2.08 0.00 -0.09 -0.01 -0.03 -0.05 -0.02 0.00 -0.02 0.01]
 [0.00 2.01 0.04 -0.00 0.00 -0.03 -0.00 -0.01 -0.00 0.02]
 [-0.09 0.04 1.99 -0.02 0.01 0.00 -0.01 -0.08 0.03 0.02]
 [-0.01 -0.00 -0.02 1.96 -0.00 -0.05 -0.01 -0.04 -0.01 -0.02]
 [-0.03 0.00 0.01 -0.00 1.98 0.06 0.00 0.04 -0.01 -0.05]
 [-0.05 -0.03 0.00 -0.05 0.06 1.94 -0.01 -0.00 -0.01 -0.02]
 [-0.02 -0.00 -0.01 -0.01 0.00 -0.01 2.04 -0.00 0.01 -0.07]
 [0.00 -0.01 -0.08 -0.04 0.04 -0.00 -0.00 1.89 -0.05 -0.02]
 [-0.02 -0.00 0.03 -0.01 -0.01 -0.01 0.01 -0.05 1.97 -0.01]
 [0.01 0.02 0.02 -0.02 -0.05 -0.02 -0.07 -0.02 -0.01 1.93]]

 [[0.00 -0.00 0.00 0.00 0.00 0.00 -0.00 0.00 0.00 -0.00]
 [-0.00 0.00 0.00 -0.00 0.00 0.00 -0.00 0.00 -0.00 -0.00]
 [0.00 0.00 0.00 -0.00 -0.00 0.00 -0.00 -0.00 -0.00 0.00]
 [0.00 -0.00 -0.00 0.00 0.00 -0.00 -0.00 -0.00 -0.00 -0.00]
 [0.00 0.00 -0.00 0.00 0.00 0.00 0.00 -0.00 0.00 0.00]
 [0.00 0.00 0.00 -0.00 0.00 0.00 -0.00 -0.00 -0.00 -0.00]
 [-0.00 -0.00 -0.00 -0.00 0.00 -0.00 0.00 0.00 -0.00 0.00]
 [0.00 0.00 -0.00 -0.00 -0.00 -0.00 0.00 0.00 -0.00 0.00]
 [0.00 -0.00 -0.00 -0.00 0.00 -0.00 -0.00 -0.00 0.00 0.00]
 [-0.00 -0.00 0.00 -0.00 0.00 -0.00 0.00 0.00 0.00 0.00]]

 [[2.07 -0.05 -0.04 0.02 0.06 -0.11 -0.08 0.03 0.05 0.02]
 [-0.05 1.91 0.10 0.01 0.07 0.00 0.01 0.02 -0.09 0.03]
 [-0.04 0.10 2.05 -0.06 0.01 0.07 -0.06 -0.11 -0.15 0.04]
 [0.02 0.01 -0.06 1.87 0.04 -0.05 -0.02 -0.10 -0.01 -0.06]
 [0.06 0.07 0.01 0.04 2.00 0.11 0.05 0.03 -0.00 -0.04]
 [-0.11 0.00 0.07 -0.05 0.11 1.91 0.00 -0.05 -0.05 -0.05]
 [-0.08 0.01 -0.06 -0.02 0.05 0.00 1.99 0.03 -0.02 -0.10]
 [0.03 0.02 -0.11 -0.10 0.03 -0.05 0.03 2.01 -0.10 0.06]
 [0.05 -0.09 -0.15 -0.01 -0.00 -0.05 -0.02 -0.10 2.16 0.01]
 [0.02 0.03 0.04 -0.06 -0.04 -0.05 -0.10 0.06 0.01 1.97]]
```



##### def white_t(y, X, params=*None*, side='Two', **kwargs):

This function constructs t-test based on White variance estimate.

**input :**

*y (ndarray):* The dependent variable.

*X (ndarray):* The explanatory variable.

*params (ndarray):* Already have parameters or not. The default is *None*.

side ()

**output :**

*t_value (ndarray):* The t-value of parameters.

*p_value (ndarray):* The p-value of parameters.

**Example**

```python
import numpy as np

X = np.random.normal(loc=0.0, scale=1.0, size=(10000,10))
b = np.random.uniform(low=-0.5, high=0.5, size=(10,1))
e = np.random.normal(loc=0.0, scale=1.0, size=(10000,1))
y = X.dot(b) + e + 1.0

re = white_t(y, X, constant=False)
np.set_printoptions(formatter={'float':'{:0.2f}'.format})
print('t_value : ', re[0], '\np_value : ', re[1])
print('b :', b.T)
=========================================================================
t_value :  [2.50 34.58 13.23 6.56 -17.64 -34.69 -16.40 13.84 28.90 30.64] 
p_value :  [0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00]
b : [[0.02 0.49 0.18 0.09 -0.25 -0.49 -0.22 0.19 0.44 0.43]]
```



##### def newey_west_t(y, X, params=*None*):

This function constructs t-test based on Newey-West variance estimate.

**input :**

*y (ndarray):* The dependent variable.

*X (ndarray):* The explanatory variable.

*params (ndarray):* Already have parameters or not. The default is None.

**output :**

*t_value (ndarray):* The t-value of parameters.

*p_value (ndarray):* The p-value of parameters.

**Example**

```python
import numpy as np

X = np.random.normal(loc=0.0, scale=1.0, size=(10000,10))
b = np.random.uniform(low=-0.5, high=0.5, size=(10,1))
e = np.random.normal(loc=0.0, scale=1.0, size=(10000,1))
y = X.dot(b) + e + 1.0

re = newey_west_t(y, X, constant=True)
np.set_printoptions(formatter={'float':'{:0.2f}'.format})
print('t_value : ', re[0], '\np_value : ', re[1])
print('b :', b.T)
=================================================================================
t_value :  [135.08 -17.19 13.80 34.95 14.61 21.31 48.56 -44.62 2.75 -28.71 51.63] 
p_value :  [0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00]
b : [[-0.14 0.10 0.25 0.11 0.16 0.40 -0.36 0.03 -0.21 0.40]]
```



