### cross_section_regress

#### class CS_regress()

##### def \__init__(self, y_list, factor):

This function initializes the class.

**input :**

*y_list (list/DataFrame):* The assets return series list.

*factor (ndarray/DataFrame):* The factor risk premium series.



##### def ts_regress(self):

This function conducts the time series regression.

**output :**

*beta(ndarray[N, c]) :* The betas for each asset.

*err_mat (ndarray):* The error matrix.



##### def ave(self):

This function conducts the average operation on the assets returns series list.

**output :**

*np.mean(self.y_list, axis=1)* : The average of the assets returns series list.



##### def cs_regress(self, beta, err_mat, constant=*True*, gls=*True*, **kwargs):

This function takes the cross-sectional regression.

**input :**

*beta (ndarray):* The betas from the output of the function ts_regress().

*err_mat (ndarray) :* The error matrix from the output of the function ts_regress().

*constant (boolean):* add constant or not. The default is add constant.

*gls (boolean):* GLS regression or OLS regression. The default is GLS regression.

**output :**

*params (array):* The params of the cross regression model.

*resid (array):* The residue of the cross regression model.

**Example**

```python
from statsmodels.regression.linear_model import GLS
from EAP.cross_section_regress import CS_regress
import numpy as np
    
X = np.random.normal(loc=0, scale=0.1, size=(2000,3))
y_list = []
for i in range(100) :
    b = np.random.uniform(low=-1, high=1, size=(3,1))
    e = np.random.normal(loc=0.0, scale=0.5, size=(2000,1))
    alpha = np.random.normal(loc=0.0, scale=0.5)
    y = X.dot(b)  + e 
    y_list.append(y)
print(np.mean(X, axis= 0)) # average return of the factor risk premium 
========================================================================
[-0.00496423  0.00146649 -0.0004722 ]
```



##### def cov_mat(self, beta, err_mat, shanken=*True*, constant=*True*, gls=*True*, **kwargs):

This function calculates the covariance matrix of the cross regression model. 

**input :**

*beta (ndarray):* The betas from the output of the function ts_regress().

*err_mat (ndarray):* The error matrix from the output of the function ts_regress().

*shanken (boolean):* Take the shanken adjustment or not. The default is *True*.

*constant (boolean):* add constant or not.

*gls (boolean):* GLS regression or OLS regression. The default is GLS regression.

**output :**

*param_cov_mat (ndarray):* The covariance matrix of the parameters.

*resid_cov_mat (ndarray):* The covariance matrix of the residue.



##### def t_test(self, param_cov_mat, params):

This function takes t-test for parameters.

**input :**

*param_cov_mat (ndarray):* The covariance matrix of the parameters from the function cov_mat.

*params (ndarray):* The parameters from the function cs_regress.

**output :**

*t_value (ndarray):* The t-value for statistical inference.

*p_value (ndarray):* The p-value for statistical inference.



##### def union_test(self, resid_cov_mat, resid):

This function takes union test for parameters.

**input :**

*resid_cov_mat (ndarray):* The covariance matrix of the residue. 

*resid (ndarray):* The residue from the function cs_regress.

**output :**

*chi_square (list):* The chi-square statistics.

*p_value (list):* The p-value corresponding to the chi-square.



##### def fit(self, **kwargs):

This function runs the cross-sectional regression and takes the statistical inference.



##### def summary(self):

This function print the summary. 

 **Example**

```python
# continue the previous code
print("\n---------------------GLS: Constant=True shanken=True------------------------\n")
re = CS_regress(y_list, X)
re.fit()
re.summary()
print("\n------------------------------------------------------------------------\n")
    
print("\n---------------------GLS: Constant=False shanken=True------------------------\n")
re = CS_regress(y_list, X)
re.fit(constant=False)
re.summary()
print("\n------------------------------------------------------------------------\n")
    
print("\n---------------------GLS: Constant=True shanken=False------------------------\n")
re = CS_regress(y_list, X)
re.fit(shanken=False)
re.summary()
print("\n------------------------------------------------------------------------\n")
    
print("\n---------------------GLS: Constant=False shanken=False------------------------\n")
re = CS_regress(y_list, X)
re.fit(constant=False, shanken=False)
re.summary()
print("\n------------------------------------------------------------------------\n")
    
print("\n---------------------OLS: Constant=True shanken=True------------------------\n")
re = CS_regress(y_list, X)
re.fit(gls=False)
re.summary()
print("\n------------------------------------------------------------------------\n")
    
print("\n---------------------OLS: Constant=False shanken=True------------------------\n")
re = CS_regress(y_list, X)
re.fit(constant=False, gls=False)
re.summary()
print("\n------------------------------------------------------------------------\n")
    
print("\n---------------------OLS: Constant=True shanken=False------------------------\n")
re = CS_regress(y_list, X)
re.fit(shanken=False, gls=False)
re.summary()
print("\n------------------------------------------------------------------------\n")
    
print("\n---------------------OLS: Constant=False shanken=False------------------------\n")
re = CS_regress(y_list, X)
re.fit(constant=False, shanken=False, gls=False)
re.summary()
print("\n------------------------------------------------------------------------\n")
================================================================================================================================
---------------------GLS: Constant=True shanken=True------------------------


-------------------------- Risk Premium ------------------------------

+-----------------------+---------------+--------------+
|         params        |    t_value    |   p_value    |
+-----------------------+---------------+--------------+
| -0.004239493082637248 | [-1.49931456] | [0.13394995] |
| 0.0018754489425203834 |  [0.64798047] |  [0.517072]  |
| -0.002623980021497935 | [-0.92194337] | [0.35666937] |
+-----------------------+---------------+--------------+

----------------------------------------------------------------------


----------------------------- Alpha test -----------------------------

+-----------------+----------------+
|    chi-square   |    p_value     |
+-----------------+----------------+
| [[89.27084753]] | [[0.69922497]] |
+-----------------+----------------+

----------------------------------------------------------------------


------------------------------------------------------------------------


---------------------GLS: Constant=False shanken=True------------------------


-------------------------- Risk Premium ------------------------------

+------------------------+---------------+--------------+
|         params         |    t_value    |   p_value    |
+------------------------+---------------+--------------+
| -0.0044337140100835495 | [-1.56801135] | [0.11703675] |
| 0.0014489988306153607  |  [0.50064261] | [0.61667778] |
| -0.0025199902303684996 |  [-0.8854116] | [0.37604117] |
+------------------------+---------------+--------------+

----------------------------------------------------------------------


----------------------------- Alpha test -----------------------------

+------------------+----------------+
|    chi-square    |    p_value     |
+------------------+----------------+
| [[123.73200185]] | [[0.03486519]] |
+------------------+----------------+

----------------------------------------------------------------------


------------------------------------------------------------------------


---------------------GLS: Constant=True shanken=False------------------------


-------------------------- Risk Premium ------------------------------

+-----------------------+---------------+--------------+
|         params        |    t_value    |   p_value    |
+-----------------------+---------------+--------------+
| -0.004239493082637248 | [-1.50013478] | [0.13373741] |
| 0.0018754489425203834 |  [0.64838824] | [0.51680835] |
| -0.002623980021497935 | [-0.92243422] | [0.35641345] |
+-----------------------+---------------+--------------+

----------------------------------------------------------------------


----------------------------- Alpha test -----------------------------

+-----------------+---------+
|    chi-square   | p_value |
+-----------------+---------+
| [[32.53458229]] |  [[1.]] |
+-----------------+---------+

----------------------------------------------------------------------


------------------------------------------------------------------------


---------------------GLS: Constant=False shanken=False------------------------


-------------------------- Risk Premium ------------------------------

+------------------------+---------------+--------------+
|         params         |    t_value    |   p_value    |
+------------------------+---------------+--------------+
| -0.0044337140100835495 | [-1.56885941] | [0.11683897] |
| 0.0014489988306153607  |  [0.50095408] | [0.6164586]  |
| -0.0025199902303684996 | [-0.88587763] | [0.37579002] |
+------------------------+---------------+--------------+

----------------------------------------------------------------------


----------------------------- Alpha test -----------------------------

+------------------+---------+
|    chi-square    | p_value |
+------------------+---------+
| [[-49.33885529]] |  [[1.]] |
+------------------+---------+

----------------------------------------------------------------------


------------------------------------------------------------------------


---------------------OLS: Constant=True shanken=True------------------------


-------------------------- Risk Premium ------------------------------

+------------------------+---------------+--------------+
|         params         |    t_value    |   p_value    |
+------------------------+---------------+--------------+
| -0.004191679829911098  | [-1.46410802] | [0.14332167] |
| 0.0026957168215120215  |  [0.91917806] | [0.35811334] |
| -0.0028054613152236852 |  [-0.9776489] | [0.3283663]  |
+------------------------+---------------+--------------+

----------------------------------------------------------------------


----------------------------- Alpha test -----------------------------

+------------------+----------------+
|    chi-square    |    p_value     |
+------------------+----------------+
| [[105.04417306]] | [[0.27097431]] |
+------------------+----------------+

----------------------------------------------------------------------


------------------------------------------------------------------------


---------------------OLS: Constant=False shanken=True------------------------


-------------------------- Risk Premium ------------------------------

+-----------------------+---------------+--------------+
|         params        |    t_value    |   p_value    |
+-----------------------+---------------+--------------+
| -0.004314458572417905 | [-1.50700942] | [0.13196625] |
|  0.002435573118655651 |  [0.83048514] | [0.40636372] |
| -0.002772551881136359 | [-0.96619054] | [0.33406572] |
+-----------------------+---------------+--------------+

----------------------------------------------------------------------


----------------------------- Alpha test -----------------------------

+------------------+----------------+
|    chi-square    |    p_value     |
+------------------+----------------+
| [[117.95438501]] | [[0.07282442]] |
+------------------+----------------+

----------------------------------------------------------------------


------------------------------------------------------------------------


---------------------OLS: Constant=True shanken=False------------------------


-------------------------- Risk Premium ------------------------------

+------------------------+---------------+--------------+
|         params         |    t_value    |   p_value    |
+------------------------+---------------+--------------+
| -0.004191679829911098  | [-1.46507223] | [0.14305847] |
| 0.0026957168215120215  |  [0.91987006] | [0.35775165] |
| -0.0028054613152236852 |  [-0.9782681] | [0.32806011] |
+------------------------+---------------+--------------+

----------------------------------------------------------------------


----------------------------- Alpha test -----------------------------

+------------------+---------------+
|    chi-square    |    p_value    |
+------------------+---------------+
| [[144.80511192]] | [[0.0011985]] |
+------------------+---------------+

----------------------------------------------------------------------


------------------------------------------------------------------------


---------------------OLS: Constant=False shanken=False------------------------


-------------------------- Risk Premium ------------------------------

+-----------------------+---------------+--------------+
|         params        |    t_value    |   p_value    |
+-----------------------+---------------+--------------+
| -0.004314458572417905 | [-1.50798575] | [0.13171619] |
|  0.002435573118655651 |  [0.8311002]  | [0.40601628] |
| -0.002772551881136359 | [-0.96679254] | [0.3337647]  |
+-----------------------+---------------+--------------+

----------------------------------------------------------------------


----------------------------- Alpha test -----------------------------

+------------------+----------------+
|    chi-square    |    p_value     |
+------------------+----------------+
| [[100.02211446]] | [[0.39645777]] |
+------------------+----------------+

----------------------------------------------------------------------


------------------------------------------------------------------------
```

