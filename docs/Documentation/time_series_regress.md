###  time_series_regress

#### class TS_regress()

This class is designed for time series regression, 


$$
r_{i,t} = \beta_if_t + \epsilon_{i,t}
$$


to obtain the beta for each asset.



##### def \__init__(self, list_y, factor):

This function initializes the object.

**input :**

*list_y (list/DataFrame):* The return matrix with i rows and t columns.

*factor (ndarray or DataFrame):* The factor risk premium return series.



##### def ts_regress(self, newey_west=*True*):

This function is for conducting the time series regression.

**input :**

*newey_west  (boolean)：* conduct the newey_west adjustment or not.

**output :**

*self.alpha (list):* The regression alpha.

*self.e_mat (ndarray):* The error matrix.



**Example**

```python
from statsmodels.base.model import Model
from statsmodels.tools.tools import add_constant
from EAP.time_series_regress import TS_regress

X = np.random.normal(loc=0.0, scale=1.0, size=(2000,10))
y_list = []
for i in range(10) :
    b = np.random.uniform(low=0.1, high=1.1, size=(10,1))
    e = np.random.normal(loc=0.0, scale=1.0, size=(2000,1))
    y = X.dot(b) + e 
    y_list.append(y)

re = TS_regress(y_list, X)
```



##### def fit(self, **kwargs):

This function runs the function, ts_regress.

**Example**

```python
# continue the previous code
re.fit()
```



##### def summary(self):

This function summarize the result, including the GRS test.

**Example**

```python
# continue the previous code
re.summary()
=============================================================================
----------------------------------- GRS Test -------------------------------- 

GRS Statistics: 
 [[0.67160203]]
GRS p_value: 
 [[0.75175471]]
------------------------------------------------------------------------------
```



##### def grs(self):

This function conducts the GRS test.

**output :**

*grs_stats (list):* The GRS statistics.

*p_value (list):* The p_value.



**Example**

```python
# continue the previous code
print(re.grs())
==============================================
(array([[0.67160203]]), array([[0.75175471]]))
```

