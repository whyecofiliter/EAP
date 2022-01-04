# Documentation

## Module

### portfolio_analysis

This package is used for portfolio analysis  including 4 steps:

1. select breakpoints
2. distribute the assets into groups
3. calculate the average and difference of groups
4. present the result

#### class ptf_analysis()

This class is for portfolio analysis containing few functions.

##### def select_breakpoints(self, character, number, perc=*None*):

This function, corresponding to the step 1,  selects the breakpoints of the sample.

**input : ** 

 *character :*  The asset characteristic by which the assets are grouped. 

 *number :* The number of the breakpoint and the number of interval is *number+1*. Once the number is given, the assets would be grouped into *number+1* groups by the average partition of the asset characteristic.  

*perc* : That perc is a list of number, represents the percentiles of the characteristic by which group the assets. Once it is set, then  *number* is invalid.     

**output :** 

*breakpoint :*  The breakpoint is the percentiles of the asset characteristic, ranging from 0% to 100%, whose in length is *number+2*.

**Example**

```python
from EAP.portfolio_analysis import Ptf_analysis as ptfa
import matplotlib.pyplot as plt
import numpy as np

# generate characteristics
character = np.random.normal(0, 100, 10000)
# generate breakpoint
breakpoint = ptfa().slect_breakpoint(character=character, number=9)
print('Breakpoint:', breakpoint)
# comapre with the true breakpoint
for i in np.linspace(0, 100, 11):
        print('True breakpoint', i, '%:', np.percentile(character,i))

========================================================================
Generated breakpoint: [-418.25352494 -127.85494153  -84.90868131  -53.27163604  -25.2394311   1.74938872   25.93867426   50.87047751   84.42711213  128.52009426  334.42181608]
True breakpoint 0.0 %: -418.25352493659074
True breakpoint 10.0 %: -127.85494153240666
True breakpoint 20.0 %: -84.90868130631613
True breakpoint 30.0 %: -53.271636036097135
True breakpoint 40.0 %: -25.239431099072657
True breakpoint 50.0 %: 1.7493887248228535
True breakpoint 60.0 %: 25.938674261710755
True breakpoint 70.0 %: 50.870477505977846
True breakpoint 80.0 %: 84.42711212754239
True breakpoint 90.0 %: 128.5200942602427
True breakpoint 100.0 %: 334.42181607589504
========================================================================
```



##### def distribute(self, character, breakpoint):

This function, corresponding to the step 2, distributes the assets into groups by characteristics  grouped by the breakpoint.

**input :**  

*character :* The characteristic by which the assets are grouped.

*breakpoint :*  The breakpoint of the characteristic.

**output : **

*label :*  an array containing the group number of each asset.

**Example**

```python
# continue the previous code
# generate the groups     
print('The Label of unique value:\n', np.sort(np.unique(ptfa().distribute(character, breakpoint))))
# plot the histogram of the label 
# each group have the same number of samples
plt.hist(ptfa().distribute(character, breakpoint))
label = ptfa().distribute(character, breakpoint)[:, 0]
# print the label
print('Label:\n', ptfa().distribute(character, breakpoint))
========================================================================
The Label of unique value:
 [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
Label:
 [[2.]
 [8.]
 [1.]
 ...
 [1.]
 [4.]
 [8.]]
```



##### def average(self, sample_return, label):

This function, corresponding to the step 3, calculates the average return for each group.

**input :**

*sample_return :* The return of each asset

*label :* The group label of each asset

**output :**

*average_return :* The average return of each group

**Example**

```python
# continue the previous code
# generate the future sample return
sample_return = character/100 + np.random.normal()
ave_ret = ptfa().average(sample_return, label)
# print the groups return
print('average return for groups:\n', ave_ret)
========================================================================
average return for groups:
 [[-1.47855293]
 [-0.75343815]
 [-0.39540537]
 [-0.0971466 ]
 [ 0.17474793]
 [ 0.42303036]
 [ 0.68905747]
 [ 0.95962212]
 [ 1.33445632]
 [ 2.03728439]]
```



#### class Univariate(ptf_analysis):

This class is designed for univariate portfolio analysis.

##### def \__init__ (self, sample, number, perc=None, factor=None, maxlag=12):

The initialization function

**input :**

*sample :*  The samples to be analyzed. Samples usually contain the future return, characteristics, time. The **DEFAULT** setting is the *1th* column is the forecast return, the *2nd* column is the characteristic, the *3rd* column or the index(if data type is Dataframe) is time label.

*number :*  The breakpoint number.

*perc :*  The breakpoint percentiles.

*factor :* The risk factor used to adjust the asset return.

*maxlag :*  The maximum lag for Newey-West adjustment.

##### def divide_by_time(self):

This function groups the sample by time.

**output :** 

*groups_by_time :* The samples group by time.

##### def average_by_time(self):

This function, using the sample group by time from function *divide_by_time*, groups the sample by the characteristic, and then calculate average return of each group samples at every time point. 

**output :** 

*average_group_time(matrix: N_T) :* The average return of groups by each characteristic-time pair.

**Example**

```python
import numpy as np
from portfolio_analysis import Univariate as uni
    
# generate time 
year=np.ones((3000,1),dtype=int)*2020
for i in range(19):
    year=np.append(year,(2019-i)*np.ones((3000,1),dtype=int))
    
# generate character
character=np.random.normal(0,1,20*3000)
# generate future return
ret=character*-0.5+np.random.normal(0,1,20*3000)
# create sample containing future return, character, time
sample=np.array([ret,character,year]).T
# initializ the univariate object
exper=uni(sample,9)
# 
data=exper.average_by_time()
print(data)
==========================================================================================================================
[[ 0.82812256  0.87549215  0.81043114  0.77480366  0.85232487  0.85599445   0.90860961  0.76600211  0.91360546  0.85921985  0.96717798  0.77677131   0.88669273  0.86895145  0.97832435  0.88494486  0.82571951  0.84777939   0.89373487  0.95906454]
 [ 0.51155724  0.4963439   0.6100762   0.47351625  0.46844971  0.53303287   0.52087477  0.43934316  0.51169633  0.61918844  0.56254028  0.50949226   0.39033219  0.49685445  0.5844816   0.48723354  0.49861094  0.43197525   0.40040156  0.57529228]
 [ 0.41566251  0.3421546   0.27117215  0.35550346  0.28884636  0.43710998   0.33146264  0.27860032  0.35956881  0.34818479  0.35692361  0.42462374   0.16909231  0.33823117  0.31762348  0.44863438  0.42785283  0.20093775   0.29664738  0.31509963]
 [ 0.21972246  0.24685649  0.29933776  0.09880866  0.13564638  0.17673649   0.14251437  0.12188551  0.1567432   0.20428427  0.15009782  0.08488247   0.20489871  0.10598241  0.12591301  0.17287433  0.11180376  0.09941738   0.22635281  0.22828588]
 [ 0.01851548  0.05771421  0.0624163   0.05368921  0.15247324  0.05839522   0.05864669  0.01863668 -0.08367879  0.09273579  0.18374921  0.12331214   0.03635538  0.05804576  0.0116589  -0.04158565  0.11655945  0.09727234   0.14038867  0.13594649]
 [-0.07910789 -0.04670755  0.08732773 -0.07361966 -0.00232509 -0.08546681  -0.15020487 -0.05302521 -0.07922696 -0.1088824  -0.01700017 -0.06742183   0.00190131  0.00961174 -0.05953252 -0.09504501 -0.0958816  -0.00355493  -0.08553405 -0.05343558]
 [-0.13094033 -0.23888179 -0.11046595 -0.11176528 -0.14017103 -0.17184142  -0.26587781 -0.14426219 -0.15687278 -0.15962335 -0.18586504 -0.2367552  -0.26761165 -0.16169935 -0.26608677 -0.16202763 -0.24272797 -0.17049684  -0.21470737 -0.13520545]
 [-0.35621842 -0.28111488 -0.42057927 -0.37219582 -0.25449753 -0.36362452  -0.34165952 -0.28564624 -0.29936621 -0.32545156 -0.28208242 -0.36730096  -0.24269836 -0.31584032 -0.34207757 -0.35185102 -0.35515763 -0.32239715  -0.2803911  -0.36334961]
 [-0.58529295 -0.54329245 -0.52006031 -0.49856708 -0.44262707 -0.4464171  -0.58846501 -0.56725297 -0.35845646 -0.52923391 -0.42119445 -0.55659388  -0.47716067 -0.4574991  -0.52123094 -0.54767832 -0.50289813 -0.45529132  -0.58429513 -0.48110405]
 [-0.81992395 -0.95766159 -0.92069685 -0.92906348 -0.84891875 -0.81670916  -0.90281776 -0.84845902 -0.90479169 -0.86860559 -0.96790821 -0.9464988  -0.88176205 -0.96118242 -0.92402295 -0.81623283 -0.81560442 -0.85841478  -0.87337267 -0.8070857 ]]
```



##### def difference(self, average_group):

This functions calculates the difference of group return, which, in detail, is the last group average return minus the first group average return. 

 **input :** 

*average_group :* The average return of groups by each characteristic-time pair.

**output :**

*result :* The matrix added with the difference of average group return.

##### def factor_adjustment(self, result):

This function calculates the group return adjusted by risk factors.

**input :**

*result :* The return table with difference sequence.

**output :**

*alpha :* The anomaly

*ttest :* The t-value of the anomaly.

##### def summary_and_test(self):

This function summarizes the result and take t-test.

**output : **

*self.average :* 

*self.ttest :* 

##### def print_summary_by_time(self):

This function print the summary grouped by time.

##### def print_summary(self):

This function print the summary grouped by characteristic and averaged by time.

**Example**

```python
# continue the previous code
exper.summary_and_test()
exper.print_summary_by_time()
========================================================================================================
+--------+-------+-------+-------+-------+--------+--------+--------+--------+--------+--------+--------+
|  Time  |   1   |   2   |   3   |   4   |   5    |   6    |   7    |   8    |   9    |   10   |  diff  |
+--------+-------+-------+-------+-------+--------+--------+--------+--------+--------+--------+--------+
| 2001.0 | 0.828 | 0.512 | 0.416 |  0.22 | 0.019  | -0.079 | -0.131 | -0.356 | -0.585 | -0.82  | -1.648 |
| 2002.0 | 0.875 | 0.496 | 0.342 | 0.247 | 0.058  | -0.047 | -0.239 | -0.281 | -0.543 | -0.958 | -1.833 |
| 2003.0 |  0.81 |  0.61 | 0.271 | 0.299 | 0.062  | 0.087  | -0.11  | -0.421 | -0.52  | -0.921 | -1.731 |
| 2004.0 | 0.775 | 0.474 | 0.356 | 0.099 | 0.054  | -0.074 | -0.112 | -0.372 | -0.499 | -0.929 | -1.704 |
| 2005.0 | 0.852 | 0.468 | 0.289 | 0.136 | 0.152  | -0.002 | -0.14  | -0.254 | -0.443 | -0.849 | -1.701 |
| 2006.0 | 0.856 | 0.533 | 0.437 | 0.177 | 0.058  | -0.085 | -0.172 | -0.364 | -0.446 | -0.817 | -1.673 |
| 2007.0 | 0.909 | 0.521 | 0.331 | 0.143 | 0.059  | -0.15  | -0.266 | -0.342 | -0.588 | -0.903 | -1.811 |
| 2008.0 | 0.766 | 0.439 | 0.279 | 0.122 | 0.019  | -0.053 | -0.144 | -0.286 | -0.567 | -0.848 | -1.614 |
| 2009.0 | 0.914 | 0.512 |  0.36 | 0.157 | -0.084 | -0.079 | -0.157 | -0.299 | -0.358 | -0.905 | -1.818 |
| 2010.0 | 0.859 | 0.619 | 0.348 | 0.204 | 0.093  | -0.109 | -0.16  | -0.325 | -0.529 | -0.869 | -1.728 |
| 2011.0 | 0.967 | 0.563 | 0.357 |  0.15 | 0.184  | -0.017 | -0.186 | -0.282 | -0.421 | -0.968 | -1.935 |
| 2012.0 | 0.777 | 0.509 | 0.425 | 0.085 | 0.123  | -0.067 | -0.237 | -0.367 | -0.557 | -0.946 | -1.723 |
| 2013.0 | 0.887 |  0.39 | 0.169 | 0.205 | 0.036  | 0.002  | -0.268 | -0.243 | -0.477 | -0.882 | -1.768 |
| 2014.0 | 0.869 | 0.497 | 0.338 | 0.106 | 0.058  |  0.01  | -0.162 | -0.316 | -0.457 | -0.961 | -1.83  |
| 2015.0 | 0.978 | 0.584 | 0.318 | 0.126 | 0.012  | -0.06  | -0.266 | -0.342 | -0.521 | -0.924 | -1.902 |
| 2016.0 | 0.885 | 0.487 | 0.449 | 0.173 | -0.042 | -0.095 | -0.162 | -0.352 | -0.548 | -0.816 | -1.701 |
| 2017.0 | 0.826 | 0.499 | 0.428 | 0.112 | 0.117  | -0.096 | -0.243 | -0.355 | -0.503 | -0.816 | -1.641 |
| 2018.0 | 0.848 | 0.432 | 0.201 | 0.099 | 0.097  | -0.004 | -0.17  | -0.322 | -0.455 | -0.858 | -1.706 |
| 2019.0 | 0.894 |  0.4  | 0.297 | 0.226 |  0.14  | -0.086 | -0.215 | -0.28  | -0.584 | -0.873 | -1.767 |
| 2020.0 | 0.959 | 0.575 | 0.315 | 0.228 | 0.136  | -0.053 | -0.135 | -0.363 | -0.481 | -0.807 | -1.766 |
+--------+-------+-------+-------+-------+--------+--------+--------+--------+--------+--------+--------+

exper.print_summary()
==================================================================================================================
+---------+--------+--------+--------+--------+-------+--------+---------+---------+---------+---------+---------+
|  Group  |   1    |   2    |   3    |   4    |   5   |   6    |    7    |    8    |    9    |    10   |   Diff  |
+---------+--------+--------+--------+--------+-------+--------+---------+---------+---------+---------+---------+
| Average | 0.867  | 0.506  | 0.336  | 0.166  | 0.068 | -0.053 |  -0.184 |  -0.326 |  -0.504 |  -0.883 |  -1.75  |
|  T-Test | 63.706 | 35.707 | 20.199 | 12.637 |  4.6  | -4.463 | -15.616 | -32.172 | -36.497 | -73.162 | -92.152 |
+---------+--------+--------+--------+--------+-------+--------+---------+---------+---------+---------+---------+

# generate factor
factor=np.random.normal(0,1.0,(20,1))
exper=uni(sample,9,factor=factor,maxlag=12)
# print(exper.summary_and_test()) # if needed
exper.print_summary()
====================================================================================================================
+---------+--------+--------+--------+--------+-------+---------+---------+---------+---------+---------+----------+
|  Group  |   1    |   2    |   3    |   4    |   5   |    6    |    7    |    8    |    9    |    10   |   Diff   |
+---------+--------+--------+--------+--------+-------+---------+---------+---------+---------+---------+----------+
| Average | 0.867  | 0.506  | 0.336  | 0.166  | 0.068 |  -0.053 |  -0.184 |  -0.326 |  -0.504 |  -0.883 |  -1.75   |
|  T-Test | 63.706 | 35.707 | 20.199 | 12.637 |  4.6  |  -4.463 | -15.616 | -32.172 | -36.497 | -73.162 | -92.152  |
|  Alpha  | 0.869  | 0.507  | 0.336  | 0.164  | 0.067 |  -0.054 |  -0.184 |  -0.326 |  -0.503 |  -0.883 |  -1.752  |
| Alpha-T | 62.39  | 69.24  | 43.377 | 16.673 | 8.204 | -12.372 | -16.679 | -65.704 | -87.618 | -93.223 | -139.881 |
+---------+--------+--------+--------+--------+-------+---------+---------+---------+---------+---------+----------+
```

 

### fama_macbeth

This module is designed for Fama_Macbeth regression(1976). 

Fama-Macbeth Regression follows two steps:

1.  Specify the model and take cross-sectional regression.
2.  Take the time-series average of regression coefficient

For more details, please read Empirical Asset Pricing: The Cross Section of Stock Returns. Bali, Engle, Murray, 2016.



#### class Fama_macbeth_regress():

Package Needed: numpy, statsmodels, scipy, prettytable.

##### def \__init__(self, sample):

**input :**

*sample :* data for analysis. The structure of the sample : 

​                The first column is dependent variable/ test portfolio return.

​                The second to the last-1 columns are independent variable/ factor loadings.

​                The last column is the time label.

##### def divide_by_time(self, sample):

This function group the sample by time.

**input :**

*sample :* The data for analysis in the \__init__ function.

**output :**

*groups_by_time :* The sample grouped by time.



##### def cross_sectional_regress(self, add_constant=True):

This function conducts the first step of Fama-Macebth Regression Fama-Macbeth, that taking the cross-sectional regression for each period.

**input :** 

*add_constant :* whether add intercept when take the cross-sectional regression

**output :**

*parameters :* The regression coefficient/factor risk premium, whose rows are the group coefficient and columns are regression variable.

*tvalue :* t value for the coefficient.

*rsquare :* The r-square.

*adjrsq :*  The adjust r-square.

*n :* The sample quantity in each group.



##### def time_series_average(self, **kwargs):

This function conducts the second step of Fama-Macbeth regression, take the time series average of cross section regression.



##### def fit(self, **kwargs):

This function fits the model by running the time_series_average function.

**Example**

```
import numpy as np
from fama_macbeth import Fama_macbeth_regress
    
# construct sample
year=np.ones((3000,1),dtype=int)*2020
for i in range(19):
    year=np.append(year,(2019-i)*np.ones((3000,1),dtype=int))
character=np.random.normal(0,1,(2,20*3000))
# print('Character:',character)
ret=np.array([-0.5,-1]).dot(character)+np.random.normal(0,1,20*3000)
sample=np.array([ret,character[0],character[1],year]).T    
# print('Sample:',sample)
# print(sample.shape)

model = Fama_macbeth_regress(sample)
result = model.fit(add_constant=False)
print(result)
=========================================================================
para_average: [-0.501 -1.003]
tvalue: [-111.857 -202.247]
R: 0.5579113793318332
ADJ_R: 0.5576164569698131
sample number N: 3000.0
```



##### def summary_by_time(self):

This function summarize the cross-section regression result at each time.

Package needed: prettytable.

**Example**

```
# continue the previous code
model.summary_by_time()
==========================================================================
+--------+-----------------+----------+--------------+---------------+
|  Year  |      Param      | R Square | Adj R Square | Sample Number |
+--------+-----------------+----------+--------------+---------------+
| 2001.0 | [-0.499 -0.990] |   0.53   |     0.53     |      3000     |
| 2002.0 | [-0.524 -0.987] |   0.56   |     0.56     |      3000     |
| 2003.0 | [-0.544 -1.015] |   0.58   |     0.58     |      3000     |
| 2004.0 | [-0.474 -0.948] |   0.53   |     0.53     |      3000     |
| 2005.0 | [-0.502 -1.007] |   0.57   |     0.57     |      3000     |
| 2006.0 | [-0.497 -0.981] |   0.55   |     0.55     |      3000     |
| 2007.0 | [-0.526 -1.020] |   0.57   |     0.57     |      3000     |
| 2008.0 | [-0.476 -1.024] |   0.56   |     0.56     |      3000     |
| 2009.0 | [-0.533 -1.011] |   0.57   |     0.57     |      3000     |
| 2010.0 | [-0.493 -1.029] |   0.57   |     0.57     |      3000     |
| 2011.0 | [-0.504 -0.975] |   0.55   |     0.55     |      3000     |
| 2012.0 | [-0.508 -1.002] |   0.56   |     0.56     |      3000     |
| 2013.0 | [-0.474 -1.015] |   0.56   |     0.56     |      3000     |
| 2014.0 | [-0.503 -0.998] |   0.55   |     0.55     |      3000     |
| 2015.0 | [-0.485 -1.034] |   0.55   |     0.55     |      3000     |
| 2016.0 | [-0.514 -1.005] |   0.57   |     0.57     |      3000     |
| 2017.0 | [-0.498 -1.016] |   0.58   |     0.58     |      3000     |
| 2018.0 | [-0.475 -0.994] |   0.55   |     0.55     |      3000     |
| 2019.0 | [-0.487 -0.974] |   0.54   |     0.54     |      3000     |
| 2020.0 | [-0.511 -1.031] |   0.56   |     0.56     |      3000     |
+--------+-----------------+----------+--------------+---------------+

```



##### def summary(self, charactername=None):

This function summarize the final result.

**input :**

*charactername :* The factors' name in the cross-section regression model.

**Example**

```python
# continue the previous code
model.summary()
================================================================================
+-----------------+---------------------+-----------+---------------+-----------+
|      Param      |     Param Tvalue    | Average R | Average adj R | Average n |
+-----------------+---------------------+-----------+---------------+-----------+
| [-0.501 -1.003] | [-111.857 -202.247] |   0.558   |     0.558     |   3000.0  |
+-----------------+---------------------+-----------+---------------+-----------+
```



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

*list_y :* The return matrix with i rows and t columns.

*factor :* The factor risk premium return series.



##### def ts_regress(self, newey_west=*True*):

This function is for conducting the time series regression.

**input :**

*newey_west ：* conduct the newey_west adjustment or not.

**output :**

*self.alpha :* The regression alpha.

*self.e_mat :* The error matrix

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

*grs_stats :* The GRS statistics.

*p_value :* The p_value.

**Example**

```python
# continue the previous code
print(re.grs())
==============================================
(array([[0.67160203]]), array([[0.75175471]]))
```



### cross_section_regress

#### class CS_regress()

##### def \__init__(self, y_list, factor):

This function initializes the class.

**input :**

*y_list :* The assets return series list.

*factor :* The factor risk premium series.



##### def ts_regress(self):

This function conducts the time series regression.

**output :**

*beta(N, c) :* The betas for each asset.

*err_mat :* The error matrix.



##### def ave(self):

This function conducts the average operation on the assets returns series list.

**output :**

*np.mean(self.y_list, axis=1)* : The average of the assets returns series list.



##### def cs_regress(self, beta, err_mat, constant=*True*, gls=*True*, **kwargs):

This function takes the cross-sectional regression.

**input :**

*beta :* The betas from the output of the function ts_regress().

*err_mat :* The error matrix from the output of the function ts_regress().

*constant :* add constant or not. The default is add constant.

*gls :* GLS regression or OLS regression. The default is GLS regression.

**output :**

*params :* The params of the cross regression model.

*resid :* The residue of the cross regression model.

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

*beta :* The betas from the output of the function ts_regress().

*err_mat :* The error matrix from the output of the function ts_regress().

*shanken :* Take the shanken adjustment or not. The default is *True*.

*constant :* add constant or not.

*gls :* GLS regression or OLS regression. The default is GLS regression.

**output :**

*param_cov_mat :* The covariance matrix of the parameters.

*resid_cov_mat :* The covariance matrix of the residue.



##### def t_test(self, param_cov_mat, params):

This function takes t-test for parameters.

**input :**

*param_cov_mat :* The covariance matrix of the parameters from the function cov_mat.

*params :* The parameters from the function cs_regress.

**output :**

*t_value :* The t-value for statistical inference.

*p_value :* The p-value for statistical inference.



##### def union_test(self, resid_cov_mat, resid):

This function takes union test for parameters.

**input :**

*resid_cov_mat :* The covariance matrix of the residue. 

*resid :* The residue from the function cs_regress.

**output :**

*chi_square :* The chi-square statistics.

*p_value :* The p-value corresponding to the chi-square.



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



### adjust

This module consists of several common adjustment method in factor analysis.

##### def ols(y, x, constant=*True*):

The function is for OLS regression, which is equal to the OLS module in package *statsmodels*. 

**input :**

*y :* The dependent variable.

*x :* The explanatory variable.

*constant :* add constant or not in OLS model. The default is *True*.

**output :**

*result :* The result of the regression.

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

*y :* The dependent variable.

*X :* The explanatory variable.

**output :**

*V_ols :* The white variance estimate

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

*y :* The dependent variable.

*X :* The explanatory variable.

*J :* The lag.

**output :**

*V_ols :* The Newey-West variance estimate.

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



##### def white_t(y, X, params=*None*, **kwargs):

This function constructs t-test based on White variance estimate.

**input :**

*y :* The dependent variable.

*X :* The explanatory variable.

*params :* Already have parameters or not. The default is *None*.

**output :**

*t_value :* The t-value of parameters.

*p_value :* The p-value of parameters.

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

*y :* The dependent variable.

*X :* The explanatory variable.

*params :* Already have parameters or not. The default is None.

**output :**

*t_value :* The t-value of parameters.

*p_value :* The p-value of parameters.

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





