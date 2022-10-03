---
sort: 3
---

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

*sample (ndarray/DataFrame):* data for analysis. The structure of the sample : 

​                The first column is dependent variable/ test portfolio return.

​                The second to the last-1 columns are independent variable/ factor loadings.

​                The last column is the time label.



##### def divide_by_time(self, sample):

This function group the sample by time.

**input :**

*sample (ndarray/DataFrame):* The data for analysis in the \__init__ function.

**output :**

*groups_by_time (list):* The sample grouped by time.



##### def cross_sectional_regress(self, add_constant=True, normalization=True, **kwargs):

This function conducts the first step of Fama-Macebth Regression Fama-Macbeth, that taking the cross-sectional regression for each period.

**input :** 

*add_constant (boolean):* whether add intercept when take the cross-sectional regression.

*normalization (boolean):* Whether conduct normalization.

**output :**

*parameters (ndarray):* The regression coefficient/factor risk premium, whose rows are the group coefficient and columns are regression variable.

*tvalue (ndarray):* t value for the coefficient.

*rsquare (list):* The r-square.

*adjrsq (list):*  The adjust r-square.

*n (list):* The sample quantity in each group.



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



#### class Factor_mimicking_portfolio

This module is designed for generating factor mimicking portfolio following the Fama-French (1993) conventions, and then calculating factor risk premium.

Fama-French (1993) conventions.



| Size\Factor  | Low(<=30%) | Medium(30%< & <=70%) | High(>70%) |
| ------------ | ---------- | -------------------- | ---------- |
| Small(<=50%) | S/L        | S/M                  | S/H        |
| Big(>50%)    | B/L        | B/M                  | B/H        |



1. Group stocks by two dimensions. One dimension is size, divided by 50% small stocks and 50% big stocks. The other is the factor, divided by 30% tail factor stocks, 30%~70% factor stocks, and 70%~100% factor stocks.

2. Calculate market value weighted portfolio return and factor risk premium.


   $$
   SMB=1/3(S/L+S/M+S/H)-1/3(B/L+B/M+B/H)
   $$

   $$
   Factor=1/2(S/H+B/H)-1/2(S/L+B/L)
   $$

3. 



3. In Fama-French (1993), the factor is book-to-market ratio, and other literatures follow the same way to construct factor mimicking portfolio. The return of each portfolio is represented by the market value weighted portfolio return.  



##### def \__init__(self, sample, perc_row=[0, 50, 100], perc_col=[0, 30, 70, 100],  percn_row=None, percn_col=None, weight=True):

This function initializes the object.

**input :**

*sample (ndarray/DataFrame):* data for analysis. The structure of the sample : 

​                The first column is dependent variable/ test portfolio return.

​                The second is the first factor.

​                The third is the second factor.

​                The fourth is the timestamp.

​                The fifth is the weight.

*perc_row (list/array):* The percentile points that divide stocks by the first factor. The **Default** percentile is [0, 50, 100].

*perc_col (list/array):* The percentile points that divide stocks by the second factor. The **Default** percentile is [0, 30, 70, 100].

*percn_row (list/array):* The percentile that divide stocks by the first factor.

*percn_col (list/array):* The percentile that divide stocks by the second factor. 

*weight (array/Series):* Whether the portfolio return calculated by weight. The **Default** is True.



##### def portfolio_return_time(self):

This function is to construct portfolio and calculate the average return and difference matrix.

**output :** 

*diff (ndarray):* The differenced portfolio return matrix.



**Example**

```python
'''
TEST Factor_mimicking_portfolio
construct sample:
    1. 20 Periods
    2. 3000 Observations for each Period
    3. Character negative with return following the return=character*-0.5+sigma where sigma~N(0,1)
'''
import numpy as np
from fama_macbeth import Factor_mimicking_portfolio
    
# construct sample
year=np.ones((3000,1),dtype=int)*2020
for i in range(19):
    year=np.append(year,(2019-i)*np.ones((3000,1),dtype=int))
character=np.random.normal(0, 1, (2, 20*3000))
weight = np.random.uniform(0, 1, (1, 20*3000))
#    print('Character:',character)
ret=np.array([-0.5,-1]).dot(character)+np.random.normal(0,1,20*3000)
sample=np.array([ret, character[0], character[1], year, weight[0]]).T    
#    print('Sample:',sample)
#    print(sample.shape)

model = Factor_mimicking_portfolio(sample)
portfolio_return_time = model.portfolio_return_time()
print('portfolio_return_time:', portfolio_return_time)
print('portfolio_return_time:', np.shape(portfolio_return_time))
========================================================================
portfolio_return_time: 
 [[[ 1.6302854   1.5920872   1.54199455  1.47560967  1.60182404
    1.48860463  1.70067317  1.57084898  1.52938766  1.54919833
    1.58910675  1.44369383  1.6489323   1.57230951  1.64104889
    1.52816059  1.53197648  1.44067358  1.55358692  1.60347805]
  [ 0.27087317  0.37938255  0.46473997  0.43118946  0.36934624
    0.3215685   0.51716485  0.3740702   0.38663032  0.44333387
    0.38917603  0.31347239  0.30760233  0.41415477  0.45250083
    0.4439825   0.35821053  0.40601508  0.40495275  0.46236114]
  [-0.77286286 -0.67348462 -0.79216628 -0.76712914 -0.73108828
   -0.74141791 -0.85011447 -0.65743291 -0.57583562 -0.80490414
   -0.64311824 -0.79901273 -0.81325556 -0.84443278 -0.80362147
   -0.75246184 -0.61693674 -0.69909426 -0.68554981 -0.6564971 ]
  [-2.40314826 -2.26557182 -2.33416083 -2.24273881 -2.33291232
   -2.23002254 -2.55078764 -2.22828188 -2.10522328 -2.35410248
   -2.23222499 -2.24270656 -2.46218786 -2.41674229 -2.44467035
   -2.28062243 -2.14891322 -2.13976784 -2.23913673 -2.25997515]]

 [[ 0.73368397  0.87086931  0.69626265  0.83424514  0.75825827
    0.65550255  0.79076344  0.77316807  0.82031302  0.83320932
    0.76971902  0.81248391  0.74568233  0.7749383   0.69168886
    0.7511554   0.70757116  0.75320784  0.78223447  0.73699303]
  [-0.55494493 -0.4097451  -0.42006211 -0.36317491 -0.47791724
   -0.40042274 -0.36004971 -0.39127871 -0.4712118  -0.33368187
   -0.48957933 -0.41103346 -0.46612991 -0.39504384 -0.34021312
   -0.41049521 -0.33209925 -0.39212119 -0.42890556 -0.40389983]
  [-1.63941363 -1.52575966 -1.55711119 -1.45657452 -1.53325438
   -1.5163362  -1.4984305  -1.50929504 -1.52485715 -1.60331314
   -1.56591033 -1.60738805 -1.77875307 -1.4963315  -1.65246163
   -1.55526031 -1.40809313 -1.49853102 -1.56149388 -1.48409324]
  [-2.3730976  -2.39662897 -2.25337384 -2.29081966 -2.29151265
   -2.17183875 -2.28919394 -2.28246311 -2.34517017 -2.43652246
   -2.33562934 -2.41987196 -2.5244354  -2.27126981 -2.34415049
   -2.30641572 -2.11566429 -2.25173887 -2.34372835 -2.22108628]]

 [[-0.89660143 -0.72121789 -0.8457319  -0.64136454 -0.84356577
   -0.83310208 -0.90990973 -0.79768091 -0.70907464 -0.71598901
   -0.81938773 -0.63120993 -0.90324997 -0.7973712  -0.94936003
   -0.77700518 -0.82440531 -0.68746574 -0.77135245 -0.86648502]
  [-0.82581811 -0.78912765 -0.88480208 -0.79436437 -0.84726347
   -0.72199124 -0.87721456 -0.7653489  -0.85784212 -0.77701574
   -0.87875536 -0.72450585 -0.77373224 -0.80919861 -0.79271394
   -0.85447771 -0.69030979 -0.79813627 -0.83385831 -0.86626098]
  [-0.86655077 -0.85227505 -0.76494491 -0.68944538 -0.8021661
   -0.77491829 -0.64831603 -0.85186213 -0.94902153 -0.798409
   -0.92279209 -0.80837532 -0.96549751 -0.65189873 -0.84884016
   -0.80279847 -0.79115639 -0.79943676 -0.87594408 -0.82759615]
  [ 0.03005066 -0.13105715  0.08078699 -0.04808085  0.04139968
    0.05818379  0.26159369 -0.05418122 -0.23994689 -0.08241999
   -0.10340436 -0.17716539 -0.06224754  0.14547248  0.10051987
   -0.02579329  0.03324893 -0.11197102 -0.10459162  0.03888887]]]
portfolio_return_time: 
 (3, 4, 20)
```



##### def portfolio_return(self):

This function is to construct factor risk premium.

**output :**

*return_row :* The first factor risk premium. The **Default** is size factor.

*return_col :* The second factor risk premium. 



##### def portfolio_return_horizon(self, period, log):

This function is to construct horizon pricing factor. For details, read *Horizon Pricing, JFQA, 2016, 51(6): 1769-1793.* 

**input :**

*period (int):* The lagged period for constructing factor risk premium return.

*log(boolean):* whether use log return.

**output :**

*return_row_multi :* The first factor risk premium. The **DEFAULT** is size factor.

*return_col_multi :* The second factor premium.



**Example**

```python
# Continue the previous code
portfolio_return = model.portfolio_return()
print('portfolio_return_row: \n', portfolio_return[0])
print('portfolio_return_row:', np.shape(portfolio_return[0]))
print('portfolio_return_col: \n', portfolio_return[1])
print('portfolio_return_col:', np.shape(portfolio_return[1]))
==============================================================
portfolio_return_row: 
 1970-01-01 00:00:00.000002001   -0.639730
1970-01-01 00:00:00.000002002   -0.623419
1970-01-01 00:00:00.000002003   -0.603673
1970-01-01 00:00:00.000002004   -0.543314
1970-01-01 00:00:00.000002005   -0.612899
1970-01-01 00:00:00.000002006   -0.567957
1970-01-01 00:00:00.000002007   -0.543462
1970-01-01 00:00:00.000002008   -0.617268
1970-01-01 00:00:00.000002009   -0.688971
1970-01-01 00:00:00.000002010   -0.593458
1970-01-01 00:00:00.000002011   -0.681085
1970-01-01 00:00:00.000002012   -0.585314
1970-01-01 00:00:00.000002013   -0.676182
1970-01-01 00:00:00.000002014   -0.528249
1970-01-01 00:00:00.000002015   -0.622599
1970-01-01 00:00:00.000002016   -0.615019
1970-01-01 00:00:00.000002017   -0.568156
1970-01-01 00:00:00.000002018   -0.599252
1970-01-01 00:00:00.000002019   -0.646437
1970-01-01 00:00:00.000002020   -0.630363
dtype: float64
portfolio_return_row: (20,)
portfolio_return_col: 
 1970-01-01 00:00:00.000002001   -1.582065
1970-01-01 00:00:00.000002002   -1.597753
1970-01-01 00:00:00.000002003   -1.502249
1970-01-01 00:00:00.000002004   -1.527213
1970-01-01 00:00:00.000002005   -1.527675
1970-01-01 00:00:00.000002006   -1.447892
1970-01-01 00:00:00.000002007   -1.526129
1970-01-01 00:00:00.000002008   -1.521642
1970-01-01 00:00:00.000002009   -1.563447
1970-01-01 00:00:00.000002010   -1.624348
1970-01-01 00:00:00.000002011   -1.557086
1970-01-01 00:00:00.000002012   -1.613248
1970-01-01 00:00:00.000002013   -1.682957
1970-01-01 00:00:00.000002014   -1.514180
1970-01-01 00:00:00.000002015   -1.562767
1970-01-01 00:00:00.000002016   -1.537610
1970-01-01 00:00:00.000002017   -1.410443
1970-01-01 00:00:00.000002018   -1.501159
1970-01-01 00:00:00.000002019   -1.562486
1970-01-01 00:00:00.000002020   -1.480724
dtype: float64
portfolio_return_col: (20,)
```

