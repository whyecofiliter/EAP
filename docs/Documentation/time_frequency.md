---
sort: 5
---

### time_frequency

This module is designed for time frequency asset pricing model, including Fourier Transform based method and Wavelet based method.



#### class Wavelet():

This class is designed for decomposing the series using wavelet method, basing on the package  **pywt**.



##### def \__init__(self, series):

**input:** 

*series (array) :* the series to be decomposed.



##### def _decompose(self, wave, mode, level=None):

This function is designed for decomposing the series using given wavelet family, mode and level.

**input :**

*wave (str) :* The chosen wavelet family, which is the same in **pywt**.

*mode (str) :* choose multiscale or single scale. 'multi' denotes multiscale. 'single' denotes single scale.

*level (int) :* the level of multiscale. If 'multi' is chosen, it must be set.

**output :**

*wave_dec (list):* The decomposed details including (CA, CN, CN-1,..., C1).



##### def _pick_details(self):

This function is designed for picking the details of decomposed series at different level.

**output:**

*pick_series (list):* pick the detail series at each level. Each elements in list only contains the details at that level. 



##### def _rebuild(self, mode='constant'):

This function is designed for rebuilding the detail series from the picked series at each level.

**input:**

*mode (str):* The recomposed method.

**output:**

*wave_rec (list):* The recomposed series from the details at each level.



##### def fit(self, wave, mode, level, output=False):

This function is designed for fitting the model.

**input :**

*wave (str) :* The chosen wavelet family, which is the same in **pywt**.

*mode (str) :* choose multiscale or single scale. 'multi' denotes multiscale. 'single' denotes single scale.

*level (int) :* the level of multiscale. If 'multi' is chosen, it must be set.

*output (boolean):* whether output the result. The **DEFAULT** is False.

**output :**

*wave_rec (list):* The recomposed series from the details at each level, if output is True.



#### class wavelet_pricing():

This module is designed for wavelet pricing model.

##### def \__init__(self, rets, factors):

**input :**

*rets (ndarray/Series/DataFrame):* the dependent variables of the return.

*factors (ndarray/Series/DataFrame):* the independent variables of factors.



##### def wavelet_dec_rec(self, **kwargs):

This function is designed for wavelet decomposing and recomposing.

**input :**

*kwargs :* the kwargs include wave family, mode, and level of wavelet.

**output :**

*rets_dec_s (list):* The recomposed detail series of returns (rets).

*factors_dec_s (list):* The recomposed detail series of factors (factors).



##### def wavelet_regression(self, **kwargs):

This function is designed for OLS regression of detail series between return and factors at each level.

**input :**

***kwargs :* the kwargs include 'constant': whether the regression includes the constant.

**output :**

*regress (list):* the regression results of OLS in package **statsmodels**.



##### def fit(self, wave, mode, level, win=None, robust=False, constant=True):

This function is designed for fitting the model.

**input :**

*wave (str) :* The chosen wavelet family, which is the same in **pywt**.

*mode (str) :* choose multiscale or single scale. 'multi' denotes multiscale. 'single' denotes single scale.

*level (int) :* the level of multiscale. If 'multi' is chosen, it must be set.

*win (int):* The rolling window if rolling regression is used.

*robust (boolean):* whether use the robust covariance matrix.

*constant (boolean):* whether includes the constant in regression. The **DEFAULT** is True.



##### def summary(self, export=False):

This function is designed for printing the summary.

**input :**

*export (boolean):* whether export the summary table. The **DEFAULT** is False.

**output :**

*df (DataFrame):* if export is True, then return the summary table. 

