### industry

This module is used for factor model using in industry.

#### class Portfolio()

This class is designed for factor investment in industry.

##### def \__init__(self, sample, number, perc=*None*, maxlag=12, weight=*False*):

**input :**

*sample (ndarray or DataFrame) :*  The samples to be analyzed. Samples usually contain the future return, characteristics, time. The **DEFAULT** setting is 

the *1th* column is the forecast return, 

the *2nd* column is the characteristic, 

the *3rd* column or the index(if data type is Dataframe) is time label.

the 4th column is firm token if weight== False otherwise is weight.

the 5th column is firm token if weight == True.

*number (int):* the breakpoint number.

*perc (list):* the breakpoint percentiles.

*maxlag (int):* maximum lag for *Newey-West* adjustment.

*weight (boolean):* whether calculate the portfolio return using weight.



##### def extract_portfolio(self, group_num):

Extract portfolio return at each timepoint.

**input :**

*group_num (int):* the index of group number.

**output :**

*self._average_group_by_time[group_num, :] (array) :* the average group return series.



##### def plot(self, group_num):

Plot portfolio return series.

**input :**

*group_num (int):* the index of group number.

**output :**

*ax (matplotlib.pyplot) :* the average group return series.



##### def plot_all(self):

Plot all portfolio return.



##### def group_turnover(self, group_num):

Calculate group turnover

**input :**

*group_num (int):* the index of group number.

**output :**

*turnover_each_time (array) :* turnover (index: group_num) at each time.



##### def turnover(self):

Calculate all group turnover.

**output :**

*turnover_allgroup (ndarray) :* turnover of all groups.



##### def compound_return(self):

Calculate compound return

**output :**

*self._group_compounnd_return (ndarray) :* group compound return.



##### def plot_compound_return(self):

Plot compound return



##### def drawdown(self, group_num):

Calculate drawdown.

**input :**

*group_num (int):* the index of group number.

**output :**

*group_drawdown (array) :* group drawdown.



##### def maxdrawdown(self):

**output :**

*self._max_drawdown (ndarray) :* max drawdown



##### def plot_drawdown(self):

Plot drawdown



##### def sharpe_ratio(self):

**output :**

*self._group_sharpe_ratio (array) :* Sharpe ratio of each group.



##### def backtest(self):

Backtest



##### def backtest_summary(self, decimals=3):

Backtest summary

**input :**

*decimals (int) :* decimals.















