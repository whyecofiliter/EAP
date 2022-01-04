'''
Test time_series_regress
'''
# %%
import sys,os
from numpy import testing

from statsmodels.base.model import Model
from statsmodels.tools.tools import add_constant
sys.path.append(os.path.abspath(".."))

from time_series_regress import TS_regress

# %%
def test_TS_regress() :
    import numpy as np

    X = np.random.normal(loc=0.0, scale=1.0, size=(2000,10))
    y_list = []
    for i in range(10) :
        b = np.random.uniform(low=0.1, high=1.1, size=(10,1))
        e = np.random.normal(loc=0.0, scale=1.0, size=(2000,1))
        y = X.dot(b) + e 
        y_list.append(y)

    re = TS_regress(y_list, X)
    re.fit()
    re.summary()
    print(re.grs())

test_TS_regress()

# %%
