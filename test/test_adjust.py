'''
Test adjust module
'''
# %%
import sys,os
from numpy import testing

from statsmodels.base.model import Model
from statsmodels.tools.tools import add_constant
sys.path.append(os.path.abspath(".."))

from adjust import newey_west, newey_west_t, ols, white_t
from adjust import white

# %% Test OLS
def test_ols() :
    import numpy as np

    X = np.random.normal(loc=0.0, scale=1.0, size=(2000,10))
    b = np.random.uniform(low=0.1, high=1.1, size=(10,1))
    e = np.random.normal(loc=0.0, scale=1.0, size=(2000,1))
    y = X.dot(b) + e + 1.0

    re = ols(y, X, constant=True)
    print('\nTrue b : ', b)
    print('\nEstimated b : ', re.params)
    print('\nresidue : ', re.resid.shape)

test_ols()

# %% test_white
def test_white() :
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

test_white()

# %% test_newey_west
def test_newey_west() :
    import numpy as np
    from statsmodels.api import add_constant
    from statsmodels.stats.sandwich_covariance import cov_hac

    X = np.random.normal(loc=0.0, scale=1.0, size=(10000,10))
    b = np.random.normal(loc=0.0, scale=1.0, size=(10,1))
    e = np.random.normal(loc=0.0, scale=1.0, size=(10000,1))
    y = X.dot(b) + e

    re = newey_west(y, X, constant=False)
    np.set_printoptions(formatter={'float':'{:0.2f}'.format})
    print(re)
#    X = add_constant(X)
    r, c = np.shape(X) 
    result = ols(y, X, constant=False)
    print('\n', cov_hac(result))

    print('\n', 1/r*X.T.dot(X).dot(re).dot(X.T.dot(X)))
    print('\n', 1/r*X.T.dot(X).dot(cov_hac(result)).dot(X.T.dot(X)))

test_newey_west()

# %% test_white_t
def test_white_t() :
    import numpy as np
    from statsmodels.api import add_constant

    X = np.random.normal(loc=0.0, scale=1.0, size=(10000,10))
    b = np.random.uniform(low=-0.5, high=0.5, size=(10,1))
    e = np.random.normal(loc=0.0, scale=1.0, size=(10000,1))
    y = X.dot(b) + e + 1.0

    re = white_t(y, X, constant=False)
    np.set_printoptions(formatter={'float':'{:0.2f}'.format})
    print('t_value : ', re[0], '\np_value : ', re[1])
    print('b :', b.T)

test_white_t()

# %% test_newey_west_t
def test_newey_west_t() :
    import numpy as np
    from statsmodels.api import add_constant

    X = np.random.normal(loc=0.0, scale=1.0, size=(10000,10))
    b = np.random.uniform(low=-0.5, high=0.5, size=(10,1))
    e = np.random.normal(loc=0.0, scale=1.0, size=(10000,1))
    y = X.dot(b) + e + 1.0

    re = newey_west_t(y, X, constant=True)
    np.set_printoptions(formatter={'float':'{:0.2f}'.format})
    print('t_value : ', re[0], '\np_value : ', re[1])
    print('b :', b.T)

test_newey_west_t()
# %%


