'''
Adjustment Method 调整方法
'''
'''
OLS regrssion
'''

from numpy import ndarray

from statsmodels.api import OLS, add_constant
import numpy as np
from numpy.linalg import inv
from scipy.stats import t

def ols(y: ndarray, x:ndarray, constant: bool=True) :
    '''
    The function is for OLS regression, which is equal to the OLS module in package *statsmodels*. 
    input :
        y (ndarray): The dependent variable.
        x (ndarray): The explanatory variable.
        constant (boolean): add constant or not in OLS model. The default is *True*.

    output :
        result (OLSRegressResult): The result of the regression.
    '''
    
    if constant == True :
        result = OLS(y,add_constant(x)).fit()
    elif constant == False :
        result = OLS(y,x).fit()
    
    return result

'''
White Estimate of Variance 
'''
def white(y: ndarray, X:ndarray, **kwargs) ->ndarray:
    '''
    White Estimate of Variance
    X(r,c): r is smaple number, c is variable number
    S0 is coviarance matrix of residue
    The Variance V is estiamted as V_ols=T(X'X)^(-1)S0(X'X)^(-1)
    The white estimation of S0 is S0=1/T*X'(X*resid_sqr)
    x_i=[xt1,xt2,...,xtk+1]'

    input :
        y (ndarray): The dependent variable.
        X (ndarray): The explanatory variable.

    output :
        V_ols (ndarray): The white variance estimate

    Example:
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
    '''

    r, c = np.shape(X)
    result = ols(y, X, **kwargs)
    resid = result.resid
    resid_sqr = np.diag(resid**2)
    if kwargs['constant'] == True:
        X = add_constant(X)
    S0 = 1 / r * X.T.dot(resid_sqr.dot(X))
    temp = X.T.dot(X)
    V_ols = r*inv(temp).dot(S0).dot(inv(temp))
    
    return V_ols

'''
Newey_West Estimate of Variance
'''
def newey_west(y:ndarray, X:ndarray, J: int=None, **kwargs) -> ndarray:
    '''
    Newey West Estimate of Variance
    X(r,c): r is smaple number, c is variable number
    S0 is coviarance matrix of residue
    The Variance V is estiamted as V_ols=T(X'X)^(-1)S0(X'X)^(-1)
    The white estimation of S0 is S0=1/T*X'(X*resid_sqr)+1/T*sum_j(sum_t(w_j*e_t*e_t-j(x_tx'_t-j+xt_-jx'_t)))
    x_i=[xt1,xt2,...,xtk+1]'

    input :
        y (ndarray): The dependent variable.
        X (ndarray): The explanatory variable.
        J (int): The lag.

    output :
        V_ols (ndarray): The Newey-West variance estimate.
    
    Example:
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
    '''

    r, c = np.shape(X)
    result = ols(y, X, **kwargs)
    resid = result.resid
    
    if J != None :
        J = J
    elif J == None :
        J = int(np.floor(4*(r/100)**(2/9)))
    
    if kwargs['constant'] == True:
        X = add_constant(X)
    r_X, c_X = np.shape(X)
    temp_cor = np.zeros((c_X, c_X))
    X = np.matrix(X)
    for j in range(1, J+1) :
        # the weight w_j= 1- j/(+J), j start from 1
        # in python however the index starts from 0, and thus j+1 is used 
        w_j = 1 - j / (1 + J)
        for t in range(j+1, r+1) :
            temp_cor = temp_cor + (1 / r) * w_j * resid[t-1] * resid[t-j-1] * \
                       (X[t-1, :].T.dot(X[t-j-1, :]) + X[t-j-1,:].T.dot(X[t-1, :]))

    resid_sqr = np.diag(resid**2)

    temp_var =  1 / r * X.T.dot(resid_sqr.dot(X))    
    S0 = temp_var + temp_cor
    temp = X.T.dot(X)
    V_ols = r * inv(temp).dot(S0).dot(inv(temp))

    return V_ols

'''
T test after adjustment of White
'''
def white_t(y:ndarray, X:ndarray, params: ndarray=None, side: str='Two', **kwargs) :
    '''
    White t test based on White Variance
    This function constructs t-test based on White variance estimate.

    input :
        y (ndarray): The dependent variable.
        X (ndarray): The explanatory variable.
        params (ndarray): Already have parameters or not. The default is *None*.

    output :
        t_value (ndarray): The t-value of parameters.
        p_value (ndarray): The p-value of parameters.
    
    Example:
    import numpy as np

    X = np.random.normal(loc=0.0, scale=1.0, size=(10000,10)) 
    b = np.random.uniform(low=-0.5, high=0.5, size=(10,1))
    e = np.random.normal(loc=0.0, scale=1.0, size=(10000,1))
    y = X.dot(b) + e + 1.0

    re = white_t(y, X, constant=False)
    np.set_printoptions(formatter={'float':'{:0.2f}'.format})
    print('t_value : ', re[0], '\np_value : ', re[1])
    print('b :', b.T)
    '''

    diagonal = np.diagonal(white(y, X, **kwargs))
    standard_error = diagonal**0.5
    if params == None :
        result = ols(y, X, **kwargs)
        params = result.params
    t_value = params / standard_error
    r, c = np.shape(X)
    if kwargs['constant'] == True :
        freedom = r - c - 1
    else :
        freedom = r - c
    
    if side == 'One':
        p_value = 1- t.cdf(np.abs(t_value), freedom)
    elif side == 'Two':
        p_value = 2 * (1- t.cdf(np.abs(t_value), freedom))
    
    return t_value, p_value

'''
T test after adjustment of Newey West
'''
def newey_west_t(y:ndarray, X:ndarray, params: ndarray=None, side: str='Two', **kwargs) :
    '''
    Newey_West t test based on Newey West Variancce 
    This function constructs t-test based on Newey-West variance estimate.
    input :
        y (ndarray): The dependent variable.
        X (ndarray): The explanatory variable.
        params (ndarray): Already have parameters or not. The default is None.

    output :
        t_value (ndarray): The t-value of parameters.
        p_value (ndarray): The p-value of parameters.
    
    Example:
    import numpy as np

    X = np.random.normal(loc=0.0, scale=1.0, size=(10000,10))
    b = np.random.uniform(low=-0.5, high=0.5, size=(10,1))
    e = np.random.normal(loc=0.0, scale=1.0, size=(10000,1))
    y = X.dot(b) + e + 1.0

    re = newey_west_t(y, X, constant=True)
    np.set_printoptions(formatter={'float':'{:0.2f}'.format})
    print('t_value : ', re[0], '\np_value : ', re[1])
    print('b :', b.T)
    '''

    diagnal = np.diagonal(newey_west(y, X, **kwargs))
    standard_error = diagnal**0.5
    if params is not None:
        t_value = params / standard_error
    else:
        constant = kwargs['constant']
        result = ols(y, X, constant=constant)
        params = result.params
        t_value = params / standard_error
    
    r, c = np.shape(X)
    if kwargs['constant'] == True :
        freedom = r - c - 1
    else :
        freedom = r - c
    
    if side == 'One':
        p_value = 1- t.cdf(np.abs(t_value), freedom)
    elif side == 'Two':
        p_value = 2 * (1- t.cdf(np.abs(t_value), freedom))

    return t_value, p_value

        