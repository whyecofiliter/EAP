'''
Adjustment Method 调整方法
'''
'''
OLS regrssion
'''
from numpy.core.fromnumeric import diagonal
from statsmodels.stats.sandwich_covariance import cov_hac_simple
from statsmodels.tools.tools import add_constant


def ols(y, x, constant=True) :
    from statsmodels.api import OLS, add_constant
    
    if constant == True :
        result = OLS(y,add_constant(x)).fit()
    elif constant == False :
        result = OLS(y,x).fit()
    
    return result

'''
White Estimate of Variance 
'''
def white(y, X, **kwargs) :
    '''
    White Estimate of Variance
    X(r,c): r is smaple number, c is variable number
    S0 is coviarance matrix of residue
    The Variance V is estiamted as V_ols=T(X'X)^(-1)S0(X'X)^(-1)
    The white estimation of S0 is S0=1/T*X'(X*resid_sqr)
    x_i=[xt1,xt2,...,xtk+1]'
    '''
    import numpy as np
    from numpy.linalg import inv
    from statsmodels.api import add_constant

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
def newey_west(y, X, J=None, **kwargs) :
    '''
    Newey West Estimate of Variance
    X(r,c): r is smaple number, c is variable number
    S0 is coviarance matrix of residue
    The Variance V is estiamted as V_ols=T(X'X)^(-1)S0(X'X)^(-1)
    The white estimation of S0 is S0=1/T*X'(X*resid_sqr)+1/T*sum_j(sum_t(w_j*e_t*e_t-j(x_tx'_t-j+xt_-jx'_t)))
    x_i=[xt1,xt2,...,xtk+1]'
    '''

    import numpy as np
    from numpy.linalg import inv
    from statsmodels.api import add_constant

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
def white_t(y, X, params=None, side='Two', **kwargs) :
    '''
    White t test based on White Variance
    '''
    import numpy as np
    from scipy.stats import t

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
def newey_west_t(y, X, params=None, side='Two', **kwargs) :
    '''
    Newey_West t test based on Newey West Variancce 
    '''
    import numpy as np
    from scipy.stats import t

    diagnal = np.diagonal(newey_west(y, X, **kwargs))
    standard_error = diagnal**0.5
    if params:
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

        