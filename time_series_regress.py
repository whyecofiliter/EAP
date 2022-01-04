'''
Time Series Analysis 时间序列分析

'''

from numpy.core.defchararray import add
from statsmodels.multivariate import factor


class TS_regress() :
    def __init__(self, list_y, factor) :
        self.list_y = list_y
        self.factor = factor

    def ts_regress(self, newey_west=True) :
        import numpy as np
        from statsmodels.api import OLS
        from statsmodels.api import add_constant
        from adjust import newey_west_t

        length = len(self.list_y)
        r, c = np.shape(self.factor)
        self.alpha = []
        self.e_mat = np.zeros((r, length))
        self.t_value = np.zeros((length, c+1))
        self.p_value = np.zeros((length, c+1))

        for i in range(length) :
            result = OLS(self.list_y[i], add_constant(self.factor)).fit()
            params = result.params
            residue = result.resid
            self.alpha.append(params[0])
            self.e_mat[:, i] = residue
            if newey_west == True :
                self.t_value[i, :], self.p_value[i, :] = newey_west_t(self.list_y[i], add_constant(self.factor), params=result.params)
            elif newey_west == False :
                self.t_value[i, :], self.p_value[i, :] = result.tvalues, result.pvalues
        
        self.alpha = np.array(self.alpha)

        return self.alpha, self.e_mat

    def fit(self, **kwargs) :
        '''
        Fit model 拟合模型
        '''

        self.ts_regress(**kwargs)
    
    def summary(self) : 
        '''
        Summary 总结
        '''
        grs_stats, grs_p = self.grs()
        print("----------------------------------- GRS Test --------------------------------\n")
        print("GRS Statistics: \n", grs_stats)
        print("GRS p_value: \n", grs_p)
        print("-----------------------------------------------------------------------------\n")

    def grs(self) :
        '''
        GRS test GRS检验
        '''
        import numpy as np
        from numpy.linalg import inv
        from scipy.stats import f

        length = len(self.list_y)
        r, c = np.shape(self.factor)
        factor_mean = np.expand_dims(np.mean(self.factor, axis=0), axis=0)
        
        temp_factor = self.factor - factor_mean
        sigma_factor_mat = 1/r*temp_factor.T.dot(temp_factor)
        error_mat = 1/r*self.e_mat.T.dot(self.e_mat)

        grs_stats = (r-length-c)/(length*(1+factor_mean.dot(sigma_factor_mat).dot(factor_mean.T)))*self.alpha.dot(error_mat).dot(self.alpha.T)
        p_value = 1 - f.cdf(grs_stats, length, r-length-c)

        return grs_stats, p_value
