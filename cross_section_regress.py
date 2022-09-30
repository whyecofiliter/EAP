'''
Cross Section Regression 截面数据回归
'''

class CS_regress() :
    def __init__(self, list_y, factor) :
        '''
        This function initializes the class.
        input :
            y_list (list/DataFrame): The assets return series list.
            factor (ndarray/DataFrame):* The factor risk premium series.
        '''
        import numpy as np

        if type(list_y).__name__ == 'DataFrame':
            self._name_y = list(list_y.columns)
            self.list_y = [np.array(list_y.iloc[:, i]) for i in range(len(list_y.columns))]
        elif type(list_y).__name__ == 'list':
            self._name_y = None
            self.list_y = list_y
        else:
            raise IOError
        
        if type(factor).__name__ == 'DataFrame':
            self._name_factor = list(factor.columns)
            self.factor = np.array(factor)
        elif type(factor).__name__ == 'ndarray':
            self._name_factor = None
            self.factor = factor
        else:
            raise IOError
    
    def ts_regress(self) :
        '''
        This function conducts the time series regression.
        output :
        beta(ndarray[N, c]) : The betas for each asset.
        err_mat (ndarray): The error matrix.
        '''
        from statsmodels.api import OLS
        from statsmodels.api import add_constant
        import numpy as np

        length = len(self.list_y)
        r, c = np.shape(self.factor)
        beta = np.zeros((length, c))
        e = np.zeros((r, length))

        for i in range(length) :
            result = OLS(self.list_y[i], add_constant(self.factor)).fit()
            params = result.params[1:]
            beta[i, :] = params
            e[:, i] = result.resid
        
        err_mat = np.cov(e.T)

        return beta, err_mat
    
    def ave(self) :
        '''
        This function conducts the average operation on the assets returns series list.
        output :
            np.mean(self.y_list, axis=1) : The average of the assets returns series list.
        '''
        import numpy as np

        return np.mean(self.list_y, axis=1)

    def cs_regress(self, beta, err_mat, constant=True, gls=True, **kwargs) :
        '''
        Cross Section Regression
        This function takes the cross-sectional regression.
        input :
            beta (ndarray): The betas from the output of the function ts_regress().
            err_mat (ndarray) : The error matrix from the output of the function ts_regress().
            constant (boolean): add constant or not. The default is add constant.
            gls (boolean): GLS regression or OLS regression. The default is GLS regression.

        output :
        params (array): The params of the cross regression model.
        resid (array): The residue of the cross regression model.

        Example:
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
        '''
        import numpy as np
        from statsmodels.api import OLS, GLS
        from statsmodels.api import add_constant

        y_mean = self.ave()
        if constant == True and gls == False :
            result = OLS(y_mean, add_constant(beta)).fit()
            
            return result.params[1:], result.resid, result.rsquared, result.rsquared_adj
        elif constant == False and gls == False :
            result = OLS(y_mean, beta).fit()
            
            return result.params, result.resid, result.rsquared, result.rsquared_adj
        elif constant == True and gls == True :
            result = GLS(y_mean, add_constant(beta), err_mat).fit()
            
            return result.params[1:], result.resid, result.rsquared, result.rsquared_adj
        elif constant == False and gls == True :
            result = GLS(y_mean, beta, err_mat).fit()         

            return result.params, result.resid, result.rsquared, result.rsquared_adj

    def cov_mat(self, beta, err_mat, shanken=True, constant=True, gls=True, **kwargs) :
        '''
        The covariance matrix
        This function calculates the covariance matrix of the cross regression model. 
        input :
            beta (ndarray): The betas from the output of the function ts_regress().
            err_mat (ndarray): The error matrix from the output of the function ts_regress().
            shanken (boolean): Take the shanken adjustment or not. The default is True.
            constant (boolean): add constant or not.
            gls (boolean): GLS regression or OLS regression. The default is GLS regression.

        output :
            param_cov_mat (ndarray): The covariance matrix of the parameters.
            resid_cov_mat (ndarray): The covariance matrix of the residue.
        '''
        import numpy as np
        from numpy.linalg import inv
        from statsmodels.api import add_constant

        length = len(self.list_y)
        r, c = np.shape(self.factor)
        I = np.identity(length)
        fac_mat = np.cov(self.factor.T)
        if shanken == True and gls == False :
            params = np.expand_dims(kwargs['params'], axis=0)
            param_cov_mat = 1/r*(inv(beta.T.dot(beta)).dot(beta.T).dot(err_mat).\
                            dot(beta).dot(inv(beta.T.dot(beta)))*(1+params.\
                            dot(inv(fac_mat)).dot(params.T))+fac_mat)
            resid_cov_mat = 1/r*(I-beta.dot(inv(beta.T.dot(beta))).dot(beta.T)).\
                            dot(err_mat).dot((I-beta.dot(inv(beta.T.dot(beta))).\
                            dot(beta.T)).T)*(1+params.dot(inv(fac_mat)).dot(params.T))
        elif shanken == False and gls == False :
            param_cov_mat = 1/r*(inv(beta.T.dot(beta)).dot(beta.T).dot(err_mat).\
                            dot(beta).dot(inv(beta.T.dot(beta)))+fac_mat)
            resid_cov_mat = 1/r*(I-beta.dot(inv(beta.T.dot(beta))).dot(beta.T)).\
                            dot(err_mat).dot((I-beta.dot(inv(beta.T.dot(beta))).dot(beta.T)).T)
        elif shanken == True and gls == True :
            params = np.expand_dims(kwargs['params'], axis=0)
            param_cov_mat = 1/r*(inv(beta.T.dot(inv(err_mat)).dot(beta))*(1+params.\
                            dot(inv(fac_mat)).dot(params.T))+fac_mat)
            resid_cov_mat = 1/r*(err_mat-beta.dot(inv(beta.T.dot(inv(err_mat))\
                            .dot(beta)).dot(beta.T)))*(1+params.dot(inv(fac_mat)).dot(params.T))
        elif shanken == False and gls == True :
            param_cov_mat = 1/r*(inv(beta.T.dot(inv(err_mat)).dot(beta))+fac_mat)
            resid_cov_mat = 1/r*(err_mat-beta.dot(inv(beta.T.dot(inv(err_mat)).dot(beta))).dot(beta.T))

        return param_cov_mat, resid_cov_mat
    
    def t_test(self, param_cov_mat, params) :
        '''
        T test for parameters
        This function takes t-test for parameters.
        input :
            param_cov_mat (ndarray): The covariance matrix of the parameters from the function cov_mat.
            params (ndarray): The parameters from the function cs_regress.

        output :
            t_value (ndarray): The t-value for statistical inference.
            p_value (ndarray): The p-value for statistical inference.
        '''
        import numpy as np
        from numpy.linalg import inv
        from scipy.stats import t
        
        r, c = np.shape(self.factor)
        params_std = np.expand_dims(np.diagonal(param_cov_mat)**0.5, axis=1)
        params = np.expand_dims(params, axis=1)
        t_value = params / params_std
        p_value = np.zeros((len(t_value),1))
        for i in range(len(p_value)):
            if t_value[i] > 0 :
                p_value[i] = 2*(1-t.cdf(t_value[i], r-1))
            elif t_value[i] <= 0 :
                p_value[i] = 2*t.cdf(t_value[i], r-1)
        return t_value, p_value

    def union_test(self, resid_cov_mat, resid) :
        '''
        Union test
        This function takes union test for parameters.
        input :
            resid_cov_mat (ndarray): The covariance matrix of the residue. 
            resid (ndarray): The residue from the function cs_regress.

        output :
            chi_square (list): The chi-square statistics.
            p_value (list): The p-value corresponding to the chi-square.
        '''
        import numpy as np
        from numpy.linalg import inv
        from scipy.stats import chi2
        
        length = len(self.list_y)
        r, c = np.shape(self.factor)
        N = length
        K = c
        alpha = np.expand_dims(resid, axis=1)
        chi_square = alpha.T.dot(inv(resid_cov_mat)).dot(alpha)
        p_value = 1-chi2.cdf(chi_square, N-K)

        return chi_square, p_value

    def fit(self,**kwargs) :
        '''
        Fit model
        This function runs the cross-sectional regression and takes the statistical inference.
        '''
        import numpy as np

        beta, err_mat = self.ts_regress()
        self.params, resid, self.rsquare, self.rsquare_adj = self.cs_regress(beta, err_mat,**kwargs)
        param_cov_mat, resid_cov_mat = self.cov_mat(beta, err_mat, **kwargs, params=self.params)
        self.params_t_value, self.params_p_value = self.t_test(param_cov_mat, self.params)
        self.alpha_test = self.union_test(resid_cov_mat, resid)
    
    def summary(self) :
        '''
        Summary

        Example:
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
        '''
        import numpy as np
        from prettytable import PrettyTable

        
        print("\n-------------------------- Risk Premium ------------------------------\n")
        table1 = PrettyTable()
        if self._name_factor != None:
            table1.add_column('variables', self._name_factor)
        table1.add_column('params', self.params)
        table1.add_column('t_value', self.params_t_value)
        table1.add_column('p_value', self.params_p_value)
        table1.add_column('R2', [self.rsquare] + ['' for i in range(len(self.params)-1)])
        table1.add_column('Adj R2', [self.rsquare_adj] + ['' for i in range(len(self.params)-1)])
        print(table1)
        print("\n----------------------------------------------------------------------\n")

        print("\n----------------------------- Alpha test -----------------------------\n")
        table2 = PrettyTable()
        table2.field_names = ["chi-square","p_value"]
        table2.add_row(self.alpha_test)
        print(table2)
        print("\n----------------------------------------------------------------------\n")



