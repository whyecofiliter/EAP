'''
Time Frenquency Pricing
'''

class wavelet():
    def __init__(self, series):
        import numpy as np

        self.series = np.reshape(series, (len(series)))

    def _decompose(self, wave, mode, level=None):
        '''
        input : 
            series : time series
            wave : wavelet family
            level : wave level 
            mode : single or multi scales

        output : 

        '''
        
        import pywt

        if mode == 'multi':
            wave_dec = pywt.wavedec(self.series, wavelet=wave, level=level)

        elif mode == 'single':
            wave_dec = pywt.dwt(self.series, wavelet=wave)

        self.mode = mode
        self.wave = wave
        self._level = level
        self.wave_dec = wave_dec

        return wave_dec
    
    def _pick_details(self):
        '''
        '''
        import numpy as np

        pick_series = list()       
        for i in range(self._level+1):
            temp_dec = list()
            for j in range(self._level+1):
                if i == j :
                    temp_dec.append(self.wave_dec[j])
                elif i != j :
                    temp_dec.append(np.zeros_like(self.wave_dec[j]))

            pick_series.append(temp_dec)
        
        return pick_series

    def _rebuild(self, mode='constant'):
        '''
        input :
            series_dec : decomposed time series
            wave : wavelet family
        output :
            wave_rec (list) : recomposed time series
        '''
        import pywt
        import numpy as np

        if self.mode == 'multi':
            pick_series = self._pick_details()
            wave_rec = list()
            for i in range(len(pick_series)):
                wave_rec.append(pywt.waverec(pick_series[i], self.wave, mode=mode))

            return wave_rec

    def fit(self, wave, mode, level, output=False):
        self._decompose(wave, mode, level)
        if output == True:
            return self._rebuild()
        elif output == False:
            self._rebuild()

'''
Wavelet pricing
'''
class wavelet_pricing():
    def __init__(self, rets, factors):
        import numpy as np

        if type(rets).__name__ == 'ndarray':
            self.rets = rets
        elif type(rets).__name__ == 'Series' or type(rets).__name__ == 'DataFrame':
            self.rets = np.array(rets)

        if type(factors).__name__ == 'ndarray':
            self.factors = factors
        elif type(factors).__name__ == 'Series' or type(factors).__name__ == 'DataFrame':
            self.factors = np.array(factors) 

    def wavelet_dec_rec(self, **kwargs):
        import numpy as np
        
        try:
            r, c = np.shape(self.factors)
        except:
            r = len(self.factors)
            c = 1

        rets_dec = wavelet(self.rets)
        if c > 1:
            factors_dec_list = [wavelet(self.factors[:, i]) for i in range(c)]
        elif c == 1:
            factors_dec_list = [wavelet(self.factors)]

        rets_dec_s = rets_dec.fit(wave=kwargs['wave'], mode=kwargs['mode'], level=kwargs['level'], output=True)
        factors_dec_s = list()
        for i in range(c):
            factors_dec_s.append(factors_dec_list[i].fit(wave=kwargs['wave'], mode=kwargs['mode'], level=kwargs['level'], output=True))
        
        self._rets_dec_s = rets_dec_s
        self._factors_dec_s = factors_dec_s
        self._level = kwargs['level']
        self._factor_num = c
        self._series_len = r
        return rets_dec_s, factors_dec_s

    def wavelet_regression(self, **kwargs):
        import statsmodels.api as sm
        from statsmodels.regression.rolling import RollingOLS
        import numpy as np
        
        regress = list()
        if kwargs['win'] == None:
            for i in range(self._level+1):
                temp_rets_s = self._rets_dec_s[i]
                temp_factor_s = np.zeros((self._series_len, self._factor_num))
            
                for j in range(self._factor_num):
                    temp_factor_s[:, j] = self._factors_dec_s[j][i]
            
                if kwargs['constant'] == True:
                    if kwargs['robust'] == False:
                        regress.append(sm.OLS(temp_rets_s, sm.add_constant(temp_factor_s)).fit())
                    elif kwargs['robust'] == True:
                        regress.append(sm.OLS(temp_rets_s, sm.add_constant(temp_factor_s)).fit(cov_type='HC0'))
                    self._constant = kwargs['constant']
                elif kwargs['constant'] == False:
                    if kwargs['robust'] == False:
                        regress.append(sm.OLS(temp_rets_s, temp_factor_s).fit())
                    elif kwargs['robust'] == True:
                        regress.append(sm.OLS(temp_rets_s, temp_factor_s).fit(cov_type='HC0'))
                    self._constant = kwargs['constant']

            return regress
        
        elif kwargs['win'] != None:
            for i in range(self._level+1):
                temp_rets_s = self._rets_dec_s[i]
                temp_factor_s = np.zeros((self._series_len, self._factor_num))
            
                for j in range(self._factor_num):
                    temp_factor_s[:, j] = self._factors_dec_s[j][i]
            
                if kwargs['constant'] == True:
                    if kwargs['robust'] == False:
                        regress.append(RollingOLS(temp_rets_s, sm.add_constant(temp_factor_s), window=kwargs['win']).fit(method='pinv', params_only=kwargs['params_only']))
                    elif kwargs['robust'] == True:
                        regress.append(RollingOLS(temp_rets_s, sm.add_constant(temp_factor_s), window=kwargs['win']).fit(method='pinv', cov_type='HC0', params_only=kwargs['params_only']))                        
                    self._constant = kwargs['constant']
                elif kwargs['constant'] == False:
                    if kwargs['robust'] == False:
                        regress.append(RollingOLS(temp_rets_s, temp_factor_s, window=kwargs['win']).fit(method='pinv', params_only=kwargs['params_only']))
                    elif kwargs['robust'] == True:
                        regress.append(RollingOLS(temp_rets_s, temp_factor_s, window=kwargs['win']).fit(method='pinv', cov_type='HC0', params_only=kwargs['params_only']))
                    self._constant = kwargs['constant']

            return regress
              
    def fit(self, wave, mode, level, win=None, robust=False, constant=True, params_only=True):
        '''
        This function is designed for fitting the model.
        input :
        wave (str) : The chosen wavelet family, which is the same in pywt.
        mode (str) : choose multiscale or single scale. 'multi' denotes multiscale. 'single' denotes single scale.
        level (int) : the level of multiscale. If 'multi' is chosen, it must be set.
        win (int): The rolling window if rolling regression is used.
        robust (boolean): whether use the robust covariance matrix.
        constant (boolean): whether includes the constant in regression. The DEFAULT is True.
        '''

        self.wavelet_dec_rec(wave=wave , mode=mode, level=level)
        self.result = self.wavelet_regression(win=win, robust=robust, constant=constant, params_only=params_only)

        return self.result

    def summary(self, export=False):
        '''
        Summarize the result
        '''
        import numpy as np
        from prettytable import PrettyTable

        result = self.result
        coef = np.zeros((self._factor_num+1, self._level+1))
        t_value = np.zeros((self._factor_num+1, self._level+1))
        r_square = list()

        for j in range(len(result)):
            coef[:, j] = np.around(result[j].params, decimals=3)
            t_value[:, j] = np.around(result[j].tvalues, decimals=3)
            r_square.append(np.around(result[j].rsquared, decimals=3))
        
        # print table
        table = PrettyTable()
        if self._constant == True:
            table.field_names = ['scale', 'alpha'] + [str(i+1) for i in range(self._factor_num)] + ['R2']
        elif self._constant == False:
            table.field_names = ['scale'] + [str(i+1) for i in range(self._factor_num)]
        
        for j in range(len(result)):
            if j != len(result)-1:
                table.add_row(['scale' + str(j+1)] + list(coef[:, -(j+1)]) + [r_square[-(j+1)]])
                table.add_row([' '] + list(t_value[:, -(j+1)]) + [' '])
            elif j == len(result)-1:
                table.add_row(['residue'] + list(coef[:, -(j+1)]) + [r_square[-(j+1)]])
                table.add_row([' '] + list(t_value[:, -(j+1)]) + [' '])

        print(table)

        if export == True:
            import pandas as pd
            try:
                from StringIO import StringIO
            except ImportError:
                from io import StringIO
            
            csv_string = table.get_csv_string()
            with StringIO(csv_string) as f:
                df = pd.read_csv(f)
            
            return df