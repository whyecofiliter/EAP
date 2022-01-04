'''
Test cross_section_regress 测试截面回归
'''

# %% Set path
import sys,os

from statsmodels.regression.linear_model import GLS
sys.path.append(os.path.abspath(".."))

from cross_section_regress import CS_regress

# %% test_CS_regress
def test_CS_regress() :
    '''
    test CS_regress module 
    '''
    import numpy as np
    
    X = np.random.normal(loc=0, scale=0.1, size=(2000,3))
    y_list = []
    for i in range(100) :
        b = np.random.uniform(low=-1, high=1, size=(3,1))
        e = np.random.normal(loc=0.0, scale=0.5, size=(2000,1))
        alpha = np.random.normal(loc=0.0, scale=0.5)
        y = X.dot(b)  + e 
        y_list.append(y)

    print(np.mean(X, axis= 0))
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

test_CS_regress()

# %%
