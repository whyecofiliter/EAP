'''
Test fama_macbeth regression module
'''
#%%
import sys,os
sys.path.append(os.path.abspath(".."))

# %% test class fama macbeth regress 
def test_fama_macbeth_regress() :
    '''
    TEST fama_macbeth_regress
    construct sample:
        1. 20 Periods
        2. 3000 Observations for each Period
        4. Character negative with return following the return=character*-0.5+sigma where sigma~N(0,1)
    '''
    import numpy as np
    from fama_macbeth import Fama_macbeth_regress
    # construct sample
    year=np.ones((3000,1),dtype=int)*2020
    for i in range(19):
        year=np.append(year,(2019-i)*np.ones((3000,1),dtype=int))
    character=np.random.normal(0,1,(2,20*3000))
#    print('Character:',character)
    ret=np.array([-0.5,-1]).dot(character)+np.random.normal(0,1,20*3000)
    sample=np.array([ret,character[0],character[1],year]).T    
#    print('Sample:',sample)
#    print(sample.shape)

    model = Fama_macbeth_regress(sample)
    result = model.fit(add_constant=False)
    print(result)
    model.summary_by_time()
    model.summary()

test_fama_macbeth_regress()    

# %% test_factor_mimicking_portfolio
def test_factor_mimicking_portfolio() :
    '''
    TEST Factor_mimicking_portfolio
    construct sample:
        1. 20 Periods
        2. 3000 Observations for each Period
        3. Character negative with return following the return=character*-0.5+sigma where sigma~N(0,1)
    '''
    import numpy as np
    from fama_macbeth import Factor_mimicking_portfolio
    
    # construct sample
    year = np.ones((3000, 1), dtype=int) * 2020
    for i in range(19):
        year = np.append(year, (2019-i) * np.ones((3000, 1), dtype=int))
    character = np.random.normal(0, 1, (2, 20*3000))
    weight = np.random.uniform(0, 1, (1, 20*3000))
#    print('Character:',character)
    ret = np.array([-0.5, -1]).dot(character) + np.random.normal(0, 1, 20*3000)
    sample = np.array([ret, character[0], character[1], year, weight[0]]).T    
#    print('Sample:',sample)
#    print(sample.shape)

    model = Factor_mimicking_portfolio(sample)
    portfolio_return_time = model.portfolio_return_time()
    print('portfolio_return_time: \n', portfolio_return_time)
    print('portfolio_return_time: \n', np.shape(portfolio_return_time))
    
    portfolio_return = model.portfolio_return()
    print('portfolio_return_row: \n', portfolio_return[0])
    print('portfolio_return_row:', np.shape(portfolio_return[0]))
    print('portfolio_return_col: \n', portfolio_return[1])
    print('portfolio_return_col:', np.shape(portfolio_return[1]))

test_factor_mimicking_portfolio()

# %%
