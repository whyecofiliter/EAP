'''
Test fama_macbeth regression module
'''
#%%
import sys,os
sys.path.append(os.path.abspath(".."))

from fama_macbeth import Fama_macbeth_regress
# %%

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
    
    
    
# %%
