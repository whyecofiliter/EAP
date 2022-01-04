# %% Set system path
import sys,os
sys.path.append(os.path.abspath(".."))
#%%
def test_ptf_analysis() :
    '''
    TEST ptf_analysis
    contruct samples:
        1. generate character 构建分类特征
        2. generate breakpoint given the character 给定特征生成间断点
        3. generate the future sample return 生成未来样本收益/数值
    '''
     
    from portfolio_analysis import ptf_analysis as ptfa
    import matplotlib.pyplot as plt
    
    import numpy as np
    
    # generate character
    character = np.random.normal(0,100,10000)
    # generate breakpoint
    breakpoint = ptfa().select_breakpoints(character=character, number=9)
    print('Generated breakpoint:', breakpoint)
    # compare with the true breakpoint
    for i in np.linspace(0, 100, 11):
        print('True breakpoint', i, '%:', np.percentile(character,i))
    
    # generate the groups     
    print('The Label of unique value:\n', np.sort(np.unique(ptfa().distribute(character, breakpoint))))
    # plot the histogram of the label 
    # each group have the same number of samples
    plt.hist(ptfa().distribute(character, breakpoint))
    label = ptfa().distribute(character, breakpoint)[:, 0]
    # print the label
    # print('Label:\n', ptfa().distribute(character, breakpoint))
    
    # generate the future sample return
    sample_return = character/100 + np.random.normal()
    ave_ret = ptfa().average(sample_return, label)
    # print the groups return
    print('average return for groups:\n', ave_ret)
        
test_ptf_analysis()

#%%
def test_univariate() :
    '''
    TEST UNIVARIATE
    construct sample:
        1. 20 Periods
        2. 3000 Observations for each Period
        3. Divided into 10 groups with 9 breakpoints
        4. Character negative with return following the return=character*-0.5+sigma where sigma~N(0,1)
    '''
    import numpy as np
    from portfolio_analysis import Univariate as uni
    
    # generate time 
    year=np.ones((3000,1),dtype=int)*2020
    for i in range(19):
        year=np.append(year,(2019-i)*np.ones((3000,1),dtype=int))
    
    # generate character
    character=np.random.normal(0,1,20*3000)
    # generate future return
    ret=character*-0.5+np.random.normal(0,1,20*3000)
    # create sample containing future return, character, time
    sample=np.array([ret,character,year]).T
    # generate the Univiriate Class
    exper=uni(sample,9)
    # 
    data=exper.average_by_time()
    print(data)
    exper.summary_and_test()
    exper.print_summary_by_time()
    exper.print_summary()
    
    # generate factor
    factor=np.random.normal(0,1.0,(20,1))
    exper=uni(sample,9,factor=factor,maxlag=12)
#    print(exper.summary_and_test())
    exper.print_summary()
    

test_univariate()
    



# %%
