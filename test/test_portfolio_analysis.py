# %% Set system path
import sys,os

sys.path.append(os.path.abspath(".."))
#%% test class ptf_analysis()
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

    # test function average for Bivariate
    character_1 = np.random.normal(0, 100, 10000)
    character_2 = np.random.normal(0, 100, 10000)
    # generate breakpoint
    breakpoint_1 = ptfa().select_breakpoints(character=character_1, number=9)
    breakpoint_2 = ptfa().select_breakpoints(character=character_2, number=9)
    # attach the label
    label_1 = ptfa().distribute(character_1, breakpoint_1)[:, 0]
    label_2 = ptfa().distribute(character_2, breakpoint_2)[:, 0]
    
    label = [label_1, label_2]
    print(label_1,'\n', label_2)
    # generate the future sample return
    sample_return = character_1/100 - character_2/100 + np.random.normal()
    ave_ret = ptfa().average(sample_return, label, cond='bi')
    print(ave_ret)

test_ptf_analysis()

#%% test class univariate()
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
    # test function average_by_time
    data=exper.average_by_time()
    print(data)
    # test function summary_and_test
    exper.summary_and_test()
    # test function print_summary_by_time
    exper.print_summary_by_time()
    # test function print_summary
    exper.print_summary()
    
    # generate factor
    factor=np.random.normal(0,1.0,(20,1))
    exper=uni(sample,9,factor=factor,maxlag=12)
    # print(exper.summary_and_test())
    exper.print_summary()
    

test_univariate()
    
# %% test class bivariate
def test_bivariate():
    '''
    TEST BIVARIATE
    construct sample:
        1. 20 Periods
        2. 3000 Observations for each Period
        3. Divided into 10 groups with 9 breakpoints
        4. first character negative with return following the return = character*-0.5 + sigma where sigma~N(0,1)
        5. second character positive with return following the return = character*0.5 + sigma where sigma~N(0,1) 
    '''

    import numpy as np
    from portfolio_analysis import Bivariate as bi
    
    # generate time 
    year = np.ones((3000,1), dtype=int)*2020
    for i in range(19):
        year = np.append(year, (2019-i)*np.ones((3000,1), dtype=int))
    
    # generate character
    character_1 = np.random.normal(0, 1, 20*3000)
    character_2 = np.random.normal(0, 1, 20*3000)

    # generate future return
    ret=character_1*-0.5 + character_2*0.5 + np.random.normal(0,1,20*3000)
    # create sample containing future return, character, time
    sample=np.array([ret,character_1, character_2, year]).T
    print(sample)
    # generate the Univiriate Class
    exper=bi(sample,9)
    # test function divide_by_time
    group_by_time = exper.divide_by_time()
    print('group by time: \n', group_by_time)
    # test function average_by_time
    average_group_time = exper.average_by_time()
    print('average_group_time: \n', average_group_time)
    print('shape of average_group_time: \n', np.shape(average_group_time))
    # test function difference
    result = exper.difference(average_group_time)
    print('result :\n', result)
    print('difference matrix :\n', np.shape(result))
    # test function summary_and_test
    average, ttest = exper.summary_and_test()
    print('average :\n', average)
    print(' shape of average :', np.shape(average))
    print('ttest :\n', ttest)
    print('shape of ttest :', np.shape(ttest))
    # test function print_summary_by_time()
    exper.print_summary_by_time()
    # test function print_summary
    exper.print_summary()
    
    # generate factor
    factor=np.random.normal(0,1.0,(20,1))
    exper=bi(sample,9,factor=factor,maxlag=12)
    exper.fit()
    exper.print_summary(export=True)
  
test_bivariate()

# %%
