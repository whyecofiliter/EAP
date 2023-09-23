# %% Set system path
import sys,os
sys.path.append(os.path.abspath(".."))

# %%
def test_plot():
    from util import plot
    import numpy as np
    from portfolio_analysis import Univariate as uni
    from portfolio_analysis import ptf_analysis as ptfa
    from util import plot
    
    # generate time 
    year=np.ones((3000,1),dtype=int)*2020
    for i in range(19):
        year=np.append(year,(2019-i)*np.ones((3000,1),dtype=int))
    
    # generate variables
    variable_1 = np.random.normal(0, 1, 20*3000)
    variable_2 = np.random.normal(0, 1, 20*3000)
    # generate character
    character=np.random.normal(0,1,20*3000)
    # generate future return
    ret=character*-0.5+np.random.normal(0,1,20*3000)
    # create sample containing future return, character, time
    sample=np.array([ret,character,year]).T

    # generate the Univiriate Class
    exper=uni(sample)
    exper.fit(4)

    plot(exper, alpha=False, pattern='scatter')

test_plot()


# %%
def test_plot_bi():
    import numpy as np
    from portfolio_analysis import Bivariate as bi
    from portfolio_analysis import ptf_analysis as ptfa
    from util import plot
    
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
    # generate the Univiriate Class
    exper=bi(sample)
    exper.fit(number=3)

    print(exper.average_result)
    print(np.shape(exper.ttest))
    plot(exper, pattern='scatter')

test_plot_bi()



# %%
