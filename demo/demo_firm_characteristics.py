'''
Firm Characteristics China
From Green 2017
'''
# %% set system path
import sys,os
sys.path.append(os.path.abspath(".."))

# %% data
import pandas as pd

month_return = pd.read_hdf('.\data\month_return.h5', key='month_return')
company_data = pd.read_hdf('.\data\last_filter_pe.h5', key='data')
trade_data = pd.read_hdf('.\data\mean_filter_trade.h5', key='data')
beta = pd.read_hdf('.\\data\\beta.h5', key='data')
cash = pd.read_hdf('.\\data\\cash.h5', key='data')
debt = pd.read_hdf('.\\data\\debt.h5', key='data')
eq_offer = pd.read_hdf('.\\data\\eq_offer.h5', key='data')
operation = pd.read_hdf('.\\data\\operation.h5', key='data')
profit = pd.read_hdf('.\\data\\profit.h5', key='data')

# %% construct characteristics
# Acronym : absacc 
# Author(s) : Bandyopadhyay, Huang, and, Wirjanto 
# Date,Journal : 2010, WP
# Definition of the characteristic-based anomaly variable : Absolute value of acc

# Acronym : acc 
# Author(s) : Sloan 
# Date,Journal : 1996, TAR
# Definition of the characteristic-based anomaly variable : 
# Annual income before extraordinary items
# (ib) minus operating cash flows (oancf )
# divided by average total assets (at); if oancf
# is missing then set to change in act -
# change in che - change in lct + change in
# dlc + change in txp-dp
delta_ca = debt['F010101A'] - debt['F010401A']# delta of curent asset





