# %% import package
import h5py
import pandas as pd
import numpy as np

# %% import data
# Monthly Return
month_return_df = pd.read_csv('G:\Data_Finance\SSE\月个股回报率文件1990-2021\TRD_Mnth.csv')

# %% Convert the csv to hdf5
month_return_hdf5 = pd.HDFStore('.\month_return.h5', mode='w', complevel=9, complib='blosc')
month_return_hdf5['month_return'] = month_return_df
month_return_hdf5.close()

# %% read the hdf5 file
mr = pd.read_hdf('.\month_return.h5', key='month_return')

#%%
import datetime as dt

temp_month_return_df = month_return_df
temp_month_return_df["Trdmnt"] = pd.to_datetime(temp_month_return_df["Trdmnt"]).apply(dt.datetime.timestamp)
temp_month_return_df["Capchgdt"] = pd.to_datetime(temp_month_return_df["Capchgdt"]).apply(dt.datetime.timestamp)

# %%
month_return_hdf5 = h5py.File("month_return.hdf5", mode="w")

# %%
mr_dset = month_return_hdf5.create_dataset("month_return", data=month_return_df.iloc[:, 0])



# %%
