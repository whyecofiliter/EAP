'''
Test time frequency
'''
# %%
import sys,os

sys.path.append(os.path.abspath(".."))

# %%
from time_frequency import wavelet 
import numpy as np

s = np.random.normal(loc=0.0, scale=1.0, size=(5001))

# %%
wl = wavelet(s)
wave_dec = wl._decompose('db2', mode='multi', level=6)

# %%
pick_series = wl._pick_details()

print(np.shape(pick_series))

# %%
wave_rec = wl._rebuild()

# %%
import matplotlib.pyplot as plt

for i in range(len(wave_rec)):
    plt.subplot(len(wave_rec), 1, i+1)
    plt.plot(wave_rec[i])

# %%
w2 = wavelet(s)
a = w2.fit('db2', mode='multi', level=6, output=True)
plt.plot(np.sum(a, axis=0))
plt.plot(s)

# %%
import numpy as np

factors = np.random.normal(loc=0.0, scale=1.0, size=(4000, 5))
e = np.random.normal(loc=0.0, scale=1.0, size=(4000, 1))
beta = np.random.normal(loc=0.0, scale=1.0, size=(5, 1))

r = factors.dot(beta) + e

# %
from time_frequency import wavelet_pricing as wlp

model = wlp(r, factors)
result = model.fit(wave='db2', mode='multi', level=6)
model.summary()

# %%
import pywt

s = np.random.normal(loc=0.0, scale=1.0, size=(5000))
plt.plot(s)

# %
coef = pywt.wavedec(s, wavelet='db2', level=6)

# % reconstruct
def pick_details(coef):
    '''
    input :
        wave_dec : 
    '''
    import numpy as np

    pick_series = list()    
    for i in range(len(coef)):
        temp_dec = list()
        for j in range(len(coef)):
            if i == j :
                temp_dec.append(coef[j])
            elif i != j :
                temp_dec.append(np.zeros_like(coef[j]))

        pick_series.append(temp_dec)
        
    return pick_series

s_pick = pick_details(coef)
s_rec = list()
for i in range(len(s_pick)):
    s_rec.append(pywt.waverec(s_pick[i], 'db2'))

# %%

s_rec_sum = np.sum(s_rec, axis=0)
plt.plot(s_rec_sum)

# %
s_rec_resid = s - s_rec_sum
plt.plot(s_rec_resid)