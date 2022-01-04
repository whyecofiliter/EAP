'''
This module is designed following the paper
"Autoencoder asset pricing models" by S. Gu, B. Kelly, D. Xiu (Journal of Econometrics, 2021, 222: 429-450.)

The factor model structure is defined as 
                r(i, t) = beta[z(i, t-1)] * f(t) + u(i, t)
where the relation of the beta and the z(~) and the relation of the factor(f) and the returns(individual/portfolio)
are modelled by autoencoder

The required packages: tensorflow, numpy, pandas 
'''

# %% packages
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras.backend import dot

# %% Construct the autoencoder for the relation of the beta and the z(~)
'''
to construct a mapping from N*P to N*K by second dimension P to K is 
to construct N autoencoders for each (P, K) pair 
'''
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class beta_character_autoencoder(Model):
    def __init__(self, K):
        super(beta_character_autoencoder, self).__init__()
        self.d1 = Dense(128, activation='relu')
        self.output_layer = Dense(K, activation = 'relu')
    
    def call(self, input):
        x = self.d1(input)
        x = self.output_layer(x)
        return x

class aggregate_beta_character_autoencder(Model):
    def __init__(self):
        super(aggregate_beta_character_autoencder, self).__init__()
    
    def call(self, N_input, K):
        import numpy as np
        
        row, col =  np.shape(N_input)
        output = np.zeros((row, K))
        for i in range(row):
            output[i, :] = beta_character_autoencoder(K).call(N_input[i, :])
        return output

# %% Construct the autoencoder for the relation of the factors and the returns
class factor_return_autoencoder(Model):
    def __init__(self, K):
        super(factor_return_autoencoder, self).__init__()
        self.output_layer = Dense(K, activation = 'relu')

    def call(self, input):
        x = self.output_layer(input)
        return x

# %% Comnbine the two autoencoder to get return
class combined_model(Model):
    def __init__(self):
        super(combined_model, self).__init__()
    
    def call(self, N_character, factor, K):
        x = aggregate_beta_character_autoencder().call(N_character, K).dot(factor_return_autoencoder(K).call(factor))
        return x

# %% Simple model
class simple_model(Model): 
    def __init__(self):
        super(simple_model, self).__init__()
    
    def call(self, N_character, factor, K):
        from tensorflow.keras import layers

        x = layers.dot((beta_character_autoencoder(K).call(N_character),(factor_return_autoencoder(K).call(factor))), axes=0)
        return x

