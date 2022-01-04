'''
This module is designed for testing the module "autoencoder" 
'''
# %% Gernerate dataset
'''
The data generating process is 
                r(i, t) = beta[z(i, t-1)] * f(t) + u(i, t).
The expectation of factor risk premium f(t) is supposed to be larger than 0, 
and its volatility is supposed to be larger than that of z(~).
The characteristic of firm z(i, t-1) is supposed to be stationary and less volatile.
The expectation of the residue u(i, t) is zero, 
and its volatility is prportional to the length of the horizon.
The beta is the linear function of the z(i, t-1):
                beta[z(i, t-1)] = z(i, t-1)*c(i) + v(i, t-1)
Following the idea from the RFS, the distributions of parameters are
                f(t) ~ Gaussian(0.1, 0.40)
                z(i, t-1) ~ Uniform(-0.1, 0.1)
                u(i,t) ~ Gaussian(0.0, 0.10) 
                c(i) ~ Gaussian(1, 0.50)
                v(i, t-1) ~ Gaussian(0.0, 0.10)
The factor number is 10
The characteristic is 30
The data length is 12 months * 30 = 360 
The data size ranges from 500 ~ 6000, including 500, 1000, 3000, 4500, 6000.
The mapping table 
sample number  1     2     3     4     5
size           500   1000  3000  4500  6000
shape          *360  *360  *360  *360  *360
'''
import numpy as np

def data_generate(dsize, dshape=360):
    ret = np.zeros(shape=(dsize, dshape))
    c = np.random.normal(loc=1.0, scale=0.5, size=(30, 10))
    factors = np.zeros(shape=(dshape, 10, 1))
    characters = np.zeros(shape=(dshape, dsize, 30))
    for i in range(dshape):
        f = np.random.normal(loc=0.1, scale=0.4, size=(10, 1))
        z = np.random.uniform(low=-0.1, high=0.1, size=(dsize, 30))
        v = np.random.normal(loc=0.0, scale=0.1, size=(dsize, 10))
        u = np.random.normal(loc=0.0, scale=0.1, size=(1,1))

        beta = z.dot(c) + v
        r = beta.dot(f) + u

        factors[i] = f
        characters[i] = z
        ret[:, i] = np.reshape(r, len(r))
    
    return ret, factors, characters

sample1 = data_generate(500)
sample2 = data_generate(1000)
sample3 = data_generate(3000)
sample4 = data_generate(4500)
sample5 = data_generate(6000)

# %% Train complex model
import sys,os
sys.path.append(os.path.abspath(".."))
from autoencoder import combined_model
import tensorflow as tf

model = combined_model()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

@tf.function
def train_step(factors, characters, ret):
    with tf.GradientTape() as tape:
        predictions = model(N_character=characters, factor=factors, K = 10)
        loss = loss_object(ret, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(ret, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()

    for i in range(360):
        train_step(sample1[2][i], sample1[1][i], sample1[0][:, i])
    
    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {train_loss.result()}, '
        f'Test Accuracy: {train_accuracy.result() * 100}'
    )

# %% Train simple model
import sys,os
sys.path.append(os.path.abspath(".."))
from autoencoder import simple_model
import tensorflow as tf

model = simple_model()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

@tf.function
def train_step(factors, characters, ret):
    with tf.GradientTape() as tape:
        predictions = model(N_character=characters, factor=factors, K = 10)
        loss = loss_object(ret, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(ret, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()

    for i in range(360):
        train_step(sample1[2][i], sample1[1][i], sample1[0][:, i])
    
    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {train_loss.result()}, '
        f'Test Accuracy: {train_accuracy.result() * 100}'
    )


# %%
