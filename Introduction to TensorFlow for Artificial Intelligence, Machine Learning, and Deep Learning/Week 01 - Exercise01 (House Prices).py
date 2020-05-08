#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Programming Assignment: Exercise 1 (Housing Prices)

# I have to build a neural network to predict the price of a house.
# The rule is as follows:
# a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.
# We predict the price for a house with 7 bedrooms.
# The output should be scaled down, it should predict the number 4 for 400, and then your answer is in the 'hundreds of thousands'


# In[4]:


import tensorflow as tf
import numpy as np # for input the sets as arrays
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([1,   2,   3,   4,   5], dtype = int)
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype = float)
model.fit(xs, ys, epochs = 500)
print(model.predict([7]))
print('hundreds of thousands')


# In[ ]:




