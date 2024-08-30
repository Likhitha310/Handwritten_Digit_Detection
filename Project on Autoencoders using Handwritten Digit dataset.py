#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model


# # Import the data

# In[2]:


(X_train,_),(X_test, _) = mnist.load_data()


# In[3]:


X_train[0][5]


# In[4]:


X_train.shape


# In[5]:


X_test.shape


# In[6]:


X_train = X_train/255.0
X_test = X_test/255.0


# In[7]:


X_train[0][5]


# In[8]:


X_train = X_train.reshape(len(X_train),28*28)
X_test = X_test.reshape(len(X_test),28*28)


# In[9]:


X_train.shape


# In[10]:


plt.figure(figsize=(10,5))
for i in range(10):
  ax = plt.subplot(1,10,i+1)
  plt.imshow(X_train[i].reshape(28,28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.suptitle('Train Data', fontsize=20)


# In[11]:


def show_visual(data, title, n=10, height=28, width=28):
  plt.figure(figsize=(10,5))
  for i in range(n):
    ax = plt.subplot(1,n,i+1)
    plt.imshow(data[i].reshape(height,width))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.suptitle(title, fontsize=20)


# In[12]:


show_visual(X_train,title='Train Data')
show_visual(X_test,title='Test Data')


# In[13]:


input_dim, output_dim = 784, 784
encode_dim = 100
hidden_dim = 256


# In[14]:


# Encoder
input_layer = Input(shape=input_dim, name="INPUT")
hidden_layer_1 = Dense(hidden_dim, activation='relu', name='HIDDEN_1')(input_layer)


# In[15]:


# Bottle Neck
bottle_neck = Dense(encode_dim, activation='relu', name='BOTTLE_NECK')(hidden_layer_1)


# In[16]:


# Decoder
hidden_layer_2 = Dense(hidden_dim, activation='relu', name='HIDDEN_2')(bottle_neck)
output_layer = Dense(output_dim, activation='sigmoid', name='OUTPUT')(hidden_layer_2)


# In[17]:


model = Model(input_layer, output_layer)


# In[18]:


model.compile(optimizer='adam', loss='binary_crossentropy')


# In[19]:


model.summary()


# In[20]:


model.fit(X_train,X_train, epochs=10)


# In[21]:


decoded_data = model.predict(X_test)


# In[22]:


get_encoded_data = Model(inputs=model.input,
                         outputs = model.get_layer('BOTTLE_NECK').output)


# In[23]:


encoded_data = get_encoded_data.predict(X_test)


# In[24]:


show_visual(X_test, title="Actual Data")
show_visual(encoded_data, title="Encoded Data", height=10, width=10)
show_visual(decoded_data, title="Decoded Data")

