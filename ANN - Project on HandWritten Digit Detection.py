#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing the dataset

# In[2]:


data = keras.datasets.mnist


# In[3]:


data


# In[4]:


(X_train, y_train),(X_test, y_test) = data.load_data()


# In[5]:


X_train.shape


# In[6]:


X_train[0][5]


# In[7]:


plt.matshow(X_train[0])


# In[8]:


for i in range(6):
  plt.matshow(X_train[i])


# In[9]:


X_test.shape


# In[10]:


X_train.ndim


# In[11]:


X_train[500].ndim


# In[12]:


# (5,28,28)
for i in range(5):
  print(1,28,28)


# In[13]:


y_train[:10]


# In[14]:


# Flattening the X_train & X_test
flat_X_train = X_train.reshape(len(X_train),28*28)
flat_X_test = X_test.reshape(len(X_test),28*28)


# In[15]:


flat_X_train.shape


# In[16]:


flat_X_train[0].ndim


# In[17]:


flat_X_train[0]


# In[18]:


X_train[0]


# * Units (Neuron): https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc
# 
# * Activation: https://keras.io/api/layers/activations/
# 
# * Compile: https://www.tutorialspoint.com/keras/keras_model_compilation.htm

# In[19]:


model = keras.Sequential([
                          keras.layers.Dense(units=10,
                                             input_shape=(784,),
                                             activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(flat_X_train,y_train, epochs=5)


# Link: https://stackoverflow.com/questions/58352326/running-the-tensorflow-2-0-code-gives-valueerror-tf-function-decorated-functio

# In[20]:


tf.config.run_functions_eagerly(True)


# In[21]:


new_X_train = X_train/255
new_X_test = X_test/255


# In[22]:


new_X_train[0][5]


# In[23]:


new_flat_X_train = new_X_train.reshape(len(new_X_train),28*28)
new_flat_X_test = new_X_test.reshape(len(new_X_test),28*28)


# In[24]:


model = keras.Sequential([
                          keras.layers.Dense(units=10,
                                             input_shape=(784,),
                                             activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(new_flat_X_train,y_train, epochs=5)


# In[25]:


model.evaluate(new_flat_X_test,y_test)


# In[26]:


y_pred = model.predict(new_flat_X_test)


# In[27]:


y_pred[0]


# In[28]:


np.argmax(y_pred[0])


# In[29]:


y_pred_label = [np.argmax(i) for i in y_pred]


# In[30]:


# y_pred_label


# In[31]:


# Confusion Matrix
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_label)


# In[32]:


plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')


# # Multiple Hidden Layes

# In[33]:


model = keras.Sequential([
                          keras.layers.Dense(units=10,
                                             input_shape=(784,),
                                             activation='sigmoid'),
                          # Second Hidden Layer
                          keras.layers.Dense(units=100,
                                             activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(new_flat_X_train,y_train, epochs=5)


# In[34]:


model = keras.Sequential([
                          keras.layers.Dense(units=100,
                                             input_shape=(784,),
                                             activation='sigmoid'),
                          # Second Hidden Layer
                          keras.layers.Dense(units=100,
                                             activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(new_flat_X_train,y_train, epochs=5)


# In[35]:


model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(units=100,
                                             activation='sigmoid'),
                          # Second Hidden Layer
                          keras.layers.Dense(units=100,
                                             activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train,y_train, epochs=5)

