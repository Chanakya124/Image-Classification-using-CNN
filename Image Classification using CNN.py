#!/usr/bin/env python
# coding: utf-8

# In[19]:


import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import warnings
warnings.filterwarnings('ignore')


# ## Load Dataset

# In[20]:


X_train = np.loadtxt(r'E:\Cnn project\Image Classification CNN Keras Dataset-20241124T114956Z-001\Image Classification CNN Keras Dataset\input.csv', delimiter = ',')
Y_train = np.loadtxt(r'E:\Cnn project\Image Classification CNN Keras Dataset-20241124T114956Z-001\Image Classification CNN Keras Dataset\labels.csv', delimiter = ',')

X_test = np.loadtxt(r'E:\Cnn project\Image Classification CNN Keras Dataset-20241124T114956Z-001\Image Classification CNN Keras Dataset\input_test.csv', delimiter = ',')
Y_test = np.loadtxt(r'E:\Cnn project\Image Classification CNN Keras Dataset-20241124T114956Z-001\Image Classification CNN Keras Dataset\labels_test.csv', delimiter = ',')


# ### Printing the Shapes of the DataSet

# In[21]:


print("Shape of X_train: ", X_train.shape)
print("Shape of Y_train: ", Y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of Y_test: ", Y_test.shape)


# ### Reshaping of the Data in Appropriate Image Sizes

# In[22]:


X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)

X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)


# In[23]:


print("Shape of X_train: ", X_train.shape)
print("Shape of Y_train: ", Y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of Y_test: ", Y_test.shape)


# ### To train the model properly the values are rescaled

# In[24]:


X_train = X_train/255 # for better training of our model we have converted it in range from 0 to 1 
X_test = X_test/255


# In[25]:


idx = random.randint(0,len(X_train))
plt.imshow(X_train[idx, :])
plt.show()


# # Model Building

# In[26]:


model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape= (100,100,3)),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation = 'relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(128,activation= 'relu'),
    Dense(1, activation= 'sigmoid')
])



# In[ ]:


#Another Method of building the model
# model = Sequential()

# model.add(Conv2D(32, (3,3), activation='relu', input_shape= (100,100,3)))
# model.add(MaxPooling2D((2,2)))   
#    # Layers are added manually       
# model.add(Conv2D(32, (3,3), activation='relu'))
# model.add(MaxPooling2D((2,2)))
          
# model.add(Flatten())
# model.add(Dense(64,activation= 'relu'))
# model.add(Dense(1, activation= 'sigmoid'))


# ## Adding cost function and back propogation

# In[27]:


model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[28]:


model.fit(X_train, Y_train, epochs=7, batch_size=32)


# ## Model Evaluation on test Data set

# In[29]:


model.evaluate(X_test, Y_test)


# # Making Prediction for the individual project

# In[32]:


idx2 = random.randint(0, len(Y_test))
plt.imshow(X_test[idx2, :])
plt.show()


y_pred = model.predict(X_test[idx2, :].reshape(1, 100, 100, 3))
print(y_pred)
    


# In[37]:


idx2 = random.randint(0, len(Y_test))
plt.imshow(X_test[idx2, :])
plt.show()

y_pred = model.predict(X_test[idx2, :].reshape(1, 100, 100, 3))
y_pred = y_pred > 0.5# As seen in above output probability of dog pic is showing the nearly Zero

if(y_pred ==0):
    pred='dog'
else:
    pred = 'cat'
    
print("Our model says it is a :", pred)


# In[ ]:





# In[ ]:




