
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# In[4]:


from pandas import Series,DataFrame


# In[37]:


import matplotlib.pyplot as plt


# In[5]:


data_train=pd.read_csv('train.csv')


# In[6]:



# In[7]:




# In[8]:




# In[38]:


fig=plt.figure()
fig.set(alpha=0.3)
data_train.Survived.value_counts().plot(kind='bar')
plt.title('jhj')
plt.ylabel('num')
plt.show()

