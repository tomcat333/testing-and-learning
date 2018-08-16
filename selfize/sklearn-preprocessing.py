
# coding: utf-8

# preprocessing 主要包含以下几种方法：
# 1,binarization
# 2,scaling
# 3,normalization
# 4,mean removal

# 首先是binarization

# 用法：preprocessing.Binarizer(threshold=1.5).transform(data)

# In[1]:


from sklearn import preprocessing
import numpy as np


# In[2]:


data=np.array([[2.2,5.9,-1.8],[5.4,-3.2,-5.1],[-1.9,4.2,3.2]])


# In[3]:


data


# In[4]:


bindata=preprocessing.Binarizer(threshold=1.5).transform(data)


# In[5]:


bindata


# 接下来是mean removal?scale?

# 用法：preprocessing.scale(data)

# In[6]:


print('mean(before)=',data.mean(axis=0))
print('standard deviation(before)=',data.std(axis=0))


# In[9]:


scaled_data=preprocessing.scale(data)
print('mean(after)=',scaled_data.mean(axis=0))
print('standard deviation(after)=',scaled_data.std(axis=0))


# In[10]:


scaled_data


# 下面是scaling，包含：1 StandardScaler,2 MinMaxScaler,3 Normalizer（只有minmaxscaler？）

# 用法：preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(data)

# In[17]:


data


# In[40]:


preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(data)


# 最后是normalization，只进行水平运算？

# 包含L1、L2两种，L1是每行绝对值和为1，L2是每行平方和为1，比例均不变

# 用法：preprocessing.normalize(data,norm='l1')

# In[21]:


data


# In[22]:


preprocessing.normalize(data,norm='l1')


# In[23]:


preprocessing.normalize(data,norm='l2')


# 下面是encoding部分

# 首先是label encoding

# 步骤：1 import（preprocessing即可），2 实例化，3 fit， 4transform

# In[42]:


labels=['setosa','versicolor','virginica']


# In[43]:


encoder=preprocessing.LabelEncoder()


# In[44]:


encoder.fit(labels)


# In[45]:


for i,item in enumerate(encoder.classes_):  ##注意这里迭代器的用法
    print(item,'=>',i)


# In[54]:


more_labels=['versicolor','versicolor','virginica','setosa','versicolor']
more_labels_encoded=encoder.transform(more_labels)
more_labels_encoded


# In[55]:


list(more_labels_encoded)


# 然后是one-hot encoding

# In[56]:


import pandas as pd
from IPython.display import display


# In[59]:


data=pd.read_csv('adult.data',header=None,index_col=False,names=['age','workclass','fnlwgt',
                                                                'education','education-num',
                                                                'martial-status','occupation',
                                                                'relationship','race','gender',
                                                                'capital-gain','capital-loss',
                                                                'hours-per-week','native-country',
                                                                'income'])


# In[60]:


data.head()


# In[72]:


data=data[['age','workclass','education','gender','hours-per-week','occupation','income']]


# In[73]:


data.head()


# In[105]:


print('original features:\n',list(data.columns),'\n')
data_dummies=pd.get_dummies(data)             ###貌似这个方法不大好，还不如直接对其中一行或几行做get_dummies
print('features after one-hot encoding:\n',list(data_dummies.columns))


# In[107]:


data_dummies.head()


# In[98]:


dummies_occupation=pd.get_dummies(data['occupation'])


# In[118]:


data.occupation.value_counts()


# In[108]:


dummies_occupation


# In[113]:


dummies_gender_income=pd.get_dummies(data[['gender','income']])


# In[114]:


dummies_gender_income


# In[115]:


features=data_dummies.ix[:,'age':'occupation_ Transport-moving']###原课堂讲法，感觉更乱，所有的feature都one-hot处理了


# In[81]:


X=features.values
y=data_dummies['income_ >50K'].values


# In[82]:


X


# In[85]:


y


# In[86]:


from sklearn.linear_model import LogisticRegression


# In[87]:


from sklearn.model_selection import train_test_split


# In[88]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)


# In[89]:


logreg=LogisticRegression()


# In[90]:


logreg.fit(X_train,y_train)


# In[96]:


print('logistic regression score on the test sset: {:.2f}'.format(logreg.score(X_test,y_test)))

