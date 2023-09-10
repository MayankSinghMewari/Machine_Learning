#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
df = pd.read_csv('D:\ML\insurance.csv')
df


# In[7]:


df['sex'] = df['sex'].astype('category')
df['sex'] = df['sex'].cat.codes
df


# In[8]:


df['smoker'] = df['smoker'].astype('category')
df['smoker'] = df['smoker'].cat.codes

df['region'] = df['region'].astype('category')
df['region'] = df['region'].cat.codes

df


# In[9]:


df.isnull().sum()


# In[10]:


x = df.drop(columns = 'charges')


# In[11]:


x


# In[12]:


y = df['charges']


# In[13]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)


# In[14]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[15]:


lr.fit(x_train, y_train)


# In[16]:


c = lr.intercept_
c


# In[17]:


n = lr.coef_
n


# In[25]:


y_pred_train = lr.predict(x_train)
y_pred_train


# In[19]:


import matplotlib.pyplot as plt


# In[26]:


plt.scatter(y_train,y_pred_train)
plt.xlabel("actual charges")
plt.ylabel("priduct charges")
plt.show()


# In[22]:


from sklearn.metrics import r2_score


# In[28]:


r2_score(y_train,y_pred_train)


# In[30]:


y_pred_test = lr.predict(x_test)


# In[33]:


import matplotlib.pyplot as plt

plt.scatter(y_test,y_pred_test)
plt.xlabel("actual charges")
plt.ylabel("priduct charges")
plt.show()


# In[34]:


r2_score(y_test,y_pred_test)


# In[ ]:




