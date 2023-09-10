#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("D:\ML\Multiple regression\insurance.csv")
df.head()


# In[3]:


df.isna().sum()


# In[4]:


df['region'].unique()


# In[5]:


df['sex'] = df['sex'].replace({'female':1,'male':2})
df['smoker'] = df['smoker'].replace({'yes':1,'no':2})
df['region'] = df['region'].replace({'southwest': 1, 'southeast': 2, 'northwest': 3, 'northeast' : 4})


# In[6]:


# defining input features and target variable
x = df.drop(columns = ['charges'])
y = df['charges']


# In[7]:


#split for traning and testing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 1)


# In[10]:


# scaling the data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)


# In[11]:


# model Developement and evaluation 
from sklearn.linear_model import LinearRegression
lin = LinearRegression()


# In[13]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 6)
x_poly_train = poly.fit_transform(x_train_scaler)
x_test_poly = poly.transform(x_test_scaler)
poly.fit(x_poly_train, y_train)
lin.fit(x_poly_train, y_train)


# In[15]:


y_pred = lin.predict(x_test_poly)


# In[19]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)


# In[20]:


y_pred_train = lin.predict(x_poly_train)
mean_absolute_error(y_train, y_pred_train)


# In[24]:


poly = PolynomialFeatures(degree =3)
x_poly_train = poly.fit_transform(x_train_scaler)
x_test_poly = poly.transform(x_test_scaler)
poly.fit(x_poly_train, y_train)
lin = LinearRegression()
lin.fit(x_poly_train, y_train)


# In[26]:


y_pred = lin.predict(x_test_poly)
mean_absolute_error(y_test,y_pred)


# In[28]:


y_pred_train = lin.predict(x_poly_train)
mean_absolute_error(y_train,y_pred_train)


# In[29]:


import matplotlib.pyplot as plt


# In[30]:


plt.scatter(y_train,y_pred_train)
plt.xlabel("actual charges")
plt.ylabel("priduct charges")
plt.show()


# In[32]:


plt.scatter(y_test,y_pred)
plt.xlabel("actual charges")
plt.ylabel("priduct charges")
plt.show()


# In[ ]:




