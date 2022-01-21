#!/usr/bin/env python
# coding: utf-8

# ### CSSM 502 Project Linear Regression Analysis

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import date
import matplotlib.dates as mdates
import seaborn as sns
import pytrends
import os
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import silhouette_score 
import scipy.cluster.hierarchy as shc 
from scipy.cluster import hierarchy
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')
from pytrends.request import TrendReq 
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

sc = StandardScaler()
lm = linear_model.LinearRegression()


# In[3]:


# Loading vaccine data
df_vac = pd.read_csv('/Users/batuh/Desktop/Total Daily Vaccination Rate in Turkey.csv', header=0, sep= ",", infer_datetime_format=True,
                          parse_dates=['Date'],
                          index_col='Date') 


# In[4]:


df_vac


# In[5]:


df_vac["Daily Total Vaccination"] = pd.to_numeric(df_vac["Daily Total Vaccination"], errors='coerce').fillna(0, downcast='infer')


# In[6]:


df_vac["Daily Total Vaccination"] = df_vac["Daily Total Vaccination"].astype(float)


# In[7]:


df_vac.describe()


# In[8]:


df_vac.dtypes


# In[9]:


# Load Google Trends data
results = pd.read_csv('/Users/batuh/Desktop/google trends.csv', header=0, sep= ",", infer_datetime_format=True,
                          parse_dates=['Date'],
                          index_col='Date') 


# In[10]:


results


# In[11]:


#deletion of missing NaN values from my google trends data. 

results_cleaned = results.dropna()


# In[12]:


results_cleaned


# ### Individual Keyword Analysis
# 
# We put all keywords into prediction model.       
# Training data ratio is 80-20, 
# Mean Square Error = 0.0, r2 = 1.0

# In[13]:


all_keywords = results[["Covid-19", "kovid-19", "corona", "korona",
                                "covid-19 belirtileri", "corona belirtileri", 
                                "kovid-19 belirtileri", "covid belirtileri", 
                                "korona belirtileri", "corona belirtisi", 
                                "korona belirtisi", "corona semptomları", 
                                "covid-19 semptomları", "covid-19 virüsü", 
                                "covid virüsü", "corona virüsü", "koronavirüs", "pandemic",
                                "pandemi", "karantina", "quarantine", "covid",
                                "kovid", "Omicron", "delta",
                                "omikron", "omicron belirtileri",
                                "omikron belirtileri", "delta belirtileri", "varyant",
                                "omicron semptomları", "omikron semptomları", "omicron öldürür mü", 
                                "omicron virüsü", "delta virüsü", "BioNTech","Turkovac","Sinovac",
                                "Sputnik","biontech aşısı","sinovac aşısı","alman aşısı","çin aşısı",
                                "türk aşısı","rus aşısı","Covid aşısı","korona aşısı","corona aşısı",
                                "covid 19 vaccine","biontech yan etkileri","alman aşısı yan etkileri",
                                "çin aşısı yan etkileri","sinovac yan etkileri", "kol ağrısı",
                                "aşı yan etkileri","corona aşısı yan etkileri","korona aşısı yan etkileri",
                                "covid-19 aşısı yan etkileri", "sinovac aşısı yan etkileri",
                                "sputnik aşısı yan etkileri", "turkovac aşısı yan etkileri", 
                                "turkovac aşısı"]]


# In[14]:


all_keywords


# In[15]:


# creating test and training data sets

train_df = all_keywords['2021-01-13':'2021-05-21'].merge(df_vac, on="Date").dropna()
test_df = all_keywords['2021-05-22':'2021-06-22'].merge(df_vac, on="Date").dropna()


# In[16]:


train_df


# In[17]:


train_df_scaled = pd.DataFrame(sc.fit_transform(train_df), columns=train_df.columns.values, index=train_df.index)
test_df_scaled = pd.DataFrame(sc.fit_transform(test_df), columns=test_df.columns.values, index=test_df.index)


# In[18]:


train_df_scaled.head()


# In[19]:


train_df_scaled.describe()


# In[20]:


m = len(train_df.columns)-1
X_train = train_df_scaled.iloc[:, 0:m]
X_test = test_df_scaled.iloc[:, 0:m]
y_train = train_df_scaled.iloc[:, m]
y_test = test_df_scaled.iloc[:, m]


# In[21]:


model = lm.fit(X_train, y_train)
predictions_1 = model.predict(X_test)
df_all = pd.DataFrame({'Actual': y_test, 'Predicted_all': predictions_1})
 
    
print(metrics.mean_squared_error(y_test, predictions_1))
print(metrics.r2_score(y_test, predictions_1))
    
prediction_dict = pd.DataFrame(columns = ['real','predicted'])
prediction_dict.real = y_test
prediction_dict.predicted = predictions_1


# In[22]:


prediction_dict.plot()

