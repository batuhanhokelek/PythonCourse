#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import re


# In[59]:


#importing data
immigration = pd.read_csv("https://raw.githubusercontent.com/carlson9/KocPythonFall2021/main/hw/immSurvey.csv")
X, y = immigration.text, immigration.sentiment


# In[60]:


# Conversion of a bag-of-words vocabulary
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range=(2,2)).fit(X)
X = vect.transform(X)


# In[61]:


#the content of the sparse matrix as dataframe
X_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())


# In[62]:


# scaling the target variable
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
y = y.values.reshape(-1, 1) # otherwise, StandardScaler returns an error
y = scaler.fit_transform(y)
y = y.flatten() # otherwise, SVR returns an error


# In[63]:


# splitting into train and test data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=44)


# In[64]:



#support vector regression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
svr = SVR()
grid_svr = GridSearchCV(svr, {'C':[0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 10]})
grid_svr.fit(X_train.toarray(), y_train)
svr_pred = grid_svr.predict(X_test.toarray())
print("Best CV parameter for c value in SVR: \n", grid_svr.best_params_)
print("Correlation matrix for test group and predictions in SVR: \n", np.corrcoef(y_test, svr_pred))


# In[65]:



# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000, random_state=44)
rf.fit(X_train.toarray(), y_train)
rf_pred = rf.predict(X_test.toarray())
print("Correlation matrix for test group and predictions in RF: \n", np.corrcoef(y_test, rf_pred))


# In[66]:



# Gaussian Process Regressor
from sklearn.gaussian_process import GaussianProcessRegressor
gaus_process_reg = GaussianProcessRegressor(normalize_y=False)
gaus_process_reg.fit(X_train.toarray(), y_train)
gpr_pred = gaus_process_reg.predict(X_test.toarray())
print("Correlation matrix for test group and predictions in GaussianPR: \n", np.corrcoef(y_test, gpr_pred))


# In[67]:



# TfIdf - term frequency inverse document frequency
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b').fit(immigration.text)
X_tfidf = vectorizer.transform(immigration.text)


# In[68]:



# train test split again
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.5, random_state=44)


# In[69]:



# Support Vector Regression after term frequency–inverse document frequency (TF-IDF) Transformation
tf_idf_svr = SVR()
tf_idf_svr.fit(X_train.toarray(), y_train)
tf_idf_pred = tf_idf_svr.predict(X_test.toarray())
print("Correlation matrix for test group and predictions in SVR after Tf-Idf: \n", np.corrcoef(y_test, tf_idf_pred))


# In[70]:



# Random Forest after Tf-Idf Transformation
tf_idf_rf = RandomForestRegressor(n_estimators=1000, random_state=44)
tf_idf_rf.fit(X_train.toarray(), y_train)
tf_idf_pred_rf = tf_idf_rf.predict(X_test.toarray())
print("Correlation matrix for test group and predictions in Random Forest after Tf-Idf: \n", np.corrcoef(y_test, tf_idf_pred_rf))


# In[71]:



# GaussianProcessRegressor after Tf-Idf Transformation
gpr_tf_idf = GaussianProcessRegressor(normalize_y=False)
gpr_tf_idf.fit(X_train.toarray(), y_train)
tf_idf_pred_gpr = gpr_tf_idf.predict(X_test.toarray())
print("Correlation matrix for test group and predictions in Gaussian Process Regressor: \n", np.corrcoef(y_test, tf_idf_pred_gpr))


# In[73]:



# exploratory visualization
plt.hist(immigration['sentiment'], bins=15, color="darkorange", edgecolor="black")
plt.title("Sentiment Distribution of Dataset")
plt.xlabel("Sentiment Score")
plt.ylabel("Counts")
plt.show()

