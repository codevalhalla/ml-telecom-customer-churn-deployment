#!/usr/bin/env python
# coding: utf-8

# # ML Telecom Churn Prediction

# In[1]:


import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn.pipeline import make_pipeline


# In[2]:


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


# In[3]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# In[4]:


df = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[5]:


y_train = df.churn


# In[6]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# In[7]:


pipeline = make_pipeline(
    DictVectorizer(),
    LogisticRegression(solver='liblinear')
)


# In[8]:


train_dict = df[categorical + numerical].to_dict(orient='records')

pipeline.fit(train_dict, y_train)


# In[10]:


with open('model.bin','wb') as f_out:
    pickle.dump(pipeline,f_out)


# In[11]:


with open('model.bin','rb') as f_in:
    pickle.load(f_in)


# In[14]:


customer = {
    'gender': 'male',
     'seniorcitizen': 0,
     'partner': 'no',
     'dependents': 'yes',
     'phoneservice': 'no',
     'multiplelines': 'no_phone_service',
     'internetservice': 'dsl',
     'onlinesecurity': 'no',
     'onlinebackup': 'yes',
     'deviceprotection': 'no',
     'techsupport': 'no',
     'streamingtv': 'no',
     'streamingmovies': 'no',
     'contract': 'month-to-month',
     'paperlessbilling': 'yes',
     'paymentmethod': 'electronic_check',
     'tenure': 6,
     'monthlycharges': 29.85,
     'totalcharges':129.85}

churn = pipeline.predict_proba(customer)[0,1]

if churn >= 0.5:
    print('Send email with promo')

else:
    print("Don't do anything")


# In[ ]:




