#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('E:\diabetes.csv')


# In[2]:


df.head()


# In[3]:


df.columns # for column names


# In[4]:


import matplotlib.pyplot as plt
df.hist(figsize=(9,9))
plt.show()


# In[5]:


df_new = df[(df.BloodPressure != 0) & (df.BMI != 0) & (df.Glucose != 0)]
print(df_new.shape)

df_new.hist(figsize=(9, 9))
plt.show()


# In[6]:


plt.hist(df.Outcome, color = 'red')
plt.title("Outcome Count")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.show()


# # Correlation

# In[7]:


from scipy.stats import pearsonr


# In[8]:


df_new.corr()


# # Regressions Model
# 
# (This is not correct model, I just made to see how to run OLS in Python)

# In[9]:


import statsmodels.api as sm

y = df["DiabetesPedigreeFunction"]
x = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age", "Outcome"]]
model = sm.OLS(y, x).fit()
predictions = model.predict(x)
model.summary()


# Now I will apply logistic regression, with ``outcome`` as a dependant variable.

# In[10]:


plt.scatter(df["BloodPressure"],df["BMI"])
plt.show()


# In[11]:


import statsmodels.formula.api as sm

model  = sm.ols(formula='BloodPressure ~ BMI', data=df).fit()
model.summary(model)


# # Logistic Regression

# Now I will take ``outcome`` as a dependant variable and run the test to study is other variables affect the ``outcome``

# In[12]:


from sklearn import preprocessing
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit


# In[13]:


y = df["Outcome"]
x = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age", "DiabetesPedigreeFunction"]]

logit_model = sm.Logit(y,x)
result=logit_model.fit()
print(result.summary())


# # Probit Regression

# In[14]:


probitmodel = Probit(y,x)
probit_model = probitmodel.fit()

print(probit_model.summary())

