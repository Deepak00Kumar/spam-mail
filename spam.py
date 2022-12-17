#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


# In[2]:


# loading the data from csv file to a pandas Dataframe
raw_mail_data = pd.read_csv('mail_data.csv')


# In[3]:


raw_mail_data.head(5)


# In[4]:


# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[5]:


# printing the first 5 rows of the dataframe
mail_data.head()


# In[6]:


# checking the number of rows and columns in the dataframe
mail_data.shape


# In[7]:


# label spam mail as 0;  ham mail as 1;
#Label encoding

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# In[8]:


# separating the data as texts and label

X = mail_data['Message']

Y = mail_data['Category']


# In[9]:


print(X)


# In[10]:


print(Y)


# In[12]:


from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[13]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# # feature extraction

# In[14]:


# transform the text data to feature vectors that can be used as input to the Logistic regression
from sklearn.feature_extraction.text import TfidfVectorizer
                                             #(mail data into numerical data)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[15]:


print(X_train)


# In[16]:


print(X_train_features)


# training the model

# Logistic Regression

# In[17]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[18]:


# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)


# Evaluating the trained model

# In[19]:


# prediction on training data
from sklearn.metrics import accuracy_score

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[20]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[21]:


# prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[22]:


print('Accuracy on test data : ', accuracy_on_test_data)


# Building a Predictive System

# In[23]:


input_mail = ["WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')


# In[ ]:





# In[ ]:




