#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import pickle


# In[2]:


df = pd.read_csv("tmdb_5000_movies.csv")
df1 = pd.read_csv("tmdb_5000_credits.csv")


# In[3]:


df.shape


# In[4]:


df1 = df1.rename(columns = {"movie_id":"id"})
df1


# In[5]:


result = pd.merge(df,df1[['id','cast','crew']],on = 'id')


# In[6]:


result


# In[7]:


def json_to_list(col):
    for index, row in result.iterrows():
        row_list= []
        x = json.loads(row[col])
        for genre in x:
            row_list.append(genre['name'])
        result.at[index, col] = row_list


# In[8]:


cols =['genres', 'keywords', 'production_companies', 'production_countries','spoken_languages','cast']
for col in cols:
    json_to_list(col)


# In[9]:


def reduce_list(col):
    for index, row in result.iterrows():
        li = row[col]
        li = li[:5]
        result.at[index, col] = li


# In[10]:


cols = ['genres','cast']
for col in cols:
    reduce_list(col)


# In[11]:


result


# In[12]:


result.isnull().sum()


# In[13]:


result = result[result['vote_average']>6]


# In[14]:


len(result.loc[(result['keywords'].str.len() == 0),:])


# In[15]:


len(result.loc[(result['genres'].str.len() == 0),:])


# In[16]:


result


# In[17]:


for index,row in result.iterrows():
    crew_list = json.loads(row['crew'])
    for x in crew_list:
        if x['job'] == 'Director':
            result.at[index,'crew'] = x['name']


# In[18]:


result


# In[19]:


result = result[['genres','keywords','overview','popularity','revenue','title','vote_average','vote_count','cast','crew']]


# In[20]:


result.rename(columns = {'crew':'director'},inplace = True)


# In[21]:


result.dropna()


# In[22]:


cols = ['genres','keywords','cast']
for col in cols:
    result = result[result[col].map(lambda d: len(d)) > 0]


# In[23]:


result


# In[24]:


result.describe()


# In[25]:


result


# In[26]:



result.shape


# In[27]:


result = result.drop('overview',axis=1)


# In[28]:


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]    


# In[29]:


features = ['cast', 'keywords', 'genres']
for feature in features:
    result[feature] = result[feature].apply(clean_data)


# In[30]:


result


# In[31]:


result['director'] = result['director'].str.replace(' ','')
result['director'] = result['director'].str.lower()
result


# In[48]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['genres'])
result['soup'] = result.apply(create_soup, axis=1)
result['soup'] = result[['soup','director']].apply(lambda x:' '.join(x), axis=1)    


# In[58]:


result['soup'][3]


# In[50]:


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(result['soup'])


# In[51]:


sig = sigmoid_kernel(count_matrix,count_matrix)


# In[52]:


result = result.reset_index()
indices = pd.Series(result.index, index=result['title'])
pickle.dump(count,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


# In[53]:


def get_recommendations(title, sig=sig):
    idx = indices[title]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:11]  
    movie_indices = [i[0] for i in sig_scores]
    return result['title'].iloc[movie_indices]


# In[59]:


get_recommendations('The Dark Knight Rises')


# In[ ]:




