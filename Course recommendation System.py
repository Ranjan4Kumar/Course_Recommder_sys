#!/usr/bin/env python
# coding: utf-8

# # Course Recommendation System on Udemy dataset
# 
# ### Algo
# + Cosine similarity
# + Linear Similairty
# 
# ### WorkFlow
# + Dataset
# + Vectorized our dataset
# + Cosine Similarity
# + ID Score
# + Recommend

# In[4]:


get_ipython().system('pip install neattext')


# In[26]:


import pandas as pd
import neattext.functions as nfx
import seaborn as sns


# In[8]:


# Load Ml/Rc Pkgs
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel


# In[9]:


df = pd.read_csv('udemy_courses.csv')


# In[10]:


df.head()


# In[11]:


df['course_title']


# In[12]:


dir(nfx)


# In[13]:


# Clean text : stopwords, specail characters
df['clean_course_title'] = df['course_title'].apply(nfx.remove_stopwords)


# In[14]:


df['clean_course_title'] = df['clean_course_title'].apply(nfx.remove_special_characters)


# In[17]:


# compare your original data with the clean data
df[['course_title' , 'clean_course_title']]


# In[19]:


# How to vectorzie our text
count_vect = CountVectorizer()
cv_mat = count_vect.fit_transform(df['clean_course_title'])


# In[20]:


# Sparse
cv_mat


# In[21]:


#Dense
cv_mat.todense()


# In[22]:


df_cv_words = pd.DataFrame(cv_mat.todense(),columns = count_vect.get_feature_names())


# In[23]:


df_cv_words.head()


# In[24]:


# Cosine similarityMatrix
cosine_sim_mat = cosine_similarity(cv_mat)


# In[25]:


cosine_sim_mat


# In[27]:


# cearting heatmap of cosine similariy 
sns.heatmap(cosine_sim_mat[0:10] , annot = True)


# In[29]:


# Get Course ID/Ondex
course_indices = pd.Series(df.index,index=df['course_title']).drop_duplicates()


# In[30]:


course_indices


# In[31]:


course_indices['How To Maximize Your Profits Trading Options']


# In[32]:


idx = course_indices['How To Maximize Your Profits Trading Options']


# In[33]:


idx


# In[34]:


scores = list(enumerate(cosine_sim_mat[idx]))


# In[35]:


#Sort our scire per cosine score
sorted_scores = sorted(scores,key=lambda x:x[1], reverse = True)


# In[37]:


# Ommit the first Value/itself
sorted_scores[1:]


# In[38]:


#Selected courses Indecis
selected_course_indeces = [i[0] for i in sorted_scores[1:]]


# In[40]:


selected_course_indeces


# In[41]:


#Select courses Score
selected_course_scores = [i[1] for i in sorted_scores[1:]]


# In[42]:


#Selected Course Scores
df['course_title'].iloc[selected_course_indeces]


# In[43]:


recommended_result = df['course_title'].iloc[selected_course_indeces]


# In[45]:


rec_df = pd.DataFrame(recommended_result)


# In[46]:


rec_df.head()


# In[47]:


rec_df['similarity_scores'] = selected_course_scores


# In[49]:


rec_df


# In[50]:


def recommend_course(title,num_of_rec = 10):
    # ID fpr the title
    idx = course_indices[title]
    # course Indices
    # Search inside cosine_sim_mat
    scores = list(enumerate(cosine_sim_mat[idx]))
    # scores
    # Sort scores
    sorted_scores = sorted(scores,key=lambda x:x[1], reverse=True)
    # recomm
    
    selected_course_indices = [i[0] for i in sorted_scores[1:]]
    selected_course_scores = [i[1] for i in sorted_scores[1:]]
    result = df['course_title'].iloc[selected_course_indices]
    rec_df = pd.DataFrame(result)
    rec_df['similarity_scores'] = selected_course_scores
    return rec_df.head(num_of_rec)


# In[51]:


recommend_course('Trading Options Basics')


# In[ ]:




