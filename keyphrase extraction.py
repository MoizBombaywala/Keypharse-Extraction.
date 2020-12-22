#!/usr/bin/env python
# coding: utf-8

# In[1]:


doc = '''
I really like the show because it is thought provoking and i like shows that make me think
The cast.
Nothing
It is the most complex and original idea I have ever seen or heard of. Also, because it delves into the topic of human emotions, but in an artificial way, if I were to be punny and serious all at once. From what I have seen of the show, I believe that we, as human beings, can compare ourselves to the androids, because we can definitely relate to them.
it has many twists and I like that
Its unpredictable so you are left wanting more
'''
       


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer

n_gram_range = (1, 1)
stop_words = "english"


# In[18]:


# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
candidates = count.get_feature_names()


# In[19]:


from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)


# In[20]:


from sklearn.metrics.pairwise import cosine_similarity

top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)


# In[21]:


keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]


# In[22]:


print(keywords)

