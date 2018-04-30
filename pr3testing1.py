
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from scipy.sparse import csr_matrix


# In[2]:


ran_state = 12


# In[3]:


with open("train.dat", "r") as fh:
    lines = fh.readlines()  


# In[4]:


len(lines)


# In[5]:


docs = [l.split() for l in lines]


# In[6]:


len(docs)


# In[7]:


#def csr_build(docs):
    #build csr w/ our input data, format: index, value, index value.... etc
    #should be even number of len for each line in doc(1 ind, 1 val)
    #index is feature number, value is count of appearance in doc
    #each line is a doc
nrows = len(docs)
nnz = 0
biggest_feature_id = 0
for d in docs:
    
    #divide by two because not all are words/indices, half are values
    nnz += len(d) / 2
    #need to only look at values so skip 1 each time..
    for w in range(0, len(d), 2):
        if(biggest_feature_id < int(d[w])):
            biggest_feature_id = int(d[w])
            
ncols = biggest_feature_id
#memory
ind = np.zeros(nnz, dtype = np.int)
val = np.zeros(nnz, dtype = np.double)
ptr = np.zeros(nrows+1, dtype = np.int)
n = 0
i = 0

for d in docs:
    for w in range(0, len(d), 2):
        ind[n] = int(d[w])
        val[n] = int(d[w+1])
        n += 1
    ptr[i+1] = n
    i += 1
    
mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    

        
        
        
        
    


# In[8]:


mat.shape[1]


# In[9]:


mat.shape


# In[ ]:


from sklearn.decomposition import TruncatedSVD


# In[ ]:


#find n_components where explained variance still > 90%, use #features-1 to test
svd = TruncatedSVD(n_components=10000, n_iter=6, random_state=ran_state)
svd_test_fit = svd.fit(mat) 

