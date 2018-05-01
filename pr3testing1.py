
# coding: utf-8

# In[14]:


import numpy as np
import scipy as sp
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from scipy.sparse import csr_matrix


# In[15]:


ran_state = 12


# In[16]:


with open("train.dat", "r") as fh:
    lines = fh.readlines()  


# In[17]:


len(lines)


# In[18]:


docs = [l.split() for l in lines]


# In[19]:


len(docs)


# In[20]:


# #def csr_build(docs):
#     #build csr w/ our input data, format: index, value, index value.... etc
#     #should be even number of len for each line in doc(1 ind, 1 val)
#     #index is feature number, value is count of appearance in doc
#     #each line is a doc
# nrows = len(docs)
# nnz = 0
# biggest_feature_id = 0
# for d in docs:
    
#     #divide by two because not all are words/indices, half are values
#     nnz += len(d) / 2
#     #need to only look at values so skip 1 each time..
#     for w in range(0, len(d), 2):
#         if(biggest_feature_id < int(d[w])):
#             biggest_feature_id = int(d[w])
            
# ncols = biggest_feature_id
# #memory
# ind = np.zeros(nnz, dtype = np.int)
# val = np.zeros(nnz, dtype = np.double)
# ptr = np.zeros(nrows+1, dtype = np.int)
# n = 0
# i = 0

# for d in docs:
#     for w in range(0, len(d), 2):
#         ind[n] = int(d[w])
#         val[n] = int(d[w+1])
#         n += 1
#     ptr[i+1] = n
#     i += 1
    
# mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    
# ####-----#### 
# #i did this first implementation same as ACT-data3 but i cannot use TruncatedSVD feature decomposition
# #on this csr_matrix because I have memory errors because the mapped ind values are extremely high
# #thus on the next implementation/example shown I map in dict to track original id to unique general id created by myself
# #which lowers the ind values by a TON for memory allocation in truncatedSVD
# #print(mat.shape[1])
# #print(mat.shape)
# #list(ind)       
# #len(ind)       
        
    


# In[21]:


#####DICT######
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
ptr = np.zeros(nrows+1, dtype = np.long)
ind_dict = dict()
n = 0
i = 0
index = 0
for d in docs:
    for w in range(0, len(d), 2):
        if not d[w] in ind_dict:
            ind_dict[d[w]] = index
            index+=1
        ind[n] = ind_dict[d[w]]
        val[n] = int(d[w+1])
        n += 1
    ptr[i+1] = n
    i += 1
    
mat = csr_matrix((val, ind, ptr), dtype=np.long)
# list(ind)
# len(ind)


# In[22]:


mat.shape[1]


# In[23]:


mat.shape


# In[24]:


mat.shape[1]-1


# In[25]:


from sklearn.decomposition import TruncatedSVD


# In[26]:


#find n_components where explained variance still > 90%, use #features-1 to test
#used n_iter=5(also default val) b/c it is higher for sparse matrices that may
#have slowly decaying spectrum
svd = TruncatedSVD(n_components=1500, n_iter=5, random_state=ran_state)
svd_test_fit = svd.fit(mat) 


# In[27]:


#explained variance graph
plt.plot(np.cumsum(svd.explained_variance_ratio_))
plt.xlabel('n_components')
plt.ylabel('cum variance')
plt.show()


# In[28]:


#calculating n_compoenets required for explained variance of 90%
#usually standard is 95% but we want to reduce dimensions a lot so it's easier
#to create distance matrix..
variance_ratios = svd.explained_variance_ratio_
total_variance = 0.0;
req_variance = 0.90
n = 0
#set correct n to 0 so i can check after it is was even set(0 return means not
#found within the n_components provided, need to increase
correct_n = 0
for ratio in variance_ratios:
    total_variance += ratio
    n += 1
    if total_variance >= req_variance:
        correct_n = n
        break
print(correct_n)
#100 not enough, increase to 1000 for truncatedSVD call.
#1000 not enough either... increase to 1500
#1500 not enough either, increase to 2200
    


# In[29]:


#using found n_components perform svd on matrix for features = correct_n
#used n_iter=5(also default val) b/c it is higher for sparse matrices that may
#have slowly decaying spectrum
tsvd = TruncatedSVD(n_components=correct_n, n_iter=5, random_state=ran_state)
reduced_X = tsvd.fit_transform(mat)


# In[30]:


reduced_X.shape


# In[32]:


reduced_X[0, :]


# In[33]:


len(reduced_X)


# In[34]:


#not allowed to use distance matrix
#test_dist_matrix = sp.spatial.distance.pdist(reduced_X, metric='euclidean')


# In[35]:


#not allowed to use distance matrix
#test_dist_matrix.shape
#shape = 36803910L, remove after


# In[36]:


len(reduced_X[0, :])


# In[172]:


def connect_comp(mat, core_border_noise_list, eps):
    unconnected_core_clusters = {}
    c_count = 0
    
    #n is for each core point to place inside a cluster by iterating n
    for n in range(len(mat)):
        clust_exiter = False
        #if point within the list is a core point
        if core_border_noise_list[n] == 0:
            #check if any elements in dict, otherwise insert first core point
            if not unconnected_core_clusters:
                #create a list(for the cluster)
                unconnected_core_clusters[c_count] = []
                #insert the id of the core point into the cluster
                unconnected_core_clusters[c_count].append(str(n))
                #increase the c_count by 1 so the next cluster created is a diff ID
                c_count += 1  
                continue
            #check if any of the points inside any(iterate) of the clusters are within eps   
            for i in range(len(unconnected_core_clusters)): #we are looking at each cluster  
                #check each element within each cluster list
                for j in range(len(unconnected_core_clusters[i])):
                    temp_cluster_key = int(unconnected_core_clusters[i][j])
                    #check if core point (n) is w/in eps of each element of each cluster list   
                    if within_eps(mat[n], mat[temp_cluster_key], eps):
                        #it is within eps of one of the points so we add to this cluster
                        unconnected_core_clusters[i].append(str(n))
                        clust_exiter = True
                        break   
                if clust_exiter:
                    break  
            else:
                #create new cluster
                unconnected_core_clusters[c_count] = []
                #put n into new cluster
                unconnected_core_clusters[c_count].append(str(n))
                c_count += 1
    return unconnected_core_clusters


# In[173]:


def add_border_points(mat, core_border_noise_list, core_cluster, eps):
    t_c = 0
    c_z = 0
    c_n = 0
    connected_cluster = core_cluster
    for n in range(len(core_border_noise_list)):
        
        cluster_exiter = False
        if core_border_noise_list[n] == 1:
            c_n += 1
            #go through every cluster in dict
            for i in range(len(core_cluster)):
                for j in range(len(core_cluster[i])):
                    temp_cluster_key = int(core_cluster[i][j])
                    if within_eps(mat[n], mat[temp_cluster_key], eps):
                        connected_cluster[i].append(str(n))
                        cluster_exiter = True
                        t_c += 1
                        break
                if cluster_exiter:
                    break
    print t_c 
    print c_n
    return connected_cluster
            


# In[194]:


def add_noise_points(core_border_noise_list, clusters):
    noise_cluster = max(clusters.iterkeys()) + 1 # k+1
    noise_indexes = []
    for i in range(len(core_border_noise_list)):
        if core_border_noise_list[i] == 2:
            noise_indexes.append(str(i))
            
    print "success - noise pts"
    sys.stdout.flush()
    clusters[noise_cluster] = noise_indexes
    return clusters


# In[195]:


#####-----DBSCAN-----#####
#reindent after all things implemented..
#what kind of input to take in? np matrix?
#def DBScan(matrix, eps, minpts):
#can use basic libraries (sqrt, min, max etc). Also, you’re allowed to use the
#scipy library to do mathematical operations like the dot products
#dot product (a dot b) = a1b1 +.. adbd.. = signal aibi
import sys
#NOTE I AM NOT USING SYS FOR DBSCAN ONLY TO DISPLAY PRINT STATEMENTS IMMEDIATELY
#BECAUSE PYTHON USES LINE BUFFERING IF INTERACTIVELY DONE(JUPYTER NOTEBOOK)

def distance(x, y):
    dist = np.sqrt(sp.dot(x,x) + sp.dot(y,y) - 2 * sp.dot(x,y))
    return dist

def within_eps(x, y, eps):
    return distance(x,y) <= eps


def core_points(mat, eps, minpts):
    #use non-core-mat after for checking these to see if border pts
    #if core then true else(border or noise check next fcn) false
    testing = 0
    core_point_index_list = np.zeros(len(mat), dtype=np.bool)
    print "finding core points"
    sys.stdout.flush()
    for i in range(len(mat)):
        #reset for every outerloop so check inner loop for 0 -> minpts
        current_min_pts = 0
        #variable to check if core point was found, if not add to non core point list
        if_nocore = False
        for j in range(len(mat)):
            #print "i: ", i, "j: ", j
            
            if i==j:
                #print "skip this i=j", i
                continue
            
            if current_min_pts == minpts-1:
                if within_eps(mat[i], mat[j], eps):
                    current_min_pts += 1
                    core_point_index_list[i]=True
                    #print "core point", i
                    if_nocore = True
                    break
            
            elif current_min_pts < minpts:
                if within_eps(mat[i], mat[j], eps):
                    current_min_pts += 1
                    #print i, "within eps of ", j
            
            else:
                core_point_index_list[i]=True
                #print "core point", i
                if_nocore = True
                break
        
        if not if_nocore:
            core_point_index_list[i]=False
            #print "non core point", i
    print "Done finding core points"
    sys.stdout.flush()
    core_border_list = border_points(mat, core_point_index_list, eps)
    connected_comps = connect_comp(mat, core_border_list, eps)
    #connect_cores
    nonoise_clusters = add_border_points(mat, core_border_list, connected_comps, eps)
    clusters = add_noise_points(core_border_list, nonoise_clusters)
    return clusters 

#for border list, 0 is core point, 1 is border point, 2 is noise point
def border_points(mat, core_point_list, eps):
    core_border_list = np.zeros(len(mat), dtype=np.int)
    print "finding border and noise points"
    sys.stdout.flush()
    #core_border_list = []
    for j in range(len(mat)):
        if core_point_list[j]:
            core_border_list[j] = 0 #0 means it is a core point
            #core_border_list.append(0)
        else:
            for i in range(len(mat)):
                if j == i:
                    continue
                
                elif core_point_list[i] and within_eps(mat[j], mat[i], eps):
                    core_border_list[j] = 1 #1 means it is a border point
                    #core_border_list.append(1)
                    break
            #this else statement only executes if the whole forloop is iterated without
            #break being called, e.g. never determined to be a border point
            else: 
                core_border_list[j] = 2 #2 means it is a noise point
                #core_border_list.append(2)
    print "Done finding border and noise points"
    sys.stdout.flush()
    return core_border_list
        
        


# In[175]:


# def connect_cores(mat, unconnected_clusters, eps):
#     connected_core_clusters = {}
#     c_count = 0
#     #for each non comp unconnected cluster:
#     for n in range(len(unconnected_clusters)):
#         if not connected_core_clusters:
#             connected_core_clusters[c_count] = []
#             connected_core_clusters[c_count].append(unconnected_clusters[n])
#         #looking through elements of nth cluster
#         for i in range(len(unconnected_clusters[n])):
#             n_temp_cluster_key = int(unconnected_clusters[n][j])           
#             #for each comparison cluster except the nth one or anything before that
#             #will skip the last since in range last element not included for up to variable
#             for j in range(n+1, len(unconnected_clusters)):
#                 #skip if n == j
#                 if n == j:
#                     continue
#                 #looking through each element of the jth cluster/comparison cluster
#                 for k in range(len(unconnected_clusters[j])):
#                     j_temp_cluster_key = int(unconnected_clusters[j][k])
#                     if within_eps(mat[n_temp_cluster_key], mat[j_temp_cluster_key], eps):
#                         #add every element of kth cluster to connected_core_clusters
#                         connected_core_clusters[c_count].append(unconnected_clusters[j])
                            
                        
            


# In[176]:


def DBScan(mat, minpts, eps):
    clusters = core_points(mat, eps, minpts)
    return clusters


# In[196]:


eps = 7
minpts = 3
mat = reduced_X
clusters = DBScan(mat, minpts, eps)


# In[213]:


###-----exporting CLUSTERS to csv-----###
import pandas as pd
##THIS IS NOT PART OF MY DBSCAN ALGORITHM, THIS IS JUST EXPORTING TO PREDICTIONS FILE##
np_preds = np.zeros(len(reduced_X), dtype=np.int)
print(len(np_preds))
for i in range(len(clusters)):
    for n in range(len(clusters[i])):
        #get the index of each of element of every cluster(i)(n)
        #as well as the index of the cluster(i) to predict
        index_label = int(clusters[i][n])
        np_preds[index_label] = i
test_df = pd.DataFrame(np_preds)
test_df.to_csv("predict.dat", index = False, header = False)


# In[181]:


# #testing add border pts
# tclus = add_border_points(reduced_X, reduced_X_core_list, unconnected_clusters, test_eps)
# core_num = 0
# for key, value in tclus.items():
#     print(key, len([item for item in value if item]))
#     core_num += len([item for item in value if item])
# print core_num


# In[182]:


# #testing connect_comp
# unconnected_clusters = connect_comp(reduced_X, reduced_X_core_list, test_eps)

# core_num = 0
# for key, value in unconnected_clusters.items():
#     print(key, len([item for item in value if item]))
#     core_num += len([item for item in value if item])
# print core_num


# In[183]:


# #testing border, noise and core point finder using core_points and border_points
# #(intenal call)
# len(reduced_X)
# test_eps = 7
# test_min = 3
# reduced_X_core_list = core_points(reduced_X, test_eps, test_min)
# print(len(reduced_X_core_list))
# unique_items, counts = np.unique(reduced_X_core_list, return_counts=True)
# print(unique_items)
# print(counts)


# In[184]:


# #testing core_points
# test_matrix = [[2,4],[5,7],[9,10],[3,1]]
# test_eps = 7
# test_min = 3
# test_matrix_core_point_list = core_points(test_matrix, test_eps, test_min)

# print(len(test_matrix_core_point_list))
# print(test_matrix_core_point_list)


# In[185]:


# #testing distance def...
# t1 = np.array([2,4,6])
# t2 = np.array([3,6,9])

# t_dist = distance(t1,t2)
# print(t_dist)


# In[186]:


# #testing within_eps
# t1 = np.array([2,4,6])
# t2 = np.array([3,6,9])

# t_eps_check = within_eps(t1,t2, 3.0)
# print(t_eps_check)
# #eucl dist is ~3.71 so > 3.0 not in eps


# In[187]:


# #testing within_eps
# t1 = np.array([2,4,6])
# t2 = np.array([3,6,9])

# t_eps_check = within_eps(t1,t2, 4.0)
# print(t_eps_check)
# #eucl dist is ~3.71 so < 4.0 not in eps

