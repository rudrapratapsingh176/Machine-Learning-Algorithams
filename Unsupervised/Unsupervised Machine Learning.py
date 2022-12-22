#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[50]:


data=pd.read_csv('https://raw.githubusercontent.com/training-ml/Files/main/Mall_Customers.csv')
data.head()


# In[12]:


x=data[['Annual Income (k$)','Spending Score (1-100)']]


# In[13]:


WCSS=[]
for i in range(1,11):
    
    kmeans=KMeans(n_clusters=i,random_state=4334)
    kmeans.fit(x)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1,11),WCSS)
plt.title('The Elbow Method')
plt.xlabel('No. of cluster')
plt.ylabel('WCSS')
plt.show()


# In[14]:


Kmeans=KMeans(n_clusters=5,random_state=3455)
y_means=Kmeans.fit_predict(x)
print(y_means)


# In[16]:


from sklearn.metrics import silhouette_score


# In[18]:


silhouette_score(x,y_means)*100


# In[20]:


test=Kmeans.predict(np.asarray([[15,39]]))
test[0]


# In[21]:


from sklearn.cluster import MiniBatchKMeans


# In[23]:


minibatch=MiniBatchKMeans(n_clusters=5)
y=minibatch.fit_predict(x)


# In[25]:


silhouette_score(x,y)*100


# In[26]:


plt.figure(figsize=(15,20))
plt.scatter(x[y_means==0]['Annual Income (k$)'],x[y_means==0]['Spending Score (1-100)'])
plt.scatter(x[y_means==1]['Annual Income (k$)'],x[y_means==1]['Spending Score (1-100)'])
plt.scatter(x[y_means==2]['Annual Income (k$)'],x[y_means==2]['Spending Score (1-100)'])
plt.scatter(x[y_means==3]['Annual Income (k$)'],x[y_means==3]['Spending Score (1-100)'])
plt.scatter(x[y_means==4]['Annual Income (k$)'],x[y_means==4]['Spending Score (1-100)'])
plt.scatter(Kmeans.cluster_centers_[:,0],Kmeans.cluster_centers_[:,1])
plt.title('Cluster of coustumers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[28]:


cluster_4_custumer=data[y_means==4]
print(cluster_4_custumer)


# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[30]:


dataset=pd.read_csv('https://raw.githubusercontent.com/training-ml/Files/main/Mall_Customers.csv')
dataset.head()


# In[31]:


x=dataset[['Annual Income (k$)','Spending Score (1-100)']]


# In[32]:


import scipy.cluster.hierarchy as sch


# In[38]:


plt.figure(figsize=(20,15))
dendo=sch.dendrogram(sch.linkage(x,method='complete'))
plt.title('Dendogram')
plt.xlabel('Custumer data')
plt.ylabel('Eucl Distance')
plt.show()


# In[40]:


from sklearn.cluster import AgglomerativeClustering


# In[42]:


group=AgglomerativeClustering(n_clusters=3)
cluster=group.fit_predict(x)
print(cluster)


# In[45]:


silhouette_score(x,cluster)*100


# In[46]:


cluster_1=dataset[cluster==0]
cluster_2=dataset[cluster==1]
cluster_3=dataset[cluster==2]


# In[47]:


print(cluster_2)


# In[49]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import DBSCAN


# In[51]:


data_1=pd.read_csv('https://raw.githubusercontent.com/training-ml/Files/main/Mall_Customers.csv')


# In[53]:


x=data_1[['Annual Income (k$)','Spending Score (1-100)']]


# In[54]:


x=StandardScaler().fit_transform(x)


# In[62]:


db=DBSCAN(eps=0.25,min_samples=3).fit(x)


# In[63]:


labels=db.labels_
print(labels)


# In[64]:


len(set(labels))


# In[65]:


(1 if -1 in labels else 0)


# In[66]:


n_clusters=len(set(labels))-(1 if -1 in labels else 0)
print(n_clusters)


# In[67]:


n_noise=list(labels).count(-1)
print(n_noise)


# In[ ]:




