#!/usr/bin/env python
# coding: utf-8

# # Clustering Geolocation Data Intelligently in Python
# We have taxi rank locations, and want to define key clusters of these taxis where we can build service stations for all taxis operating in that region.
#
# ## Prerequisites
# - Basic Matplotlib skills for plotting 2-D data clearly.
# - Basic understanding of Pandas and how to use it for data manipulation.
# - The concepts behind clustering algorithms, although we will go through this throughout the project.
#
# ## Project Outline
#
# [**Task 1**](#task1): Exploratory Data Analysis
#
# [**Task 2**](#task2): Visualizing Geographical Data
#
# [**Task 3**](#task3): Clustering Strength / Performance Metric
#
# [**Task 4**](#task4): K-Means Clustering
#
# [**Task 5**](#task5): DBSCAN
#
# [**Task 6**](#task6): HDBSCAN
#
# [**Task 7**](#task7): Addressing Outliers
#
# [**Further Reading**](#further)

# In[1]:


from flask import Flask,render_template,url_for,request
import matplotlib
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import pandas as pd
import numpy as np
from flask import Flask
from tqdm import tqdm

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

from ipywidgets import interactive

from collections import defaultdict
#

import hdbscan
import folium
import re


cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
        '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
        '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
        '#000075', '#808080']*10


# In[4]:


# Task 1: Exploratory Data Analysis

app=Flask(__name__)

# In[5]:


df = pd.read_csv('Data/taxi_data.csv')


# In[6]:


df.head()


# In[7]:


df.duplicated(subset=['LON', 'LAT']).values.any()


# In[8]:


df.isna().values.any()


# In[9]:


print(f'Before dropping NaNs and dupes\t:\tdf.shape = {df.shape}')
df.dropna(inplace=True)
df.drop_duplicates(subset=['LON', 'LAT'], keep='first', inplace=True)
print(f'After dropping NaNs and dupes\t:\tdf.shape = {df.shape}')


# In[ ]:


df.head()


# In[11]:


X = np.array(df[['LON', 'LAT']], dtype='float64')


# In[12]:


plt.scatter(X[:,0], X[:,1], alpha=0.2, s=50)


# <a id='task2'></a>
# # Task 2: Visualizing Geographical Data
#

# In[13]:


m = folium.Map(location=[df.LAT.mean(), df.LON.mean()], zoom_start=9,
               tiles='Stamen Toner')

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row.LAT, row.LON],
        radius=5,
        popup=re.sub(r'[^a-zA-Z ]+', '', row.NAME),
        color='#1787FE',
        fill=True,
        fill_colour='#1787FE'
    ).add_to(m)


# In[20]:


m




# <a id='task3'></a>
# # Task 3: Clustering Strength / Performance Metric

# In[ ]:


X_blobs, _ = make_blobs(n_samples=1000, centers=10, n_features=2,
                        cluster_std=0.5, random_state=4)


# In[ ]:


plt.scatter(X_blobs[:,0], X_blobs[:,1], alpha=0.2)


# In[ ]:


class_predictions = np.load('Data/sample_clusters.npy')


# In[ ]:


unique_clusters = np.unique(class_predictions)
for unique_cluster in unique_clusters:
    X = X_blobs[class_predictions==unique_cluster]
    plt.scatter(X[:,0], X[:,1], alpha=0.2, c=cols[unique_cluster])


# In[ ]:


silhouette_score(X_blobs, class_predictions)


# In[ ]:


class_predictions = np.load('Data/sample_clusters_improved.npy')
unique_clusters = np.unique(class_predictions)
for unique_cluster in unique_clusters:
    X = X_blobs[class_predictions==unique_cluster]
    plt.scatter(X[:,0], X[:,1], alpha=0.2, c=cols[unique_cluster])


# In[ ]:


silhouette_score(X_blobs, class_predictions)


# <a id='task4'></a>
# # Task 4: K-Means Clustering

# In[ ]:


X_blobs, _ = make_blobs(n_samples=1000, centers=50,
                        n_features=2, cluster_std=1, random_state=4)


# In[ ]:


data = defaultdict(dict)
for x in range(1,21):
    model = KMeans(n_clusters=3, random_state=17,
                   max_iter=x, n_init=1).fit(X_blobs)

    data[x]['class_predictions'] = model.predict(X_blobs)
    data[x]['centroids'] = model.cluster_centers_
    data[x]['unique_classes'] = np.unique(class_predictions)


# In[ ]:


def f(x):
    class_predictions = data[x]['class_predictions']
    centroids = data[x]['centroids']
    unique_classes = data[x]['unique_classes']

    for unique_class in unique_classes:
            plt.scatter(X_blobs[class_predictions==unique_class][:,0],
                        X_blobs[class_predictions==unique_class][:,1],
                        alpha=0.3, c=cols[unique_class])
    plt.scatter(centroids[:,0], centroids[:,1], s=200, c='#000000', marker='v')
    plt.ylim([-15,15]); plt.xlim([-15,15])
    plt.title('How K-Means Clusters')

interactive_plot = interactive(f, x=(1, 20))
output = interactive_plot.children[-1]
output.layout.height = '350px'
interactive_plot


# In[ ]:


X = np.array(df[['LON', 'LAT']], dtype='float64')
k = 70
model = KMeans(n_clusters=k, random_state=17).fit(X)
class_predictions = model.predict(X)
df[f'CLUSTER_kmeans{k}'] = class_predictions


# In[ ]:


df.head()


# In[ ]:


def create_map(df, cluster_column):
    m = folium.Map(location=[df.LAT.mean(), df.LON.mean()], zoom_start=9, tiles='Stamen Toner')

    for _, row in df.iterrows():

        if row[cluster_column] == -1:
            cluster_colour = '#000000'
        else:
            cluster_colour = cols[row[cluster_column]]

        folium.CircleMarker(
            location= [row['LAT'], row['LON']],
            radius=5,
            popup= row[cluster_column],
            color=cluster_colour,
            fill=True,
            fill_color=cluster_colour
        ).add_to(m)

    return m

m = create_map(df, 'CLUSTER_kmeans70')
print(f'K={k}')
print(f'Silhouette Score: {silhouette_score(X, class_predictions)}')

m.save('kmeans_70.html')
print(m)

# In[ ]:


m


# In[ ]:


best_silhouette, best_k = -1, 0

for k in tqdm(range(2, 100)):
    model = KMeans(n_clusters=k, random_state=1).fit(X)
    class_predictions = model.predict(X)

    curr_silhouette = silhouette_score(X, class_predictions)
    if curr_silhouette > best_silhouette:
        best_k = k
        best_silhouette = curr_silhouette

print(f'K={best_k}')
print(f'Silhouette Score: {best_silhouette}')


# <a id='task5'></a>
# # Task 5: DBSCAN
# Density-Based Spatial Clustering of Applications with Noise

# In[ ]:


# code for indexing out certain values
dummy = np.array([-1, -1, -1, 2, 3, 4, 5, -1])

new = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(dummy)])


# In[ ]:


model = DBSCAN(eps=0.01, min_samples=5).fit(X)
class_predictions = model.labels_

df['CLUSTERS_DBSCAN'] = class_predictions


# In[ ]:


m = create_map(df, 'CLUSTERS_DBSCAN')


print(f'Number of clusters found: {len(np.unique(class_predictions))}')
print(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')

print(f'Silhouette ignoring outliers: {silhouette_score(X[class_predictions!=-1], class_predictions[class_predictions!=-1])}')

no_outliers = 0
no_outliers = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(class_predictions)])
print(f'Silhouette outliers as singletons: {silhouette_score(X, no_outliers)}')


# In[ ]:





# <a id='task6'></a>
# # Task 6: HDBSCAN
# Hierarchical DBSCAN

# In[ ]:


model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2,
                        cluster_selection_epsilon=0.01)
#min_cluster_size
#min_samples
#cluster_slection_epsilon

class_predictions = model.fit_predict(X)
df['CLUSTER_HDBSCAN'] = class_predictions


# In[ ]:


m = create_map(df, 'CLUSTER_HDBSCAN')

print(f'Number of clusters found: {len(np.unique(class_predictions))-1}')
print(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')

print(f'Silhouette ignoring outliers: {silhouette_score(X[class_predictions!=-1], class_predictions[class_predictions!=-1])}')

no_outliers = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(class_predictions)])
print(f'Silhouette outliers as singletons: {silhouette_score(X, no_outliers)}')




# In[ ]:


#get_ipython().run_line_magic('pinfo', 'hdbscan.HDBSCAN')


# <a id='task7'></a>
# # Task 7: Addressing Outliers
#

# In[ ]:


classifier = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


df_train = df[df.CLUSTER_HDBSCAN!=-1]
df_predict = df[df.CLUSTER_HDBSCAN==-1]


# In[ ]:


X_train = np.array(df_train[['LON', 'LAT']], dtype='float64')
y_train = np.array(df_train['CLUSTER_HDBSCAN'])

X_predict = np.array(df_predict[['LON', 'LAT']], dtype='float64')


# In[ ]:


classifier.fit(X_train, y_train)


# In[ ]:


predictions = classifier.predict(X_predict)


# In[ ]:


df['CLUSTER_hybrid'] = df['CLUSTER_HDBSCAN']


# In[ ]:


df.loc[df.CLUSTER_HDBSCAN==-1, 'CLUSTER_hybrid'] = predictions


# In[ ]:


m = create_map(df, 'CLUSTER_hybrid')


# In[ ]:


m


# In[ ]:


class_predictions = df.CLUSTER_hybrid
print(f'Number of clusters found: {len(np.unique(class_predictions))}')
print(f'Silhouette: {silhouette_score(X, class_predictions)}')

m.save('templates/hybrid.html')






# In[ ]:


df['CLUSTER_hybrid'].value_counts().plot.hist(bins=70, alpha=0.4,
                                              label='Hybrid')
df['CLUSTER_kmeans70'].value_counts().plot.hist(bins=70, alpha=0.4,
                                               label='K-Means (70)')
plt.legend()
plt.title('Comparing Hybrid and K-Means Approaches')
plt.xlabel('Cluster Sizes')


# In[ ]:
@app.route("/")
def welcome():
    return render_template("project.html")


    
@app.route("/output")
def intiate():
    return render_template("ProjectOutcome1.html")



if __name__=="__main__":
    app.run(debug=True)



# <a id='further'></a>
# # Further Reading
#
# For some additional reading, feel free to check out [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), and [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/) clustering respectively.
#
# It may be of use to also check out [other forms of clustering](https://scikit-learn.org/stable/modules/clustering.html) that are commonly used and available in the scikit-learn library. HDBSCAN documentation also includes [a good methodology](https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html) for choosing your clustering algorithm based on your dataset and other limiting factors.

# In[ ]:

