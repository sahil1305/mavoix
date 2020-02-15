# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 02:19:16 2020

@author: sahil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

job= pd.read_excel("H:\\all datasets\\mavoix_ml_sample_dataset.xlsx")

#required column for the model building 
ds_wd = job.iloc[:,[2,3,4,5,6,7,8,9,11,12,14,15]]

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ds_wd)

k = list(range(2,15))

TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Selecting 2 clusters
model=KMeans(n_clusters=2) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
#1 is for web development and 0 is for data science
job['ds_wd']=md

job.ds_wd.value_counts() #192 applied for Web Development and 200 applied for Data Science
