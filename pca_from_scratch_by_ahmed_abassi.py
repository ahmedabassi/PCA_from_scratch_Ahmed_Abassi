# -*- coding: utf-8 -*-
"""PCA from scratch_by Ahmed Abassi.ipynb

#Loading the Dataset

link for the dataset: https://drive.google.com/file/d/1hL-zgI69Agp_YADulY5zCdJ-_uSEBb68/view?usp=sharing
"""

from google.colab import drive
drive.mount("/content/drive",force_remount = True)

import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/images.csv")

"""#EDA"""

df.head(3)

df.shape

import math as m 
IMG_dim= int(m.sqrt(2304))
#our image is 48x48
IMG_dim

import numpy as np
image1=np.asarray(df.iloc[0])
image1=image1.reshape(IMG_dim,IMG_dim)

from matplotlib import pyplot as plt
plt.imshow(image1)
plt.show()

"""#normalizing and standarizing"""

#calculating means
df_center=df.copy()
means=[]
for col in df.columns:
  means.append(sum(df[col])/df.shape[0])
len(means)

#normalizing (might take some time :p)
for i in range(df.shape[1]):
  df_center.iloc[:,i] = df_center.iloc[:,i] - means[i]

"""Pixel values are from 0 to 255 so there is no need to scale them down (dividing by Standard deviation)

#Covariance matrix
"""

cov=np.matmul(np.transpose(df_center.values),df_center.values)

"""#SVD """

u, s, vh = np.linalg.svd(cov)

"""#sorting our eigenvalues and eigenvectors"""

def bubbleSort(u,s):
    n = len(s)
    for i in range(n):
        for j in range(0, n-i-1):
            if s[j] < s[j+1] :
                s[j], s[j+1] = s[j+1], s[j]
                u[:,j],u[:,j+1] = u[:,j+1],u[:,j]

"""#Applying PCA

Now we will apply PCA to obtain the first 10 principal components, report the proportion of
variance explained (PVE) for each of the principal components, reshape each of the principal component
to a 48x48 matrix and show them (eigenfaces).
"""

def get_PC(u,s,k):
  return u[:,k] , (s[k]/np.sum(s))*100

n=IMG_dim
sum=0
for i in range(10):
  plt.figure()
  image,PVE = get_PC(u,s,i)
  sum+=PVE
  print('PCA {a} has PVE of {b:.2f}%'.format(a=i+1,b=PVE))
  image=image.reshape(n,n)
  plt.title("PVE = {b:.2f}%".format(b=PVE))
  plt.imshow(image)
  path =str(i)+".png"
  #plt.savefig("/content/drive/MyDrive/pics/"+path)
  plt.show()

"""#Number of PCs used vs PVE

We will now obtain first k principal components and report PVE for k ∈ {1, 10, 50, 100, 500}.
"""

def plot():
  ks=[1, 10, 50, 100, 500]
  PVEs=[]
  for k in ks:
    sum=0
    for i in range(k):
      noNeed,PVE = get_PC(u,s,i)
      sum+=PVE
    PVEs.append(round(sum,2))

  plt.plot(ks,PVEs)
  plt.xlabel("Number of PCs used")
  plt.ylabel("PVEs")
  plt.title("Number of PCs used vs PVE")
  plt.show()

plot()

"""#Reconstructing Image

We will now reconstruct an image using the principal components we obtained previously. we will use first k principal components to analyze and reconstruct the first image in the
dataset where k ∈ [1, 10, 50, 100, 500].
"""

ks=[1, 10, 50, 100, 500]
image1=df.iloc[0,:]
def reconstruct(k): 
  new_image = np.array([0 for i in range(2304)])
  for i in range(k):
    PCA,PVE = get_PC(u,s,i)
    new_image = new_image + np.sum(np.dot(image1,PCA)) * PCA
  return new_image

from matplotlib import pyplot as plt
for k in ks:
  a = reconstruct(k)
  n=len(a)
  print("using {kk} eigenfaces to reconstruct the image".format(kk=k))
  plt.imshow(a.reshape(int(m.sqrt(n)),int(m.sqrt(n))))
  path =str(k)+".png"
  plt.savefig("/content/drive/MyDrive/pics/"+path)
  plt.show()

"""#PCA using scikit learn"""

from sklearn.preprocessing import StandardScaler
x = df.values
x = StandardScaler().fit_transform(x) # normalizing the features

np.mean(x),np.std(x)

new_df = pd.DataFrame(x,columns=df.columns)
new_df.head(5)

from sklearn.decomposition import PCA
pca_img = PCA(n_components=10)
principalComponents = pca_img.fit_transform(x)

principalComponents.shape

principalComponents

df.shape

for i in range(len(pca_img.explained_variance_ratio_)):
  print('Explained variance for principal component {i}: {e}'.format(i=i+1,e=pca_img.explained_variance_ratio_[i]))