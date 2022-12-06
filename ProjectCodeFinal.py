#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install xgboost')


# #### Importing all the required libraries

# In[2]:


import itertools
import pickle
import random
import matplotlib
import math
import copy
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, plot_precision_recall_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter


# #### Reading dataset path and loading images

# In[3]:


print("Loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images("./data/training/")))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath, 0)
    image = cv2.resize(image, (40, 40))
    image = np.reshape(image, 1600)
    data.append(image)

    label = imagePath[-7:-4]
    if label == "pos":
        label = 1
    else:
        label = 0
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


# #### Displaying array sample

# In[4]:


# displaying image array
print(data[:4])

# displaying labels
print(labels[:4])


# #### Displaying training image

# In[5]:


for i, images in enumerate(imagePaths[:4]):
    img = cv2.imread(images)
    img = cv2.resize(img, (100, 100))
    plt.subplot(2, 2, i + 1)
    plt.title(labels[i])
    plt.imshow(img)
    plt.grid(False)
    plt.axis('off')
plt.show()


# #### Splitting dataset into train-test

# In[6]:


trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=3)


# In[7]:


trainX.shape, testX.shape


# #### NCA-XGBoosting

# In[16]:


dim = len(trainX[0])
n_classes = len(np.unique(trainY))


# In[17]:


nca = make_pipeline(
    StandardScaler(),
    NeighborhoodComponentsAnalysis(n_components=2, random_state=3),
)


# In[18]:


xgb = XGBClassifier(n_estimators=3)


# In[19]:


nca.fit(trainX, trainY)


# In[20]:


xgb.fit(nca.transform(trainX), trainY)


# In[21]:


print("Accuracy score -->" ,accuracy_score(xgb.predict(nca.transform(testX)), testY))


# In[22]:


print(classification_report(testY, xgb.predict(nca.transform(testX))))


# In[23]:


confusion_matrix(testY, xgb.predict(nca.transform(testX)))


# In[24]:


plot_confusion_matrix(estimator=xgb, X=nca.transform(testX), y_true=testY, cmap="summer_r")
plt.show()


# In[25]:


plot_precision_recall_curve(estimator=xgb, X=nca.transform(testX), y=testY)
plt.show()


# #### KNN Classifier

# In[26]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[27]:


knn.fit(trainX, trainY)


# In[28]:


print("Accuracy score -->" ,accuracy_score(knn.predict(testX), testY))


# In[29]:


print(classification_report(testY, knn.predict(testX)))


# In[30]:


confusion_matrix(testY, knn.predict(testX))


# In[31]:


plot_confusion_matrix(estimator=knn, X=testX, y_true=testY, cmap="summer_r")
plt.show()


# In[32]:


plot_precision_recall_curve(estimator=knn, X=testX, y=testY)
plt.show()


# #### Adaboost Classifier

# In[33]:


ada = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1.0,
                         algorithm='SAMME.R')


# In[34]:


ada.fit(trainX, trainY)


# In[35]:


print("Accuracy score -->" ,accuracy_score(ada.predict(testX), testY))


# In[36]:


print(classification_report(testY, ada.predict(testX)))


# In[37]:


confusion_matrix(testY, ada.predict(testX))


# In[38]:


plot_confusion_matrix(estimator=ada, X=testX, y_true=testY, cmap="summer_r")
plt.show()


# In[39]:


plot_precision_recall_curve(estimator=ada, X=testX, y=testY)
plt.show()

