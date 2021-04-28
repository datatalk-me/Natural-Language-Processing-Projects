#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:24:12 2021

@author: sandeep
"""


import pandas as pd
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])


# Data cleaning and preprocessing

import re
import nltk


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


wnl = WordNetLemmatizer()

corpus = []

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [wnl.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating Bag of words Model
    
from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf = TfidfVectorizer()
X = Tfidf.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

# Train Test Split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state=0)

# Training the model
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)
y_pred = spam_detect_model.predict(X_test)


# Confusion Matrix
from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(y_test,y_pred)

# checking the accuracy score

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,y_pred)
