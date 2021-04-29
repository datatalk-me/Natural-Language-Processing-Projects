#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:07:17 2021

@author: sandeep
"""


import pandas as pd

df = pd.read_csv('Data.csv')


train = df[df['Date']<'20150101']
test = df[df['Date']>'20141231']


# Removing Punctuations

data = train.iloc[:,2:27]
data.replace("[^a-zA-z]"," ",regex = True,inplace=True)

# Remaining column names for ease of access

list1 = [i for i in range(25)]

new_index = [str(i) for i in list1]

data.columns = new_index
data.head()


# Converting HeadLines into lowercase

for index in new_index:
    data[index] = data[index].str.lower()
    
    
# Combining each row into a paragraph
    
headlines= []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# implement bag of words
countvector = CountVectorizer(ngram_range=(2,2))
traindataset = countvector.fit_transform(headlines)

# implement Random Forest Classifier
randomclassifier = RandomForestClassifier(n_estimators = 200,criterion = 'entropy')
randomclassifier.fit(traindataset,train['Label'])

# Predict for the test dataset

test_transform = []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
    test_dataset = countvector.transform(test_transform)
    predictions = randomclassifier.predict(test_dataset)
    
# import library to check accuracy
    
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix = confusion_matrix(test['Label'],predictions)
print(matrix)

score =  accuracy_score(test['Label'],predictions)
print(score)

report = classification_report(test['Label'],predictions)
print(report)