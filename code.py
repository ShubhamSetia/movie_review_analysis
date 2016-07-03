# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:30:41 2016

@author: shubham
"""
import pandas as pd  #importing pandas
from bs4 import BeautifulSoup #importing beautiful soup to clean the data
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer 
import numpy as np
from sklearn.naive_bayes import BernoulliNB
#from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

def review_to_words(raw_review):
    """Function to  convert raw review to string of words.
    Takes a single string as input.
    Cleans it using beutiful soup .
    Removes punctuations. Converts them to lower case string and 
    splits them. returns a single string."""
    #remove html tags
    review_text=BeautifulSoup(raw_review,"lxml").get_text()
    
    #removing non-letters
    letters_only=re.sub("[^a-zA-Z]"," ", review_text)
    
    #converting the letters to lower case
    lower_case=letters_only.lower()
    
    #spliting lower case letters
    words=lower_case.split()
    
    #converting english stopwords list to set
    stops=set(stopwords.words("english"))
   
    #removing stop words
    processed_review=[w for w in words if not w in stops]
   
    #joining words back into one string separated by space , and returning the result
    return " ".join(processed_review)

#clean_review = review_to_words( train["review"][0] )
#print clean_review   

#storing number of reivews
num_reviews=train["review"].size

print "Cleaning and parsing the training set movie reviews...\n"
#initializing empty list to store reviews
clean_train_review=[]

#looping over all the reivews and cleaning them
for i in xrange(0,num_reviews):
    #printing message after multiple of 1000 reviews are cleaned
    if i%1000==0:
        print "Review %d of %d" %(i+1,num_reviews)
    clean_train_review.append(review_to_words(train["review"][i]))

    
print "Creating Bag of Words............ \n "   
#initializing countvectorizer object
#vectorizer=CountVectorizer(binary='true')
#creating feature vector using fit_transform
#train_data_features=vectorizer.fit_transform(clean_train_review)
#converting the above into an array
train_data_features=TfidfVectorizer().fit_transform(clean_train_review)
train_data_features=np.array(train_data_features)
print train_data_features

#train_data_features=train_data_features.toarray()
#print train_data_features.shape
#print train_data_features
#storing feature vector
#vocab=vectorizer.get_feature_names()
#print vocab
#dist=np.sum(train_data_features, axis=0)
"""for tag,count in zip(vocab,dist):
    print count,tag
"""
"""
classifier = BernoulliNB().fit(train_data_features,[0,1])
predicted = classifier.predict(["good","nice"])
print predicted
"""
