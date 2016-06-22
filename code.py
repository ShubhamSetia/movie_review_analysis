# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:30:41 2016

@author: shubham
"""
import pandas as pd  #importing pandas
from bs4 import BeautifulSoup #importing beautiful soup to clean the data
import re
import nltk
from nltk.corpus import stopwords
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
#test
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
    
    