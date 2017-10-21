import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from stemming.porter2 import stem
import pickle
#function to remove the html tags, punctuation marks, removing stopwords and returning the clean review
def clean_review(raw_review):
	#removing html tags
	text_review = BeautifulSoup(raw_review,"lxml").get_text()
	#removing punctuations
	letters_only = re.sub("[^a-z,A-Z]"," ",text_review)
	#converting letters to lower case
	lower_case = letters_only.lower()
	#spliting the letters into words
	words = lower_case.split()
	#storing stopwords in a set for faster processing
	stops = set(stopwords.words("english"))
	#removing stopwords
	useable_words = [w for w in words if not w in stops]
	#stemming the review.
	stemmed_words=[stem(word) for word in useable_words]
	#joining words back to get the clean review string and returning it
	return " ".join(stemmed_words)


filename = 'finalized_model.sav'
# load the model from disk
classifier = BernoulliNB()
classifier= pickle.load(open(filename, 'rb'))
vectorizer=CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)
                             
filename1 = 'finalized_vector.sav'
vectorizer= pickle.load(open(filename1, 'rb'))
review = raw_input("Enter the review > ")
#print review
test=[]
test.append(clean_review(review))
test_data_features = vectorizer.transform(test)
test_data_features = test_data_features.toarray()
result = classifier.predict(test_data_features)
print result
	
