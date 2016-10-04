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
	#joining words back to get the clean review string and returning it
	return " ".join(useable_words)



#loading the review file
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
#number of reviews
num_review=train["review"].size
print "Cleaning and preparing review training set reviews for further processing.........."
#declaring empty list to store processed reviews
clean_train_review=[]
#using loop to clean all the reviews and store them in  the list
for i in xrange(0,num_review):
    #printing message after multiple of 1000 reviews are cleaned
    if i%1000==0:
        print "Review %d of %d" %(i+1,num_review)
    clean_train_review.append(clean_review(train["review"][i]))

print "Training set reviews are cleaned and ready for further processing\n"
print "Creating Bag of Words .................\n"
#initializing count vectorizer object
vectorizer=CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)
train_data_features=vectorizer.fit_transform(clean_train_review)
#train_data_features=TfidfTransformer(use_idf="false").fit_transform(train_data_features)
train_data_features=train_data_features.toarray()
print train_data_features.shape
forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(train_data_features,train["sentiment"])
# Read the test data
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )



# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = []

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)

    clean_test_reviews.append( clean_review( test["review"][i] ) )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
print "Output copied to file."
