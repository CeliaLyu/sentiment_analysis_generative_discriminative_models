import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import os
import warnings
import os.path
import pickle

"""
This part of code refer to:
https://medium.com/@pyashpq56/sentiment-analysis-on-imdb-movie-review-d004f3e470bd
"""

imdb_data=pd.read_csv('IMDBDataset.csv')
print(imdb_data.shape)
print(imdb_data.head(10))

# Text normalization
#Tokenization of text
tokenizer = ToktokTokenizer()
#Setting English stopwords
nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')

# Removing html strips and noise text
#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()
#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)
#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

# Removing special characters
#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

# Text stemming
#Stemming the text
def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

# #Apply function on review column
# imdb_data['review']=imdb_data['review'].apply(denoise_text)
# #Apply function on review column
# imdb_data['review']=imdb_data['review'].apply(remove_special_characters)
# #Apply function on review column
# imdb_data['review'] = imdb_data['review'].apply(simple_stemmer)

# Removing stopwords and normalization
#set stopwords to english
stop = set(stopwords.words('english'))

print(stop)

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

if os.path.exists('processed_imdb.pkl'):
    with open('processed_imdb.pkl', 'rb') as f:
        imdb_data['review'] = pickle.load(f)
else:
    # Apply function on review column
    imdb_data['review'] = imdb_data['review'].apply(denoise_text)
    # Apply function on review column
    imdb_data['review'] = imdb_data['review'].apply(remove_special_characters)
    # Apply function on review column
    imdb_data['review'] = imdb_data['review'].apply(simple_stemmer)
    # Apply function on review column
    imdb_data['review'] = imdb_data['review'].apply(remove_stopwords)

    with open('processed_imdb.pkl', 'wb+') as f:
        pickle.dump(imdb_data['review'], f, protocol=pickle.HIGHEST_PROTOCOL)

#normalized train reviews
norm_train_reviews = imdb_data.review[:40000]
print(norm_train_reviews[0])

#Normalized test reviews
norm_test_reviews = imdb_data.review[40000:]
print(norm_test_reviews[45005])

# Bag of words Model
#Count vectorizer for bag of words
cv=CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 3))
#transformed train reviews
cv_train_reviews = cv.fit_transform(norm_train_reviews)
#transformed test reviews
cv_test_reviews = cv.transform(norm_test_reviews)

print("-------------------------------")
print("cv_train_reviews[0]:")
print(type(cv_train_reviews[0]))
print(cv_train_reviews[0].shape)
cv_train_reviews_array0 = cv_test_reviews[0].toarray()
print(cv_train_reviews_array0)
print(cv_train_reviews_array0.shape)
# print(cv.vocabulary_)
# print("cv.decode(cv_train_reviews[0]):")
# print(cv_train_reviews[0].decode())
print("-------------------------------")

print('BOW_cv_train:', cv_train_reviews.shape)
print('BOW_cv_test:', cv_test_reviews.shape)
#vocab=cv.get_feature_names()-toget feature names

#labeling the sentient data
lb = LabelBinarizer()
#transformed sentiment data
sentiment_data = lb.fit_transform(imdb_data['sentiment'])
print(sentiment_data.shape)

#Spliting the sentiment data
train_sentiments = sentiment_data[:40000]
test_sentiments = sentiment_data[40000:]
print(train_sentiments)
print(test_sentiments)

#training the model
mnb = MultinomialNB()
#fitting the svm for bag of words
mnb_bow = mnb.fit(cv_train_reviews, train_sentiments)
print(mnb_bow)

print("---------------------------")
print(mnb_bow.get_params())
print(mnb_bow.class_count_)
print(mnb_bow.class_log_prior_)
print(mnb_bow.classes_)
print(mnb_bow.feature_count_)
print(mnb_bow.feature_count_[:,:10])
class0_idx = np.where(mnb_bow.feature_count_[0] != 0)[0]
class1_idx = np.where(mnb_bow.feature_count_[1] != 0)[0]

reviews_num = 100
review1_list = []
sentiment1_list = ["negative"]*reviews_num
for i in range(reviews_num):
    length = np.random.randint(10, 31)
    class1_values = np.random.choice(class1_idx, size=length, replace=False)
    key_list = list(cv.vocabulary_.keys())
    val_list = list(cv.vocabulary_.values())
    review1 = ''
    for value1 in class1_values:
        position = val_list.index(value1)
        review1 = review1 + key_list[position]
    # print(review1)
    review1_list.append(review1)

cv_syn_review1 = cv.transform(review1_list)
print(mnb.predict(cv_syn_review1))

data1 = {
  "review": review1_list,
  "sentiment": sentiment1_list
}
df = pd.DataFrame(data1)
# os.makedirs('folder/subfolder', exist_ok=True)
df.to_csv('class1.csv', index=False)


print(class0_idx)
print(class0_idx.shape)
print(class1_idx)
print(class1_idx.shape)
print(mnb_bow.feature_log_prob_)
print(mnb_bow.n_features_in_)
print("---------------------------")

#Predicting the model for bag of words
mnb_bow_predict = mnb.predict(cv_test_reviews)
print(mnb_bow_predict)

#Accuracy score for bag of words
mnb_bow_score = accuracy_score(test_sentiments, mnb_bow_predict)
print("mnb_bow_score :", mnb_bow_score)

#Classification report for bag of words
mnb_bow_report = classification_report(test_sentiments, mnb_bow_predict, target_names=['Positive', 'Negative'])
print(mnb_bow_report)

#confusion matrix for bag of words
cm_bow = confusion_matrix(test_sentiments, mnb_bow_predict, labels=[1, 0])
print(cm_bow)


# test synthetic data
print("-----------sssssssssssssssss-------------")
synth_data = pd.read_csv('class1.csv')
norm_synth_reviews = synth_data.review
#transformed test reviews
cv_synth_reviews = cv.transform(norm_synth_reviews)

# #labeling the sentient data
# lb = LabelBinarizer()
# #transformed sentiment data
# sync_sentiment_data = lb.fit_transform(synth_data['sentiment'])

#Predicting the model for bag of words
mnb_bow_sync_predict = mnb.predict(cv_synth_reviews)
print(mnb_bow_sync_predict)

sync_sentiment_data = np.ones((reviews_num, 1), dtype=np.int8)

#Accuracy score for bag of words
mnb_bow_sync_score = accuracy_score(sync_sentiment_data, mnb_bow_sync_predict)
print("mnb_bow_score :", mnb_bow_sync_score)
