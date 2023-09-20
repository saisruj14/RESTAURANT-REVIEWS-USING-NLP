#Importing essential libraries
import numpy as np
import pandas as pd
#Loading the data
data = pd.read_csv('/content/drive/MyDrive/Restaurant_Reviews.tsv',delimiter='\t',quoting = 3)
#Importing essential libraries and downloading packages
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range (0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1500)
X = tfidf_vectorizer.fit_transform(corpus).toarray()
y = data.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
#Model training using SVM Algorithm and PCA reduction for dimension
from sklearn.svm import SVC
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)

classifier = SVC(kernel='linear', C=1.0, random_state=0)
classifier.fit(X_train_pca, y_train)

X_test_pca = pca.transform(X_test)

y_pred = classifier.predict(X_test_pca)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

X_test_pca = pca.transform(X_test)

# Use the trained SVM classifier to make predictions on the PCA-transformed test data
y_pred = classifier.predict(X_test_pca)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm
classifier = SVC(kernel='linear', C=0.1, random_state=0) 

classifier.fit(X_train, y_train)
def predict_sentiment(sample_review):
  sample_review = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_review)
  sample_review = sample_review.lower()
  sample_review_words = sample_review.split()
  sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
  ps = PorterStemmer()
  final_review = [ps.stem(word) for word in sample_review_words]
  final_review = ' '.join(final_review)

  temp = tfidf_vectorizer.transform([final_review]).toarray()
  return classifier.predict(temp)


