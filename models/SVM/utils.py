import pandas as pd 
import nltk
import re
import string
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def clean_text_tokenized(text):

    ps = nltk.PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove punctuation
    text = ''.join([word.lower() for word in text if word not in string.punctuation])
    # Tokenize
    tokens = re.split(r'\W+', text)
    tokens = list(filter(None,tokens))
    # Stem
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text