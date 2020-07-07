import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.preprocessing import FunctionTransformer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

pickle_dir = os.path.join(os.path.dirname(__file__), 'tokenizer.pickle')


with open(pickle_dir, 'rb') as handle:
    tokenizer = pickle.load(handle)

def RemoveHTML(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    return text

def TextToSequence(text):
    # loading
    text = pd.Series(text)
    text = pad_sequences(tokenizer.texts_to_sequences((pd.Series(text))),maxlen=500)
    return text
