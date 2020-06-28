import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model   

import pickle
from joblib import dump, load


if __name__ == '__main__':
    
    # demo

    # load model
    model = load_model("rnn_model.h5")
    
    # load text_transformer
    text_transformer = load("text_transformer.joblib")

    

    text = 'awesome movie, but somehwere I got lost'

    sequence = text_transformer.transform(text)
    
    V = len(tokenizer.word_index) # number of unique tokens
    T = 1 # sequence length
    D = 32 # Embedding dimensionality
    M = 10 # Hidden state dimensionality

    pred = model.predict(sequence)
    
    print(pred)


