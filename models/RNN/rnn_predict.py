from tensorflow.keras.models import load_model   
from models.RNN.transformers import RemoveHTML, TextToSequence
import pickle
from joblib import dump, load
import os

# load tokenizer
text_transformer_dir = os.path.join(os.path.dirname(__file__), 'text_transformer.joblib')
rnn_model_dir = os.path.join(os.path.dirname(__file__), 'rnn_model.h5')

# load text_transformer
text_transformer = load(text_transformer_dir)

# load model
model = load_model(rnn_model_dir)

# demo text
#text = 'awful movie'

def rnn_make_predictions(text):

    # transform text
    sequence = text_transformer.transform(text)
    # make prediction
    pred = model.predict(sequence)[0][0]
    
    return pred


if __name__ == '__main__':
    rnn_make_predictions(text)