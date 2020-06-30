from tensorflow.keras.models import load_model   
from models.RNN.transformers import RemoveHTML, TextToSequence
import pickle
from joblib import dump, load

# load tokenizer

# load text_transformer
text_transformer = load("text_transformer.joblib")

# load model
model = load_model("rnn_model.h5")

# demo text
#text = 'awful movie'

def rnn_make_predictions(text):

    # transform text
    sequence = text_transformer.transform(text)
    # make prediction
    pred = model.predict(sequence)[0][0]
    print(pred)
    
    return pred


if __name__ == '__main__':
    rnn_make_predictions(text)