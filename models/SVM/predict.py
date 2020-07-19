from models.SVM.utils import clean_text_tokenized, ps, stopwords
from joblib import load
import os 

pipe_dir = os.path.join(os.path.dirname(__file__), 'pipelines', 'svm_pipe.joblib')

svm_pipe = load(pipe_dir)

def svm_make_prediction(text):
    pred = svm_pipe.predict_proba([text])[0][1]
    return pred