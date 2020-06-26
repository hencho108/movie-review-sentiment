#http://www.corelangs.com/css/box/center-div.html
#https://community.plotly.com/t/horizontally-center-image/15253


import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from joblib import load

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import nltk
import re
import string
import time
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
stopwords = nltk.corpus.stopwords.words('english')

wn = nltk.WordNetLemmatizer()
def clean_text_tokenized(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove punctuation
    text = ''.join([word.lower() for word in text if word not in string.punctuation])
    # Tokenize
    tokens = re.split('\W+', text)
    # Stem
    text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    return text
pipeline = load('training/svc_pipeline.joblib')


external_stylesheets = [
    #"https://use.fontawesome.com/releases/v5.0.7/css/all.css",
    #'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'
    'https://stackpath.bootstrapcdn.com/bootswatch/4.5.0/flatly/bootstrap.min.css'
    #'https://fonts.googleapis.com/css?family=Roboto&display=swap'
]

#external_script = "https://raw.githubusercontent.com/MarwanDebbiche/post-tuto-deployment/master/src/dash/assets/gtag.js"

app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(
        id='page-content',
        style={
            'width':'80%', 
            'max-width':'500px',
            'margin':'0 auto',
            'margin-top':'10px',
            'padding':'5px',
            'height':'500px', 
            'border-style':'solid',
            'border-width':'thin'
        }
    )
]
)

home_layout = html.Div([
    html.Div(
        'Rate a Movie',
        id='title', 
        style={
            'border-style':'dashed',
            'border-width':'thin'
        }
    ),
    html.Div(
        [
            dcc.Textarea(
                className="form-control z-depth-1",
                id="review",
                rows="8",
                placeholder="Write something here..."
            ),
            dbc.Progress(
                id='progress',
                value=0,
                striped=False,
                color='success',
                style={
                    'margin-top':'5px'
                }
            )
        ],
        style={
            'border-style':'dashed',
            'border-width':'thin',
            'margin-top':'5px'
        }
    ),
    html.Div(
        [
            dbc.Button(
                'Info', 
                color='info', 
                className='mr-1'
            )
        ],
        style={
            'border-style':'dashed',
            'border-width':'thin',
            'margin-top':'5px'
        } 
    )
    ]
)


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/':
        return home_layout
    else:
        return [
            html.Div(
                [
                    html.Img(
                        src="./assets/404.png",
                        style={
                            "width": "50%"
                        }
                    ),
                ],
                className="form-review"
            ),
            dcc.Link("Go to Home", href="/"),
        ]

@app.callback(
    [Output('progress','value'), Output('progress','children')],
    [Input('review','value')]
)
def update_progress(review):
    if review is None:
        bar_value = 0
        pred_sentiment = 0
    else:
        pred_sentiment = pipeline.predict_proba([review])[0][1]
        #pred_sentiment = round(pred_sentiment * 100, 1)
        #pred_sentiment_text = f'{pred_sentiment}%'
        bar_value = pred_sentiment * 100
    return bar_value, pred_sentiment


if __name__ == '__main__':
    app.run_server(debug=True)