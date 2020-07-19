import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import pickle
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
#from flask import Flask, request, jsonify
#from models.SVM.utils import clean_text_tokenized
#from models.SVM.predict import svm_make_prediction
import pandas as pd
import random


########################
import nltk
import os
from joblib import load
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


pipe_dir = os.path.join(os.path.dirname(__file__), 'models','SVM','pipelines', 'svm_pipe.joblib')

svm_pipe = load(pipe_dir)

def svm_make_prediction(text):
    pred = svm_pipe.predict_proba([text])[0][1]
    return pred
#######################

movies  = pd.read_csv('./movie scraper/data/movies.csv', sep=';')

external_stylesheets = [
    dbc.themes.FLATLY,
    'https://use.fontawesome.com/releases/v5.7.2/css/all.css'
]

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True
)

server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(
        id='page-content',
        style={
            'width':'80%',
            'max-width':'320px',
            'min-width':'320px',
            'margin':'0 auto',
            'margin-top':'10px',
            'padding':'5px',
            'height':'600px',
            #'border':'1px solid grey'
        }
    )
], style={'height':'100vh','width':'100vw'}
)

home_layout = html.Div([
    html.H3(
        'Review this movie 🎬',
        id='title',
        className='display-5',
        style={'text-align':'center', 'font-weight':'bold'}
    ),

    # Movie title wrapper
    html.Div(
        [
            html.Div(
                [
                    html.Img(
                        id='poster',
                        style={
                            'position':'relative',
                            'top':'50%',
                            'left':'50%',
                            'transform':'translate(-50%,-50%)'
                        }
                    )
                ],
                style={
                    'height':'80px',
                    'width':'50px',
                    'float':'left',
                    #'border':'1px solid red'
                }
            ),
            html.Div(
                [
                html.H4(
                        id='movie_title',
                        style={
                            'display':'table-cell',
                            'vertical-align':'middle'
                        }
                    ),
                ],
                style={
                    #'border':'1px solid green',
                    'float':'left',
                    'margin-left':'5px',
                    'max-width':'240px',
                    'height':'80px',
                    'overflow': 'auto',
                    'display':'table',
                    'vertical-align':'middle'
                }
            ),
        ],
        className='alert alert-dismissible alert-light',
        style={
            'height':'90px',
            'padding':'5px',
            'overflow': 'hidden',
            #'border':'1px solid lightgrey',
            #'border-radius':'15px''
        }
    ),

    # Review and results wrapper
    html.Div(
        [
            dcc.Textarea(
                className="form-control z-depth-1",
                id="review",
                rows="6",
                placeholder="Write a review here..."
            ),
            html.H5(
                'Classifying sentiment...',
                style={'margin-top':'10px'}
            ),
            dbc.Progress(
                id='progress',
                value=0,
                striped=False,
                color='success',
                style={
                    'height':'20px',
                    'margin-top':'10px'
                }
            ),
            html.H4(
                id='decision',
                style={'margin-top':'10px', 'font-weight':'bold'}
            ),
            html.H5(
                'Is our classification correct? 🤔',
                style={'margin-top':'25px'}
            ),
            dbc.ButtonGroup(
                [
                    dbc.Button('Yes', id='button_yes', className='btn btn-primary', active=True),
                    dbc.Button('No', id='button_no', className='btn btn-primary', active=False)
                ],
            ),
            dbc.Button(
                [
                    html.Span(
                        'Submit',
                        style={'margin-right':'10px'}
                    ),
                    html.I(
                        className='fas fa-flag-checkered'
                    )
                ],
                id='submit_button',
                color='primary',
                className='btn btn-primary btn-lg btn-block',
                n_clicks_timestamp=0,
                style={'margin-top':'25px'}
            ),
            dbc.Button(
                [
                    html.Span(
                        'Review another movie',
                        style={'margin-right':'10px'}
                    ),
                    html.I(
                        className='fas fa-redo-alt'
                    )
                ],
                id='shuffle_button',
                color='secondary',
                className='btn btn-primary btn-lg btn-block',
                n_clicks_timestamp=0
            ),
            html.H5(
                'Change classifier:',
                style={'margin-top':'15px'}
            ),
            dbc.ButtonGroup(
                [
                    dbc.Button('SVM', id='button_svm', className='btn btn-primary', active=True),
                    dbc.Button('None', id='button_none', className='btn btn-primary', active=False)
                ],
            )
        ],
        style={
            #'border':'1px solid green',
            'margin-top':'10px',
            'text-align':'center'
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
                    html.H2('404 Page not found')
                ],
            ),
            dcc.Link("Go to Home", href="/"),
        ]


@app.callback(
    [
        Output('progress','value'),
        Output('progress','children'),
        Output('progress','color'),
        Output('decision','children')
    ],
    [
        Input('review','value'),
        Input('button_svm','active'),
        Input('button_none','active')
    ]
)
def update_progress(review, button_svm, button_none):

    if review is not None and review.strip() != '':
        if button_svm:
            pred_sentiment = 0#svm_make_prediction(review)
        elif button_none:
            pred_sentiment = 0

        pred_sentiment = round(pred_sentiment * 100, 1)
        pred_sentiment_text = f'{pred_sentiment}%'
        bar_value = pred_sentiment

        if pred_sentiment >= 67:
            color = 'success'
        elif 33 < pred_sentiment < 67:
            color = 'warning'
        else:
            color = 'danger'

        if pred_sentiment >= 50:
            decision = 'Sentiment: Positive 👍'
        else:
            decision = 'Sentiment: Negative 👎'
        return bar_value, pred_sentiment_text, color, decision
    else:
        return 0, None, None, 'Classification: ---'


@app.callback(
    [
        Output('poster','src'),
        Output('movie_title','children'),
        Output('review', 'value')
    ],
    [
        Input('submit_button','n_clicks_timestamp'),
        Input('shuffle_button','n_clicks_timestamp')
    ]
)
def shuffle_movie(submit_click_ts, shuffle_click_ts):
    random_movie = movies.sample(1).to_dict(orient='records')[0]
    movie_title = random_movie['title']
    poster_url = random_movie['poster_url']
    movie_year = random_movie['year']
    movie_title_year = f'{movie_title} ({movie_year})'

    return poster_url, movie_title_year, ''


@app.callback(
    [
        Output('button_yes','active'),
        Output('button_no','active')
    ],
    [
        Input('button_yes','n_clicks'),
        Input('button_no','n_clicks')
    ]
)
def toggle_yesno_buttons(yes_clicks, no_clicks):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if not any([yes_clicks, no_clicks]):
        return False, False
    elif button_id == 'button_yes':
        return True, False
    elif button_id == 'button_no':
        return False,True


@app.callback(
    [
        Output('button_svm','active'),
        Output('button_none','active')
    ],
    [
        Input('button_svm','n_clicks'),
        Input('button_none','n_clicks')
    ]
)
def toggle_model_buttons(svm_clicks, rnn_clicks):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'button_svm':
        return True, False
    elif button_id == 'button_none':
        return False, True


if __name__ == '__main__':
    app.run_server(debug=True)#, host='0.0.0.0')
