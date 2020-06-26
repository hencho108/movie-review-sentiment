#http://www.corelangs.com/css/box/center-div.html
#https://community.plotly.com/t/horizontally-center-image/15253


import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

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
        n_chars = 0
    else:
        n_chars = len(str(review))
    return n_chars, n_chars


if __name__ == '__main__':
    app.run_server(debug=True)