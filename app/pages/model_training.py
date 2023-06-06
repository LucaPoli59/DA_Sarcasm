import dash
from dash import html, dcc
import plotly.express as px
import dash_bootstrap_components as dbc
import constants
import os
import pandas as pd

train_history = pd.read_csv(os.path.join(constants.MODEL_DIR, "history.csv"))

layout = dbc.Container(className="fluid", children=[
    html.Center(html.H1("Addestramento del modello", className="display-3 my-4")),
    html.Center(html.H3("Immagine del modello", className="my-4")),
    # html.Div([
    #     html.Img(src=dash.get_asset_url("model_img.png"), style={'width': '50%'}, className="d-inline-block"),
    #     html.Img(src=dash.get_asset_url("model_img_bert.png"), style={'width': '50%'}, className="d-inline-block"),
    # ], className="d-inline-block"),
    html.Img(src=dash.get_asset_url("model_img_h.png"), className="img-fluid"),
    html.Img(src=dash.get_asset_url("model_img_bert_h.png"), className="img-fluid"),


    html.Hr(className="my-5"),
    html.Center(html.H3("History di training del modello")),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=px.line(train_history, x="epoch", y=["loss", "val_loss"]))),
        dbc.Col(dcc.Graph(figure=px.line(train_history, x="epoch", y=["accuracy", "val_accuracy"])))
    ]),

    html.Hr(className="my-5"),
    html.Center(html.H3("Risultati sul dataset di validation")),
])
