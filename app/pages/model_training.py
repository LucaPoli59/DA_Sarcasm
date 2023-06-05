import dash
from dash import html
import dash_bootstrap_components as dbc
import constants
import os

layout = dbc.Container(className="fluid", children=[
    html.Center(html.H1("Addestramento del modello", className="display-3 my-4")),
    html.Center(html.H3("Immagine del modello", className="my-4")),
    html.Img(src=dash.get_asset_url("model.png"), className="img-fluid", width="800px", height="450px"),

    html.Hr(className="my-5"),
    html.Center(html.H3("History di training del modello")),

    html.Hr(className="my-5"),
    html.Center(html.H3("Risultati sul dataset di validation")),
])
