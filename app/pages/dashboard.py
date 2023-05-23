from dash import html, dash_table, dcc, callback, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
import constants
import timeit


df_s = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "data_full_sample.csv")).iloc[:, 1:]
df_t = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "train.csv"), index_col="index")
df_t['date'] = pd.to_datetime(df_t['date'])
df_t['sarcastic'] = df_t['sarcastic'].astype('int')

df_t_len = df_t[['sarcastic', 'text', 'parent']].copy()
df_t_len[['text', 'parent']] = df_t_len[['text', 'parent']].applymap(lambda x: len(x.split()))
df_t_len = df_t_len.loc[abs(zscore(df_t_len['text']) <= 3) & abs(zscore(df_t_len['parent']) <= 3)]

len_range = df_t_len.describe().loc[['min', 'max'], ['text', 'parent']]

layout = dbc.Container(className="fluid", children=[
    dbc.Row(html.Center(html.H1("Dashboard", className="display-3 my-4"))),
    dbc.Row([
        dbc.Col(html.Div([
            html.H3("Informazioni generali sul dataset iniziale"),
            html.P("Numero di righe: " + str(df_s.shape[0])),
            html.P("Numero di righe nulle: " + str(df_s.isna().sum().sum())),
            html.P("Numero di righe duplicate: " + str(df_s.duplicated().sum())),
            html.H5("Colonne e tipo")] +
            [html.P(col + ": " + str(df_s[col].dtype), style={'margin-bottom': '2px'}) for col in df_s.columns]
        ))
    ]),
    html.Hr(className="my-3"),
    dbc.Row([
        dbc.Col(html.Div([
            html.H3("Informazioni generali sul dataset di training"),
            html.P("Numero di righe: " + str(df_t.shape[0])),
            html.P("Proporzione split tra train e validation: " +
                   str(round((df_t.shape[0] / df_s.shape[0])*100, 2)) + "%"),
            html.H5("Colonne e tipo")] +
            [html.P(col + ": " + str(df_t[col].dtype), style={'margin-bottom': '2px'}) for col in df_t.columns])),
        dbc.Col(html.Div([
            html.H3("Distribuzione del target"),
            dcc.Graph(figure=px.histogram(df_t['sarcastic'],
                                          text_auto=True, histnorm='percent').update_layout(bargap=0.2))
        ]))

    ]),
    html.Hr(className="my-5"),
    dbc.Row(dash_table.DataTable(data=df_t.to_dict('records'), page_size=3,
                                 style_data={'whiteSpace': 'normal', 'height': 'auto'})),
    html.Hr(className="my-5"),
    dbc.Row(dcc.Graph(id="len_graph")),
    dcc.RangeSlider(id="len_slider", min=len_range['text']['min'], max=len_range['text']['max'],
                    step=round((len_range['text'][-1] - len_range['text'][0])/30), value=len_range['text'].values)

])


@callback(Output(component_id='len_graph', component_property='figure'),
          [Input(component_id='len_slider', component_property='value')])
def update_len_graph(value):
    return px.histogram(df_t_len.loc[df_t_len['text'].between(*value)], x="text", color="sarcastic",
                        histfunc='count', histnorm=None, text_auto=True, log_x=False, barnorm='percent')
