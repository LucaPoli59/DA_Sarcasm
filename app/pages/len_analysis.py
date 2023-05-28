from dash import html, dash_table, dcc, callback, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from scipy.stats import zscore
import os

import constants
from general_data import target_info_rate


len_dfs = {'text': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "len_text.csv")),
           'parent': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "len_parent.csv"))}

for df_name, df in len_dfs.items():
    df = df.rename(columns={df.columns[0]: 'len'})
    df['prop'] = round(df['prop'] * 100, 0)
    len_dfs[df_name] = df.loc[abs(zscore(df['len']) < 3)]

tot_range = {}
for df_name, df in len_dfs.items():
    tot_range[df_name] = df['tot'].quantile(np.arange(0, 1.01, 0.01))[::-1]
    tot_range[df_name].index = ((1 - tot_range[df_name].index.values) * 100).round().astype(int)

len_frequency = {df_name: df['len'].repeat(df['tot']) for df_name, df in len_dfs.items()}

layout = dbc.Container(className="fluid", children=[
    dbc.Container(className="d-flex flex-column justify-content-center align-items-center my-5", children=[
        html.Center(html.H1(id='len_title', children="Analisi della lunghezza del testo")),
        dbc.RadioItems(id="feature_len_selector", options={'text': 'Testo', 'parent': 'Parent'}, value='text',
                       inline=True, className="date-group-items justify-content-center mt-4"),
    ]),
    html.Center(html.H5("Distribuzione lunghezza sentenze rispetto al numero di campioni")),
    dcc.Graph(id="len_tot_graph", className="mb-3"),

    html.Center(html.H5("Distribuzione Rateo informativo rispetto alla lunghezza delle sentenze")),
    dcc.Graph(id="len_info_rate_graph"),

    dbc.Container(className="mt-3", children=[
        dbc.Label("Numero di campioni per gruppo:"),
        dcc.RangeSlider(id="len_tot_slider", min=0, max=100, step=1, dots=False, value=(0, 50), className="mt-1",
                        allowCross=False),

    ]),
])


@callback([Output(component_id='len_tot_slider', component_property='marks'),
           Output(component_id='len_tot_slider', component_property='value'),
           Output(component_id='len_title', component_property='children'),
           Output(component_id='len_tot_graph', component_property='figure')],
          [Input(component_id='feature_len_selector', component_property='value')])
def update_len_sliders_graph(ft_s):
    tot_slider_marks = {mark: str(round(v)) for mark, v in tot_range[ft_s].iloc[::10].items()}

    title = "Distribuzione della lunghezza del "
    if ft_s == 'text':
        title += "testo"
    else:
        title += "parent"

    len_tot_graph = px.box(len_frequency[ft_s], x='len', labels={'len': 'Numero di parole nelle sentenze'})

    return tot_slider_marks, (0, 50), title, len_tot_graph


@callback(Output(component_id='len_info_rate_graph', component_property='figure'),
          [Input(component_id='feature_len_selector', component_property='value'),
           Input(component_id='len_tot_slider', component_property='value')])
def update_len_info_rate_graph(ft_s, tot_value):
    df_len = len_dfs[ft_s]
    tot_value = (tot_range[ft_s][tot_value[0]], tot_range[ft_s][tot_value[1]])[::-1]

    return px.bar(df_len.loc[df_len['tot'].between(*tot_value) & df_len['len']],
                  x="len", y="info_rate", text_auto=True, hover_data=['prop', 'tot'], range_y=[0, 51],
                  labels={'len': 'Numero di parole nelle sentenze', 'prop': 'Sarcastica (%)',
                          'info_rate': 'Rateo informativo', 'tot': 'Numero campioni'}
                  ).update_layout(xaxis=dict(rangeslider=dict(visible=True)))
