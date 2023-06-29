import os

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from dash import html, dcc, callback, Input, Output, Patch

import constants
from general_data import target_info_rate

len_dfs = {'text': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "len_text.csv")),
           'parent': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "len_parent.csv"))}
len_text_parent = pd.read_csv(os.path.join(constants.DATA_SP_PATH, "len_text_parent.csv"))

for df_name, df in len_dfs.items():
    df = df.rename(columns={df.columns[0]: 'len'})
    df['prop'] = round(df['prop'] * 100, 0)
    len_dfs[df_name] = df

tot_range = {}
for df_name, df in list(len_dfs.items()) + [('text_parent', len_text_parent)]:
    tot_range[df_name] = df['tot'].quantile(np.arange(0, 1.01, 0.01))[::-1]
    tot_range[df_name].index = ((1 - tot_range[df_name].index.values) * 100).round().astype(int)

len_frequency = {df_name: df['len'].repeat(df['tot']) for df_name, df in len_dfs.items()}


@callback([Output(component_id='len_tot_slider', component_property='marks'),
           Output(component_id='len_tot_slider', component_property='value'),
           Output(component_id='len_title', component_property='children'),
           Output(component_id='len_tot_graph', component_property='figure')],
          [Input(component_id='feature_len_selector', component_property='value')], prevent_initial_call=True)
def update_len_sliders_graph(ft_s, patch=True):
    marks = {mark: str(round(v)) for mark, v in tot_range[ft_s].iloc[::10].items()}

    title = "Distribuzione della lunghezza del "
    if ft_s == 'text':
        title += "testo"
    else:
        title += "parent"

    if patch:
        graph = Patch()
        graph["data"][0]["x"] = len_frequency[ft_s].values
    else:
        graph = px.box(len_frequency[ft_s], x='len', labels={'len': 'Numero di parole nelle sentenze'})

    return marks, (0, 50), title, graph


@callback(Output(component_id='len_info_rate_graph', component_property='figure'),
          [Input(component_id='feature_len_selector', component_property='value'),
           Input(component_id='len_tot_slider', component_property='value')], prevent_initial_call=True)
def update_len_info_rate_graph(ft_s, tot_value, patch=True):
    df_len = len_dfs[ft_s]
    tot_value = (tot_range[ft_s][tot_value[0]], tot_range[ft_s][tot_value[1]])[::-1]
    df_len = df_len.loc[df_len['tot'].between(*tot_value)]

    if patch:
        graph = Patch()
        graph["data"][0]["x"] = df_len['len'].values
        graph["data"][0]["y"] = df_len['info_rate'].values
        graph["data"][0]["customdata"] = df_len[['prop', 'tot']].values
    else:
        graph = px.bar(df_len,
                       x="len", y="info_rate", text_auto=True, hover_data=['prop', 'tot'], range_y=[0, 51],
                       labels={'len': 'Numero di parole nelle sentenze', 'prop': 'Sarcastica (%)',
                               'info_rate': 'Rateo informativo', 'tot': 'Numero campioni'}
                       ).update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    return graph


@callback(Output(component_id='len_s_info_rate_graph', component_property='figure'),
          [Input(component_id='len_s_tot_slider', component_property='value')], prevent_initial_call=True)
def update_len_s_info_rate_graph(tot_value, patch=True):
    tot_value = (tot_range['text_parent'][tot_value[0]], tot_range['text_parent'][tot_value[1]])[::-1]
    df_len = len_text_parent.loc[len_text_parent['tot'].between(*tot_value)]

    if patch:
        graph = Patch()
        graph["data"][0]["x"] = df_len['parent'].values
        graph["data"][0]["y"] = df_len['text'].values
        graph["data"][0]["z"] = df_len['info_rate'].values
        graph["data"][0]["customdata"] = df_len[['prop', 'tot']].values
        graph["data"][1]["x"] = df_len['parent'].values  # aggiornamento del marginal del parent
        graph["data"][2]["y"] = df_len['text'].values  # aggiornamento del marginal del testo
    else:
        graph = px.density_heatmap(df_len, x='parent', y='text', z='info_rate', marginal_y='box',
                                   range_color=(0, round(100 * target_info_rate)),
                                   color_continuous_scale=constants.COLOR_SCALE, hover_data=['prop', 'tot'],
                                   labels={'parent': 'Lunghezza sentenze del Parent',
                                           'text': 'Lunghezza sentenze nel Testo',
                                           'prop': 'Sarcastica (%)', 'info_rate': 'Rateo informativo',
                                           'tot': 'Numero campioni'}, histfunc='avg', marginal_x='box')
    return graph


len_tot_slider_marks, len_tot_slider_value, len_title, len_tot_graph = update_len_sliders_graph('text', patch=False)
len_info_rate_graph = update_len_info_rate_graph('text', len_tot_slider_value, patch=False)
len_s_info_rate_graph = update_len_s_info_rate_graph(len_tot_slider_value, patch=False)


layout = dbc.Container(className="fluid", children=[
    dbc.Container(className="d-flex flex-column justify-content-center align-items-center my-5", children=[
        html.Center(html.H1(id='len_title', children=len_title)),
        dbc.RadioItems(id="feature_len_selector", options={'text': 'Testo', 'parent': 'Parent'}, value='text',
                       inline=True, className="date-group-items justify-content-center mt-4"),
    ]),
    html.Center(html.H5("Distribuzione lunghezza sentenze rispetto al numero di campioni")),
    dcc.Graph(id="len_tot_graph", className="mb-3", figure=len_tot_graph),

    html.Center(html.H5("Distribuzione Rateo informativo rispetto alla lunghezza delle sentenze")),
    dcc.Graph(id="len_info_rate_graph", figure=len_info_rate_graph),

    dbc.Container(className="mt-3", children=[
        dbc.Label("Numero di campioni per gruppo:"),
        dcc.RangeSlider(id="len_tot_slider", min=0, max=100, step=1, dots=False, value=len_tot_slider_value,
                        className="mt-1", allowCross=False, marks=len_tot_slider_marks),

    ]),

    html.Hr(className="my-5"),
    html.Center(html.H5("Distribuzione Rateo informativo rispetto alle due lunghezze simultaneamente")),
    dcc.Graph(id="len_s_info_rate_graph", figure=len_s_info_rate_graph),

    dbc.Container(className="mt-3", children=[
        dbc.Label("Numero di campioni per gruppo:"),
        dcc.RangeSlider(id="len_s_tot_slider", min=0, max=100, step=1, dots=False, value=(0, 50), className="mt-1",
                        allowCross=False,
                        marks={mark: str(round(v)) for mark, v in tot_range['text_parent'].iloc[::10].items()}),

    ])

])
