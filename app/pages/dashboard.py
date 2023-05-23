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

target_info_rate = df_t['sarcastic'].value_counts(normalize=True).max()

len_dfs = {'text': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "len_text.csv")),
           'parent': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "len_parent.csv"))}

for df_name, df in len_dfs.items():
    df = df.rename(columns={df.columns[0]: 'len'})
    df['prop_s'] = df['prop'] - target_info_rate
    df['prop'] = round(df['prop'] * 100, 0)
    len_dfs[df_name] = df.loc[abs(zscore(df['len']) < 3)]

len_range = {}
tot_range = {}
for df_name, df in len_dfs.items():
    len_range[df_name] = df['len'].quantile(np.arange(0, 1.01, 0.01))
    len_range[df_name].index = (len_range[df_name].index.values * 100).round().astype(int)
    tot_range[df_name] = df['tot'].quantile(np.arange(0, 1.01, 0.01))
    tot_range[df_name].index = (tot_range[df_name].index.values * 100).round().astype(int)

layout = dbc.Container(className="fluid", children=[
    dbc.Row(html.Center(html.H1("Dashboard", className="display-3 my-4"))),
    dbc.Row([
        dbc.Col(html.Div([
                             html.H3("Informazioni generali sul dataset iniziale"),
                             html.P("Numero di righe: " + str(df_s.shape[0])),
                             html.P("Numero di righe nulle: " + str(df_s.isna().sum().sum())),
                             html.P("Numero di righe duplicate: " + str(df_s.duplicated().sum())),
                             html.H5("Colonne e tipo")] +
                         [html.P(col + ": " + str(df_s[col].dtype), style={'margin-bottom': '2px'}) for col in
                          df_s.columns]
                         ))
    ]),
    html.Hr(className="my-3"),
    dbc.Row([
        dbc.Col(html.Div([
                             html.H3("Informazioni generali sul dataset di training"),
                             html.P("Numero di righe: " + str(df_t.shape[0])),
                             html.P("Proporzione split tra train e validation: " +
                                    str(round((df_t.shape[0] / df_s.shape[0]) * 100, 2)) + "%"),
                             html.H5("Colonne e tipo")] +
                         [html.P(col + ": " + str(df_t[col].dtype), style={'margin-bottom': '2px'}) for col in
                          df_t.columns])),
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
    dbc.Row(dcc.RadioItems(id="feature_len_selector", options={'text': 'Testo', 'parent': 'Parent'}, value='text',
                           inline=True)),
    dbc.Row(dcc.Graph(id="len_graph")),
    dbc.Row(dcc.RangeSlider(id="len_slider", min=0, max=100, step=1, dots=False, value=(50, 100), allowCross=False)),
    dbc.Row(dcc.RangeSlider(id="tot_slider", min=0, max=100, step=1, dots=False, value=(50, 100), allowCross=False)),
])


@callback([Output(component_id='len_slider', component_property='marks'),
           Output(component_id='len_slider', component_property='value'),
           Output(component_id='tot_slider', component_property='marks'),
           Output(component_id='tot_slider', component_property='value')],
            [Input(component_id='feature_len_selector', component_property='value')])
def update_sliders(ft_s):
    len_slider_marks = {mark: str(round(v)) for mark, v in len_range[ft_s].iloc[::20].items()}
    tot_slider_marks = {mark: str(round(v)) for mark, v in tot_range[ft_s].iloc[::10].items()}

    return len_slider_marks, (0, 50), tot_slider_marks, (50, 100)


@callback(Output(component_id='len_graph', component_property='figure'),
          [Input(component_id='feature_len_selector', component_property='value'),
           Input(component_id='len_slider', component_property='value'),
           Input(component_id='tot_slider', component_property='value')])
def update_len_graph(ft_s, len_value, tot_value):
    df_len = len_dfs[ft_s]
    len_value = len_range[ft_s][len_value[0]], len_range[ft_s][len_value[1]]
    tot_value = tot_range[ft_s][tot_value[0]], tot_range[ft_s][tot_value[1]]

    return px.bar(df_len.loc[df_len['len'].between(*len_value) & df_len['tot'].between(*tot_value)],
                  x="len", y="prop_s", text_auto=True, hover_data=['prop', 'tot'], range_y=[-0.51, 0.51],
                  labels={'len': 'Numero parole', 'prop': 'Sarcastica (%)',
                          'prop_s': 'Rateo informativo', 'tot': 'Numero campioni'})