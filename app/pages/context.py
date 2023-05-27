from dash import html, dash_table, dcc, callback, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
import datetime as dt

import constants
from general_data import df_train, target_info_rate


def compute_unique_sp(df_dict, threshold_list):
    """
    Funzione che calcola per ogni dataframe, passato nel relativo dizionario, la percentuale di elementi unici,
     i cui ratei informativi superano le soglie

    :param df_dict: dizionario dei dataframe delle feature contenti i valori unici
    :type df_dict: dict
    :param threshold_list: soglie in cui calcolare il numero di elementi unici sarcastici
    :type threshold_list: list or np.array
    :return: dataframe con i risultati
    :rtype: pd.DataFrame
    """
    ris = pd.DataFrame(columns=list(df_dict.keys()), index=threshold_list).apply(
        lambda feature: list(map(
            lambda threshold: (abs(df_dict[feature.name]['prop'] - target_info_rate * 100) > threshold).sum(),
            feature.index.values)
        ))
    ris = ris.iloc[1:] / ris.iloc[0] * 100
    ris.index.name, ris.columns.name = "Threshold", "Feature"

    return ris


dfs_sp = {'author': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "author.csv"), index_col="element"),
          'parent': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "parent.csv"), index_col="element"),
          'date': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "date.csv"), index_col="element"),
          'subreddit': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "subreddit.csv"), index_col="element")}

for df_name in dfs_sp.keys():
    dfs_sp[df_name] = dfs_sp[df_name].loc[dfs_sp[df_name]['tot'] > 1]
    dfs_sp[df_name] = dfs_sp[df_name].sort_values(by='tot', ascending=False).reset_index()
    dfs_sp[df_name]['info_rate'] = abs(dfs_sp[df_name]['prop'] - target_info_rate) * 100
    dfs_sp[df_name]['prop'] = round(dfs_sp[df_name]['prop'] * 100)

dfs_sp['date']['element'] = pd.to_datetime(dfs_sp['date']['element']).apply(lambda x: x.strftime("%d-%m-%Y"))


unique_stats = compute_unique_sp(dfs_sp, np.arange(0, 50, 10))

layout = dbc.Container(className="fluid", children=[
    html.Center(html.H1("Context Exploring", className="display-3 my-4")),

    html.Center(html.H5("Distribuzione delle feature del contesto rispetto al numero di testi per ogni feature unica")),
    dbc.Row([dbc.Col(dcc.Graph(figure=px.box(df['tot'], points='outliers', orientation='v',
                                             labels={'value': 'Numero di campioni', 'variable': feature})),
                     class_name="col-sm-3")
             for feature, df in dfs_sp.items()], class_name="mx-2"),

    html.Hr(className="my-3"),
    html.Center(html.H5("Distribuzione delle feature del contesto rispetto al numero di testi per ogni feature unica")),
    dcc.Graph(figure=px.bar(unique_stats.transpose(), barmode='group',
                            labels={'value': 'Percentuale di elementi unici'})),

    html.Hr(className="my-4"),
    dbc.Container(className="d-flex flex-column justify-content-center align-items-center my-5", children=[
        html.Center(html.H3(id='context_info_rate_title',
                            children="Distribuzione del rateo informativo degli elementi di Autore")),
        dbc.RadioItems(id="context_info_rate_selector", inline=True,
                       options={'author': 'Autore', 'parent': 'Parent', 'date': 'Data', 'subreddit': 'Subreddit'},
                       value='author', className="date-group-items justify-content-center mt-4"),
    ]),
    dcc.Graph(id="context_info_rate_graph"),
    dbc.Container(className="mt-3", children=[
        dbc.Label("Numero di testi associati all'elemento:"),
        dcc.RangeSlider(id="context_tot_slider", min=0, max=100, dots=False, value=(0, 100), className="mt-1",
                        allowCross=False, tooltip={'always_visible': False, 'placement': 'top'}),

    ]),

])


@callback([Output(component_id='context_tot_slider', component_property='min'),
           Output(component_id='context_tot_slider', component_property='max'),
           Output(component_id='context_tot_slider', component_property='value'),
           Output(component_id='context_info_rate_title', component_property='children')],
          [Input(component_id='context_info_rate_selector', component_property='value')])
def update_context_slider_graph(ft_s):

    title = "Distribuzione del rateo informativo degli elementi di "
    match ft_s:
        case 'author':
            title += "Autore"
        case 'parent':
            title += "Parent"
        case 'date':
            title += "Data"
        case 'subreddit':
            title += "Subreddit"

    range_min, range_max = dfs_sp[ft_s]['tot'].min(), dfs_sp[ft_s]['tot'].max()

    return range_min, range_max, (range_min, range_max), title


@callback(Output(component_id='context_info_rate_graph', component_property='figure'),
          [Input(component_id='context_info_rate_selector', component_property='value'),
           Input(component_id='context_tot_slider', component_property='value')])
def update_context_info_rate_graph(ft_s, tot_value):
    dataframe = dfs_sp[ft_s]
    dataframe = dataframe.loc[dataframe['tot'].between(*tot_value)]

    fig = px.scatter(dataframe, x='element', hover_name='element', range_y=[-1, 52], range_x=[-0.3, 19.3],
                      y="info_rate", hover_data={'prop': True, 'tot': True, 'element': False, 'info_rate': False},
                      labels={'prop': 'Sarcastica (%)', 'info_rate': 'Rateo informativo', 'element': 'Elementi',
                              'tot': "Numero di testi"}, color_continuous_scale='blackbody', color='prop')
    fig = fig.update_layout(xaxis=dict(ticktext=dataframe['element'].str.slice(-10), tickvals=dataframe['element']))

    return fig
