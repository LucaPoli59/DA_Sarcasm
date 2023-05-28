from dash import html, dash_table, dcc, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
import timeit
from wordcloud import WordCloud
import json
import constants
from general_data import df_train, target_info_rate


def get_wc_fig(wc, info_rate):
    dummy_scatter = go.Scatter(x=[None], y=[None], mode='markers', marker=dict(
        colorscale=constants.COLOR_SCALE, showscale=True, cmin=info_rate.min(), cmax=info_rate.max(),
        colorbar=dict(title="Rateo Informativo")))

    return {'img': go.Image(z=wc), 'color_bar': dummy_scatter}

start = timeit.default_timer()

selector_options = {'tokenized': 'Normale', 'nsw': 'No Stop Words', 'st': 'Stemming',
                    'nsw_st': 'No Stop Words e Stemming'}

dfs_sp = {name: pd.read_csv(os.path.join(constants.DATA_SP_PATH, "text_" + name + ".csv"), index_col="element")
          for name in selector_options.keys()}

for name in dfs_sp.keys():
    dfs_sp[name]['tot_s'] = round(dfs_sp[name]['tot'] / dfs_sp[name]['tot'].sum() * 100, 2)
    dfs_sp[name]['prop'] = round(dfs_sp[name]['prop'] * 100)

info_rate_graphs_dict = {}
for name in selector_options.keys():
    df = dfs_sp[name]

    with open(os.path.join(constants.DATA_WC_PATH, "text_" + name + ".json"), 'r') as f:
        wc_array = np.array(json.load(f))
    
    bar_fig = px.bar(df.reset_index(), x='element', y='tot_s', color='info_rate',
                     color_continuous_scale=constants.COLOR_SCALE,
                     range_x=[-0.3, 9.3], range_y=[0, df['tot_s'].max() + 0.1], hover_name='element',
                     hover_data={'tot': True, 'tot_s': False, 'info_rate': True, 'element': False, 'prop': True},
                     labels={'element': 'Parola', 'tot_s': 'Numero di campioni (%)', 'prop': 'Sarcastica (%)',
                             'tot': 'Numero di campioni', 'info_rate': 'Rateo informativo'})
    
    hist_fig = px.histogram(df, x='tot', y='info_rate', histfunc='avg', nbins=100, marginal='rug',
                            hover_data=['tot', 'tot_s', 'info_rate'],
                            labels={'element': 'Parola', 'tot': 'Numero di campioni',
                                    'tot_s': 'Numero di campioni (%)', 'info_rate': 'Rateo informativo'})
    
    info_rate_graphs_dict[name] = {'bar': bar_fig, 'hist': hist_fig, 'wc': get_wc_fig(wc_array, df['info_rate'])}


wc_layout = go.Layout(margin={"t": 20, "b": 0, "r": 0, "l": 0, "pad": 0}, xaxis={"visible": False},
                      yaxis={"visible": False}, hovermode=False)

end = timeit.default_timer()
print("Tempo di caricamento: ", end - start)

layout = dbc.Container(className="fluid", children=[
    html.Center(html.H1("Text Analysis", className="display-3 my-4")),

    dbc.Container(className="d-flex flex-column justify-content-center align-items-center my-5", children=[
        html.Center(html.H3(id='text_info_rate_title',
                            children="Distribuzione del rateo informativo degli elementi del testo normale")),
        dbc.RadioItems(id="text_info_rate_text_selector", inline=True,
                       options=selector_options, value='tokenized',
                       className="date-group-items justify-content-center my-2"),
        dbc.Container(className="d-inline-flex justify-content-center align-items-center my-2 gap-3", children=[
            dbc.Label("Tramite:"),
            dbc.Checklist(id="text_info_rate_graph_selector", inline=True, value=['wc', 'bar'],
                          options={'wc': 'Word Cloud', 'bar': 'Grafo a barre'},
                          className="date-group-items justify-content-center")
        ])
    ]),

    html.Div(id="text_info_rate_graphs_wc_bar", className="my-3"),
    dcc.Graph(id="text_info_rate_hist_graph")
])


@callback([Output(component_id="text_info_rate_graphs_wc_bar", component_property="children"),
           Output(component_id="text_info_rate_hist_graph", component_property="figure")],
          [Input(component_id="text_info_rate_text_selector", component_property="value"),
           Input(component_id="text_info_rate_graph_selector", component_property="value")])
def update_info_rate_graphs(text_selector, graph_selector):
    graphs = info_rate_graphs_dict[text_selector]

    if 'bar' in graph_selector and 'wc' in graph_selector:
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=go.Figure(data=graphs['wc']['img'], layout=wc_layout)), className="col-sm-6"),
            dbc.Col(dcc.Graph(figure=graphs['bar']), className="col-sm-6")]), graphs['hist']
    elif 'bar' in graph_selector:
        return dcc.Graph(figure=graphs['bar']), graphs['hist']
    elif 'wc' in graph_selector:
        return dcc.Graph(figure=go.Figure(data=list(graphs['wc'].values()), layout=wc_layout)), graphs['hist']

    return [], hist_fig
