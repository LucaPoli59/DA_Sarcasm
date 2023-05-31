from dash import html, dcc, callback, Input, Output
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
from dash_ag_grid import AgGrid


def get_wc_fig(wc, info_rate):
    dummy_scatter = go.Scatter(x=[None], y=[None], mode='markers', marker=dict(
        colorscale=constants.COLOR_SCALE, showscale=True, cmin=info_rate.min(), cmax=info_rate.max(),
        colorbar=dict(title="Rateo Informativo")))

    return {'img': go.Image(z=wc), 'color_bar': dummy_scatter}


selector_options = {'tokenized': 'Normale', 'punctuation': 'Punteggiatura', 'nsw': 'Senza Stopwords', 'st': 'Stemming',
                    'nsw_st': 'Senza Stopwords e Stemming'}

dfs_sp = {name: pd.read_csv(os.path.join(constants.DATA_SP_PATH, "text_" + name + ".csv"), index_col="element")
          for name in selector_options.keys()}
df_texts = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "train_text.csv"), index_col="index")

for name in dfs_sp.keys():
    dfs_sp[name]['tot_s'] = dfs_sp[name]['tot'] / dfs_sp[name]['tot'].sum() * 100
    dfs_sp[name]['prop'] = round(dfs_sp[name]['prop'] * 100)

dfs_info_stats = pd.DataFrame(index=pd.Index(dfs_sp.keys(), name='feature'), columns=['avg', 'std']).drop('punctuation')
dfs_info_stats['avg'] = [np.average(dfs_sp[name]['info_rate'].values, weights=dfs_sp[name]['tot_s'].values)
                         for name in dfs_info_stats.index]
dfs_info_stats['std'] = [np.sqrt(np.cov(dfs_sp[name]['info_rate'].values, aweights=dfs_sp[name]['tot_s'].values))
                         for name in dfs_info_stats.index]

sp_cols_to_grid = {'element': 'Testo', 'tot_s': 'Frequenza %', 'tot': 'Frequenza', 'prop': 'Proporzione sarcastica %',
                   'info_rate': 'Rateo informativo'}
sp_grids = {name: AgGrid(
    rowData=dfs_sp[name].reset_index()[list(sp_cols_to_grid.keys())].to_dict('records'),
    columnDefs=[{'field': col, 'headerName': sp_cols_to_grid[col]} for col, col_name in sp_cols_to_grid.items()],
    defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 125,
                   "wrapText": True, 'autoHeight': True},
    columnSize='sizeToFit',
) for name in dfs_sp.keys()}

info_rate_graphs_dict = {}
for name in selector_options.keys():
    df = dfs_sp[name]

    with open(os.path.join(constants.DATA_WC_PATH, "text_" + name + ".json"), 'r') as f:
        wc_array = np.array(json.load(f))

    bar = px.bar(df.reset_index(), x='element', y='tot_s', color='info_rate',
                 color_continuous_scale=constants.COLOR_SCALE,
                 range_x=[-0.3, 14.3], range_y=[0, df['tot_s'].max() + 0.1], hover_name='element',
                 hover_data={'tot': True, 'tot_s': False, 'info_rate': True, 'element': False, 'prop': True},
                 labels={'element': 'Parola', 'tot_s': 'Numero di campioni (%)', 'prop': 'Sarcastica (%)',
                         'tot': 'Numero di campioni', 'info_rate': 'Rateo informativo'}
                 )

    hist = px.histogram(df, x='tot_s', y='info_rate', histfunc='avg', nbins=100, marginal='rug',
                        hover_data=['tot', 'info_rate'],
                        labels={'element': 'Parola', 'tot': 'Numero di campioni',
                                'tot_s': 'Numero di campioni (%)', 'info_rate': 'Rateo informativo'})

    box = px.box(df['info_rate'], labels={'value': 'valore', 'variable': 'Rateo informativo'}
                 ).update_layout(xaxis={'visible': True, 'showticklabels': False})

    info_rate_graphs_dict[name] = {'bar': bar, 'hist': hist, 'wc': get_wc_fig(wc_array, df['info_rate']), 'box': box}

wc_layout = go.Layout(margin={"t": 20, "b": 0, "r": 0, "l": 0, "pad": 0}, xaxis={"visible": False},
                      yaxis={"visible": False}, hovermode=False)


@callback([Output("text_info_rate_graph_wc", "figure"), Output("text_info_rate_graphs_wc", "figure"),
           Output("text_info_rate_graph_bar", "figure"), Output("text_info_rate_graphs_bar", "figure")],
          [Input("text_info_rate_text_selector", "value")],
          prevent_initial_call=True)
def update_info_rate_graphs(text_selector):
    graphs = info_rate_graphs_dict[text_selector]

    return go.Figure(data=list(graphs['wc'].values()), layout=wc_layout), \
           go.Figure(data=graphs['wc']['img'], layout=wc_layout), *[graphs['bar']] * 2


@callback([Output("text_info_graph_tab1", "label_class_name"), Output("text_info_graph_tab2", "label_class_name"),
           Output("text_info_graph_tab3", "label_class_name")], [Input("text_info_graph_selector", "active_tab")])
def update_tabs_active_style(graph_selector):
    return ["text-primary fs-5" if graph_selector == 'wc' else "text-dark fs-5",
            "text-primary fs-5" if graph_selector == 'wc_bar' else "text-dark fs-5",
            "text-primary fs-5" if graph_selector == 'bar' else "text-dark fs-5"]


@callback([Output("text_info_rate_graph_hist", "figure"), Output("text_info_rate_graph_box", "figure")],
          [Input("text_info_rate_text_selector", "value")])
def update_info_rate_stats(text_selector):
    graphs = info_rate_graphs_dict[text_selector]
    return graphs['hist'], graphs['box']


@callback([Output("text_info_rate_grid", "children")], [Input("text_info_rate_text_selector", "value")],
          prevent_initial_call=True)
def update_sp_grid(text_selector):
    return [sp_grids[text_selector]]


def_graph = update_info_rate_graphs('tokenized')

layout = dbc.Container(className="fluid", children=[
    html.Center(html.H1("Text Analysis", className="display-3 my-4")),

    html.Center(html.H3(className='my-3', children="Testi a confronto")),
    html.Div(AgGrid(
        rowData=df_texts.to_dict('records'),
        columnDefs=[{'field': col, 'headerName': selector_options[col]} for col in df_texts.columns],
        defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 125,
                       "wrapText": True, 'autoHeight': True},
        columnSize='sizeToFit',
    )),

    html.Hr(className="my-5"),
    html.Center(html.H3("Rateo informativo medio")),
    dcc.Graph(figure=px.bar(dfs_info_stats.rename(selector_options), orientation='v', barmode='group',
                            labels={'value': 'Rateo informativo', 'feature': 'Tipo di testo'})),
    html.Hr(className="my-5"),
    dbc.Container(className="border border-secondary", children=[
        dbc.Container(className="d-flex flex-column justify-content-center align-items-center mt-3", children=[
            html.Center(html.H2("Analisi del rateo informativo degli elementi di ogni tipo di testo")),
            dbc.RadioItems(id="text_info_rate_text_selector", inline=True,
                           options=selector_options, value='tokenized',
                           className="date-group-items justify-content-center my-2"),
        ]),
        dcc.Loading(type="circle", children=[dbc.Row(className="mt-3", children=[
            html.Center(html.H5("Distribuzione generale")),
            dbc.Col(dcc.Graph(id="text_info_rate_graph_hist"), className="col-sm-9"),
            dbc.Col(dcc.Graph(id="text_info_rate_graph_box"), className="col-sm-3")
        ])]),
        html.Hr(),
        html.Center(html.H5("Approfondimento del rateo per le parole")),
        html.Hr(),
        dbc.Tabs(className="nav nav-tabs nav-justified justify-content-center align-items-center", active_tab='wc_bar',
                 id="text_info_graph_selector", children=[
                dbc.Tab(label='Word Cloud', tab_id='wc', label_class_name="text-dark fs-5", id="text_info_graph_tab1",
                        children=dcc.Loading(type="circle",
                                             children=dcc.Graph(figure=def_graph[0], id='text_info_rate_graph_wc'))
                        ),

                dbc.Tab(label='Entrambi', tab_id='wc_bar', label_class_name="text-dark fs-5", id="text_info_graph_tab2",
                        children=[
                            dcc.Loading(type="circle", children=[
                                dbc.Row([
                                    dbc.Col(dcc.Graph(figure=def_graph[1], id='text_info_rate_graphs_wc'),
                                            className="col-sm-6"),
                                    dbc.Col(dcc.Graph(figure=def_graph[2], id='text_info_rate_graphs_bar'),
                                            className="col-sm-6")
                                ])
                            ])
                        ]),
                dbc.Tab(label='Grafo a barre', tab_id='bar', label_class_name="text-dark fs-5",
                        id="text_info_graph_tab3",
                        children=dcc.Loading(type="circle",
                                             children=dcc.Graph(figure=def_graph[3], id='text_info_rate_graph_bar'))
                        ),
            ]),
        html.Hr(className="my-5"),
        dcc.Loading(type="circle", children=html.Div(children=[sp_grids['tokenized']], id="text_info_rate_grid")),
    ]),

])
