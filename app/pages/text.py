from dash import html, dcc, callback, Input, Output, Patch, State
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os
import timeit
import json
import constants
from dash_ag_grid import AgGrid

from general_data import target_info_rate


start = timeit.default_timer()
#
selector_options = {'tokenized': 'Normale', 'punctuation': 'Punteggiatura', 'nsw': 'Senza Stopwords', 'st': 'Stemming',
                    'nsw_st': 'Senza Stopwords e Stemming'}

dfs_sp = {name: pd.read_csv(os.path.join(constants.DATA_SP_PATH, "text_" + name + ".csv"), index_col="element")
          for name in selector_options.keys()}
df_texts = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "train_text.csv"), index_col="index")

wc_array = {}
for name in dfs_sp.keys():
    dfs_sp[name]['tot_s'] = dfs_sp[name]['tot'] / dfs_sp[name]['tot'].sum() * 100
    dfs_sp[name]['prop'] = round(dfs_sp[name]['prop'] * 100)
    dfs_sp[name] = dfs_sp[name].sort_values(by='tot_s', ascending=False)

    with open(os.path.join(constants.DATA_WC_PATH, "text_" + name + ".json"), 'r') as f:
        wc_array[name] = np.array(json.load(f))

sp_cols_to_grid = {'element': 'Testo', 'tot_s': 'Frequenza %', 'tot': 'Frequenza', 'prop': 'Proporzione sarcastica %',
                   'info_rate': 'Rateo informativo'}

sp_grids = {name: AgGrid(
    rowData=dfs_sp[name].reset_index()[list(sp_cols_to_grid.keys())].to_dict('records'),
    columnDefs=[{'field': col, 'headerName': sp_cols_to_grid[col]} for col, col_name in sp_cols_to_grid.items()],
    defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 125,
                   "wrapText": True, 'autoHeight': True},
    columnSize='sizeToFit',
) for name in dfs_sp.keys()}

dfs_info_stats = pd.DataFrame(index=pd.Index(dfs_sp.keys(), name='feature'), columns=['avg', 'std']).drop('punctuation')
dfs_info_stats['avg'] = [np.average(dfs_sp[name]['info_rate'].values, weights=dfs_sp[name]['tot_s'].values)
                         for name in dfs_info_stats.index]
dfs_info_stats['std'] = [np.sqrt(np.cov(dfs_sp[name]['info_rate'].values, aweights=dfs_sp[name]['tot_s'].values))
                         for name in dfs_info_stats.index]


wc_layout = go.Layout(margin={"t": 20, "b": 0, "r": 0, "l": 0, "pad": 0}, xaxis={"visible": False},
                      yaxis={"visible": False}, hovermode=False)
dummy_scatter = go.Scatter(x=[None], y=[None], mode='markers', marker=dict(
        colorscale=constants.COLOR_SCALE, showscale=True, cmin=0, cmax=round(100 * target_info_rate, 4),
        colorbar=dict(title="Rateo Informativo")))


@callback([Output("text_info_rate_graph_wc", "figure", allow_duplicate=True),
           Output("text_info_rate_graph_bar", "figure", allow_duplicate=True),
           Output("text_info_rate_graph_hist", "figure"), Output("text_info_rate_graph_box", "figure")],
          [Input("text_info_rate_text_selector", "value")], prevent_initial_call=True)
def update_info_rate_graph(text_selector, patch=True):
    df = dfs_sp[text_selector]
    bar_df = df.head(100).reset_index()

    if patch:
        wc, bar, hist, box = Patch(), Patch(), Patch(), Patch()

        wc['data'][0]['z'] = wc_array[text_selector]

        bar['data'][0]['x'] = bar_df['element'].values
        bar['data'][0]['y'] = bar_df['tot_s'].values
        bar['data'][0]['marker']['color'] = bar_df['info_rate'].values
        bar['data'][0]['hovertext'] = bar_df['element'].values
        bar['data'][0]['customdata'] = bar_df[['tot', 'info_rate', 'prop']].values

        hist['data'][0]['x'] = df['tot_s'].values
        hist['data'][0]['y'] = df['info_rate'].values
        hist['data'][0]['customdata'] = df[['tot']].values

        box['data'][0]['y'] = df['info_rate'].values

    else:
        wc = go.Figure(data=go.Image(z=wc_array[text_selector]), layout=wc_layout)

        bar = px.bar(bar_df, x='element', y='tot_s', color='info_rate',
                     color_continuous_scale=constants.COLOR_SCALE, range_color=(0, round(100 * target_info_rate)),
                     range_x=[-0.3, 14.3], hover_name='element',
                     hover_data={'tot': True, 'tot_s': False, 'info_rate': True, 'element': False, 'prop': True},
                     labels={'element': 'Parola', 'tot_s': 'Numero di campioni (%)', 'prop': 'Sarcastica (%)',
                             'tot': 'Numero di campioni', 'info_rate': 'Rateo informativo'})
        hist = px.histogram(df, x='tot_s', y='info_rate', histfunc='avg', nbins=100, marginal='rug',
                            hover_data=['tot', 'info_rate'],
                            labels={'element': 'Parola', 'tot': 'Numero di campioni',
                                    'tot_s': 'Numero di campioni (%)', 'info_rate': 'Rateo informativo'})

        box = px.box(df['info_rate'], labels={'value': 'valore', 'variable': 'Rateo informativo'}
                     ).update_layout(xaxis={'visible': True, 'showticklabels': False})

    return wc, bar, hist, box


@callback([Output("text_info_rate_grid", "children")], [Input("text_info_rate_text_selector", "value")],
          prevent_initial_call=True)
def update_sp_grid(text_selector):
    return [sp_grids[text_selector]]


def_grid = update_sp_grid('tokenized')
def_graph = update_info_rate_graph('tokenized', patch=False)

end = timeit.default_timer()
print('text page loaded, in', end - start, 'seconds')

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
            dbc.Col(dcc.Graph(id="text_info_rate_graph_hist", figure=def_graph[2]), className="col-sm-9"),
            dbc.Col(dcc.Graph(id="text_info_rate_graph_box", figure=def_graph[3]), className="col-sm-3")
        ])]),
        html.Hr(),
        html.Center(html.H5("Approfondimento del rateo per le parole")),
        html.Hr(),
        dbc.Tabs(className="nav nav-tabs nav-justified justify-content-center align-items-center", active_tab='wc_bar',
                 id="text_info_graph_selector", children=[
                dbc.Tab(label='Word Cloud', tab_id='wc', label_class_name="text-dark fs-5", id="text_info_graph_tab1"),

                dbc.Tab(label='Entrambi', tab_id='wc_bar', label_class_name="text-dark fs-5",
                        id="text_info_graph_tab2"),
                dbc.Tab(label='Grafo a barre', tab_id='bar', label_class_name="text-dark fs-5",
                        id="text_info_graph_tab3"),
            ]),
        html.Div(dcc.Loading(type="circle", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=def_graph[0], id='text_info_rate_graph_wc'),
                        className="col-6", id="text_info_rate_graph_wc_col"),
                dbc.Col(dcc.Graph(figure=def_graph[1], id='text_info_rate_graph_bar'),
                        className="col-6", id="text_info_rate_graph_bar_col")
            ])
        ])),
        html.Hr(className="my-5"),
        # dcc.Loading(type="circle", children=[html.Div(children=[def_grid], id="text_info_rate_grid")]),
    ]),

])


@callback([Output("text_info_graph_tab1", "label_class_name"), Output("text_info_graph_tab2", "label_class_name"),
           Output("text_info_graph_tab3", "label_class_name")], [Input("text_info_graph_selector", "active_tab")])
def update_tabs_active_style(graph_selector):
    return ["text-primary fs-5" if graph_selector == 'wc' else "text-dark fs-5",
            "text-primary fs-5" if graph_selector == 'wc_bar' else "text-dark fs-5",
            "text-primary fs-5" if graph_selector == 'bar' else "text-dark fs-5"]


@callback([Output("text_info_graph_selector", "active_tab")],
          [Input("text_info_rate_text_selector", "value")], prevent_initial_call=True)
def update_active_tab(text_selector):
    return ['wc_bar']


@callback([Output("text_info_rate_graph_wc", "figure"), Output("text_info_rate_graph_bar", "figure"),
           Output("text_info_rate_graph_wc", "style"), Output("text_info_rate_graph_bar", "style"),
           Output("text_info_rate_graph_wc_col", "className"), Output("text_info_rate_graph_bar_col", "className")],
          [Input("text_info_graph_selector", "active_tab")], State("text_info_rate_graph_wc", "figure"))
def select_info_rate_graph(graph_selector, prev_wc):
    graph_wc, graph_bar = Patch(), Patch()

    styles, class_cols = [], []
    match graph_selector:
        case 'wc':
            if len(prev_wc['data']) == 1:
                graph_wc['data'].append(dummy_scatter)

            styles = {'display': 'block'}, {'display': 'none'}
            class_cols = ['col-12', '']
        case 'bar':
            graph_bar['layout']['xaxis']['range'] = [-0.3, 25.3]

            styles = {'display': 'none'}, {'display': 'block'}
            class_cols = ['', 'col-12']
        case 'wc_bar':
            if len(prev_wc['data']) == 2:
                del graph_wc['data'][1]
            graph_bar['layout']['xaxis']['range'] = [-0.3, 14.3]

            styles = {'display': 'block'}, {'display': 'block'}
            class_cols = ['col-6', 'col-6']

    return graph_wc, graph_bar, styles[0], styles[1], class_cols[0], class_cols[1]
