import dash_bootstrap_components as dbc
import plotly.express as px
from dash import html, dcc
from dash_ag_grid import AgGrid
import dash

import general_data as gd

dash.register_page(__name__, path="/", name="Dashboard", title="Dashboard", order=0, nav=True)


cols_to_grid = {'sarcastic': 'Sarcastico', 'text': 'Testo', 'author': 'Autore', 'date': 'Data',
                'subreddit': 'Subreddit', 'parent': 'Parent'}

ag_grid = AgGrid(
    rowData=gd.df_train[list(cols_to_grid.keys())].to_dict('records'),
    columnDefs=[{'field': col, 'headerName': cols_to_grid[col]} for col, col_name in cols_to_grid.items()],
    defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 125,
                   "wrapText": True, 'autoHeight': True},
    columnSize='sizeToFit',
)

layout = dbc.Container(className="fluid", children=[
    html.Center(html.H1("Dashboard", className="display-3 my-4")),
    html.Div(
        [
            html.H3("Informazioni generali sul dataset iniziale"),
            html.P("Numero di righe: " + str(gd.df_full_stats['len'])),
            html.P("Numero di righe nulle: " + str(gd.df_full_stats['null_values'])),
            html.P("Numero di righe duplicate: " + str(gd.df_full_stats['duplicated'])),
            html.H5("Colonne e tipo")] +
        [
            html.P(var + ": " + str(gd.df_full_stats_dt[var]), style={'margin-bottom': '2px'})
            for var in gd.df_full_stats_dt.index
        ]
    ),
    html.Hr(className="my-3"),
    dbc.Row([
        dbc.Col(html.Div(
            [
                html.H3("Informazioni generali sul dataset di training"),
                html.P("Numero di righe: " + str(gd.df_train_stats['len'])),
                html.P("Proporzione split tra train e validation: " +
                       str(round((gd.df_train_stats['len'] / gd.df_full_stats['len']) * 100, 2)) + "%"),
                html.H5("Colonne e tipo")] +
            [
                html.P(var + ": " + str(gd.df_train_stats_dt[var]), style={'margin-bottom': '2px'})
                for var in gd.df_train_stats_dt.index
            ]
        )),
        dbc.Col(html.Div([
            html.H3("Distribuzione del target"),
            dcc.Graph(figure=px.bar(gd.target_distribution, text_auto=True,
                                    labels={'value': 'Numero di righe (%)',
                                            'false': 'Non sarcastico', 'true': 'Sarcastico'}))
        ]))
    ]),
    html.Hr(className="my-5"),
    html.Center(html.H3("Dataset di training")),
    html.Div(className="my-4", children=ag_grid),
])
