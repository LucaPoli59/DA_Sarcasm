import dash_bootstrap_components as dbc
import plotly.express as px
from dash import html, dcc
from dash_ag_grid import AgGrid

# from general_data import df_full, df_train
#
# cols_to_grid = {'sarcastic': 'Sarcastico', 'text': 'Testo', 'author': 'Autore', 'date': 'Data',
#                 'subreddit': 'Subreddit', 'parent': 'Parent'}
#
# ag_grid = AgGrid(
#     rowData=df_train[list(cols_to_grid.keys())].to_dict('records'),
#     columnDefs=[{'field': col, 'headerName': cols_to_grid[col]} for col, col_name in cols_to_grid.items()],
#     defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 125,
#                    "wrapText": True, 'autoHeight': True},
#     columnSize='sizeToFit',
# )
print("dashboard page loaded")

layout = dbc.Container(className="fluid", children=[" "
    # html.Center(html.H1("Dashboard", className="display-3 my-4")),
    # html.Div(
    #     [
    #         html.H3("Informazioni generali sul dataset iniziale"),
    #         html.P("Numero di righe: " + str(df_full.shape[0])),
    #         html.P("Numero di righe nulle: " + str(df_full.isna().sum().sum())),
    #         html.P("Numero di righe duplicate: " + str(df_full.duplicated().sum())),
    #         html.H5("Colonne e tipo")] +
    #     [
    #         html.P(col + ": " + str(df_full[col].dtype), style={'margin-bottom': '2px'}) for col in df_full.columns
    #     ]
    # ),
    # html.Hr(className="my-3"),
    # dbc.Row([
    #     dbc.Col(html.Div([
    #                          html.H3("Informazioni generali sul dataset di training"),
    #                          html.P("Numero di righe: " + str(df_train.shape[0])),
    #                          html.P("Proporzione split tra train e validation: " +
    #                                 str(round((df_train.shape[0] / df_full.shape[0]) * 100, 2)) + "%"),
    #                          html.H5("Colonne e tipo")] +
    #                      [html.P(col + ": " + str(df_train[col].dtype), style={'margin-bottom': '2px'}) for col in
    #                       df_train.columns])),
    #     dbc.Col(html.Div([
    #         html.H3("Distribuzione del target"),
    #         dcc.Graph(figure=px.histogram(df_train['sarcastic'],
    #                                       text_auto=True, histnorm='percent').update_layout(bargap=0.2))
    #     ]))
    # ]),
    # html.Hr(className="my-5"),
    # html.Center(html.H3("Dataset di training")),
    # html.Div(className="my-4", children=ag_grid),
])
