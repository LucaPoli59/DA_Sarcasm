from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    dbc.Row(html.Center(html.H1("Dashboard", className="display-3 my-3"))),
    dbc.Row([
        dbc.Col(html.Div("This is content for the first tab")),
        dbc.Col(html.Div("This is content for the second tab")),
    ])
])
