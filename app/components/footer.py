from dash import html
import dash_bootstrap_components as dbc


def footer():
    layout = html.Div([
        dbc.Footer(
            children=[
                dbc.Row(
                    dbc.Col(
                        html.P("This is a demo", className="mb-0")
                    )
                )
            ]
        )
    ])
    return layout