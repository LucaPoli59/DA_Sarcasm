from dash import html
import dash_mantine_components as dmc


def footer():
    layout = dmc.Footer(
        children=[
            html.P("Footer test"),
        ],
        height=60,
        fixed=True,
        style={"background-color": "#333333", "color": "white", "text-align": "center", "padding-top": "20px"},
    )
    return layout
