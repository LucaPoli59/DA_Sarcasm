import dash
from dash import html
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc


app = dash.Dash(__name__, use_pages=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
                suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Div([
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink(page['name'], href=page['relative_path']))
            for page in dash.page_registry.values() if page['nav'] is True],
            brand="Project: Sarcasm Detection with transformers",
            brand_href="/",
            color="dark",
            dark=True,
        )
    ]),


    dash.page_container,


    dmc.Footer(
        children=[
            html.P("Data Analytics: Sarcasm Detection with transformers, Luca Poli [852027]"),
        ],
        height=60,
        fixed=False,
        style={"background-color": "#333333", "color": "white", "text-align": "center", "padding-top": "20px",
               "margin-top": "20px"}
    )
])


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)