from dash import html
import dash_bootstrap_components as dbc


def navbar():
    layout = html.Div([
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Home", href="/home")),
                dbc.NavItem(dbc.NavLink("Page 1", href="/page1")),
            ],
            brand="Demo",
            brand_href="/home",
            color="dark",
            dark=True,
        )
    ])
    return layout




