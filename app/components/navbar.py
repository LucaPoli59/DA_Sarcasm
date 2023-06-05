from dash import html
import dash_bootstrap_components as dbc


def navbar():
    layout = html.Div([
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Dashboard", href="/dashboard")),
                dbc.NavItem(dbc.NavLink("Sentence Len Analysis", href="/len_analysis")),
                dbc.NavItem(dbc.NavLink("Context Exploring", href="/context")),
                dbc.NavItem(dbc.NavLink("Text Analysis", href="/text")),
                dbc.NavItem(dbc.NavLink("Model Training", href="/model_training")),
                dbc.NavItem(dbc.NavLink("Model Demo", href="/model_demo")),
            ],
            brand="Demo",
            brand_href="/home",
            color="dark",
            dark=True,
        )
    ])
    return layout




