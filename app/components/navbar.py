import dash_bootstrap_components as dbc
from dash import html


def navbar():
    layout = html.Div([
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Dashboard", href="/")),
                dbc.NavItem(dbc.NavLink("Sentence Len Analysis", href="/len_analysis")),
                dbc.NavItem(dbc.NavLink("Context Exploring", href="/context")),
                dbc.NavItem(dbc.NavLink("Text Analysis", href="/text")),
                dbc.NavItem(dbc.NavLink("Model Training", href="/model_training")),
                dbc.NavItem(dbc.NavLink("Model Testing", href="/model_testing")),
            ],
            brand="Progetto: Sarcasm Detection",
            brand_href="/home",
            color="dark",
            dark=True,
        )
    ])
    return layout




