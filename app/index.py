
from dash import html, dcc
from dash.dependencies import Input, Output

from app import app
from pages import home, page1
from components import navbar, footer

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar.navbar(),
    html.Div(id='page-content', children=[]),
    footer.footer()
])


@app.callback(Output(component_id='page-content', component_property='children'),
              [Input(component_id='url', component_property='pathname')])
def display_page(pathname):
    if pathname == '/page1':
        return page1.layout

    return home.layout


if __name__ == '__main__':
    app.run_server(debug=True)