
from dash import html, dcc
from dash.dependencies import Input, Output

from app import app
from pages import context, text, model_training, model_testing, dashboard
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

    match pathname:
        case '/context':
            return context.layout
        case '/text':
            return text.layout
        case '/model_training':
            return model_training.layout
        case '/model_testing':
            return model_testing.layout
        case _:
            return dashboard.layout



if __name__ == '__main__':
    app.run_server(debug=True)