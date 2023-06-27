from dash import html, dcc, callback, Input, Output

from app import app
from pages import dashboard, context
# from pages import context, text, dashboard, len_analysis
from components import navbar, footer

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar.navbar(),
    html.Div(id='page-content', children=[dashboard.layout]),
    footer.footer()
])


@callback(Output(component_id='page-content', component_property='children'),
          [Input(component_id='url', component_property='pathname')],
          prevent_initial_call=True)
def display_page(pathname):
    match pathname:
        case '/context':
            return context.layout
        # case '/text':
        #     return text.layout
        # case '/len_analysis':
        #     return len_analysis.layout
        # case '/model_training':
        #     return model_training.layout
        # case '/model_demo':
        #     return model_demo.layout
        case _:
            return dashboard.layout

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
