import os
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import html, dcc, dash_table
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import constants

train_history = pd.read_csv(os.path.join(constants.MODEL_DIR, "history.csv"), index_col=0)
cmp_val = pd.read_csv(os.path.join(constants.MODEL_DIR, "compare_val.csv"), index_col=0)

val_fpr, val_tpr, val_thresholds = roc_curve(cmp_val['True'], cmp_val['score'])
val_roc = pd.DataFrame({'fpr': val_fpr, 'tpr': val_tpr}, index=pd.Index(val_thresholds, name='thresholds'))
val_roc.columns.name = 'Rateo'
val_auc = auc(val_fpr, val_tpr)
val_roc_curve = px.area(x=val_roc['fpr'], y=val_roc['tpr'],
                        labels={'x': 'Rateo falsi positivi', 'y': 'Rateo veri positivi'})
val_roc_curve = val_roc_curve.add_shape(type='line', line={'dash': 'dash'}, x0=0, x1=1, y0=0, y1=1)
val_roc_curve = val_roc_curve.update_yaxes(
    scaleanchor="x", scaleratio=1).update_xaxes(constrain='domain').update_layout(title_text=f'AUC = {val_auc:.4f}',
                                                                                  title_x=0.5)

val_rp = classification_report(cmp_val['True'], cmp_val['Predicted'], target_names=['Non sarcastico', 'Sarcastico'],
                               output_dict=True)
val_rp_acc = round(val_rp.pop('accuracy'), 3)
val_rp = pd.DataFrame(val_rp).loc[['precision', 'recall', 'f1-score'], ['Non sarcastico', 'Sarcastico']].transpose()
val_rp = val_rp.round(3).reset_index().rename(columns={'index': 'Classe'})

layout = dbc.Container(className="fluid", children=[
    html.Center(html.H1("Addestramento del modello", className="display-3 my-4")),
    html.Center(html.H3("Immagine del modello", className="my-4")),
    html.Img(src=dash.get_asset_url("model_img.png"), className="img-fluid"),
    html.Img(src=dash.get_asset_url("model_img_bert.png"), className="img-fluid mt-3"),

    html.Hr(className="my-5"),
    html.Center(html.H3("History di training del modello")),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=px.line(train_history[["loss", "val_loss"]].rename(
            columns={"loss": "Training", "val_loss": "Validation"}),
                                         labels={'value': 'Loss', 'variable': 'Tipo di dataset'}))),
        dbc.Col(dcc.Graph(figure=px.line(train_history[["binary_accuracy", "val_binary_accuracy"]].rename(
            columns={"binary_accuracy": "Training", "val_binary_accuracy": "Validation"}),
                                         labels={'value': 'Accuracy', 'variable': 'Tipo di dataset'})))
    ]),

    html.Hr(className="my-5"),
    html.Center(html.H3("Risultati sul dataset di validation")),
    dbc.Row([
        dbc.Col([
            html.Center(html.H5("Confusion Matrix")),
            dcc.Graph(figure=px.imshow(confusion_matrix(cmp_val['True'], cmp_val['Predicted'], normalize='true'),
                                       x=['Non Sarcastico', 'Sarcastico'], y=['Non Sarcastico ', 'Sarcastico '],
                                       text_auto=True, color_continuous_scale='blues',
                                       labels={'color': 'Percentuale', 'x': 'Classe predetta', 'y': 'Classe reale'}
                                       ).update_layout(coloraxis_showscale=False))
        ], className="col-6"),
        dbc.Col([
            dbc.Row(html.Center(html.H5("Report della classificazione"))),
            dbc.Row(html.P("Accuracy: " + str(val_rp_acc))),
            dbc.Row(dash_table.DataTable(val_rp.to_dict('records'))),

        ], className="col-6"),
    ], className="d-flex align-items-center"),
    html.Center(html.H5("Confronto tra la probabilità predetta e la classe reale")),
    dcc.Graph(figure=px.histogram(cmp_val, x='score', color="True",
                                  labels={'score': 'Probabilità predetta', 'True': 'Classe reale',
                                          'count': 'Numero di istanze', 0: 'Non sarcastico', 1: 'Sarcastico'})),
    dbc.Row([
        dbc.Col([
            html.Center(html.H5("Falsi positivi e Veri positivi ad ogni soglia")),
            dcc.Graph(figure=px.line(val_roc, labels={'fpr': 'Rateo falsi positivi', 'tpr': 'Rateo veri positivi',
                                                      'thresholds': 'soglia'}
                                     ).update_yaxes(scaleanchor="x", scaleratio=1
                                                    ).update_xaxes(range=[0, 1], constrain='domain'))
        ], className="col-6"),
        dbc.Col([
            html.Center(html.H5("Curva ROC")),
            dcc.Graph(figure=val_roc_curve)

        ], className="col-6")
    ])

])
