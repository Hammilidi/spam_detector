
import dash
from dash import html, dcc, Output, Input, State
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import dash_bootstrap_components as dbc
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Fonction de prétraitement du texte
    """
    if pd.isna(text):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression de la ponctuation et des caractères spéciaux
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Suppression des stopwords et stemming
    processed_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            processed_tokens.append(stemmer.stem(token))
    
    return ' '.join(processed_tokens)


def predict_spam(text, model, vectorizer):
    """
    Fonction pour prédire si un email est spam ou non
    
    Args:
        text (str): Texte de l'email à analyser
        model: Modèle de classification entraîné
        vectorizer: Vectoriseur TF-IDF entraîné
    
    Returns:
        dict: Résultat de la prédiction avec probabilités
    """
    try:
        # Prétraitement du texte
        processed_text = preprocess_text(text)
        
        if not processed_text.strip():
            return {
                'prediction': 'ham',
                'probability': 0.5,
                'confidence': 'Low',
                'error': 'Texte vide après prétraitement'
            }
        
        # Vectorisation
        text_vectorized = vectorizer.transform([processed_text])
        
        # Prédiction
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Interprétation des résultats
        label = 'spam' if prediction == 1 else 'ham'
        confidence_score = max(probabilities)
        
        if confidence_score > 0.8:
            confidence = 'High'
        elif confidence_score > 0.6:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        return {
            'prediction': label,
            'probability': confidence_score,
            'confidence': confidence,
            'probabilities': {
                'ham': probabilities[0],
                'spam': probabilities[1]
            }
        }
        
    except Exception as e:
        return {
            'prediction': 'ham',
            'probability': 0.5,
            'confidence': 'Low',
            'error': str(e)
        }


model = joblib.load("../models/spam_detection_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
# preprocessor = preprocess_text(text)
# predictor = predict_spam(model, vectorizer, preprocessor)

df = pd.read_csv("../data/DataSet_Emails.csv")
df["text"] = df["text"].fillna("")
df["length"] = df["text"].apply(lambda x: len(str(x)))


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("\U0001F4E7 Spam Classifier", className="my-3 text-center"),
    html.P("Ce modèle prédit si un email est SPAM ou HAM (non-spam).", className="mb-4 text-center"),
    
    dbc.Card([
        dbc.CardHeader("\U0001F50D Tester un email"),
        dbc.CardBody([
            dcc.Textarea(id='email-input', placeholder="Entrer le contenu de l'email", style={"width": "100%", "height": 150}),
            html.Br(),
            dbc.Button("Prédire", id='predict-button', color="primary", className="mt-2"),
            html.Div(id='prediction-output', className="mt-3")
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader("\U0001F4CA Analyse exploratoire (EDA)"),
        dbc.CardBody([
            html.H5("Aperçu des données"),
            dcc.Graph(figure=px.histogram(df, x="label", title="Distribution des classes")),

            html.H5("Longueur des messages"),
            dcc.Graph(figure=px.histogram(df, x="length", color="label", nbins=50, title="Longueur des messages par classe"))
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader("\U0001F4C9 Évaluation du modèle"),
        dbc.CardBody([
            html.H5("Matrice de confusion"),
            dcc.Graph(id='confusion-matrix'),

            html.H5("Rapport de classification"),
            dcc.Loading(dcc.Graph(id='classification-report'))
        ])
    ]),

    html.Hr(),
    html.P("\U0001F4E6 Projet IA · Modèle TF-IDF + Classifieur supervisé · By YONLI Fidelis", className="text-center")
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('email-input', 'value')
)
def predict_email(n, text):
    if n and text:
        result = predict_spam(text, model, vectorizer)
        if result["prediction"] == "spam":
            return dbc.Alert(f"\u274c SPAM détecté (confiance : {result['probability']:.2f}, {result['confidence']})", color="danger")
        else:
            return dbc.Alert(f"\u2705 HAM (non-spam) (confiance : {result['probability']:.2f}, {result['confidence']})", color="success")
    return ""


@app.callback(
    Output('confusion-matrix', 'figure'),
    Output('classification-report', 'figure'),
    Input('predict-button', 'n_clicks')
)
def update_eval_section(n):
    if not n:
        raise dash.exceptions.PreventUpdate

    df["text"] = df["text"].fillna("")  # sécurité
    X_transformed = vectorizer.transform(df["text"])
    y_true = df["label"]
    y_pred = model.predict(X_transformed)

    cm = confusion_matrix(y_true, y_pred)
    labels = ["HAM", "SPAM"]
    fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                       color_continuous_scale='Blues',
                       labels=dict(x="Prédiction", y="Réel", color="Nombre"))

    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "Classe"})
    fig_report = px.bar(df_report[df_report["Classe"].isin(labels)],
                        x="Classe", y="f1-score",
                        title="F1-score par classe", color="Classe")

    return fig_cm, fig_report

if __name__ == '__main__':
    app.run(debug=True, port=8051)
    