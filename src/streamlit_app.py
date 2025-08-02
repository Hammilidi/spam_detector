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

# Configuration de la page
st.set_page_config(
    page_title="BMSecurity - Détecteur de Spam",
    page_icon="🛡️",
    layout="wide"
)

# Fonction de prétraitement (identique à celle du notebook)
@st.cache_data
def preprocess_text(text):
    """Fonction de prétraitement du texte"""
    if pd.isna(text) or not text.strip():
        return ""

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

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

# Chargement du modèle et du vectoriseur
@st.cache_resource
def load_models():
    """Chargement des modèles sauvegardés"""
    try:
        with open('../models/spam_detection_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        return None, None

# Fonction de prédiction
def predict_email(text, model, vectorizer):
    """Prédiction pour un email donné"""
    try:
        # Prétraitement
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

        # Interprétation
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

def main():
    st.title("🛡️ BMSecurity - Système de Détection de Spam")
    st.markdown("*Solution intelligente pour la sécurité des communications*")
    st.markdown("---")

    # Chargement des modèles
    model, vectorizer = load_models()

    if model is None or vectorizer is None:
        st.error("Impossible de charger les modèles. Veuillez vérifier les fichiers.")
        return

    # Sidebar pour la navigation
    st.sidebar.title("🚀 Navigation")
    page = st.sidebar.selectbox(
        "Choisir une section",
        ["🏠 Accueil", "📧 Détection", "📊 Statistiques", "🔍 Analyse", "ℹ️ À propos"]
    )

    if page == "🏠 Accueil":
        st.header("🏠 Bienvenue dans le système de détection de spam")

        st.write("""
        ### 🎯 Objectif
        Cette application utilise des techniques avancées de **Machine Learning** et de 
        **Traitement du Langage Naturel (NLP)** pour détecter automatiquement les emails 
        malveillants et protéger vos communications.
        """)

        # Métriques de performance basées sur les graphiques fournis
        st.subheader("📈 Performances des Modèles")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🎯 Accuracy (SVM)", "98.7%", "↗️ Meilleur")
        with col2:
            st.metric("🔍 Precision (SVM)", "98.2%", "↗️ Excellent")
        with col3:
            st.metric("📡 Recall (SVM)", "99.4%", "↗️ Optimal")
        with col4:
            st.metric("⚖️ F1-Score (SVM)", "98.8%", "↗️ Top")

        # Fonctionnalités
        st.subheader("🛠️ Fonctionnalités")

        col1, col2 = st.columns(2)

        with col1:
            st.write("""
            **🔧 Prétraitement Avancé:**
            - Tokenisation intelligente
            - Suppression des mots vides
            - Stemming et normalisation
            - Vectorisation TF-IDF
            """)

        with col2:
            st.write("""
            **🤖 Algorithmes Utilisés:**
            - Support Vector Machine (SVM) - **Recommandé**
            - Naive Bayes Multinomial
            - Decision Trees optimisés
            - Validation croisée 5-fold
            """)

        # Instructions
        st.subheader("🚀 Comment utiliser l'application")
        st.write("""
        1. **📧 Détection** : Analysez un email en temps réel
        2. **📊 Statistiques** : Consultez les performances détaillées des 3 modèles
        3. **🔍 Analyse** : Explorez les patterns dans les données (wordclouds, distributions)
        4. **ℹ️ À propos** : Découvrez les détails techniques
        """)

    elif page == "📧 Détection":
        st.header("📧 Analyser un Email")
        st.write("Collez le contenu de votre email ci-dessous pour une analyse instantanée.")

        # Zone de saisie
        email_text = st.text_area(
            "📝 Contenu de l'email :",
            placeholder="Collez ici le texte de l'email à analyser...",
            height=200
        )

        # Exemples d'emails
        st.subheader("💡 Exemples d'emails à tester")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📧 Email Légitime", type="secondary"):
                st.session_state.email_example = "Hello team, I wanted to remind you about our meeting tomorrow at 3 PM in conference room A. Please bring your quarterly reports and be prepared to discuss the upcoming project milestones. Looking forward to seeing everyone there. Best regards, John"

        with col2:
            if st.button("🚨 Email Suspect", type="secondary"):
                st.session_state.email_example = "URGENT! CONGRATULATIONS! You have won $1,000,000 in our international lottery! Send your bank details immediately to claim your prize. Act now before this offer expires! Click here to claim your money now!"

        # Afficher l'exemple sélectionné
        if 'email_example' in st.session_state:
            email_text = st.text_area(
                "📝 Contenu de l'email (exemple sélectionné) :",
                value=st.session_state.email_example,
                height=200,
                key="email_content"
            )

        # Bouton d'analyse
        if st.button("🔍 Analyser l'Email", type="primary"):
            if email_text.strip():
                with st.spinner("🔄 Analyse en cours..."):
                    result = predict_email(email_text, model, vectorizer)

                # Affichage des résultats
                st.markdown("---")
                st.subheader("📋 Résultats de l'Analyse")

                # Résultat principal
                if result['prediction'] == 'spam':
                    st.error(f"🚨 **SPAM DÉTECTÉ**")
                    st.markdown(f"**Niveau de confiance :** {result['confidence']} ({result['probability']:.1%})")
                else:
                    st.success(f"✅ **EMAIL LÉGITIME**")
                    st.markdown(f"**Niveau de confiance :** {result['confidence']} ({result['probability']:.1%})")

                # Détails des probabilités
                st.subheader("📊 Probabilités Détaillées")

                # Graphique en barres
                fig, ax = plt.subplots(figsize=(8, 4))
                categories = ['Email Légitime', 'Spam']
                probabilities = [result['probabilities']['ham'], result['probabilities']['spam']]
                colors = ['green' if result['prediction'] == 'ham' else 'lightgreen',
                         'red' if result['prediction'] == 'spam' else 'lightcoral']

                bars = ax.bar(categories, probabilities, color=colors, alpha=0.7)
                ax.set_ylabel('Probabilité')
                ax.set_title('Distribution des Probabilités')
                ax.set_ylim(0, 1)

                # Ajout des valeurs sur les barres
                for bar, prob in zip(bars, probabilities):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')

                st.pyplot(fig)
                plt.close()

                # Gauge de confiance
                st.subheader("🎯 Niveau de Confiance")
                confidence_value = result['probability']

                # Création d'une gauge simple
                fig, ax = plt.subplots(figsize=(8, 3))

                # Barre de progression
                ax.barh(0, confidence_value, color='green' if confidence_value > 0.7 else 'orange' if confidence_value > 0.5 else 'red', alpha=0.7)
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xlabel('Niveau de Confiance')
                ax.set_title(f'Confiance: {confidence_value:.1%} ({result["confidence"]})')
                ax.set_yticks([])

                # Marqueurs de seuils
                ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Seuil Moyen')
                ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Seuil Élevé')
                ax.legend()

                st.pyplot(fig)
                plt.close()

                # Recommandations
                st.subheader("💡 Recommandations")

                if result['confidence'] == 'High':
                    if result['prediction'] == 'spam':
                        st.warning("""
                        ⚠️ **Action Recommandée :** 
                        - Supprimez cet email immédiatement
                        - Ne cliquez sur aucun lien
                        - Signalez-le comme spam
                        """)
                    else:
                        st.info("""
                        ✅ **Email Fiable :** 
                        - Cet email semble légitime
                        - Vous pouvez le traiter normalement
                        """)
                else:
                    st.warning("""
                    🤔 **Vérification Manuelle Recommandée :** 
                    - Le modèle n'est pas certain
                    - Examinez l'email plus attentivement
                    - Vérifiez l'expéditeur et le contenu
                    """)

            else:
                st.warning("⚠️ Veuillez saisir un email à analyser.")

    elif page == "📊 Statistiques":
        st.header("📊 Performances et Statistiques des Modèles")

        # Interprétation des résultats basée sur les graphiques
        st.subheader("🎯 Analyse Comparative des Modèles")
        
        st.write("""
        Basé sur l'analyse des performances des trois modèles testés, voici les résultats détaillés :
        """)

        # Métriques détaillées avec interprétation
        st.subheader("📈 Résultats de Performance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🌳 Decision Tree")
            st.metric("Accuracy", "95.6%")
            st.metric("Precision", "96.1%") 
            st.metric("Recall", "95.9%")
            st.metric("F1-Score", "96.0%")

        with col2:
            st.markdown("### 📊 Naive Bayes")
            st.metric("Accuracy", "98.1%")
            st.metric("Precision", "97.7%")
            st.metric("Recall", "98.7%") 
            st.metric("F1-Score", "98.2%")

        with col3:
            st.markdown("### 🎯 SVM (Recommandé)")
            st.metric("Accuracy", "98.7%", "🏆")
            st.metric("Precision", "98.2%", "🏆")
            st.metric("Recall", "99.4%", "🏆") 
            st.metric("F1-Score", "98.8%", "🏆")

        # Graphique de comparaison des performances
        st.subheader("📊 Comparaison Visuelle des Performances")
        try:
            st.image("plots/comparaison_performances_models.png", 
                    caption="Comparaison des métriques de performance entre les trois modèles", 
                    use_container_width=True)
        except:
            st.info("Graphique de comparaison non disponible. Veuillez vérifier le fichier.")

        # Validation croisée
        st.subheader("🔄 Validation Croisée")
        st.write("""
        **Analyse de la stabilité des modèles :**
        - **Decision Tree** : 95.6% ± 0.002 (Stable mais performance plus faible)
        - **Naive Bayes** : 98.1% ± 0.001 (Très stable et performant)
        - **SVM** : 98.7% ± 0.001 (Le plus stable ET le plus performant)
        """)

        try:
            st.image("plots/confusion_models.png", 
                    caption="Matrices de confusion pour les trois modèles", 
                    use_container_width=True)
        except:
            st.info("Matrices de confusion non disponibles. Veuillez vérifier le fichier.")

        # Interprétation des matrices de confusion
        st.subheader("🔍 Analyse des Matrices de Confusion")
        
        st.write("""
        **Observations clés :**
        
        1. **SVM** montre les meilleures performances avec :
           - Très peu de faux positifs (80 emails légitimes classés comme spam)
           - Très peu de faux négatifs (21 spams non détectés)
           - Meilleur équilibre global
        
        2. **Naive Bayes** :
           - Performance solide avec 76 faux positifs et 43 faux négatifs
           - Bon compromis entre précision et rappel
        
        3. **Decision Tree** :
           - Plus de faux positifs (125) et faux négatifs (134)
           - Performance inférieure mais toujours acceptable
        """)

        # Recommandation
        st.success("""
        **🏆 Recommandation :** Le modèle **SVM** est recommandé pour la production grâce à :
        - Sa précision exceptionnelle (98.7%)
        - Son excellent rappel (99.4% - détecte presque tous les spams)
        - Sa stabilité en validation croisée
        """)

    elif page == "🔍 Analyse":
        st.header("🔍 Analyse Exploratoire et Visualisations")

        # Nuages de mots
        st.subheader("☁️ Analyse des Mots Fréquents")
        
        st.write("""
        Les nuages de mots révèlent les patterns linguistiques distinctifs entre 
        les emails légitimes et les spams :
        """)

        try:
            st.image("plots/word_cloud.png", 
                    caption="Comparaison des mots fréquents : SPAM vs Emails Légitimes", 
                    use_container_width=True)
        except:
            st.info("Nuages de mots non disponibles. Veuillez vérifier le fichier.")

        # Interprétation des word clouds
        st.subheader("📝 Interprétation des Patterns Linguistiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚨 Mots Typiques des SPAMS")
            st.write("""
            - **"money", "make", "win"** : Promesses financières
            - **"free", "offer", "company"** : Offres trop belles
            - **"want", "need", "one"** : Langage urgent/pressant
            - **"email", "business"** : Contexte commercial agressif
            """)
            
        with col2:
            st.markdown("#### ✅ Mots Typiques des Emails Légitimes")
            st.write("""
            - **"enron", "energy", "gas"** : Contexte professionnel spécifique
            - **"meet", "time", "work"** : Communication d'entreprise
            - **"thank", "question", "help"** : Ton poli et collaboratif
            - **"project", "business"** : Discussions professionnelles
            """)

        # Distribution des longueurs
        st.subheader("📏 Analyse des Caractéristiques Textuelles")
        
        try:
            st.image("plots/distribution.png", 
                    caption="Distribution des longueurs de texte et nombre de mots par classe", 
                    use_container_width=True)
        except:
            st.info("Graphiques de distribution non disponibles. Veuillez vérifier le fichier.")

        # Interprétation des distributions
        st.subheader("📊 Insights sur les Caractéristiques Textuelles")
        
        st.write("""
        **Observations importantes :**
        
        1. **Longueur des textes :**
           - Les emails **légitimes** sont généralement plus longs (~1650 caractères en moyenne)
           - Les **spams** sont plus courts (~1350 caractères) mais plus variables
        
        2. **Nombre de mots :**
           - Les emails **légitimes** contiennent plus de mots (~350 mots)
           - Les **spams** sont plus concis (~260 mots) pour un impact rapide
        
        3. **Implications pour la détection :**
           - La longueur peut être un indicateur utile
           - Les spams privilégient la concision et l'impact
           - Les emails légitimes tendent vers plus de détails
        """)

        # Features importantes
        st.subheader("🔍 Features les Plus Discriminantes")
        
        st.write("""
        **Caractéristiques clés identifiées par les modèles :**
        
        - **Vocabulaire financier** : "money", "win", "prize", "cash"
        - **Urgence artificielle** : "urgent", "act now", "limited time"
        - **Formulations suspectes** : "click here", "free", "guarantee"
        - **Ton professionnel vs commercial** : Différence marquée dans le registre de langue
        """)

    elif page == "ℹ️ À propos":
        st.header("ℹ️ À Propos du Projet")

        st.markdown("""
        ## 🎯 Contexte du Projet

        Ce système de détection de spam a été développé par **BMSecurity** dans le cadre 
        du renforcement de la sécurité des communications. Il constitue la base d'une 
        solution évolutive destinée à être intégrée aux plateformes de messagerie.

        ## 🔬 Approche Technique

        ### **Prétraitement du Texte**
        - Conversion en minuscules
        - Suppression de la ponctuation
        - Tokenisation avec NLTK
        - Suppression des mots vides (stopwords)
        - Stemming avec PorterStemmer

        ### **Extraction des Caractéristiques**
        - Vectorisation TF-IDF (Term Frequency-Inverse Document Frequency)
        - Matrice de 5000 caractéristiques maximum
        - Support des unigrammes et bigrammes

        ### **Modèles Testés et Résultats**
        - **Support Vector Machine (SVM)** : 98.7% accuracy - **Modèle sélectionné**
        - **Naive Bayes Multinomial** : 98.1% accuracy - Très bon second choix
        - **Decision Tree Classifier** : 95.6% accuracy - Performance acceptable

        ### **Validation et Optimisation**
        - Validation croisée 5-fold
        - Optimisation des hyperparamètres avec GridSearchCV
        - Métriques : Accuracy, Precision, Recall, F1-Score

        ## 📊 Architecture du Système

        ```
        Email Input → Preprocessing → TF-IDF → SVM Model → Classification
                         ↓              ↓         ↓           ↓
                    Tokenization   Vectorization  Prediction  Result
                    Stemming       Feature         Probability  Confidence
                    Cleaning       Extraction      Score       Level
        ```

        ## 🛠️ Technologies Utilisées

        - **Python 3.8+**
        - **Scikit-learn** : Algorithmes ML et preprocessing
        - **NLTK** : Traitement du langage naturel
        - **Pandas** : Manipulation des données
        - **Streamlit** : Interface web interactive
        - **Matplotlib/Seaborn** : Visualisations
        - **WordCloud** : Nuages de mots

        ## 📈 Performances Finales (Modèle SVM)

        | Métrique | Score | Interprétation |
        |----------|-------|----------------|
        | Accuracy | 98.7% | Excellent taux de classification correcte |
        | Precision | 98.2% | Très peu de faux positifs |
        | Recall | 99.4% | Détecte presque tous les spams |
        | F1-Score | 98.8% | Excellent équilibre précision/rappel |

        ## 🔍 Insights Clés du Projet

        1. **Le modèle SVM s'est révélé supérieur** aux autres approches
        2. **Les spams utilisent un vocabulaire financier spécifique** ("money", "win", "free")
        3. **Les emails légitimes sont généralement plus longs** et détaillés
        4. **La validation croisée confirme la stabilité** des performances

        ## 🚀 Perspectives d'Amélioration

        - **Deep Learning** : Intégration de réseaux de neurones (LSTM, Transformers)
        - **Features Avancées** : Analyse des métadonnées, patterns temporels
        - **Apprentissage en Continu** : Mise à jour automatique du modèle
        - **Multi-langues** : Support d'autres langues que l'anglais

        ## 👥 Équipe de Développement

        **BMSecurity - Intelligence Artificielle Team**

        *Ce projet a été réalisé avec passion ❤️ et expertise pour protéger 
        vos communications contre les menaces numériques.*

        ---

        📧 **Contact :** yonlifidelis2@gmail.com  
        🌐 **LinkedIn :** www.linkedin.com/in/yonlifidele 
        📅 **Version :** 1.0.0 (2025)
        """)

if __name__ == "__main__":
    main()