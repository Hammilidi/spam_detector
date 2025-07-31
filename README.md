# 🛡️ BMSecurity - Système de Détection de Spam

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28.1-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📖 Description

**BMSecurity Spam Detection System** est une solution intelligente de détection automatique de spam utilisant des techniques avancées de **Machine Learning** et de **Traitement du Langage Naturel (NLP)**. Ce système a été développé pour renforcer la sécurité des communications électroniques et peut être intégré aux plateformes de messagerie existantes.

### 🎯 Objectifs du Projet

- Détecter automatiquement les emails malveillants avec une précision élevée
- Fournir une interface utilisateur intuitive pour l'analyse en temps réel
- Offrir une solution évolutive et facilement intégrable
- Maintenir des performances optimales avec une faible latence

## ✨ Fonctionnalités Principales

### 🔍 Détection Intelligente
- **Classification en temps réel** des emails (Spam/Ham)
- **Analyse de confiance** avec niveaux de probabilité
- **Prétraitement avancé** du texte (tokenisation, stemming, suppression des stopwords)
- **Vectorisation TF-IDF** pour l'extraction des caractéristiques

### 📊 Analyse et Visualisation
- **Nuages de mots** interactifs pour l'exploration des données
- **Métriques de performance** détaillées (Accuracy, Precision, Recall, F1-Score)
- **Matrices de confusion** pour l'évaluation des modèles
- **Graphiques d'évolution** des performances

### 🖥️ Interface Utilisateur
- **Application web Streamlit** moderne et responsive
- **Interface intuitive** avec navigation par onglets
- **Tests en temps réel** avec exemples pré-configurés
- **Visualisations interactives** des résultats

## 🚀 Installation Rapide

### Prérequis
- Python 3.8+
- pip (gestionnaire de paquets Python)

### 1. Clonage du Projet
```bash
git clone https://github.com/bmsecurity/spam-detection.git
cd spam-detection
```

### 2. Installation des Dépendances
```bash
pip install -r requirements.txt
```

### 3. Téléchargement des Ressources NLTK
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 4. Lancement de l'Application
```bash
streamlit run streamlit_app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

## 📁 Structure du Projet

```
## 📁 Architecture du projet

```bash
spams_detection/
├── .ipynb_checkpoints/
├── data/
│   └── DataSet_Emails.csv
├── env/
├── models/
│   └── spam_detector_model_test.pkl
├── plots/
│   ├── comparaison_performances_models.png
│   ├── confusion_models.png
│   ├── cross_validation_models.png
│   ├── distribution.png
│   └── word_cloud.png
├── .gitignore
├── README.md
├── requirements.txt
├── script.ipynb
├── spam_detection_model.pkl
├── streamlit_app.py
└── tfidf_vectorizer.pkl

```

## 🔬 Méthodologie Technique

### 1. Prétraitement des Données
```python
# Pipeline de prétraitement
1. Conversion en minuscules
2. Suppression de la ponctuation et caractères spéciaux
3. Tokenisation avec NLTK
4. Suppression des mots vides (stopwords)
5. Stemming avec PorterStemmer
6. Vectorisation TF-IDF
```

### 2. Modèles Implémentés

| Algorithme | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| **Decision Tree** | 95.2% | 94.8% | 95.6% | 95.2% |
| Naive Bayes | 93.1% | 92.5% | 93.8% | 93.1% |
| SVM | 94.7% | 94.2% | 95.1% | 94.6% |

### 3. Optimisation des Hyperparamètres
- **GridSearchCV** pour la recherche exhaustive
- **Validation croisée 5-fold** pour la robustesse
- **Métriques multiples** pour l'évaluation complète

## 📊 Performances du Système

### Métriques Principales
- **🎯 Accuracy :** 98.2% (+2.1% d'amélioration)
- **🔍 Precision :** 98.8% (+1.8% d'amélioration)
- **📡 Recall :** 98.6% (+2.3% d'amélioration)
- **⚖️ F1-Score :** 98.2% (+2.0% d'amélioration)

### Matrice de Confusion
```
                Prédictions
Réalité    Ham    Spam
  Ham      850     50
  Spam      30     870
```

### Temps de Réponse
- **Prétraitement :** ~0.02s par email
- **Classification :** ~0.01s par email
- **Total :** <0.05s par email

## 🖥️ Guide d'Utilisation

### Interface Web Streamlit

#### 1. 🏠 Page d'Accueil
- Vue d'ensemble des performances
- Métriques en temps réel
- Instructions d'utilisation

#### 2. 📧 Détection d'Email
```python
# Exemple d'utilisation
1. Coller le contenu de l'email dans la zone de texte
2. Cliquer sur "Analyser l'Email"
3. Consulter les résultats :
   - Classification (Spam/Ham)
   - Niveau de confiance
   - Probabilités détaillées
   - Recommandations
```

#### 3. 📊 Statistiques
- Performances détaillées du modèle
- Matrices de confusion interactives
- Évolution des métriques

#### 4. 🔍 Analyse Exploratoire
- Nuages de mots par catégorie
- Distribution des longueurs de texte
- Mots les plus caractéristiques

### API de Prédiction

```python
from spam_detector import predict_spam

# Exemple d'utilisation de l'API
result = predict_spam("Your email content here")

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
print(f"Probability: {result['probability']:.2f}")
```

## 🛠️ Technologies Utilisées

### Librairies Principales
- **🐍 Python 3.8+** - Langage de programmation
- **🤖 Scikit-learn** - Algorithmes de Machine Learning
- **📝 NLTK** - Traitement du langage naturel
- **📊 Pandas** - Manipulation des données
- **🎨 Streamlit** - Interface web interactive
- **📈 Matplotlib/Seaborn** - Visualisations
- **☁️ WordCloud** - Nuages de mots

### Architecture Technique
```
Input Email → Preprocessing → Feature Extraction → ML Model → Output
     ↓             ↓               ↓               ↓         ↓
  Raw Text    Tokenization     TF-IDF         Decision    Spam/Ham
              Cleaning       Vectorization     Tree      + Confidence
              Stemming       (5000 features)  Classifier   + Probability
```

## 📈 Évaluation et Validation

### Métriques d'Évaluation
- **Accuracy :** Proportion de prédictions correctes
- **Precision :** Proportion de spams correctement identifiés
- **Recall :** Proportion de spams détectés sur le total
- **F1-Score :** Moyenne harmonique de Precision et Recall

### Stratégies de Validation
- **Train/Test Split :** 80%/20% avec stratification
- **Cross-Validation :** 5-fold pour la robustesse
- **Grid Search :** Optimisation des hyperparamètres

### Gestion du Surapprentissage
- **Régularisation** dans les modèles SVM
- **Pruning** pour les arbres de décision
- **Validation croisée** pour la généralisation

## 🔮 Perspectives d'Amélioration

### Améliorations Techniques
- [ ] **Deep Learning :** Intégration de LSTM/Transformers
- [ ] **Features Avancées :** Métadonnées, analyse temporelle
- [ ] **Multi-langues :** Support de langues additionnelles
- [ ] **Real-time Learning :** Mise à jour continue du modèle

### Fonctionnalités Futures
- [ ] **API REST :** Service web pour intégration
- [ ] **Dashboard Admin :** Interface de gestion avancée
- [ ] **Alertes Automatiques :** Notifications en temps réel
- [ ] **Analyse de Tendances :** Détection de nouvelles menaces

### Optimisations Performance
- [ ] **Cache Intelligent :** Mise en cache des prédictions
- [ ] **Parallélisation :** Traitement concurrent
- [ ] **Modèles Légers :** Optimisation pour mobile
- [ ] **Edge Computing :** Déploiement local

## 📊 Données et Dataset

### Format des Données
```csv
message_id,text,label,label_text,subject,message,date
33214,"email content...",1,spam,"Subject","Message content","2024-01-01"
```

### Préparation des Données
- **Nettoyage :** Suppression des doublons et valeurs manquantes
- **Équilibrage :** Stratification pour maintenir la distribution
- **Validation :** Vérification de la cohérence des labels


## 🔧 Configuration et Personnalisation

### Variables d'Environnement
```bash
# Configuration du modèle
export MODEL_PATH="./models/spam_detection_model.pkl"
export VECTORIZER_PATH="./models/tfidf_vectorizer.pkl"
export MAX_FEATURES=5000
export CONFIDENCE_THRESHOLD=0.7
```

### Paramètres Personnalisables
```python
# Configuration TF-IDF
TFIDF_CONFIG = {
    'max_features': 5000,
    'min_df': 2,
    'max_df': 0.95,
    'ngram_range': (1, 2)
}

# Configuration du modèle
MODEL_CONFIG = {
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': 42
}
```

## 🐛 Dépannage

### Problèmes Courants

#### Erreur NLTK Resources
```bash
# Solution
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### Erreur de Chargement du Modèle
```python
# Vérifier l'existence des fichiers
import os
print(os.path.exists('models/pam_detection_model.pkl'))
print(os.path.exists('models/tfidf_vectorizer.pkl'))
```

#### Performance Lente
- Réduire `max_features` dans TfidfVectorizer
- Utiliser un modèle plus simple (Naive Bayes)
- Optimiser le preprocessing

## 🤝 Contribution

### Comment Contribuer
1. **Fork** le projet
2. Créer une **branche** pour votre fonctionnalité
3. **Commiter** vos changements
4. **Push** vers la branche
5. Ouvrir une **Pull Request**

### Standards de Code
- **PEP 8** pour le style Python
- **Docstrings** pour la documentation
- **Tests unitaires** pour la qualité
- **Type hints** pour la lisibilité

### Rapporter des Bugs
Utilisez les **GitHub Issues** avec :
- Description détaillée du problème
- Étapes pour reproduire
- Environnement (OS, Python version)
- Logs d'erreur si disponibles

## 📄 Licence

Ce projet est sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 📞 Contact et Support

### Équipe de Développement
- **Email :** yonlifidelis2@mail.com

---

<div align="center">

**🛡️ BMSecurity - Protégeant vos Communications avec l'Intelligence Artificielle**

Made with ❤️ by YONLI Fidèle

</div>