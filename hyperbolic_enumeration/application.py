import streamlit as st
import openturns.viewer as otv
import matplotlib.pyplot as plt
import pandas as pd
import plot_hyperbolic_anisotropic_rule_lib as pharlib

# Configuration de la page
st.set_page_config(page_title="Stratification Chaos Polynomial", layout="wide")

# --- INTERFACE STREAMLIT ---

st.title("Visualisation de la stratification des multi-indices")
st.markdown("""
Cette application visualise la répartition des multi-indices pour le chaos polynomial 
selon une norme $q$ pondérée (énumération anisotrope hyperbolique).
""")

# Barre latérale pour les widgets
st.sidebar.header("Paramètres")
q_param = st.sidebar.slider("Valeur de q", min_value=0.1, max_value=1.0, value=0.7, step=0.05)
w1 = st.sidebar.slider("Poids w1", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
w2 = st.sidebar.slider("Poids w2", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
nb_strates = st.sidebar.number_input("Nombre de strates", min_value=2, max_value=15, value=8)

# Sélecteur pour l'indice marginal maximum
max_marginal = st.sidebar.number_input(
    "Indice marginal maximum", 
    min_value=1, 
    max_value=16, 
    value=8, 
    step=1
)

weights = [w1, w2]

# Calculs (Utilisation de nb_strates ici)
graph, data_table, maximum_q_norm = pharlib.plot_hyperbolic_anisotropic_rule(
    weights, q_param, maximum_strata_index=int(nb_strates), maximum_marginal_index=max_marginal
)

# Préparation unique du DataFrame avec correction
df = pd.DataFrame(data_table, columns=["Indice", "Couche", "Degré", "Multi-indice", "q-norm"])
# Conversion systématique en chaîne pour PyArrow
df["Multi-indice"] = df["Multi-indice"].apply(lambda x: str(list(x)))

# --- INDICATEUR GÉNÉRAL ---
# Calcul du nombre total
nb_points = len(df)

# Création de deux colonnes pour les métriques
m1, m2 = st.columns(2)

with m1:
    st.metric(label="Nombre total de multi-indices", value=nb_points)

with m2:
    # Formatage optionnel à 4 décimales pour la propreté visuelle
    st.metric(label="Q-Norme maximale", value=f"{maximum_q_norm:.4f}")


st.divider() # Petite ligne de séparation visuelle

# Affichage du graphique et du tableau
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Graphique des strates")
    # OpenTURNS View
    view = otv.View(graph, axes_kw={"aspect": "equal"}, figure_kw={"figsize": (5, 4)})
    fig = view.getFigure()
    st.pyplot(fig)

with col2:
    st.subheader("Données des indices")

    # On utilise le DataFrame déjà corrigé
    st.dataframe(df, width='stretch', hide_index=True, height=400)

st.info("Les lignes de niveau représentent la frontière théorique de chaque strate selon la norme q.")