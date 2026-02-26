import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Iris Classifier Pro", layout="wide")

# Título y Descripción
st.title("🌸 Clasificador Dinámico: Dataset IRIS")
st.markdown("""
Esta aplicación permite entrenar un modelo de Machine Learning en tiempo real, 
analizar su desempeño y probarlo con datos personalizados.
""")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    target_names = iris.target_names
    return df, target_names, iris.feature_names

df, target_names, feature_names = load_data()

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("⚙️ Configuración del Modelo")
test_size = st.sidebar.slider("Tamaño del set de prueba (%)", 10, 50, 20) / 100
n_estimators = st.sidebar.select_slider("Número de árboles (n_estimators)", options=[10, 50, 100, 200], value=100)

# --- ENTRENAMIENTO ---
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- DISEÑO DE LA APP (Pestañas) ---
tab1, tab2, tab3 = st.tabs(["📊 Exploración y Desempeño", "🧠 Predicción en Vivo", "📝 Reporte Técnico"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Clases")
        fig_scatter = px.scatter_matrix(df, dimensions=feature_names, color="target",
                                      labels={'target': 'Especie'},
                                      title="Matriz de Dispersión Iris")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.subheader("Métricas de Desempeño")
        st.metric("Exactitud (Accuracy)", f"{acc:.2%}")
        
        # Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        st.pyplot(fig_cm)

with tab2:
    st.subheader("🧪 Probar el Modelo")
    st.write("Ajusta los parámetros de la flor para ver la predicción en tiempo real:")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        s_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
        s_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    
    with col_input2:
        p_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
        p_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)
    
    # Predicción manual
    input_data = [[s_length, s_width, p_length, p_width]]
    prediction = clf.predict(input_data)
    prediction_proba = clf.predict_proba(input_data)
    
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.success(f"### Predicción: **{target_names[prediction[0]].upper()}**")
    
    with res_col2:
        # Gráfica de Probabilidades
        prob_df = pd.DataFrame(prediction_proba, columns=target_names)
        st.bar_chart(prob_df.T)

with tab3:
    st.subheader("Análisis de Importancia")
    importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    st.bar_chart(importances)
    st.info("Esta gráfica muestra qué características (pétalo o sépalo) fueron más determinantes para el modelo.")
    
    st.subheader("Reporte de Clasificación")
    st.text(classification_report(y_test, y_pred, target_names=target_names))
