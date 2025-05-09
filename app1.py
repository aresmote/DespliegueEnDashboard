
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Configuración inicial
st.set_page_config(layout="wide", page_title="Análisis de Datos Airbnb", page_icon="🏠")

# Iconos para cada vista
icons = {
    "univariado": "📊",
    "lineal_simple": "📈",
    "logistica": "🔮",
    "multiple": "🧮"
}

# Cargar datasets
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "OTTAWA":
        data = pd.read_csv('ottawa_original_clean.csv')
    elif dataset_name == "MÉXICO":
        data = pd.read_csv('mexico_original_clean.csv')
    elif dataset_name == "LOS ÁNGELES":
        data = pd.read_csv('california_original_clean.csv')
    elif dataset_name == "BARCELONA":
        data = pd.read_csv('barcelona_original_clean.csv')
    return data

# Sidebar principal
st.sidebar.title("Menu de Navegación")
st.sidebar.image("airbnb.jpg", width=300)

# Selección de dataset
dataset = st.sidebar.selectbox("Selecciona Dataset", ["OTTAWA", "MÉXICO", "LOS ÁNGELES", "BARCELONA"],)
data = load_data(dataset)

# Preprocesamiento básico
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
text_cols = data.select_dtypes(include=['object']).columns

# Selección de vista
view = st.sidebar.selectbox(
    "Selecciona Tipo de Análisis",
    ["Análisis Univariado", "Regresión Lineal Simple", "Regresión Logística", "Regresión Múltiple"],
    format_func=lambda x: f"{icons.get(x.split()[1].lower(), '❓')} {x}"

)

# --------------------------
# VISTA 1: ANÁLISIS UNIVARIADO
# --------------------------
if view == "Análisis Univariado":
    st.title(f"{icons['univariado']} Análisis Univariado - {dataset}")
    
    if dataset == "OTTAWA":
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image("ottawa.jpg", width=600)
    elif dataset == "MÉXICO":
        col1, col2, col3 = st.columns([1,2,1])
        with col2: 
            st.image("CDMX.jpeg", width=600)
    elif dataset == "LOS ÁNGELES":
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image("LA.jpg", width=600)
    elif dataset == "BARCELONA":
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image("barcelona.jpg", width=600)
    
    # Fila de métricas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Registros", len(data))
    col2.metric("Variables Numéricas", len(numeric_cols))
    col3.metric("Variables Categóricas", len(text_cols))
    col4.metric("Valores Faltantes", data.isnull().sum().sum())
    
    st.title("Dataset Completo")
    st.dataframe(data, height=300)
    
    # Selector de tipo de gráfico
    chart_type = st.selectbox(
        "Tipo de Gráfico",
        ["Histograma", "Boxplot", "Diagrama de Barras", "Gráfico de Torta", "Heatmap"]
    )
    
    # Multiselect para variables
    selected_vars = st.multiselect(
        "Selecciona variables para visualizar",
        numeric_cols if chart_type in ["Histograma", "Boxplot", "Heatmap"] else text_cols
    )
    
    # Generación de gráficos
    if selected_vars:
        if chart_type == "Histograma":
            fig = px.histogram(data, x=selected_vars[0], nbins=30, title=f"Distribución de {selected_vars[0]}")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Boxplot":
            fig = px.box(data, y=selected_vars, title="Diagrama de Caja")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Diagrama de Barras":
            if len(selected_vars) == 1:
                fig = px.bar(data[selected_vars[0]].value_counts(), title=f"Frecuencia de {selected_vars[0]}")
                st.plotly_chart(fig, use_container_width=True)
                
        elif chart_type == "Gráfico de Torta":
            fig = px.pie(data, names=selected_vars[0], title=f"Distribución de {selected_vars[0]}")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Heatmap":
            corr = data[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de resumen
    if st.checkbox("Mostrar estadísticas descriptivas"):
        st.dataframe(data.describe(), use_container_width=True)

# --------------------------
# VISTA 2: REGRESIÓN LINEAL SIMPLE
# --------------------------
elif view == "Regresión Lineal Simple":
    st.title(f"{icons['lineal_simple']} Regresión Lineal Simple - {dataset}")
    
    # Selección de variables
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Variable Independiente (X)", numeric_cols)
    with col2:
        y_var = st.selectbox("Variable Dependiente (Y)", numeric_cols)
    
    # Gráfico de dispersión con línea de regresión
    fig = px.scatter(data, x=x_var, y=y_var, trendline="ols", 
                    title=f"Regresión Lineal: {y_var} ~ {x_var}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Entrenamiento del modelo
    X = data[[x_var]].values
    y = data[y_var].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Resultados del modelo
    st.subheader("Resultados del Modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("Intercepto", round(model.intercept_, 4))
    col2.metric("Coeficiente", round(model.coef_[0], 4))
    col3.metric("Error Cuadrático Medio", round(mse, 2))
    
    # Gráfico de residuales
    residuals = y_test - y_pred
    fig = px.scatter(x=y_pred, y=residuals, 
                    labels={"x": "Predicciones", "y": "Residuales"},
                    title="Análisis de Residuales")
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# VISTA 3: REGRESIÓN LOGÍSTICA
# --------------------------
elif view == "Regresión Logística":
    st.title(f"{icons['logistica']} Regresión Logística - {dataset}")
    
    # Verificar si hay variable binaria
    binary_vars = [col for col in numeric_cols if data[col].nunique() == 2]
    
    if not binary_vars:
        st.warning("No se encontraron variables binarias para regresión logística.")
        if st.checkbox("Crear variable binaria artificial"):
            selected_num = st.selectbox("Selecciona variable numérica para binarizar", numeric_cols)
            median_val = data[selected_num].median()
            data['target'] = (data[selected_num] > median_val).astype(int)
            binary_vars = ['target']
            st.success(f"Variable binaria creada: 'target' (1 si {selected_num} > {median_val:.2f})")
    
    if binary_vars:
        # Selección de variables
        col1, col2 = st.columns(2)
        with col1:
            y_var = st.selectbox("Variable Objetivo (Y)", binary_vars)
        with col2:
            x_var = st.selectbox("Variable Predictora (X)", numeric_cols)
        
        # Gráfico de dispersión con regresión logística
        fig = px.scatter(data, x=x_var, y=y_var, 
                        title=f"Regresión Logística: {y_var} ~ {x_var}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Ajustar modelo
        X = data[[x_var]].values
        y = data[y_var].values
        
        # Estandarizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Métricas
        from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Mostrar métricas
        st.subheader("📊 Métricas del Modelo")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Exactitud", f"{accuracy*100:.2f}%")
        col2.metric("Sensibilidad", f"{recall*100:.2f}%")
        col3.metric("Precisión", f"{precision*100:.2f}%")
        col4.metric("F1-Score", f"{f1*100:.2f}%")
        
        # Matriz de confusión corregida
        cm = confusion_matrix(y_test, y_pred)
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicho 0', 'Predicho 1'],
            y=['Real 0', 'Real 1'],
            text=[[f"VN: {cm[0,0]}", f"FP: {cm[0,1]}"], 
                 [f"FN: {cm[1,0]}", f"VP: {cm[1,1]}"]],
            texttemplate="%{text}",
            colorscale='Blues',
            showscale=False
        ))
        
        fig.update_layout(
            title="Matriz de Confusión",
            xaxis_title="Predicción",
            yaxis_title="Valor Real"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explicación de métricas
        with st.expander("🔍 Explicación de las Métricas"):
            st.markdown("""
            - **Exactitud (Accuracy):** Porcentaje de predicciones correctas
            - **Sensibilidad (Recall):** Capacidad de detectar casos positivos
            - **Precisión:** Exactitud en predicciones positivas
            - **F1-Score:** Balance entre precisión y recall
            """)

# --------------------------
# VISTA 4: REGRESIÓN MÚLTIPLE
# --------------------------
elif view == "Regresión Múltiple":
    st.title(f"{icons['multiple']} Regresión Múltiple - {dataset}")
    
    # Selección de variables
    y_var = st.selectbox("Variable Dependiente (Y)", numeric_cols)
    x_vars = st.multiselect("Variables Independientes (X)", numeric_cols.drop(y_var))
    
    if x_vars:
        # Entrenamiento del modelo
        X = data[x_vars].values
        y = data[y_var].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Resultados del modelo
        st.subheader("Resultados del Modelo")
        st.write(f"**R²:** {model.score(X_test, y_test):.4f}")
        st.write(f"**Error Cuadrático Medio (MSE):** {mse:.4f}")
        
        # Coeficientes
        st.subheader("Coeficientes del Modelo")
        coef_df = pd.DataFrame({
            "Variable": ["Intercepto"] + x_vars,
            "Coeficiente": [model.intercept_] + list(model.coef_)
        })
        st.dataframe(coef_df, use_container_width=True)
        
        # Gráfico de importancia de variables
        st.subheader("Importancia de Variables")
        fig = px.bar(coef_df[1:].sort_values("Coeficiente", key=abs), 
                    x="Variable", y="Coeficiente", 
                    color="Coeficiente", color_continuous_scale="Viridis",
                    title="Magnitud de Coeficientes")
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de predicciones vs reales
        st.subheader("Predicciones vs Valores Reales")
        fig = px.scatter(x=y_test, y=y_pred, 
                        labels={"x": "Valores Reales", "y": "Predicciones"},
                        title="Comparación de Predicciones")
        fig.add_shape(type="line", x0=min(y_test), y0=min(y_test), 
                     x1=max(y_test), y1=max(y_test),
                     line=dict(color="Red", width=2, dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Dashboard de Análisis Airbnb**")
st.sidebar.markdown("Inteligencia de Negocios")
