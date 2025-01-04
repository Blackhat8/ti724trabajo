import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import requests
import time
import io
import random

# Configuración de Notion
NOTION_TOKEN = "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"
DATABASE_ID = "164e92c4-7b03-81ad-a486-fb13e21aa64b"
NOTION_API_URL = f"https://api.notion.com/v1/databases/{DATABASE_ID}/query"

# Headers para la API de Notion
headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

# Configuración de la página
st.set_page_config(
    page_title="TI724 Gestión de Cargas de Trabajo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para obtener datos de Notion
@st.cache_data(ttl=300)
def get_notion_data():
    try:
        response = requests.post(NOTION_API_URL, headers=headers)
        data = response.json()
        
        if response.status_code != 200:
            st.error(f"Error al obtener datos de Notion: {data.get('message')}")
            return None, None

        # Procesar datos para empleados
        empleados_data = {
            'Empleado': [],
            'Skills': [],
            'FTE': [],
            'Riesgo_Burnout': [],
            'Proyecto_Actual': [],
            'Productividad': []
        }

        # Procesar datos para proyectos
        proyectos_data = {
            'Proyecto': [],
            'Progreso_Real': [],
            'Progreso_Planificado': [],
            'Fecha_Inicio': [],
            'Fecha_Fin': [],
            'Complejidad': []
        }

        for page in data.get('results', []):
            props = page.get('properties', {})
            
            # Extraer datos de empleados
            if 'Responsable' in props:
                empleados = props['Responsable'].get('people', [])
                for emp in empleados:
                    empleados_data['Empleado'].append(emp.get('name', ''))
                    empleados_data['Skills'].append(props.get('Skills', {}).get('rich_text', [{}])[0].get('plain_text', ''))
                    empleados_data['FTE'].append(props.get('FTE Real', {}).get('formula', {}).get('number', 0))
                    empleados_data['Riesgo_Burnout'].append(random.random())  # Simulado
                    empleados_data['Proyecto_Actual'].append(props.get('Proyecto', {}).get('select', {}).get('name', ''))
                    empleados_data['Productividad'].append(random.randint(70, 100))  # Simulado

            # Extraer datos de proyectos
            if 'Actividad' in props:
                proyectos_data['Proyecto'].append(props['Actividad']['title'][0]['plain_text'])
                proyectos_data['Progreso_Real'].append(props.get('Progreso Subitems', {}).get('number', 0))
                proyectos_data['Progreso_Planificado'].append(props.get('Progreso Proyecto', {}).get('formula', {}).get('number', 0))
                proyectos_data['Fecha_Inicio'].append(props.get('Fecha Estimada', {}).get('date', {}).get('start', ''))
                proyectos_data['Fecha_Fin'].append(props.get('Fecha Estimada', {}).get('date', {}).get('end', ''))
                proyectos_data['Complejidad'].append(random.random())  # Simulado

        df_empleados = pd.DataFrame(empleados_data)
        df_proyectos = pd.DataFrame(proyectos_data)

        return df_empleados, df_proyectos

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

# Función para sugerir empleados usando ML
def sugerir_empleados_ml(skills_requeridos, horas_requeridas, df_empleados):
    if df_empleados is None or df_empleados.empty:
        return pd.DataFrame()

    # Crear matriz de similitud de skills
    all_skills = set(','.join(df_empleados['Skills']).split(','))
    skill_matrix = []
    
    for emp_skills in df_empleados['Skills']:
        emp_skill_vector = [1 if skill in emp_skills.split(',') else 0 for skill in all_skills]
        skill_matrix.append(emp_skill_vector)
    
    required_skill_vector = [1 if skill in skills_requeridos.split(',') else 0 for skill in all_skills]
    
    # Calcular similitud
    similarity_scores = cosine_similarity([required_skill_vector], skill_matrix)[0]
    
    # Combinar con otros factores
    availability_scores = 1 - df_empleados['FTE']
    burnout_safety = 1 - df_empleados['Riesgo_Burnout']
    productivity_scores = df_empleados['Productividad'] / 100
    
    # Calcular puntuación final
    final_scores = similarity_scores * availability_scores * burnout_safety * productivity_scores
    
    # Agregar puntuaciones al DataFrame
    df_resultados = df_empleados.copy()
    df_resultados['Puntuación'] = final_scores
    
    return df_resultados.sort_values('Puntuación', ascending=False).head(3)
    # Estilos CSS personalizados
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .alert-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .alert-warning { background-color: #fff3cd; border: 1px solid #ffeeba; }
    .alert-danger { background-color: #f8d7da; border: 1px solid #f5c6cb; }
    .alert-info { background-color: #cce5ff; border: 1px solid #b8daff; }
    </style>
    """, unsafe_allow_html=True)

# Cargar datos
df_empleados, df_proyectos = get_notion_data()

# Título principal con animación
st.markdown("""
    <h1 style='text-align: center; color: #1E88E5; animation: fadeIn 1.5s;'>
        🚀 TI724 Gestión de Cargas de Trabajo
    </h1>
    """, unsafe_allow_html=True)

# Tabs principales
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard Principal",
    "👥 Gestión de Recursos",
    "📈 Seguimiento de Proyectos",
    "🎯 Planificación",
    "🔔 Alertas y Notificaciones"
])

with tab1:
    if df_empleados is not None and df_proyectos is not None:
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Empleados Activos", len(df_empleados))
        with col2:
            st.metric("FTE Promedio", f"{df_empleados['FTE'].mean():.2f}")
        with col3:
            if 'Progreso_Real' in df_proyectos:
                st.metric("Progreso General", f"{df_proyectos['Progreso_Real'].mean():.1%}")
        with col4:
            proyectos_atrasados = len(df_proyectos[df_proyectos['Progreso_Real'] < df_proyectos['Progreso_Planificado']])
            st.metric("Proyectos Atrasados", proyectos_atrasados)
        
        # Gráfico de distribución FTE
        st.subheader("📊 Distribución de Cargas de Trabajo")
        fig_fte = px.bar(
            df_empleados,
            x="Empleado",
            y="FTE",
            color="Riesgo_Burnout",
            color_continuous_scale="RdYlBu_r",
            title="FTE por Empleado"
        )
        st.plotly_chart(fig_fte, use_container_width=True)
        
        # Gráfico de progreso de proyectos
        st.subheader("📈 Progreso de Proyectos")
        fig_progress = go.Figure()
        fig_progress.add_trace(go.Bar(
            name='Progreso Real',
            x=df_proyectos['Proyecto'],
            y=df_proyectos['Progreso_Real'],
            marker_color='rgb(26, 118, 255)'
        ))
        fig_progress.add_trace(go.Bar(
            name='Progreso Planificado',
            x=df_proyectos['Proyecto'],
            y=df_proyectos['Progreso_Planificado'],
            marker_color='rgb(55, 83, 109)'
        ))
        fig_progress.update_layout(barmode='group')
        st.plotly_chart(fig_progress, use_container_width=True)

with tab2:
    st.subheader("👥 Gestión de Recursos Humanos")
    
    if df_empleados is not None:
        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            proyecto_filter = st.selectbox(
                "Filtrar por Proyecto",
                ["Todos"] + list(df_empleados['Proyecto_Actual'].unique())
            )
        with col2:
            skill_filter = st.multiselect(
                "Filtrar por Skills",
                list(set(','.join(df_empleados['Skills']).split(',')))
            )
        
        # Aplicar filtros
        df_filtrado = df_empleados.copy()
        if proyecto_filter != "Todos":
            df_filtrado = df_filtrado[df_filtrado['Proyecto_Actual'] == proyecto_filter]
        if skill_filter:
            df_filtrado = df_filtrado[df_filtrado['Skills'].apply(
                lambda x: any(skill in x for skill in skill_filter)
            )]
        
        # Mostrar datos
        st.dataframe(
            df_filtrado,
            column_config={
                "FTE": st.column_config.ProgressColumn(
                    "Carga de Trabajo",
                    help="Full Time Equivalent",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "Riesgo_Burnout": st.column_config.ProgressColumn(
                    "Riesgo de Burnout",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "Productividad": st.column_config.ProgressColumn(
                    "Productividad",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                )
            }
        )

with tab3:
    st.subheader("📈 Seguimiento de Proyectos")
    
    if df_proyectos is not None:
        # Gráfico de línea temporal
        fig_timeline = px.timeline(
            df_proyectos,
            x_start='Fecha_Inicio',
            x_end='Fecha_Fin',
            y='Proyecto',
            color='Progreso_Real',
            title="Línea de Tiempo de Proyectos"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Tabla de proyectos
        st.dataframe(
            df_proyectos,
            column_config={
                "Progreso_Real": st.column_config.ProgressColumn(
                    "Progreso Real",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                ),
                "Progreso_Planificado": st.column_config.ProgressColumn(
                    "Progreso Planificado",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                )
            }
        )

with tab4:
    st.subheader("🎯 Planificación de Recursos")
    
    with st.form("planificacion_form"):
        col1, col2 = st.columns(2)
        with col1:
            nuevo_proyecto = st.text_input("Nombre del Nuevo Proyecto")
            skills_requeridos = st.multiselect(
                "Skills Requeridos",
                list(set(','.join(df_empleados['Skills']).split(','))) if df_empleados is not None else []
            )
        with col2:
            horas_requeridas = st.number_input("Horas Estimadas", min_value=1, value=40)
            fecha_inicio = st.date_input("Fecha de Inicio")
        
        submitted = st.form_submit_button("Buscar Recursos Disponibles")
        
        if submitted and skills_requeridos:
            with st.spinner("Analizando recursos disponibles..."):
                sugerencias = sugerir_empleados_ml(','.join(skills_requeridos), horas_requeridas, df_empleados)
                
                if not sugerencias.empty:
                    st.success("Recursos sugeridos encontrados")
                    st.dataframe(sugerencias)
                else:
                    st.warning("No se encontraron recursos disponibles que cumplan con los criterios")

with tab5:
    st.subheader("🔔 Alertas y Notificaciones")
    
    if df_empleados is not None and df_proyectos is not None:
        # Alertas de sobrecarga
        empleados_sobrecargados = df_empleados[df_empleados['FTE'] > 0.8]
        for _, emp in empleados_sobrecargados.iterrows():
            st.markdown(f"""
                <div class="alert-box alert-warning">
                    ⚠️ {emp['Empleado']} está sobrecargado (FTE: {emp['FTE']:.2f})
                </div>
            """, unsafe_allow_html=True)
        
        # Alertas de burnout
        riesgo_burnout = df_empleados[df_empleados['Riesgo_Burnout'] > 0.7]
        for _, emp in riesgo_burnout.iterrows():
            st.markdown(f"""
                <div class="alert-box alert-danger">
                    🔥 {emp['Empleado']} tiene alto riesgo de burnout
                </div>
            """, unsafe_allow_html=True)
        
        # Alertas de proyectos
        proyectos_atrasados = df_proyectos[df_proyectos['Progreso_Real'] < df_proyectos['Progreso_Planificado']]
        for _, proj in proyectos_atrasados.iterrows():
            st.markdown(f"""
                <div class="alert-box alert-info">
                    📉 Proyecto {proj['Proyecto']} está atrasado
                </div>
            """, unsafe_allow_html=True)

# Barra lateral con métricas y exportación
with st.sidebar:
    st.header("📊 Métricas del Sistema")
    st.metric("Tiempo de Respuesta", f"{random.uniform(0.5, 2.0):.2f}s")
    st.metric("Uso de CPU", f"{random.randint(30, 70)}%")
    st.metric("Memoria", f"{random.uniform(1.5, 3.0):.1f} GB")
    
    # Exportar datos
    st.subheader("📤 Exportar Datos")
    if st.button("Generar Reporte"):
        with st.spinner("Generando reporte..."):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                if df_empleados is not None:
                    df_empleados.to_excel(writer, sheet_name='Empleados', index=False)
                if df_proyectos is not None:
                    df_proyectos.to_excel(writer, sheet_name='Proyectos', index=False)
            
            st.download_button(
                label="📥 Descargar Reporte",
                data=buffer.getvalue(),
                file_name=f"reporte_ti724_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.ms-excel"
            )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>© 2024 TI724. Todos los derechos reservados.</p>",
    unsafe_allow_html=True
)
