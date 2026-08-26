import streamlit as st
import pandas as pd
import os
import numpy as np
import re
from scipy.integrate import quad
import plotly.graph_objects as go

st.set_page_config(page_title="MinGeoChem", page_icon="🌋", layout="wide")
st.title("🌋 MinGeoChem")
st.markdown("Herramienta didáctica para la visualización de propiedades ópticas, cristalográficas y geoquímicas. *Desarrollado por Ernesto Álvarez.*")

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
@st.cache_data
def load_optical_data():
    ruta_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'minerales_ugr_FINAL.csv')
    df = pd.read_csv(ruta_csv, encoding='utf-8-sig')
    return df

@st.cache_data
def load_thermo_data():
    ruta_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.csv')
    if os.path.exists(ruta_csv):
        df = pd.read_csv(ruta_csv, encoding='utf-8-sig')
        for col in ['DeltaH', 'DeltaS', 'a', 'b', 'c', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['DeltaH', 'DeltaS', 'a', 'b', 'c', 'Volume'])
    return None

df_optica = load_optical_data()
df_thermo = load_thermo_data()

# ==========================================
# 2. FUNCIONES AUXILIARES
# ==========================================
def limpiar_imagenes(url_string):
    if pd.isna(url_string): return []
    imagenes = str(url_string).replace('\n', '').split('|')
    blacklist = ['sil', 'nesop', 'cuadp', 'sorop', 'ciclop', 'inop', 'filop', 'tectop', 'imag', 'camp', 'anal', 'den', 'desc', 'amp', 'mas', 'tit', 'rest', 'sinesc', 'escol', 'inicio', 'propiedades', 'tutorial', 'identifica', 'nuevo', 'ol.gif', 'imagcom', 'imanal', 'immas']
    buenas = []
    for img in imagenes:
        img = img.strip()
        if not img.startswith('http'): continue
        img_lower = img.lower()
        if any(b in img_lower for b in blacklist): continue
        if not img_lower.endswith('.jpg'): continue
        if img not in buenas:
            buenas.append(img)
    return buenas

def get_imagenes_con_caption(url_string):
    imgs = limpiar_imagenes(url_string)
    result = []
    for img in imgs:
        fname = img.split('/')[-1].lower().replace('.jpg', '')
        if fname.endswith('x'):
            result.append((img, "Vista en Nicol Cruzado (XPL)"))
        else:
            result.append((img, "Vista en Nicol Paralelo (PPL)"))
    return result

def parsear_propiedades(texto, mineral):
    if pd.isna(texto): return pd.DataFrame({"Propiedad": ["N/A"], "Valor": ["N/A"]})
    texto = str(texto).replace("(Mg,Fe)2SiO4 (Mg,Fe)2SiO4", "").strip()
    texto = re.sub(r'\s+', ' ', texto)
    inicio = texto.find(mineral.upper())
    if inicio != -1:
        texto = texto[inicio:]
        final = texto.find("Seleccione")
        if final != -1: texto = texto[:final]
    match_formula = re.search(r'^[A-ZÁÉÍÓÚÑ\(\) ]+([A-Za-z0-9\(\)\s\.,]+?)Hábito', texto)
    formula = match_formula.group(1).strip() if match_formula else "N/A"
    keys = ["Hábito", "Exfoliación", "Maclas", "Color/Pleocroismo", "Relieve", "Ng", "Nm", "Np", "Birrefringencia", "Color de interferencia", "Ángulo de extinción", "Signo de elongación", "Tipo óptico", "Signo Óptico", "2V", "Otros datos", "Alteración"]
    pattern = '|'.join([re.escape(k) for k in keys]) + r':'
    matches = list(re.finditer(pattern, texto))
    data = {"Propiedad": ["Fórmula Química"], "Valor": [formula]}
    for i, match in enumerate(matches):
        key = match.group(0).replace(':', '')
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(texto)
        value = texto[start:end].strip()
        data["Propiedad"].append(key)
        data["Valor"].append(value)
    return pd.DataFrame(data)

def cp_mineral(T, a, b, c):
    return a + b*T - c/(T**2)

def get_crystal_vertices(a, b, c, alpha, beta, gamma):
    alpha, beta, gamma = np.radians([alpha, beta, gamma])
    v1 = np.array([a, 0, 0])
    v2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(0, c**2 - cx**2 - cy**2))
    v3 = np.array([cx, cy, cz])
    return [np.array([0,0,0]), v1, v2, v3, v1+v2, v1+v3, v2+v3, v1+v2+v3]

def calculate_clapeyron_curve(phase1, phase2, T_ref_K=773.15, P_ref_bar=4000, T_final_K=1000):
    """Calcula la curva P-T dinámica para una reacción de 2 fases."""
    reaccion = {phase1: -1, phase2: 1}
    dH_ref = dS_ref = dV_total = 0
    
    for min, coef in reaccion.items():
        datos = df_thermo[df_thermo['Phase'] == min].iloc[0]
        dH_ref += coef * datos['DeltaH'] * 1000
        dS_ref += coef * datos['DeltaS']
        dV_total += coef * datos['Volume']
        
    if dV_total == 0: return None, None
    
    def get_dS_at_T(T):
        dS = 0
        for min, coef in reaccion.items():
            datos = df_thermo[df_thermo['Phase'] == min].iloc[0]
            int_S = quad(lambda t: cp_mineral(t, datos['a'], datos['b'], datos['c'])/t, 298.15, T)[0]
            dS += coef * (datos['DeltaS'] + int_S)
        return dS

    T_curva = np.linspace(T_ref_K, T_final_K, 25)
    P_curva_bar = [P_ref_bar]
    
    for i in range(1, len(T_curva)):
        dS_prev = get_dS_at_T(T_curva[i-1])
        dS_curr = get_dS_at_T(T_curva[i])
        dP_dT = ((dS_prev + dS_curr) / 2) / dV_total
        dT = T_curva[i] - T_curva[i-1]
        P_curva_bar.append(P_curva_bar[-1] + dP_dT * dT)
    
    T_C = T_curva - 273.15
    P_GPa = np.array(P_curva_bar) / 10000
    return T_C, P_GPa

# ==========================================
# 3. NAVEGACIÓN ESTABLE (SIN BUGS)
# ==========================================
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "📖 Atlas Óptico"

opciones_tabs = ["📖 Atlas Óptico", "📐 Simetría 3D", "🧪 Geoquímica", "🎮 Quiz", "🌋 Diagrama de Fases"]

st.session_state.active_tab = st.radio(
    "Navegación", opciones_tabs, 
    index=opciones_tabs.index(st.session_state.active_tab),
    horizontal=True, label_visibility="collapsed"
)
st.markdown("---")

# ==========================================
# 4. LÓGICA DE PESTAÑAS (IF / ELIF)
# ==========================================

# ------------------------------------------
# PESTAÑA 1: ATLAS ÓPTICO
# ------------------------------------------
if st.session_state.active_tab == "📖 Atlas Óptico":
    st.subheader("Selecciona un mineral")
    mineral_seleccionado = st.selectbox("Mineral:", df_optica['Mineral'].unique(), key='sel_mineral')
    fila = df_optica[df_optica['Mineral'] == mineral_seleccionado].iloc[0]
    st.header(f"🔬 {mineral_seleccionado}")
    
    imagenes = get_imagenes_con_caption(fila['Imagenes'])
    df_props = parsear_propiedades(fila['Texto'], mineral_seleccionado)
    
    st.subheader("Vistas al Microscopio")
    if imagenes:
        cols = st.columns(min(len(imagenes), 4))
        for i, col in enumerate(cols):
            if i < len(imagenes):
                with col:
                    st.image(imagenes[i][0], use_container_width=True)
                    st.caption(imagenes[i][1])
    else:
        st.warning("No se encontraron imágenes válidas.")
    
    st.subheader("Propiedades Ópticas")
    st.dataframe(df_props, use_container_width=True, hide_index=True)
    st.markdown(f"**Fuente:** [Universidad de Granada]({fila['URL']})")

# ------------------------------------------
# PESTAÑA 2: SIMETRÍA 3D
# ------------------------------------------
elif st.session_state.active_tab == "📐 Simetría 3D":
    st.header("Visualizador de Sistemas Cristalinos")
    sistemas = {
        "Cúbico (Isométrico)": {"a":1, "b":1, "c":1, "alpha":90, "beta":90, "gamma":90},
        "Tetragonal": {"a":1, "b":1, "c":1.5, "alpha":90, "beta":90, "gamma":90},
        "Ortorrómbico": {"a":1, "b":1.5, "c":2, "alpha":90, "beta":90, "gamma":90},
        "Monoclínico": {"a":1, "b":1.5, "c":2, "alpha":90, "beta":105, "gamma":90},
        "Triclínico": {"a":1, "b":1.2, "c":1.5, "alpha":85, "beta":95, "gamma":90},
        "Trigonal/Romboédrico": {"a":1, "b":1, "c":1, "alpha":55, "beta":55, "gamma":55},
        "Hexagonal": {"a":1, "b":1, "c":1.5, "alpha":90, "beta":90, "gamma":120}
    }
    sistema_sel = st.selectbox("Elige un sistema cristalino:", list(sistemas.keys()))
    sim = sistemas[sistema_sel]
    st.write(f"**Parámetros:** a={sim['a']}, b={sim['b']}, c={sim['c']} | α={sim['alpha']}°, β={sim['beta']}°, γ={sim['gamma']}°")
    v = get_crystal_vertices(sim['a'], sim['b'], sim['c'], sim['alpha'], sim['beta'], sim['gamma'])
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[0, v[1][0]*1.5], y=[0, v[1][1]*1.5], z=[0, v[1][2]*1.5], mode='lines+text', line=dict(color='red', width=10), text=['', 'a'], name='Eje a'))
    fig.add_trace(go.Scatter3d(x=[0, v[2][0]*1.5], y=[0, v[2][1]*1.5], z=[0, v[2][2]*1.5], mode='lines+text', line=dict(color='green', width=10), text=['', 'b'], name='Eje b'))
    fig.add_trace(go.Scatter3d(x=[0, v[3][0]*1.5], y=[0, v[3][1]*1.5], z=[0, v[3][2]*1.5], mode='lines+text', line=dict(color='blue', width=10), text=['', 'c'], name='Eje c'))
    edges = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]
    for e in edges:
        p1, p2 = v[e[0]], v[e[1]]
        fig.add_trace(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]], mode='lines', line=dict(color='black', width=4), showlegend=False))
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))), margin=dict(l=0, r=0, t=0, b=0), height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# PESTAÑA 3: GEOQUÍMICA
# ------------------------------------------
elif st.session_state.active_tab == "🧪 Geoquímica":
    st.header("🧪 Calculadora Termodinámica de Reacciones")
    
    if df_thermo is not None:
        st.subheader("1. Cálculo de ΔG (Ley de Hess e Integración Cp)")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Reactivos**")
            r1 = st.selectbox("Reactivo 1", df_thermo['Phase'].unique(), key='r1')
            r2 = st.selectbox("Reactivo 2", df_thermo['Phase'].unique(), key='r2')
        with col2:
            st.write("**Productos**")
            p1 = st.selectbox("Producto 1", df_thermo['Phase'].unique(), key='p1')
            p2 = st.selectbox("Producto 2", df_thermo['Phase'].unique(), key='p2')
            
        T_input = st.number_input("Temperatura final (°C)", min_value=0.0, max_value=6000.0, value=500.0, step=50.0, key='temp_input')
        T_final_K = T_input + 273.15
        
        if st.button("Calcular ΔG", key='btn_calc_thermo'):
            if T_final_K <= 298.15:
                st.error("La temperatura final debe ser mayor a 25°C.")
            else:
                reaccion = {r1: -1, r2: -1, p1: 1, p2: 1}
                dH_ref = dS_ref = 0
                for min, coef in reaccion.items():
                    datos = df_thermo[df_thermo['Phase'] == min].iloc[0]
                    dH_ref += coef * datos['DeltaH'] * 1000
                    dS_ref += coef * datos['DeltaS']
                
                temperaturas = np.linspace(298.15, T_final_K, 20)
                resultados = []
                for T in temperaturas:
                    dH_T = dH_ref
                    dS_T = dS_ref
                    for min, coef in reaccion.items():
                        datos = df_thermo[df_thermo['Phase'] == min].iloc[0]
                        dH_T += coef * quad(cp_mineral, 298.15, T, args=(datos['a'], datos['b'], datos['c']))[0]
                        dS_T += coef * quad(lambda t: cp_mineral(t, datos['a'], datos['b'], datos['c'])/t, 298.15, T)[0]
                    dG = dH_T - T * dS_T
                    resultados.append({'Temp (°C)': T - 273.15, 'ΔH (kJ/mol)': dH_T/1000, 'ΔS (J/K·mol)': dS_T, 'ΔG (kJ/mol)': dG/1000})
                
                df_result = pd.DataFrame(resultados)
                st.dataframe(df_result.style.applymap(lambda x: 'color: green' if x < 0 else 'color: red', subset=['ΔG (kJ/mol)']), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("2. Generador de Fronteras de Fase (Diagrama P-T-Profundidad)")
        
        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1: f1 = st.selectbox("Fase 1", df_thermo['Phase'].unique(), key='f1_diag')
        with c_f2: f2 = st.selectbox("Fase 2", df_thermo['Phase'].unique(), key='f2_diag')
        with c_f3:
            usar_f3 = st.checkbox("Calcular Punto Triple (Fase 3)")
            f3 = None
            if usar_f3:
                f3 = st.selectbox("Fase 3", df_thermo['Phase'].unique(), key='f3_diag')
                
        st.write("**Condiciones de referencia (Punto inicial de la curva)**")
        col_ref1, col_ref2, col_ref3 = st.columns(3)
        with col_ref1:
            T_ref_input = st.number_input("Temp. referencia (°C)", min_value=0.0, max_value=2000.0, value=500.0, key='t_ref_diag')
        with col_ref2:
            P_ref_input = st.number_input("Presión referencia (GPa)", min_value=0.0, max_value=10.0, value=0.4, step=0.1, key='p_ref_diag')
        with col_ref3:
            T_max_diag = st.number_input("Temp. máxima diagrama (°C)", min_value=100, max_value=2000, value=800, key='t_max_diag')
            
        if st.button("Generar Fronteras", key='btn_gen_fronteras'):
            T_ref_K_val = T_ref_input + 273.15
            P_ref_bar_val = P_ref_input * 10000
            
            # Curva 1: Fase 1 <-> Fase 2
            T1, P1 = calculate_clapeyron_curve(f1, f2, T_ref_K=T_ref_K_val, P_ref_bar=P_ref_bar_val, T_final_K=T_max_diag+273.15)
            if T1 is not None:
                st.session_state.curva1 = {'T': T1, 'P': P1, 'fase_low': f1, 'fase_high': f2}
            
            if usar_f3 and f3:
                # Curva 2: Fase 2 <-> Fase 3
                T2, P2 = calculate_clapeyron_curve(f2, f3, T_ref_K=T_ref_K_val, P_ref_bar=P_ref_bar_val, T_final_K=T_max_diag+273.15)
                if T2 is not None:
                    st.session_state.curva2 = {'T': T2, 'P': P2, 'fase_low': f2, 'fase_high': f3}
                
                # Curva 3 (NUEVA): Fase 1 <-> Fase 3
                T3, P3 = calculate_clapeyron_curve(f1, f3, T_ref_K=T_ref_K_val, P_ref_bar=P_ref_bar_val, T_final_K=T_max_diag+273.15)
                if T3 is not None:
                    st.session_state.curva3 = {'T': T3, 'P': P3, 'fase_low': f1, 'fase_high': f3}
            else:
                if 'curva2' in st.session_state: del st.session_state.curva2
                if 'curva3' in st.session_state: del st.session_state.curva3
                
            st.success("Fronteras calculadas. Ve a la pestaña '🌋 Diagrama de Fases'.")

        st.markdown("---")
        st.subheader("Regla de las Fases de Gibbs ($f = c - p + 2$)")
        col_g1, col_g2 = st.columns(2)
        with col_g1: c_comp = st.number_input("Componentes (c)", min_value=1, max_value=10, value=2, key='c_comp')
        with col_g2: p_fases = st.number_input("Fases (p)", min_value=1, max_value=10, value=2, key='p_fases')
        f_libertad = c_comp - p_fases + 2
        if f_libertad < 0: st.error(f"Grados de libertad (f) = {f_libertad}. No válido.")
        elif f_libertad == 0: st.warning(f"f = {f_libertad}. INVARIANTE (Punto triple/cuádruple).")
        elif f_libertad == 1: st.info(f"f = {f_libertad}. UNIVARIANTE (Curva).")
        else: st.success(f"f = {f_libertad}. DIVARIANTE (Campo).")
    else:
        st.error("No se encontró 'database.csv'.")

# ------------------------------------------
# PESTAÑA 4: QUIZ
# ------------------------------------------
elif st.session_state.active_tab == "🎮 Quiz":
    st.header("Retador Mineralógico")
    st.write("¿Puedes identificar el mineral solo por su vista al microscopio?")
    if 'minerales_con_imagen' not in st.session_state:
        valid_mins = []
        for m in df_optica['Mineral'].unique():
            if get_imagenes_con_caption(df_optica[df_optica['Mineral'] == m].iloc[0]['Imagenes']):
                valid_mins.append(m)
        st.session_state.minerales_con_imagen = valid_mins if valid_mins else list(df_optica['Mineral'].unique())
        
    if 'score' not in st.session_state:
        st.session_state.score = 0
        st.session_state.correct = 0
        st.session_state.incorrect = 0
        st.session_state.quiz_mineral = np.random.choice(st.session_state.minerales_con_imagen)
        st.session_state.quiz_answered = False
        st.session_state.quiz_feedback = ""
        st.session_state.quiz_options = None
        
    col1, col2, col3 = st.columns(3)
    col1.metric("Puntos", st.session_state.score)
    col2.metric("Correctas", st.session_state.correct)
    col3.metric("Incorrectas", st.session_state.incorrect)
    
    mineral_correcto = st.session_state.quiz_mineral
    fila_quiz = df_optica[df_optica['Mineral'] == mineral_correcto].iloc[0]
    imagenes_q = get_imagenes_con_caption(fila_quiz['Imagenes'])
    
    ppl_img = xpl_img = None
    for img, cap in imagenes_q:
        if 'XPL' in cap and xpl_img is None: xpl_img = (img, cap)
        elif 'PPL' in cap and ppl_img is None: ppl_img = (img, cap)
            
    if ppl_img or xpl_img:
        cols = st.columns(2)
        with cols[0]:
            if ppl_img: st.image(ppl_img[0], use_container_width=True); st.caption(ppl_img[1])
            else: st.warning("No hay imagen PPL")
        with cols[1]:
            if xpl_img: st.image(xpl_img[0], use_container_width=True); st.caption(xpl_img[1])
            else: st.warning("No hay imagen XPL")
    else:
        df_props = parsear_propiedades(fila_quiz['Texto'], mineral_correcto)
        formula = df_props[df_props['Propiedad'] == 'Fórmula Química']['Valor'].values[0] if 'Fórmula Química' in df_props['Propiedad'].values else "N/A"
        st.info(f"Pista visual no disponible. Fórmula química: {formula}")
    
    if st.session_state.quiz_options is None:
        opciones = np.random.choice(st.session_state.minerales_con_imagen, 3, replace=False)
        if mineral_correcto not in opciones:
            opciones = np.append(opciones, mineral_correcto)
        np.random.shuffle(opciones)
        st.session_state.quiz_options = opciones
        
    selected = st.radio("¿Qué mineral es?", st.session_state.quiz_options, key='quiz_mineral_selector')
    
    if not st.session_state.quiz_answered:
        if st.button("Comprobar", key='btn_check_quiz'):
            st.session_state.quiz_answered = True
            if selected == mineral_correcto:
                st.session_state.quiz_feedback = "success"
                st.session_state.score += 10
                st.session_state.correct += 1
            else:
                st.session_state.quiz_feedback = "error"
                st.session_state.score -= 5
                st.session_state.incorrect += 1
    else:
        if st.session_state.quiz_feedback == "success": st.success(f"✅ ¡Correcto! Es {mineral_correcto}.")
        else: st.error(f"❌ Incorrecto. La respuesta correcta era {mineral_correcto}.")
        if st.button("Siguiente mineral", key='btn_next_quiz'):
            st.session_state.quiz_mineral = np.random.choice(st.session_state.minerales_con_imagen)
            st.session_state.quiz_options = None
            st.session_state.quiz_answered = False
            st.session_state.quiz_feedback = ""

# ------------------------------------------
# PESTAÑA 5: DIAGRAMA DE FASES 3D
# ------------------------------------------
elif st.session_state.active_tab == "🌋 Diagrama de Fases":
    st.header("🌋 Diagrama de Fases P-T-Profundidad (Clapeyron)")
    
    if 'curva1' in st.session_state:
        curvas = [st.session_state.curva1]
        if 'curva2' in st.session_state: curvas.append(st.session_state.curva2)
        if 'curva3' in st.session_state: curvas.append(st.session_state.curva3)
        
        fig = go.Figure()
        max_Z, max_Y, max_X = 50, 1.0, 1000
        
        # Nombres de las fases para las etiquetas flotantes
        fase_baja_p = curvas[0]['fase_low']
        fase_media_p = curvas[0]['fase_high']
        fase_alta_p = curvas[1]['fase_high'] if len(curvas) >= 2 else None

        colores = ['red', 'blue', 'green'] # Rojo, Azul, Verde para las 3 curvas

        for i, c in enumerate(curvas):
            T_C = c['T']
            P_GPa = c['P']
            Depth_km = P_GPa * 33
            
            valid_idx = (P_GPa > 0) & (Depth_km > 0)
            T_C = T_C[valid_idx]
            P_GPa = P_GPa[valid_idx]
            Depth_km = Depth_km[valid_idx]
            
            if len(T_C) > 2:
                max_Z = max(max_Z, np.max(Depth_km) * 1.5)
                max_Y = max(max_Y, np.max(P_GPa) * 1.5)
                max_X = max(max_X, np.max(T_C) * 1.2)
                
                color_line = colores[i % 3]
                nombre_linea = c['fase_high'] # La línea se nombra por la fase de alta P
                
                # Pared semitransparente y línea
                T_grid, Z_grid = np.meshgrid(T_C, np.linspace(0, max_Z, 5))
                P_grid = np.tile(P_GPa, (5, 1))
                fig.add_trace(go.Surface(x=T_grid, y=P_grid, z=Z_grid, opacity=0.3, colorscale=[[0, 'lightgray'], [1, 'lightgray']], showscale=False, hoverinfo='skip', name=nombre_linea))
                
                fig.add_trace(go.Scatter3d(x=T_C, y=P_GPa, z=Depth_km, mode='lines+markers', name=nombre_linea, line=dict(color=color_line, width=10), marker=dict(size=4)))
                fig.add_trace(go.Scatter3d(x=T_C, y=P_GPa, z=np.zeros_like(T_C), mode='lines', line=dict(color='black', width=3, dash='dash'), showlegend=False))

        # Lógica del Punto Triple (Intersección de las 2 primeras curvas)
        if len(curvas) >= 2:
            c1, c2 = curvas[0], curvas[1]
            v1 = (c1['P'] > 0)
            v2 = (c2['P'] > 0)
            
            if np.sum(v1) > 2 and np.sum(v2) > 2:
                T1_K = c1['T'][v1] + 273.15
                P1_bar = c1['P'][v1] * 10000
                T2_K = c2['T'][v2] + 273.15
                P2_bar = c2['P'][v2] * 10000
                
                T_common = np.linspace(max(T1_K.min(), T2_K.min()), min(T1_K.max(), T2_K.max()), 800)
                
                if len(T_common) > 2:
                    P1i = np.interp(T_common, T1_K, P1_bar)
                    P2i = np.interp(T_common, T2_K, P2_bar)
                    diff = P1i - P2i
                    cambios = np.where(np.diff(np.sign(diff)))[0]
                    
                    if len(cambios) > 0:
                        k = cambios[0]
                        T_triple_K = T_common[k]
                        P_triple_bar = P1i[k]
                        T_triple_C = T_triple_K - 273.15
                        P_triple_GPa = P_triple_bar / 10000
                        Depth_triple = P_triple_GPa * 33
                        
                        if P_triple_GPa > 0:
                            max_Z = max(max_Z, Depth_triple * 1.2)
                            max_Y = max(max_Y, P_triple_GPa * 1.2)
                            fig.add_trace(go.Scatter3d(
                                x=[T_triple_C], y=[P_triple_GPa], z=[Depth_triple],
                                mode='markers+text', marker=dict(size=10, color='black', symbol='diamond'),
                                text=['  Punto triple'], textposition='top center', name='Punto triple'
                            ))
                            st.info(f"📍 **Punto triple aproximado:** T ≈ {T_triple_C:.0f} °C, P ≈ {P_triple_GPa:.2f} GPa (~{Depth_triple:.1f} km profundidad)")
                        else:
                            st.info("La intersección cae en presión negativa (fuera del rango físico). Prueba con otras referencias.")
                    else:
                        st.info("No se encontró intersección entre las dos fronteras en este rango.")
        
        # Etiquetas 3D de los campos de estabilidad
        mid_X = max_X * 0.5
        top_Z = max_Z * 0.9
        fig.add_trace(go.Scatter3d(x=[mid_X], y=[max_Y*0.15], z=[top_Z], mode='text', text=[fase_baja_p], textfont=dict(color='darkred', size=18), showlegend=False))
        if fase_alta_p:
            fig.add_trace(go.Scatter3d(x=[mid_X], y=[max_Y*0.5], z=[top_Z], mode='text', text=[fase_media_p], textfont=dict(color='darkgreen', size=18), showlegend=False))
            fig.add_trace(go.Scatter3d(x=[mid_X], y=[max_Y*0.85], z=[top_Z], mode='text', text=[fase_alta_p], textfont=dict(color='darkblue', size=18), showlegend=False))
        else:
            fig.add_trace(go.Scatter3d(x=[mid_X], y=[max_Y*0.85], z=[top_Z], mode='text', text=[fase_media_p], textfont=dict(color='darkblue', size=18), showlegend=False))

        # Marcos de la caja
        fig.add_trace(go.Scatter3d(x=[0, max_X], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=3), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, max_Y], z=[0, 0], mode='lines', line=dict(color='black', width=3), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, max_Z], mode='lines', line=dict(color='black', width=3), showlegend=False))
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Temperatura (°C)', range=[0, max_X], backgroundcolor='rgb(240,240,240)'),
                yaxis=dict(title='Presión (GPa)', range=[0, max_Y], backgroundcolor='rgb(240,240,240)'),
                zaxis=dict(title='Profundidad (km)', range=[0, max_Z], backgroundcolor='rgb(240,240,240)'),
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
            ),
            margin=dict(l=0, r=0, t=40, b=0), height=650
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Ve a la pestaña '🧪 Geoquímica', genera las fronteras de fase y vuelve aquí.")
