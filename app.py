import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

st.title("Herramienta de Análisis Cinético y Estabilidad")
st.write("Análisis de degradación y predicción de vida útil de productos farmacéuticos.")

# 1. Ingreso de Datos Experimentales y Unidades
st.subheader("1. Datos de Concentración vs Tiempo")

col_u1, col_u2 = st.columns(2)
with col_u1:
    u_t = st.text_input("Unidad de tiempo (ej. min, h, días):", "min")
with col_u2:
    u_c = st.text_input("Unidad de concentración (ej. mg/L, M, %):", "mg/L")

t_input = st.text_input(f"Ingresa los tiempos separados por comas:", "0, 5, 10, 15, 20")
c_input = st.text_input(f"Ingresa las concentraciones separadas por comas:", "100, 85, 70, 55, 40")

if t_input and c_input:
    t = np.array([float(i.strip()) for i in t_input.split(",")])
    c = np.array([float(i.strip()) for i in c_input.split(",")])
    
    if len(t) == len(c):
        # 2. Cálculos para los 3 órdenes
        df = pd.DataFrame({f"Tiempo ({u_t})": t, f"[A] ({u_c})": c})
        df["ln[A]"] = np.log(c)
        df[f"1/[A] ({u_c}⁻¹)"] = 1 / c
        
        st.write("Tabla de datos calculada:", df)
        
        # Regresiones
        reg_0 = linregress(t, df[f"[A] ({u_c})"])
        reg_1 = linregress(t, df["ln[A]"])
        reg_2 = linregress(t, df[f"1/[A] ({u_c}⁻¹)"])
        
        r2_0, r2_1, r2_2 = reg_0.rvalue**2, reg_1.rvalue**2, reg_2.rvalue**2
        
        st.subheader("2. Gráficos y Modelos")
        col1, col2, col3 = st.columns(3)
        
        # Gráficos (Orden 0)
        with col1:
            fig, ax = plt.subplots()
            ax.plot(t, df[f"[A] ({u_c})"], 'o', label="Datos")
            ax.plot(t, reg_0.intercept + reg_0.slope * t, 'r', label="Ajuste")
            ax.set_title("Orden 0: [A] vs t")
            ax.set_xlabel(f"Tiempo ({u_t})")
            ax.set_ylabel(f"[A] ({u_c})")
            st.pyplot(fig)
            st.write(f"**R²:** {r2_0:.4f}")
            st.write(f"**Pendiente:** {reg_0.slope:.4f}")
            st.write(f"**Intercepto:** {reg_0.intercept:.4f}")
            
        # Gráficos (Orden 1)
        with col2:
            fig, ax = plt.subplots()
            ax.plot(t, df["ln[A]"], 'o')
            ax.plot(t, reg_1.intercept + reg_1.slope * t, 'r')
            ax.set_title("Orden 1: ln[A] vs t")
            ax.set_xlabel(f"Tiempo ({u_t})")
            ax.set_ylabel("ln[A]")
            st.pyplot(fig)
            st.write(f"**R²:** {r2_1:.4f}")
            st.write(f"**Pendiente:** {reg_1.slope:.4f}")
            st.write(f"**Intercepto:** {reg_1.intercept:.4f}")
            
        # Gráficos (Orden 2)
        with col3:
            fig, ax = plt.subplots()
            ax.plot(t, df[f"1/[A] ({u_c}⁻¹)"], 'o')
            ax.plot(t, reg_2.intercept + reg_2.slope * t, 'r')
            ax.set_title("Orden 2: 1/[A] vs t")
            ax.set_xlabel(f"Tiempo ({u_t})")
            ax.set_ylabel(f"1/[A] ({u_c}⁻¹)")
            st.pyplot(fig)
            st.write(f"**R²:** {r2_2:.4f}")
            st.write(f"**Pendiente:** {reg_2.slope:.4f}")
            st.write(f"**Intercepto:** {reg_2.intercept:.4f}")
            
        # 3. Determinación automática del orden
        r2_values = {0: r2_0, 1: r2_1, 2: r2_2}
        best_order = max(r2_values, key=r2_values.get)
        best_r2 = r2_values[best_order]
        st.subheader(f"3. Orden de Reacción Más Probable: Orden {best_order}")
        
        # 4. Parámetros Cinéticos
        C0 = c[0]
        if best_order == 0:
            k_best = -reg_0.slope
            k_units = f"{u_c}/{u_t}"
            t50, t98, t95, t90 = (C0*0.5)/k_best, (C0*0.02)/k_best, (C0*0.05)/k_best, (C0*0.1)/k_best
        elif best_order == 1:
            k_best = -reg_1.slope
            k_units = f"{u_t}⁻¹"
            t50, t98, t95, t90 = -np.log(0.5)/k_best, -np.log(0.98)/k_best, -np.log(0.95)/k_best, -np.log(0.90)/k_best
        else:
            k_best = reg_2.slope
            k_units = f"({u_c}·{u_t})⁻¹"
            t50 = 0.5 / (0.5 * k_best * C0)
            t98 = 0.02 / (0.98 * k_best * C0)
            t95 = 0.05 / (0.95 * k_best * C0)
            t90 = 0.10 / (0.90 * k_best * C0)
            
        st.write(f"**Constante cinética (k):** {k_best:.4f} {k_units}")
        st.write(f"**Tiempos de vida:** t50 = {t50:.2f} {u_t} | t98 = {t98:.2f} {u_t} | t95 = {t95:.2f} {u_t} | t90 = {t90:.2f} {u_t}")

st.divider()

# 5. Módulo Arrhenius
st.subheader("4 y 5. Módulo Arrhenius y Predicción")
temp_input = st.text_input("Temperaturas (°C) separadas por comas:", "25, 40, 50, 60")
k_input = st.text_input("Constantes k respectivas separadas por comas:", "0.002, 0.008, 0.018, 0.035")

if temp_input and k_input and 'k_units' in locals():
    temps = np.array([float(i.strip()) for i in temp_input.split(",")])
    ks = np.array([float(i.strip()) for i in k_input.split(",")])
    
    T_kelvin = temps + 273.15
    inv_T = 1 / T_kelvin
    ln_k = np.log(ks)
    
    # Tabla de 1/T y ln(k)
    df_arrhenius = pd.DataFrame({
        "T (°C)": temps,
        "T (K)": T_kelvin,
        "1/T (K⁻¹)": inv_T,
        f"k ({k_units})": ks,
        "ln(k)": ln_k
    })
    st.write("**Tabla de datos de Arrhenius calculada:**")
    st.dataframe(df_arrhenius)
    
    reg_arrhenius = linregress(inv_T, ln_k)
    R = 8.314 # J/(mol K)
    Ea = -reg_arrhenius.slope * R / 1000 # kJ/mol
    A = np.exp(reg_arrhenius.intercept)
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        # Gráfica de Arrhenius
        fig2, ax2 = plt.subplots()
        ax2.plot(inv_T, ln_k, 'o', label="Datos")
        ax2.plot(inv_T, reg_arrhenius.intercept + reg_arrhenius.slope * inv_T, 'r', label="Ajuste")
        ax2.set_xlabel("1/T (K⁻¹)")
        ax2.set_ylabel("ln(k)")
        ax2.set_title("Gráfica de Arrhenius: ln(k) vs 1/T")
        ax2.legend()
        st.pyplot(fig2)
        
    with col_a2:
        st.write(f"**Pendiente:** {reg_arrhenius.slope:.4f}")
        st.write(f"**Intercepto (ln A):** {reg_arrhenius.intercept:.4f}")
        st.write(f"**Energía de Activación (Ea):** {Ea:.2f} kJ/mol")
        st.write(f"**Factor Preexponencial (A):** {A:.2e}")
    
    st.divider()
    
    # 6. Predicción y Tiempos
    st.subheader("Predicción de Estabilidad a Nueva Temperatura")
    t_pred = st.number_input("Temperatura para predecir k y tiempos (°C):", value=30.0)
    k_pred = A * np.exp(-Ea * 1000 / (R * (t_pred + 273.15)))
    
    # Recalcular tiempos con la k predicha
    if best_order == 0:
        t50_p, t98_p, t95_p, t90_p = (C0*0.5)/k_pred, (C0*0.02)/k_pred, (C0*0.05)/k_pred, (C0*0.1)/k_pred
    elif best_order == 1:
        t50_p, t98_p, t95_p, t90_p = -np.log(0.5)/k_pred, -np.log(0.98)/k_pred, -np.log(0.95)/k_pred, -np.log(0.90)/k_pred
    else:
        t50_p = 0.5 / (0.5 * k_pred * C0)
        t98_p = 0.02 / (0.98 * k_pred * C0)
        t95_p = 0.05 / (0.95 * k_pred * C0)
        t90_p = 0.10 / (0.90 * k_pred * C0)
        
    st.write(f"**k predicha a {t_pred}°C:** {k_pred:.6f} {k_units}")
    st.write(f"**Nuevos tiempos estimados a {t_pred}°C:** t50 = {t50_p:.2f} {u_t} | t98 = {t98_p:.2f} {u_t} | t95 = {t95_p:.2f} {u_t} | t90 = {t90_p:.2f} {u_t}")

    st.divider()
    
    # 7. Conclusiones Automáticas
    st.subheader("6. Conclusiones Automáticas del Análisis")
    conclusiones = f"""
    📌 **Análisis Cinético:** Al evaluar los datos experimentales, el sistema determinó que la degradación del principio activo se ajusta mejor a un modelo de **Orden {best_order}**, presentando el mayor coeficiente de correlación (R² = {best_r2:.4f}). Bajo estas condiciones, la constante de velocidad aparente (k) es de **{k_best:.4f} {k_units}**.
    
    📌 **Estabilidad Base:** Con el modelo cinético establecido, se calculó que el tiempo necesario para que el fármaco alcance el 90% de su concentración original (t90 o vida útil) es de **{t90:.2f} {u_t}**.
    
    📌 **Comportamiento Termodinámico:** Mediante el estudio a diferentes temperaturas y la aplicación del modelo de Arrhenius, se obtuvo una Energía de Activación (Ea) de **{Ea:.2f} kJ/mol**. Esta energía representa la barrera termodinámica que deben superar las moléculas para degradarse.
    
    📌 **Predicción de Estabilidad:** Al proyectar el comportamiento a una nueva temperatura de **{t_pred}°C**, el modelo predice que la constante de degradación será de **{k_pred:.6f} {k_units}**. Esto impacta directamente la estabilidad del producto, estimando un nuevo tiempo de vida útil (t90) de **{t90_p:.2f} {u_t}**.
    """
    st.info(conclusiones)