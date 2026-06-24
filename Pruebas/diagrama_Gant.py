import matplotlib.pyplot as plt
import pandas as pd
import datetime
import textwrap  # Importamos librería para dividir texto

# --- 1. DATOS Y LÓGICA ---
fases = [
    ("Fase 1: Estudio y planificación", 35),
    ("Fase 2: Creación de datasets y preparación de datos", 21),
    ("Fase 3: Programación y desarrollo del modelo", 35),
    ("Fase 4: Evaluación y selección del modelo", 7),
    ("Fase 5: Documentación y entrega final", 42),
]

fecha_inicio = datetime.date(2026, 2, 1)

tareas = []
inicio = fecha_inicio
for nombre, duracion in fases:
    fin = inicio + datetime.timedelta(days=duracion)
    tareas.append({"Fase": nombre, "Inicio": inicio, "Fin": fin, "Duración (días)": duracion})
    inicio = fin

df = pd.DataFrame(tareas)

# --- MODIFICACIÓN DE FECHAS (Simultaneidad) ---
tarea_objetivo = "WORD DEVELOPMENT AND GOOGLE PRESENTATIONS OF SECOND SESSION"
mask = df['Fase'] == tarea_objetivo
# Forzamos las fechas
if not df[mask].empty:
    df.loc[mask, 'Inicio'] = datetime.date(2026, 3, 15)
    df.loc[mask, 'Fin'] = datetime.date(2026, 3, 22)

# Invertir orden para graficar de arriba a abajo
df_reversed = df.iloc[::-1]

# --- 2. CONFIGURACIÓN DEL GRÁFICO ---

# Aumentamos el ancho (14) y la altura (8) para dar espacio
fig, ax = plt.subplots(figsize=(14, 9))  # Aumenté un poco la altura a 9 para las etiquetas de deadline

# Función para cortar las frases largas en varias líneas
def romper_texto(texto, ancho=30):
    return textwrap.fill(texto, width=ancho)

# Aplicamos el corte de texto a las etiquetas
etiquetas_cortadas = [romper_texto(t) for t in df_reversed["Fase"]]

# Dibujar las barras
for i, row in enumerate(df_reversed.itertuples()):
    inicio_ord = row.Inicio.toordinal()
    fin_ord = row.Fin.toordinal()
    duracion = (row.Fin - row.Inicio).days
    
    # Color diferente para la tarea modificada
    color = "orange" if row.Fase == tarea_objetivo else "skyblue"
    
    ax.barh(i, fin_ord - inicio_ord, left=inicio_ord, height=0.6, color=color, edgecolor="black")
    # Texto de duración al lado de la barra
    ax.text(fin_ord + 1, i, f"{duracion} d", va="center", ha="left", fontsize=9, fontweight='bold')

# --- 3. AÑADIR DEADLINES (Líneas Rojas) ---
# Definimos las fechas de entrega
deadlines = [
    ("1ST DELIVERY", datetime.date(2026, 2, 22)),
    ("2ND DELIVERY", datetime.date(2026, 3, 22)),
    ("FINAL DELIVERY", datetime.date(2026, 5, 10))
]

for label, fecha in deadlines:
    fecha_ord = fecha.toordinal()
    
    # Línea vertical roja
    ax.axvline(x=fecha_ord, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    # Etiqueta de texto encima del gráfico
    # Usamos len(df) para poner el texto un poco por encima de la última barra
    ax.text(fecha_ord, len(df) - 0.5, f"{label}\n{fecha.strftime('%d-%b')}", 
            color='red', ha='center', va='bottom', fontsize=9, fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# --- 4. FORMATO FINAL ---

ax.set_xlabel("DATE", fontsize=12)
ax.set_title("GANTT DIAGRAM - COURSE PROJECT", fontsize=14, pad=30) # Más pad para dejar sitio a las etiquetas rojas

# Asignar las etiquetas cortadas al eje Y
ax.set_yticks(range(len(df_reversed)))
ax.set_yticklabels(etiquetas_cortadas, fontsize=10)

# Ajuste de fechas en eje X
# Calculamos min y max considerando también las deadlines para que no se corten
min_fecha = min(df["Inicio"].min(), min(d[1] for d in deadlines))
max_fecha = max(df["Fin"].max(), max(d[1] for d in deadlines))

fechas = pd.date_range(start=min_fecha - datetime.timedelta(days=5), 
                       end=max_fecha + datetime.timedelta(days=5), freq="W")

ax.set_xticks([d.toordinal() for d in fechas])
ax.set_xticklabels([d.strftime("%d-%b") for d in fechas], rotation=0, ha='center')

# Cuadrícula
ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()