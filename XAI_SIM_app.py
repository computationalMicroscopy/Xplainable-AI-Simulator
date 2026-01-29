import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Bayes-Netz Simulator", layout="wide")

st.title("🎓 Bayes-Netzwerk Simulator für Schulungen")
st.markdown("""
Dieses Tool visualisiert, wie **bedingte Wahrscheinlichkeitstabellen (CPTs)** aus Daten gelernt werden.
1. Definiere die Struktur (wer ist Elternteil von wem?).
2. Gib Trainingsdaten im One-Hot-Format ein (0 oder 1).
3. Beobachte, wie sich die CPTs berechnen.
""")

# --- SEITENLEISTE: Struktur-Definition ---
st.sidebar.header("1. Netzwerk-Struktur")
nodes = ["Knoten_A", "Knoten_B", "Knoten_C", "Knoten_D"]
structure = {}

for node in nodes:
    parents = st.sidebar.multiselect(
        f"Eltern für {node}:",
        [n for n in nodes if n != node],
        key=f"parents_{node}"
    )
    structure[node] = parents

# --- HAUPTBEREICH: Dateneingabe ---
st.header("2. Trainingsdaten (One-Hot Encoding)")
st.info("Gib hier 0 oder 1 für jeden Zustand ein. Jede Zeile repräsentiert eine Beobachtung.")

# Initialdaten
default_data = pd.DataFrame(
    [[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 1]], 
    columns=nodes
)

edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

# --- BERECHNUNG DER CPTs ---
st.header("3. Bedingte Wahrscheinlichkeitstabellen (CPTs)")

def calculate_cpt(df, target, parents):
    if not parents:
        # Prior Wahrscheinlichkeit (keine Eltern)
        prob = df[target].mean()
        return pd.DataFrame({f"P({target}=1)": [prob], f"P({target}=0)": [1-prob]})
    
    # Gruppieren nach Eltern-Zuständen
    counts = df.groupby(parents)[target].value_counts(normalize=True).unstack(fill_value=0)
    
    # Sicherstellen, dass beide Spalten (0 und 1) existieren
    if 1 not in counts.columns: counts[1] = 0.0
    if 0 not in counts.columns: counts[0] = 0.0
    
    counts = counts.rename(columns={1: f"P({target}=1)", 0: f"P({target}=0)"})
    return counts

cols = st.columns(2)
for i, node in enumerate(nodes):
    with cols[i % 2]:
        st.subheader(f"CPT für {node}")
        if structure[node]:
            st.write(f"Abhängig von: {', '.join(structure[node])}")
        else:
            st.write("Unabhängiger Wurzelknoten")
            
        try:
            cpt_df = calculate_cpt(edited_df, node, structure[node])
            st.dataframe(cpt_df.style.format("{:.2%}"))
        except Exception as e:
            st.warning(f"Nicht genügend Datenkombinationen für {node}")

# --- LOGIK-CHECK ---
st.divider()
st.sidebar.success("💡 **Tipp für die Schulung:** Ändere einen Wert in der Tabelle oben auf 0 und beobachte, wie die Wahrscheinlichkeiten in den CPTs sofort "live" neu berechnet werden.")
