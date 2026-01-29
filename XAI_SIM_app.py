import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Bayes-Netz Simulator", layout="wide")

st.title("🎓 Bayes-Netzwerk Simulator für KI-Schulungen")

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

# --- VISUALISIERUNG ---
st.header("Netzwerk-Struktur")
dot = "digraph { rankdir=LR; node [style=filled, fillcolor=lightblue]; "
for node, parents in structure.items():
    dot += f"{node}; "
    for p in parents:
        dot += f"{p} -> {node}; "
dot += "}"
st.graphviz_chart(dot)



# --- DATENEINGABE ---
st.header("Trainingsdaten (One-Hot)")
st.info("Bearbeite die Tabelle, um die Wahrscheinlichkeiten live zu verändern.")

default_data = pd.DataFrame(
    [[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]], 
    columns=nodes
)
edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

# --- CPT BERECHNUNG ---
st.header("Bedingte Wahrscheinlichkeitstabellen (CPTs)")

cpts = {} # Speicher für die berechneten Tabellen

def calculate_cpt(df, target, parents):
    if not parents:
        prob = df[target].mean()
        res = pd.DataFrame({0: [1-prob], 1: [prob]}, index=["Prior"])
    else:
        res = df.groupby(parents)[target].value_counts(normalize=True).unstack(fill_value=0)
        if 0 not in res.columns: res[0] = 0.0
        if 1 not in res.columns: res[1] = 0.0
    
    res.columns = [f"{target}=0", f"{target}=1"]
    return res

cols = st.columns(2)
for i, node in enumerate(nodes):
    with cols[i % 2]:
        st.subheader(f"Tabelle für {node}")
        try:
            cpt_df = calculate_cpt(edited_df, node, structure[node])
            cpts[node] = cpt_df
            st.dataframe(cpt_df.style.format("{:.2%}"))
        except Exception:
            st.warning(f"Keine Daten für Kombinationen in {node}")

# --- INFERENZ (WAS-WÄRE-WENN) ---
st.divider()
st.header("Inferenz: Auswirkungen beobachten")
st.markdown("Setze hier eine Bedingung fest (Evidenz), um zu sehen, wie sie die Vorhersage beeinflusst.")

inf_cols = st.columns(len(nodes))
evidence = {}
for i, node in enumerate(nodes):
    with inf_cols[i]:
        choice = st.selectbox(f"Zustand {node}", ["Unbekannt", "0", "1"], key=f"inf_{node}")
        if choice != "Unbekannt":
            evidence[node] = int(choice)

if st.button("Berechne Wahrscheinlichkeit für gewählte Evidenz"):
    # Vereinfachte Darstellung: Wir filtern die Tabelle nach der Evidenz
    query_df = edited_df.copy()
    for node, val in evidence.items():
        query_df = query_df[query_df[node] == val]
    
    if query_df.empty:
        st.error("Diese Kombination kommt in den Trainingsdaten nicht vor!")
    else:
        st.success("Wahrscheinlichkeiten basierend auf aktueller Evidenz:")
        st.dataframe(query_df.mean().to_frame("Wahrscheinlichkeit (Zustand=1)").style.format("{:.2%}"))

st.sidebar.success('💡 **Tipp:** Ändere die Eltern-Beziehungen links, um zu sehen, wie die Tabellen unten komplexer werden (Kombinatorik!).')
