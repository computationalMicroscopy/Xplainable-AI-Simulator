import streamlit as st
import pandas as pd
import numpy as np

# Konfiguration
st.set_page_config(page_title="Bayes-Netz Explorer", layout="wide")

# Session State Initialisierung
if 'edges' not in st.session_state:
    st.session_state.edges = []
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(
        [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]], 
        columns=["A", "B", "C", "D"]
    )

st.title("🔗 Interaktiver Bayes-Simulator")
st.markdown("Verbinde die Knoten grafisch und starte das Forward Sampling.")

# --- 1. GRAFISCHER AUFBAU (Interaktive Auswahl) ---
st.sidebar.header("Netzwerk Design")
nodes = ["A", "B", "C", "D"]

with st.sidebar.expander("Kanten definieren", expanded=True):
    source = st.selectbox("Von (Ursache)", nodes)
    target = st.selectbox("Nach (Wirkung)", [n for n in nodes if n != source])
    if st.button("Kante hinzufügen"):
        edge = (source, target)
        if edge not in st.session_state.edges:
            st.session_state.edges.append(edge)
    if st.button("Alle Kanten löschen"):
        st.session_state.edges = []

# Visualisierung des Graphen
dot = "digraph { rankdir=LR; node [style=filled, fillcolor='#F0F2F6', fontname='Arial']; "
for n in nodes:
    dot += f"{n}; "
for s, t in st.session_state.edges:
    dot += f"{s} -> {t}; "
dot += "}"

st.graphviz_chart(dot)



# --- 2. TRAININGSDATEN ---
st.header("1. Trainingsdaten (One-Hot)")
edited_df = st.data_editor(st.session_state.data, num_rows="dynamic", use_container_width=True)

# CPT Berechnung
def get_cpt(df, target):
    parents = [s for s, t in st.session_state.edges if t == target]
    if not parents:
        p1 = df[target].mean()
        return {"type": "prior", "p1": p1}
    else:
        # Berechne Wahrscheinlichkeit für jede Eltern-Kombination
        cpt = df.groupby(parents)[target].mean().to_dict()
        return {"type": "conditional", "parents": parents, "probs": cpt}

# --- 3. FORWARD SAMPLING INFERENZ ---
st.header("2. Forward Sampling Inferenz")
num_samples = st.slider("Anzahl der Samples", 10, 1000, 500)

if st.button("Simulation starten"):
    # Einfache topologische Sortierung (für 4 Knoten manuell/simpel)
    # In einer Schulung: Wir gehen davon aus, dass der User keinen Zyklus baut
    samples = pd.DataFrame(columns=nodes)
    
    for _ in range(num_samples):
        sample = {}
        # Wir müssen sicherstellen, dass Eltern vor Kindern berechnet werden
        # Für diesen Simulator nutzen wir eine feste Reihenfolge oder prüfen Abhängigkeiten
        nodes_to_process = nodes.copy()
        while nodes_to_process:
            for n in nodes_to_process:
                parents = [s for s, t in st.session_state.edges if t == n]
                if all(p in sample for p in parents):
                    cpt = get_cpt(edited_df, n)
                    if cpt["type"] == "prior":
                        prob = cpt["p1"]
                    else:
                        parent_vals = tuple(sample[p] for p in cpt["parents"])
                        # Fallback falls Kombination in Daten nicht existiert
                        prob = cpt["probs"].get(parent_vals if len(parent_vals) > 1 else parent_vals[0], 0.5)
                    
                    sample[n] = 1 if np.random.rand() < prob else 0
                    nodes_to_process.remove(n)
                    break
        samples = pd.concat([samples, pd.DataFrame([sample])], ignore_index=True)

    st.subheader("Ergebnis der Simulation (Generierte Daten)")
    st.dataframe(samples.head(10))
    
    st.subheader("Vergleich: Training vs. Simulation (P(X=1))")
    comparison = pd.DataFrame({
        "Original (Training)": edited_df.mean(),
        "Simuliert (Sampling)": samples.mean()
    })
    st.bar_chart(comparison)

# --- 4. CPT ANZEIGE ---
st.divider()
st.header("3. Lokale CPTs (Details)")
cols = st.columns(4)
for i, n in enumerate(nodes):
    with cols[i]:
        st.write(f"**Knoten {n}**")
        cpt = get_cpt(edited_df, n)
        if cpt["type"] == "prior":
            st.table(pd.DataFrame({"P(1)": [cpt["p1"]]}))
        else:
            st.write("Bedingt durch:", cpt['parents'])
            st.table(pd.DataFrame.from_dict(cpt["probs"], orient='index', columns=['P(1)']))

st.sidebar.info("Schulungs-Hinweis: Forward Sampling zeigt, wie das Netz 'denkt'. Die Balken sollten bei genug Samples dem Original ähneln.")
