import streamlit as st
import pandas as pd
import numpy as np
import itertools

st.set_page_config(page_title="Bayes-Schulungs-Simulator", layout="wide")

# --- INITIALISIERUNG ---
if 'nodes_config' not in st.session_state:
    st.session_state.nodes_config = {
        "A": ["Kalt", "Warm"],
        "B": ["Aus", "An"],
        "C": ["Niedrig", "Hoch"],
        "D": ["Fehler", "OK"]
    }
if 'edges' not in st.session_state:
    st.session_state.edges = []
if 'training_data' not in st.session_state:
    # Start-Daten mit 0 und 1 für One-Hot Logik
    st.session_state.training_data = pd.DataFrame(
        [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]], 
        columns=["A", "B", "C", "D"]
    )

st.title("🎓 Bayes-Netz: Training & Simulation")

# --- 1. STRUKTUR-EDITOR (SIDEBAR) ---
with st.sidebar:
    st.header("1. Netzwerk-Design")
    for n in list(st.session_state.nodes_config.keys()):
        states = st.text_input(f"Zustände {n}", ",".join(st.session_state.nodes_config[n]), key=f"s_{n}")
        st.session_state.nodes_config[n] = [s.strip() for s in states.split(",")]

    src = st.selectbox("Ursache (Eltern)", list(st.session_state.nodes_config.keys()))
    tgt = st.selectbox("Wirkung (Kind)", [n for n in st.session_state.nodes_config.keys() if n != src])
    if st.button("Verbindung hinzufügen ➕"):
        if (src, tgt) not in st.session_state.edges:
            st.session_state.edges.append((src, tgt))
    if st.button("Struktur zurücksetzen 🗑️"):
        st.session_state.edges = []
        st.rerun()

# --- 2. TRAININGSDATEN (ONE-HOT) ---
st.header("2. Training aus Daten")
st.markdown("Gib hier Beobachtungen ein (0 = erster Zustand, 1 = zweiter Zustand etc.).")
trained_df = st.data_editor(st.session_state.training_data, num_rows="dynamic", use_container_width=True)

# Graph zur Visualisierung
dot = "digraph { rankdir=LR; node [style=filled, fillcolor='#E1F5FE', shape=box]; "
for n in st.session_state.nodes_config:
    dot += f'{n}; '
for s, t in st.session_state.edges:
    dot += f"{s} -> {t}; "
dot += "}"
st.graphviz_chart(dot)


# --- 3. CPT BERECHNUNG (TRAINING) ---
st.header("3. Bedingte Wahrscheinlichkeitstabellen (CPTs)")
st.info("Diese Tabellen wurden aus den obigen Daten gelernt. Du kannst sie manuell überschreiben.")

cpt_storage = {}

for n in st.session_state.nodes_config.keys():
    parents = [s for s, t in st.session_state.edges if t == n]
    states = st.session_state.nodes_config[n]
    
    # Training: Häufigkeiten zählen
    if not parents:
        # Prior lernen
        counts = trained_df[n].value_counts(normalize=True).to_dict()
        # Mapping von Index (0,1..) auf Namen (Kalt, Warm..)
        init_values = [counts.get(i, 1/len(states)) * 100 for i in range(len(states))]
        df_cpt = pd.DataFrame([init_values], columns=states, index=["Wahrscheinlichkeit (%)"])
    else:
        # Bedingt lernen
        parent_states = [st.session_state.nodes_config[p] for p in parents]
        combinations = list(itertools.product(*parent_states))
        comb_indices = list(itertools.product(*[range(len(st.session_state.nodes_config[p])) for p in parents]))
        
        row_labels = [" | ".join(map(str, combo)) for combo in combinations]
        df_cpt = pd.DataFrame(0.0, index=row_labels, columns=states)
        
        # MLE Training aus dem DataFrame
        for i, (comb_idx, label) in enumerate(zip(comb_indices, row_labels)):
            subset = trained_df
            for p_idx, p_name in enumerate(parents):
                subset = subset[subset[p_name] == comb_idx[p_idx]]
            
            if len(subset) > 0:
                dist = subset[n].value_counts(normalize=True).to_dict()
                for s_idx, s_name in enumerate(states):
                    df_cpt.loc[label, s_name] = dist.get(s_idx, 0.0) * 100
            else:
                df_cpt.iloc[i] = 100 / len(states) # Laplace Smoothing / Uniform bei fehlenden Daten

    # Manueller Editor für die CPT
    edited_cpt = st.data_editor(df_cpt, key=f"ed_{n}")
    
    # Normalisierte Daten für Simulation speichern
    normalized = edited_cpt.div(edited_cpt.sum(axis=1), axis=0).fillna(1/len(states))
    if not parents:
        cpt_storage[n] = normalized.iloc[0].to_dict()
    else:
        cpt_storage[n] = {"parents": parents, "lookup": {comb: normalized.iloc[i].to_dict() for i, comb in enumerate(combinations)}}

# --- 4. FORWARD SAMPLING ---
st.divider()
st.header("4. Inferenz (Simulation)")
num_samples = st.slider("Samples ziehen", 100, 5000, 1000)

if st.button("🚀 Forward Sampling starten"):
    all_nodes = list(st.session_state.nodes_config.keys())
    final_samples = []
    
    for _ in range(num_samples):
        current_sample = {}
        to_process = all_nodes.copy()
        while to_process:
            for n in to_process:
                parents = [s for s, t in st.session_state.edges if t == n]
                if all(p in current_sample for p in parents):
                    states = st.session_state.nodes_config[n]
                    if not parents:
                        probs = [cpt_storage[n][s] for s in states]
                    else:
                        p_vals = tuple(current_sample[p] for p in parents)
                        probs = [cpt_storage[n]["lookup"][p_vals][s] for s in states]
                    
                    current_sample[n] = np.random.choice(states, p=probs)
                    to_process.remove(n)
                    break
        final_samples.append(current_sample)
    
    res_df = pd.DataFrame(final_samples)
    cols = st.columns(len(all_nodes))
    for i, n in enumerate(all_nodes):
        with cols[i]:
            st.write(f"**Verteilung {n}**")
            st.bar_chart(res_df[n].value_counts(normalize=True))

st.sidebar.info("Schulungs-Tipp: Ändere die 0/1 Werte in der Trainingsdaten-Tabelle und beobachte, wie 'Training' die CPTs unten sofort beeinflusst!")
