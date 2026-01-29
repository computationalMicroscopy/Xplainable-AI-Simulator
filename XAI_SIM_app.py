import streamlit as st
import pandas as pd
import numpy as np
import itertools

st.set_page_config(page_title="Bayes-Expert-Simulator", layout="wide")

# --- SESSION STATE INITIALISIERUNG ---
if 'nodes_config' not in st.session_state:
    st.session_state.nodes_config = {
        "A": ["Kalt", "Warm"],
        "B": ["Aus", "An"],
        "C": ["Niedrig", "Hoch"],
        "D": ["Fehler", "OK"]
    }
if 'edges' not in st.session_state:
    st.session_state.edges = []

st.title("🧠 Bayes-Netzwerk Experte & Simulator")

# --- 1. KNOTEN & STRUKTUR ---
with st.sidebar:
    st.header("1. Konfiguration")
    
    st.subheader("Zustände bearbeiten")
    for n in list(st.session_state.nodes_config.keys()):
        states_str = st.text_input(f"Zustände für {n}", 
                                   value=",".join(st.session_state.nodes_config[n]),
                                   key=f"input_{n}")
        st.session_state.nodes_config[n] = [s.strip() for s in states_str.split(",")]

    st.subheader("Struktur")
    src = st.selectbox("Ursache", list(st.session_state.nodes_config.keys()))
    tgt = st.selectbox("Wirkung", [n for n in st.session_state.nodes_config.keys() if n != src])
    if st.button("Verbindung hinzufügen ➕"):
        if (src, tgt) not in st.session_state.edges:
            st.session_state.edges.append((src, tgt))
    if st.button("Kanten löschen 🗑️"):
        st.session_state.edges = []
        st.rerun()

# --- 2. GRAPH-VISUALISIERUNG ---
st.header("Netzwerk-Graph")
dot = "digraph { rankdir=LR; node [style=filled, fillcolor='#E1F5FE', shape=box, fontname='Arial']; "
for n, states in st.session_state.nodes_config.items():
    dot += f'{n} [label="{n}\\n({"/".join(states)})"]; '
for s, t in st.session_state.edges:
    dot += f"{s} -> {t}; "
dot += "}"
st.graphviz_chart(dot)

# --- 3. DIE CPTs ---
st.header("2. Bedingte Wahrscheinlichkeitstabellen (CPTs)")
st.info("Trage Wahrscheinlichkeiten ein (Werte werden pro Zeile automatisch auf 100% normalisiert).")

cpt_data = {}

for n in st.session_state.nodes_config.keys():
    parents = [s for s, t in st.session_state.edges if t == n]
    st.write(f"### Tabelle für Knoten: {n}")
    
    states = st.session_state.nodes_config[n]
    
    if not parents:
        # Wurzelknoten: Einfache Zeile
        df_prior = pd.DataFrame([[100/len(states)] * len(states)], columns=states, index=["Basiswahrscheinlichkeit (%)"])
        edited = st.data_editor(df_prior, key=f"editor_{n}")
        # Normalisierung
        row_sum = edited.iloc[0].sum()
        cpt_data[n] = (edited.iloc[0] / row_sum).to_dict() if row_sum > 0 else {s: 1/len(states) for s in states}
    else:
        # Bedingte Tabelle: MultiIndex fixen durch String-Kombination
        parent_states = [st.session_state.nodes_config[p] for p in parents]
        combinations = list(itertools.product(*parent_states))
        
        # Erzeuge lesbare Zeilenbeschriftungen: "Zustand_P1, Zustand_P2"
        row_labels = [" | ".join(map(str, combo)) for combo in combinations]
        
        df_cond = pd.DataFrame(
            100/len(states), 
            index=row_labels,
            columns=states
        )
        df_cond.index.name = "Eltern-Zustände (" + " & ".join(parents) + ")"
        
        edited = st.data_editor(df_cond, key=f"editor_{n}")
        
        # Zurück-Mapping für das Sampling
        normalized_df = edited.div(edited.sum(axis=1), axis=0).fillna(1/len(states))
        # Wir speichern das Mapping von der Kombinations-Tuple zur Wahrscheinlichkeitsverteilung
        cpt_dict = {}
        for i, combo in enumerate(combinations):
            cpt_dict[combo] = normalized_df.iloc[i].to_dict()
        cpt_data[n] = {"type": "conditional", "parents": parents, "lookup": cpt_dict}

# --- 4. FORWARD SAMPLING ---
st.divider()
st.header("3. Inferenz via Forward Sampling")
num_samples = st.slider("Samples", 100, 5000, 1000)

if st.button("🚀 Simulation starten"):
    results = []
    
    # Topologische Sortierung (simpel für kleine Graphen)
    all_nodes = list(st.session_state.nodes_config.keys())
    
    for _ in range(num_samples):
        sample = {}
        remaining = all_nodes.copy()
        
        while remaining:
            for n in remaining:
                parents = [s for s, t in st.session_state.edges if t == n]
                if all(p in sample for p in parents):
                    states = st.session_state.nodes_config[n]
                    
                    if not parents:
                        probs_dict = cpt_data[n]
                    else:
                        p_vals = tuple(sample[p] for p in parents)
                        probs_dict = cpt_data[n]["lookup"][p_vals]
                    
                    p_list = [probs_dict[s] for s in states]
                    sample[n] = np.random.choice(states, p=p_list)
                    remaining.remove(n)
                    break
        results.append(sample)
    
    res_df = pd.DataFrame(results)
    
    # Visualisierung der Ergebnisse
    cols = st.columns(len(all_nodes))
    for i, n in enumerate(all_nodes):
        with cols[i]:
            st.write(f"**Ergebnis {n}**")
            counts = res_df[n].value_counts(normalize=True)
            st.bar_chart(counts)
