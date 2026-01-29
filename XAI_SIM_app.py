import streamlit as st
import pandas as pd
import numpy as np

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
    
    # Knoten-Zustände definieren
    st.subheader("Zustände bearbeiten")
    for n in st.session_state.nodes_config.keys():
        states_str = st.text_input(f"Zustände für {n} (Komma-getrennt)", 
                                   value=",".join(st.session_state.nodes_config[n]))
        st.session_state.nodes_config[n] = [s.strip() for s in states_str.split(",")]

    st.subheader("Struktur (Drag & Drop Ersatz)")
    src = st.selectbox("Ursache", list(st.session_state.nodes_config.keys()))
    tgt = st.selectbox("Wirkung", [n for n in st.session_state.nodes_config.keys() if n != src])
    if st.button("Verbindung hinzufügen ➕"):
        if (src, tgt) not in st.session_state.edges:
            st.session_state.edges.append((src, tgt))
    if st.button("Reset 🗑️"):
        st.session_state.edges = []

# --- 2. GRAPH-VISUALISIERUNG ---
col_graph, col_data = st.columns([1, 1])

with col_graph:
    st.subheader("Netzwerk-Graph")
    dot = "digraph { rankdir=LR; node [style=filled, fillcolor='#E1F5FE', shape=ellipse]; "
    for n in st.session_state.nodes_config.keys():
        dot += f'{n} [label="{n}\\n({"/".join(st.session_state.nodes_config[n])})"]; '
    for s, t in st.session_state.edges:
        dot += f"{s} -> {t}; "
    dot += "}"
    st.graphviz_chart(dot)

# --- 3. DIE CPTs (DAS HERZSTÜCK) ---
st.header("2. Bedingte Wahrscheinlichkeitstabellen (CPTs)")
st.info("Hier kannst du die Wahrscheinlichkeiten für jeden Zustand direkt eingeben.")

cpt_data = {}

for n in st.session_state.nodes_config.keys():
    parents = [s for s, t in st.session_state.edges if t == n]
    st.write(f"### Tabelle für Knoten: {n}")
    
    if not parents:
        # Prior Tabelle
        df_prior = pd.DataFrame([1/len(st.session_state.nodes_config[n])] * len(st.session_state.nodes_config[n]), 
                                index=st.session_state.nodes_config[n], columns=["Wahrscheinlichkeit"])
        edited = st.data_editor(df_prior.T, key=f"cpt_{n}")
        cpt_data[n] = edited.iloc[0].to_dict()
    else:
        # Bedingte Tabelle erstellen
        import itertools
        parent_states = [st.session_state.nodes_config[p] for p in parents]
        combinations = list(itertools.product(*parent_states))
        
        index_names = parents
        prob_cols = st.session_state.nodes_config[n]
        
        df_cond = pd.DataFrame(
            1/len(prob_cols), 
            index=pd.MultiIndex.from_tuples(combinations, names=index_names),
            columns=prob_cols
        )
        edited = st.data_editor(df_cond, key=f"cpt_{n}")
        cpt_data[n] = edited

# --- 4. FORWARD SAMPLING (INFERENZ) ---
st.divider()
st.header("3. Forward Sampling Simulation")
num_samples = st.number_input("Anzahl der Simulationen", 100, 10000, 1000)

if st.button("🚀 Simulation starten"):
    results = []
    
    for _ in range(num_samples):
        sample = {}
        processed = set()
        
        while len(processed) < len(st.session_state.nodes_config):
            for n in st.session_state.nodes_config.keys():
                if n in processed: continue
                
                parents = [s for s, t in st.session_state.edges if t == n]
                if all(p in processed for p in parents):
                    states = st.session_state.nodes_config[n]
                    
                    if not parents:
                        probs = [cpt_data[n][s] for s in states]
                    else:
                        p_vals = tuple(sample[p] for p in parents)
                        # MultiIndex Zugriff
                        row = cpt_data[n].loc[p_vals if len(parents) > 1 else p_vals[0]]
                        probs = [row[s] for s in states]
                    
                    # Normalisieren falls User-Eingabe nicht 1 ergibt
                    probs = np.array(probs) / np.sum(probs)
                    sample[n] = np.random.choice(states, p=probs)
                    processed.add(n)
        results.append(sample)
    
    res_df = pd.DataFrame(results)
    st.subheader("Ergebnis der Verteilung")
    
    for n in st.session_state.nodes_config.keys():
        st.write(f"**Verteilung {n}:**")
        st.bar_chart(res_df[n].value_counts(normalize=True))

st.sidebar.markdown("---")
st.sidebar.help("Tipp: Wenn du die CPTs änderst, achte darauf, dass die Summe pro Zeile 1.0 ergeben sollte. Das Tool normalisiert aber automatisch nach.")
