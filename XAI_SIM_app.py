import streamlit as st
import pandas as pd
import numpy as np
import itertools

st.set_page_config(page_title="Bayes-Schulungs-Simulator Pro", layout="wide")

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
    st.session_state.training_data = pd.DataFrame(
        [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]], 
        columns=["A", "B", "C", "D"]
    )
if 'cpt_values' not in st.session_state:
    st.session_state.cpt_values = {}

st.title("🎓 Bayes-Netz: Training & Beliebige bedingte Inferenz")

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
            st.session_state.cpt_values = {} # Reset CPTs bei Strukturänderung
    if st.button("Struktur zurücksetzen 🗑️"):
        st.session_state.edges = []
        st.session_state.cpt_values = {}
        st.rerun()

# --- 2. TRAININGSDATEN (ONE-HOT) ---
st.header("2. Trainingsdaten (One-Hot)")
st.markdown("Nutze 0 für den 1. Zustand, 1 für den 2. Zustand usw.")
trained_df = st.data_editor(st.session_state.training_data, num_rows="dynamic", use_container_width=True)

# Graph zur Visualisierung
dot = "digraph { rankdir=LR; node [style=filled, fillcolor='#E1F5FE', shape=box, fontname='Arial']; "
for n in st.session_state.nodes_config:
    dot += f'{n}; '
for s, t in st.session_state.edges:
    dot += f"{s} -> {t}; "
dot += "}"
st.graphviz_chart(dot)

# --- 3. TRAININGSKNOPF & CPT BERECHNUNG ---
st.header("3. Bedingte Wahrscheinlichkeitstabellen (CPTs)")

if st.button("🎯 Wahrscheinlichkeiten aus Daten lernen"):
    new_cpts = {}
    for n in st.session_state.nodes_config.keys():
        parents = [s for s, t in st.session_state.edges if t == n]
        states = st.session_state.nodes_config[n]
        
        if not parents:
            counts = trained_df[n].value_counts(normalize=True).to_dict()
            vals = [counts.get(i, 1/len(states)) * 100 for i in range(len(states))]
            new_cpts[n] = pd.DataFrame([vals], columns=states, index=["Basiswahrscheinlichkeit (%)"])
        else:
            parent_states = [st.session_state.nodes_config[p] for p in parents]
            combinations = list(itertools.product(*parent_states))
            comb_indices = list(itertools.product(*[range(len(st.session_state.nodes_config[p])) for p in parents]))
            row_labels = [" | ".join(map(str, combo)) for combo in combinations]
            
            df_cpt = pd.DataFrame(0.0, index=row_labels, columns=states)
            for i, (comb_idx, label) in enumerate(zip(comb_indices, row_labels)):
                subset = trained_df
                for p_idx, p_name in enumerate(parents):
                    subset = subset[subset[p_name] == comb_idx[p_idx]]
                
                if len(subset) > 0:
                    dist = subset[n].value_counts(normalize=True).to_dict()
                    for s_idx, s_name in enumerate(states):
                        df_cpt.loc[label, s_name] = dist.get(s_idx, 0.0) * 100
                else:
                    df_cpt.iloc[i] = 100 / len(states)
            new_cpts[n] = df_cpt
    st.session_state.cpt_values = new_cpts
    st.success("Training abgeschlossen!")

# Anzeige und manuelles Editieren der CPTs
cpt_storage_for_sim = {}
for n in st.session_state.nodes_config.keys():
    parents = [s for s, t in st.session_state.edges if t == n]
    states = st.session_state.nodes_config[n]
    st.write(f"### CPT für {n}")
    
    # Sicherstellen, dass die Tabelle zur aktuellen Struktur passt
    parent_states = [st.session_state.nodes_config[p] for p in parents]
    p_combs = list(itertools.product(*parent_states))
    row_labels = [" | ".join(map(str, combo)) for combo in p_combs] if parents else ["Basiswahrscheinlichkeit (%)"]
    
    if n not in st.session_state.cpt_values or len(st.session_state.cpt_values[n]) != len(row_labels):
        df_init = pd.DataFrame(100/len(states), index=row_labels, columns=states)
        st.session_state.cpt_values[n] = df_init

    edited_cpt = st.data_editor(st.session_state.cpt_values[n], key=f"editor_{n}")
    
    # Normalisierung
    normalized = edited_cpt.div(edited_cpt.sum(axis=1), axis=0).fillna(1/len(states))
    
    if not parents:
        cpt_storage_for_sim[n] = normalized.iloc[0].to_dict()
    else:
        # Hier lag der Fehler: Wir nutzen nun direkt die Zeilen des aktuellen Editors
        cpt_storage_for_sim[n] = {
            "parents": parents, 
            "lookup": {p_combs[i]: normalized.iloc[i].to_dict() for i in range(len(p_combs))}
        }

# --- 4. BEDINGTE INFERENZ ---
st.divider()
st.header("4. Inferenz: Beliebige bedingte Wahrscheinlichkeiten")
st.markdown("Setze Bedingungen fest. Das Modell berechnet die Wahrscheinlichkeiten unter dieser Evidenz.")

evidence = {}
ev_cols = st.columns(len(st.session_state.nodes_config))
for i, n in enumerate(st.session_state.nodes_config.keys()):
    with ev_cols[i]:
        options = ["Keine"] + st.session_state.nodes_config[n]
        sel = st.selectbox(f"Bedingung für {n}", options, key=f"ev_{n}")
        if sel != "Keine":
            evidence[n] = sel

num_samples = st.slider("Samples", 500, 10000, 2000)

if st.button("🚀 Bedingte Wahrscheinlichkeiten berechnen"):
    all_nodes = list(st.session_state.nodes_config.keys())
    valid_samples = []
    attempts = 0
    max_attempts = num_samples * 50 
    
    with st.spinner("Simuliere..."):
        while len(valid_samples) < num_samples and attempts < max_attempts:
            attempts += 1
            current_sample = {}
            nodes_to_process = all_nodes.copy()
            is_valid = True
            
            while nodes_to_process:
                for n in nodes_to_process:
                    parents = [s for s, t in st.session_state.edges if t == n]
                    if all(p in current_sample for p in parents):
                        states = st.session_state.nodes_config[n]
                        if not parents:
                            probs = [cpt_storage_for_sim[n][s] for s in states]
                        else:
                            p_vals = tuple(current_sample[p] for p in parents)
                            probs = [cpt_storage_for_sim[n]["lookup"][p_vals][s] for s in states]
                        
                        sampled_val = np.random.choice(states, p=probs)
                        if n in evidence and sampled_val != evidence[n]:
                            is_valid = False
                            break
                        current_sample[n] = sampled_val
                        nodes_to_process.remove(n)
                        break
                if not is_valid: break
            if is_valid: valid_samples.append(current_sample)
        
    if valid_samples:
        res_df = pd.DataFrame(valid_samples)
        st.success(f"Erfolg! {len(valid_samples)} passende Samples gefunden.")
        
        # Einzelwahrscheinlichkeiten
        res_cols = st.columns(len(all_nodes))
        for i, n in enumerate(all_nodes):
            with res_cols[i]:
                st.write(f"**P({n} | Evidenz)**")
                dist = res_df[n].value_counts(normalize=True).sort_index()
                st.bar_chart(dist)
                st.table(dist.to_frame("P"))

        # Beliebige bedingte Abfrage (Joint Distribution)
        remaining = [n for n in all_nodes if n not in evidence]
        if len(remaining) > 1:
            st.subheader(f"Gemeinsame bedingte Verteilung: P({', '.join(remaining)} | Evidenz)")
            joint = res_df.groupby(remaining).size() / len(res_df)
            st.dataframe(joint.to_frame("Wahrscheinlichkeit").style.format("{:.2%}"))
    else:
        st.error("Keine Samples gefunden. Evidenz zu unwahrscheinlich.")

st.sidebar.info("Tipp: Wenn du die Struktur änderst, werden die CPTs automatisch zurückgesetzt, um Index-Fehler zu vermeiden.")
