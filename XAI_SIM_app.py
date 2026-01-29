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
    st.session_state.training_data = pd.DataFrame(
        [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]], 
        columns=["A", "B", "C", "D"]
    )
if 'cpt_values' not in st.session_state:
    st.session_state.cpt_values = {}

st.title("🎓 Bayes-Netz: Training & Bedingte Inferenz")

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
st.header("2. Trainingsdaten (One-Hot)")
st.markdown("Nutze 0 für den 1. Zustand, 1 für den 2. Zustand usw.")
trained_df = st.data_editor(st.session_state.training_data, num_rows="dynamic", use_container_width=True)

# Graph zur Visualisierung
dot = "digraph { rankdir=LR; node [style=filled, fillcolor='#E1F5FE', shape=box]; "
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
    st.success("Erfolgreich trainiert!")

# Anzeige und manuelles Editieren der CPTs
cpt_storage_for_sim = {}
for n in st.session_state.nodes_config.keys():
    parents = [s for s, t in st.session_state.edges if t == n]
    states = st.session_state.nodes_config[n]
    
    st.write(f"### CPT für {n}")
    
    # Default Werte falls noch nicht trainiert
    if n not in st.session_state.cpt_values:
        if not parents:
            df_init = pd.DataFrame([[100/len(states)] * len(states)], columns=states, index=["Basiswahrscheinlichkeit (%)"])
        else:
            parent_states = [st.session_state.nodes_config[p] for p in parents]
            row_labels = [" | ".join(map(str, combo)) for combo in itertools.product(*parent_states)]
            df_init = pd.DataFrame(100/len(states), index=row_labels, columns=states)
        st.session_state.cpt_values[n] = df_init

    edited_cpt = st.data_editor(st.session_state.cpt_values[n], key=f"editor_{n}")
    
    # Für Simulation aufbereiten
    normalized = edited_cpt.div(edited_cpt.sum(axis=1), axis=0).fillna(1/len(states))
    if not parents:
        cpt_storage_for_sim[n] = normalized.iloc[0].to_dict()
    else:
        parent_states = [st.session_state.nodes_config[p] for p in parents]
        combinations = list(itertools.product(*parent_states))
        cpt_storage_for_sim[n] = {
            "parents": parents, 
            "lookup": {comb: normalized.iloc[i].to_dict() for i, comb in enumerate(combinations)}
        }

# --- 4. BEDINGTE INFERENZ & FORWARD SAMPLING ---
st.divider()
st.header("4. Inferenz: Bedingte Wahrscheinlichkeiten")
st.markdown("Setze Bedingungen fest, um bedingte Verteilungen zu berechnen (z.B. $P(Knoten | Evidenz)$).")

evidence = {}
ev_cols = st.columns(len(st.session_state.nodes_config))
for i, n in enumerate(st.session_state.nodes_config.keys()):
    with ev_cols[i]:
        options = ["Keine"] + st.session_state.nodes_config[n]
        sel = st.selectbox(f"Evidenz für {n}", options, key=f"ev_{n}")
        if sel != "Keine":
            evidence[n] = sel

num_samples = st.slider("Anzahl Samples", 500, 10000, 2000)

if st.button("🚀 Bedingte Inferenz berechnen"):
    all_nodes = list(st.session_state.nodes_config.keys())
    valid_samples = []
    attempts = 0
    max_attempts = num_samples * 10 # Sicherheitsstopp bei seltener Evidenz
    
    with st.status("Simuliere...") as status:
        while len(valid_samples) < num_samples and attempts < max_attempts:
            attempts += 1
            current_sample = {}
            to_process = all_nodes.copy()
            is_valid = True
            
            while to_process:
                for n in to_process:
                    parents = [s for s, t in st.session_state.edges if t == n]
                    if all(p in current_sample for p in parents):
                        states = st.session_state.nodes_config[n]
                        if not parents:
                            probs = [cpt_storage_for_sim[n][s] for s in states]
                        else:
                            p_vals = tuple(current_sample[p] for p in parents)
                            probs = [cpt_storage_for_sim[n]["lookup"][p_vals][s] for s in states]
                        
                        sampled_val = np.random.choice(states, p=probs)
                        
                        # Evidenz-Check (Rejection Sampling)
                        if n in evidence and sampled_val != evidence[n]:
                            is_valid = False
                            break
                        
                        current_sample[n] = sampled_val
                        to_process.remove(n)
                        break
                if not is_valid: break
            
            if is_valid:
                valid_samples.append(current_sample)
        
        if len(valid_samples) < num_samples:
            st.warning(f"Nur {len(valid_samples)} Samples gefunden, die der Evidenz entsprechen.")
        status.update(label="Simulation abgeschlossen!", state="complete")

    if valid_samples:
        res_df = pd.DataFrame(valid_samples)
        st.subheader("Bedingte Verteilungen")
        cols = st.columns(len(all_nodes))
        for i, n in enumerate(all_nodes):
            with cols[i]:
                st.write(f"**P({n} | Evidenz)**")
                dist = res_df[n].value_counts(normalize=True).sort_index()
                st.bar_chart(dist)
    else:
        st.error("Keine Samples gefunden! Die gewählte Evidenz ist laut Modell unmöglich oder extrem unwahrscheinlich.")

st.sidebar.markdown("---")
st.sidebar.info("Schulungs-Hinweis: Das 'Lernen' befüllt die CPTs basierend auf den relativen Häufigkeiten der 0/1-Werte in der Tabelle.")
