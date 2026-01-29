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

st.title("🎓 Bayes-Netz: Training & Komplexe bedingte Inferenz")

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
            st.session_state.cpt_values = {} 
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

cpt_storage_for_sim = {}
for n in st.session_state.nodes_config.keys():
    parents = [s for s, t in st.session_state.edges if t == n]
    states = st.session_state.nodes_config[n]
    st.write(f"### CPT für {n}")
    
    parent_states = [st.session_state.nodes_config[p] for p in parents]
    p_combs = list(itertools.product(*parent_states))
    row_labels = [" | ".join(map(str, combo)) for combo in p_combs] if parents else ["Basiswahrscheinlichkeit (%)"]
    
    if n not in st.session_state.cpt_values or len(st.session_state.cpt_values[n]) != len(row_labels):
        df_init = pd.DataFrame(100/len(states), index=row_labels, columns=states)
        st.session_state.cpt_values[n] = df_init

    edited_cpt = st.data_editor(st.session_state.cpt_values[n], key=f"editor_{n}")
    normalized = edited_cpt.div(edited_cpt.sum(axis=1), axis=0).fillna(1/len(states))
    
    if not parents:
        cpt_storage_for_sim[n] = normalized.iloc[0].to_dict()
    else:
        cpt_storage_for_sim[n] = {
            "parents": parents, 
            "lookup": {p_combs[i]: normalized.iloc[i].to_dict() for i in range(len(p_combs))}
        }

# --- 4. KOMPLEXE INFERENZ ---
st.divider()
st.header("4. Inferenz: Beliebige bedingte Abfragen")
st.markdown("Berechne $P(Ziel | Bedingung)$. Wähle Variablen aus, deren gemeinsame Verteilung dich interessiert.")

col_ev, col_target = st.columns(2)

with col_ev:
    st.subheader("Bedingung (Evidenz)")
    evidence = {}
    for n in st.session_state.nodes_config.keys():
        opt = ["Keine"] + st.session_state.nodes_config[n]
        sel = st.selectbox(f"Fixiere {n}", opt, key=f"ev_inf_{n}")
        if sel != "Keine":
            evidence[n] = sel

with col_target:
    st.subheader("Zielvariablen (Joint Query)")
    query_nodes = st.multiselect("Wahrscheinlichkeit für Kombination von:", 
                                 [n for n in st.session_state.nodes_config.keys() if n not in evidence],
                                 default=[n for n in st.session_state.nodes_config.keys() if n not in evidence][:1])

num_samples = st.slider("Samples", 1000, 20000, 5000)

if st.button("🚀 Inferenz berechnen"):
    all_nodes = list(st.session_state.nodes_config.keys())
    valid_samples = []
    attempts = 0
    
    with st.spinner("Rejection Sampling läuft..."):
        while len(valid_samples) < num_samples and attempts < num_samples * 100:
            attempts += 1
            sample, to_proc, valid = {}, all_nodes.copy(), True
            while to_proc:
                for n in to_proc:
                    pars = [s for s, t in st.session_state.edges if t == n]
                    if all(p in sample for p in pars):
                        stts = st.session_state.nodes_config[n]
                        prbs = [cpt_storage_for_sim[n][s] for s in stts] if not pars else [cpt_storage_for_sim[n]["lookup"][tuple(sample[p] for p in pars)][s] for s in stts]
                        val = np.random.choice(stts, p=prbs)
                        if n in evidence and val != evidence[n]:
                            valid = False; break
                        sample[n] = val; to_proc.remove(n); break
                if not valid: break
            if valid: valid_samples.append(sample)
        
    if valid_samples:
        df_res = pd.DataFrame(valid_samples)
        st.success(f"Erfolg: {len(valid_samples)} Samples nach {attempts} Versuchen.")
        
        if query_nodes:
            st.subheader(f"Gemeinsame Wahrscheinlichkeit P({', '.join(query_nodes)} | Evidenz)")
            joint = df_res.groupby(query_nodes).size() / len(df_res)
            st.table(joint.to_frame("Wahrscheinlichkeit").style.format("{:.2%}"))
            
            for qn in query_nodes:
                st.write(f"**Einzel-Marginal P({qn} | Evidenz)**")
                st.bar_chart(df_res[qn].value_counts(normalize=True))
    else:
        st.error("Evidenz zu unwahrscheinlich für Rejection Sampling.")

st.sidebar.info("Schulungs-Tipp: Nutze die 'Joint Query' um zu zeigen, dass $P(A,B|E)$ nicht einfach $P(A|E) \cdot P(B|E)$ sein muss (Abhängigkeiten!).")
