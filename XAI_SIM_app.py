import streamlit as st
import pandas as pd
import numpy as np
import itertools

st.set_page_config(page_title="Bayes-Schulungs-Simulator Pro", layout="wide")

# --- INITIALISIERUNG (ROBUST) ---
# Wir löschen den State, falls die Struktur veraltet ist, um KeyErrors zu vermeiden
if 'nodes_config' in st.session_state:
    if not isinstance(st.session_state.nodes_config["A"], dict):
        del st.session_state.nodes_config

if 'nodes_config' not in st.session_state:
    st.session_state.nodes_config = {
        "A": {"name": "Wetter", "states": ["Sonne", "Regen"]},
        "B": {"name": "Sprinkler", "states": ["Aus", "An"]},
        "C": {"name": "Rasen", "states": ["Trocken", "Nass"]},
        "D": {"name": "Stimmung", "states": ["Gut", "Schlecht"]}
    }
if 'edges' not in st.session_state:
    st.session_state.edges = []
if 'training_data' not in st.session_state:
    st.session_state.training_data = pd.DataFrame(
        [[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]], 
        columns=["A", "B", "C", "D"]
    )
if 'cpt_values' not in st.session_state:
    st.session_state.cpt_values = {}

st.title("🎓 Bayes-Netz: Training, Benennung & Inferenz")

# --- 1. STRUKTUR-EDITOR (SIDEBAR) ---
with st.sidebar:
    st.header("1. Netzwerk-Design")
    for n in ["A", "B", "C", "D"]:
        st.subheader(f"Knoten {n}")
        # Namen und Zustände anpassen
        st.session_state.nodes_config[n]["name"] = st.text_input(
            f"Anzeigename für {n}", 
            st.session_state.nodes_config[n]["name"], 
            key=f"name_input_{n}"
        )
        
        states_str = st.text_input(
            f"Zustände für {st.session_state.nodes_config[n]['name']}", 
            ",".join(st.session_state.nodes_config[n]["states"]), 
            key=f"states_input_{n}"
        )
        st.session_state.nodes_config[n]["states"] = [s.strip() for s in states_str.split(",")]

    st.divider()
    st.subheader("Kanten (Struktur)")
    node_opts = {n: st.session_state.nodes_config[n]["name"] for n in ["A", "B", "C", "D"]}
    src_id = st.selectbox("Ursache", options=list(node_opts.keys()), format_func=lambda x: node_opts[x])
    tgt_id = st.selectbox("Wirkung", options=[n for n in node_opts.keys() if n != src_id], format_func=lambda x: node_opts[x])
    
    if st.button("Verbindung hinzufügen ➕"):
        if (src_id, tgt_id) not in st.session_state.edges:
            st.session_state.edges.append((src_id, tgt_id))
            st.session_state.cpt_values = {} 
            st.rerun()
    if st.button("Struktur zurücksetzen 🗑️"):
        st.session_state.edges = []
        st.session_state.cpt_values = {}
        st.rerun()

# --- 2. TRAININGSDATEN ---
st.header("2. Trainingsdaten (One-Hot / Index-basiert)")
st.info("Spalte A entspricht Knoten A, etc. Nutze 0, 1, 2... für die Zustände.")
trained_df = st.data_editor(st.session_state.training_data, num_rows="dynamic", use_container_width=True)

# Graph Visualisierung
st.subheader("Visualisierung des Netzwerks")
dot = "digraph { rankdir=LR; node [style=filled, fillcolor='#E1F5FE', shape=box, fontname='Arial']; "
for nid, cfg in st.session_state.nodes_config.items():
    dot += f'{nid} [label="{cfg["name"]}\\n({"/".join(cfg["states"])})"]; '
for s, t in st.session_state.edges:
    dot += f"{s} -> {t}; "
dot += "}"
st.graphviz_chart(dot)

# --- 3. TRAINING & CPTs ---
st.header("3. Wahrscheinlichkeitstabellen (CPTs)")

if st.button("🎯 Aus Daten lernen"):
    new_cpts = {}
    for n in ["A", "B", "C", "D"]:
        parents = [s for s, t in st.session_state.edges if t == n]
        states = st.session_state.nodes_config[n]["states"]
        
        if not parents:
            counts = trained_df[n].value_counts(normalize=True).to_dict()
            vals = [counts.get(i, 1/len(states)) * 100 for i in range(len(states))]
            new_cpts[n] = pd.DataFrame([vals], columns=states, index=["Basis (%)"])
        else:
            p_states_list = [st.session_state.nodes_config[p]["states"] for p in parents]
            combinations = list(itertools.product(*p_states_list))
            comb_indices = list(itertools.product(*[range(len(st.session_state.nodes_config[p]["states"])) for p in parents]))
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
    st.rerun()

cpt_storage_for_sim = {}
for n in ["A", "B", "C", "D"]:
    cfg = st.session_state.nodes_config[n]
    parents = [s for s, t in st.session_state.edges if t == n]
    p_states_list = [st.session_state.nodes_config[p]["states"] for p in parents]
    p_combs = list(itertools.product(*p_states_list))
    row_labels = [" | ".join(map(str, combo)) for combo in p_combs] if parents else ["Basis (%)"]
    
    if n not in st.session_state.cpt_values or len(st.session_state.cpt_values[n]) != len(row_labels):
        st.session_state.cpt_values[n] = pd.DataFrame(100/len(cfg["states"]), index=row_labels, columns=cfg["states"])

    st.write(f"### CPT: {cfg['name']}")
    edited_cpt = st.data_editor(st.session_state.cpt_values[n], key=f"editor_{n}")
    
    # Für Inferenz normalisieren
    norm = edited_cpt.div(edited_cpt.sum(axis=1), axis=0).fillna(1/len(cfg["states"]))
    if not parents:
        cpt_storage_for_sim[n] = norm.iloc[0].to_dict()
    else:
        cpt_storage_for_sim[n] = {"parents": parents, "lookup": {p_combs[i]: norm.iloc[i].to_dict() for i in range(len(p_combs))}}

# --- 4. INFERENZ ---
st.divider()
st.header("4. Gezielte Abfrage: P(Ziel | Bedingung)")

col_ev, col_target = st.columns(2)
with col_ev:
    st.subheader("Bedingung (Evidenz)")
    evidence = {}
    for n in ["A", "B", "C", "D"]:
        cfg = st.session_state.nodes_config[n]
        sel = st.selectbox(f"Fixiere {cfg['name']}", ["Keine"] + cfg["states"], key=f"ev_inf_{n}")
        if sel != "Keine": evidence[n] = sel

with col_target:
    st.subheader("Ziel")
    avail_ids = [n for n in ["A", "B", "C", "D"] if n not in evidence]
    query_ids = st.multiselect("Zielknoten:", avail_ids, format_func=lambda x: st.session_state.nodes_config[x]["name"], key="query_select")
    target_vals = {}
    for qid in query_ids:
        target_vals[qid] = st.selectbox(f"Wert für {st.session_state.nodes_config[qid]['name']}", st.session_state.nodes_config[qid]["states"], key=f"tval_{qid}")

if st.button("🚀 Wahrscheinlichkeit berechnen"):
    num_samples = 5000
    valid_samples = []
    attempts = 0
    while len(valid_samples) < num_samples and attempts < num_samples * 50:
        attempts += 1
        sample, to_proc, valid = {}, ["A", "B", "C", "D"], True
        while to_proc:
            for n in to_proc:
                pars = [s for s, t in st.session_state.edges if t == n]
                if all(p in sample for p in pars):
                    stts = st.session_state.nodes_config[n]["states"]
                    probs = [cpt_storage_for_sim[n][s] for s in stts] if not pars else [cpt_storage_for_sim[n]["lookup"][tuple(sample[p] for p in pars)][s] for s in stts]
                    val = np.random.choice(stts, p=probs)
                    if n in evidence and val != evidence[n]:
                        valid = False; break
                    sample[n] = val; to_proc.remove(n); break
            if not valid: break
        if valid: valid_samples.append(sample)
        
    if valid_samples:
        df_res = pd.DataFrame(valid_samples)
        t_str = ", ".join([f"{st.session_state.nodes_config[k]['name']}={v}" for k, v in target_vals.items()])
        e_str = ", ".join([f"{st.session_state.nodes_config[k]['name']}={v}" for k, v in evidence.items()])
        
        if query_ids:
            mask = True
            for k, v in target_vals.items(): mask &= (df_res[k] == v)
            st.metric(f"P({t_str} | {e_str if e_str else '∅'})", f"{mask.mean():.2%}")
        
        st.subheader("Einzel-Marginale")
        res_cols = st.columns(4)
        for i, n in enumerate(["A", "B", "C", "D"]):
            with res_cols[i]:
                st.write(f"**{st.session_state.nodes_config[n]['name']}**")
                st.bar_chart(df_res[n].value_counts(normalize=True))
    else:
        st.error("Bedingung ist zu unwahrscheinlich für Sampling.")
