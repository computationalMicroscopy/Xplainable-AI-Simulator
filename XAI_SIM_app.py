import streamlit as st
import pandas as pd
import numpy as np
import itertools

st.set_page_config(page_title="Bayes-Schulungs-Simulator Pro", layout="wide")

# --- INITIALISIERUNG ---
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
        new_name = st.text_input(f"Name für {n}", st.session_state.nodes_config[n]["name"], key=f"name_{n}")
        st.session_state.nodes_config[n]["name"] = new_name
        
        states_str = st.text_input(f"Zustände für {new_name}", ",".join(st.session_state.nodes_config[n]["states"]), key=f"s_{n}")
        st.session_state.nodes_config[n]["states"] = [s.strip() for s in states_str.split(",")]

    st.divider()
    st.subheader("Struktur")
    # Nutzung der Anzeigenamen in der Auswahl
    node_opts = {n: st.session_state.nodes_config[n]["name"] for n in ["A", "B", "C", "D"]}
    src_id = st.selectbox("Ursache", options=list(node_opts.keys()), format_func=lambda x: node_opts[x])
    tgt_id = st.selectbox("Wirkung", options=[n for n in node_opts.keys() if n != src_id], format_func=lambda x: node_opts[x])
    
    if st.button("Verbindung hinzufügen ➕"):
        if (src_id, tgt_id) not in st.session_state.edges:
            st.session_state.edges.append((src_id, tgt_id))
            st.session_state.cpt_values = {} 
    if st.button("Struktur zurücksetzen 🗑️"):
        st.session_state.edges = []
        st.session_state.cpt_values = {}
        st.rerun()

# --- 2. TRAININGSDATEN (ONE-HOT) ---
st.header("2. Trainingsdaten (One-Hot)")
st.markdown("Nutze 0 für den 1. Zustand, 1 für den 2. Zustand usw. Spalten-IDs korrespondieren mit den Knoten.")
trained_df = st.data_editor(st.session_state.training_data, num_rows="dynamic", use_container_width=True)

# Graph zur Visualisierung
st.subheader("Aktuelle Netzwerk-Struktur")
dot = "digraph { rankdir=LR; node [style=filled, fillcolor='#E1F5FE', shape=box, fontname='Arial']; "
for nid, cfg in st.session_state.nodes_config.items():
    dot += f'{nid} [label="{cfg["name"]}\\n({"/".join(cfg["states"])})"]; '
for s, t in st.session_state.edges:
    dot += f"{s} -> {t}; "
dot += "}"
st.graphviz_chart(dot)



# --- 3. TRAININGSKNOPF & CPT BERECHNUNG ---
st.header("3. Bedingte Wahrscheinlichkeitstabellen (CPTs)")

if st.button("🎯 Wahrscheinlichkeiten aus Daten lernen"):
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
    st.success("Erfolgreich trainiert!")

cpt_storage_for_sim = {}
for n in ["A", "B", "C", "D"]:
    name = st.session_state.nodes_config[n]["name"]
    parents = [s for s, t in st.session_state.edges if t == n]
    states = st.session_state.nodes_config[n]["states"]
    st.write(f"### CPT für: {name} ({n})")
    
    p_states_list = [st.session_state.nodes_config[p]["states"] for p in parents]
    p_combs = list(itertools.product(*p_states_list))
    row_labels = [" | ".join(map(str, combo)) for combo in p_combs] if parents else ["Basis (%)"]
    
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

# --- 4. SPEZIFISCHE BEDINGTE INFERENZ ---
st.divider()
st.header("4. Inferenz: Gezielte Wahrscheinlichkeitsabfrage")

col_ev, col_target = st.columns(2)
with col_ev:
    st.subheader("Bedingung (Evidenz)")
    evidence = {}
    for n in ["A", "B", "C", "D"]:
        cfg = st.session_state.nodes_config[n]
        opt = ["Keine"] + cfg["states"]
        sel = st.selectbox(f"Fixiere {cfg['name']}", opt, key=f"ev_inf_{n}")
        if sel != "Keine": evidence[n] = sel

with col_target:
    st.subheader("Zielvariablen & Ausprägungen")
    target_selection = {}
    avail_ids = [n for n in ["A", "B", "C", "D"] if n not in evidence]
    query_ids = st.multiselect("Zielknoten wählen:", avail_ids, 
                                format_func=lambda x: st.session_state.nodes_config[x]["name"],
                                default=avail_ids[:1] if avail_ids else [])
    for qid in query_ids:
        cfg = st.session_state.nodes_config[qid]
        target_selection[qid] = st.selectbox(f"Wert für {cfg['name']}", cfg["states"], key=f"target_val_{qid}")

num_samples = st.slider("Samples", 1000, 20000, 5000)

if st.button("🚀 Wahrscheinlichkeit berechnen"):
    valid_samples = []
    attempts = 0
    with st.spinner("Sampling läuft..."):
        while len(valid_samples) < num_samples and attempts < num_samples * 100:
            attempts += 1
            sample, to_proc, valid = {}, ["A", "B", "C", "D"], True
            while to_proc:
                for n in to_proc:
                    pars = [s for s, t in st.session_state.edges if t == n]
                    if all(p in sample for p in pars):
                        stts = st.session_state.nodes_config[n]["states"]
                        prbs = [cpt_storage_for_sim[n][s] for s in stts] if not pars else [cpt_storage_for_sim[n]["lookup"][tuple(sample[p] for p in pars)][s] for s in stts]
                        val = np.random.choice(stts, p=prbs)
                        if n in evidence and val != evidence[n]:
                            valid = False; break
                        sample[n] = val; to_proc.remove(n); break
                if not valid: break
            if valid: valid_samples.append(sample)
        
    if valid_samples:
        df_res = pd.DataFrame(valid_samples)
        t_str = ", ".join([f"{st.session_state.nodes_config[k]['name']}={v}" for k, v in target_selection.items()])
        e_str = ", ".join([f"{st.session_state.nodes_config[k]['name']}={v}" for k, v in evidence.items()])
        full_expr = f"P({t_str} | {e_str if e_str else '∅'})"
        
        if target_selection:
            mask = True
            for k, v in target_selection.items(): mask &= (df_res[k] == v)
            prob_value = mask.mean()
            st.metric(label="Berechneter Ausdruck", value=full_expr)
            st.info(f"Ergebnis: **{prob_value:.2%}**")
        
        st.divider()
        st.subheader("Einzel-Verteilungen unter der Bedingung")
        res_cols = st.columns(4)
        for i, n in enumerate(["A", "B", "C", "D"]):
            with res_cols[i]:
                st.write(f"**{st.session_state.nodes_config[n]['name']}**")
                dist = df_res[n].value_counts(normalize=True).sort_index()
                st.bar_chart(dist)
    else:
        st.error("Bedingung zu unwahrscheinlich.")
