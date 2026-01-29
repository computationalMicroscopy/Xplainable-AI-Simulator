import streamlit as st
import pandas as pd
import numpy as np
import itertools

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="Bayes Expert Pro | Enterprise Edition", layout="wide", initial_sidebar_state="expanded")

# --- HIGH-CONTRAST PROFESSIONAL DESIGN (DARK MODE OPTIMIZED) ---
st.markdown("""
    <style>
    /* Haupt-Hintergrund und Textfarben */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Karten-Design für Sektionen */
    div.stBlock {
        background-color: #1a1c24;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Metriken (Ergebnisse) hervorheben */
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons: High Contrast Blue */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3.5em;
        background-color: #2563eb;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3b82f6;
        border: 1px solid #ffffff;
    }

    /* Sidebar-Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Überschriften */
    h1, h2, h3 {
        color: #3b82f6;
        font-family: 'Segoe UI', Roboto, sans-serif;
        letter-spacing: -0.5px;
    }

    /* Trennlinien */
    hr {
        border: 0;
        border-top: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALISIERUNG (ROBUST) ---
if 'nodes_config' in st.session_state:
    if not isinstance(st.session_state.nodes_config["A"], dict):
        del st.session_state.nodes_config

if 'nodes_config' not in st.session_state:
    st.session_state.nodes_config = {
        "A": {"name": "Wetter", "states": ["Sonne", "Regen", "Bewölkt"]},
        "B": {"name": "Sprinkler", "states": ["Aus", "An"]},
        "C": {"name": "Rasen", "states": ["Trocken", "Nass"]},
        "D": {"name": "Stimmung", "states": ["Gut", "Neutral", "Schlecht"]}
    }
if 'edges' not in st.session_state:
    st.session_state.edges = []

def get_one_hot_columns():
    cols = []
    for nid in ["A", "B", "C", "D"]:
        cfg = st.session_state.nodes_config[nid]
        for state in cfg["states"]:
            cols.append(f"{nid}_{state}")
    return cols

if 'training_data' not in st.session_state:
    st.session_state.training_data = pd.DataFrame(columns=get_one_hot_columns())

if 'cpt_values' not in st.session_state:
    st.session_state.cpt_values = {}

# --- HEADER ---
st.title("🛡️ Bayes Expert Pro | Enterprise Edition")
st.markdown("<p style='color: #8b949e;'>Professionelle Kausal-Analyse & Soft-Data Training</p>", unsafe_allow_html=True)

# --- 1. STRUKTUR-EDITOR (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Konfiguration")
    with st.expander("📝 Knoten & Zustände", expanded=True):
        for n in ["A", "B", "C", "D"]:
            st.markdown(f"**Knoten {n}**")
            st.session_state.nodes_config[n]["name"] = st.text_input(
                f"Bezeichnung", st.session_state.nodes_config[n]["name"], key=f"name_input_{n}")
            
            states_str = st.text_input(
                f"Zustände", ",".join(st.session_state.nodes_config[n]["states"]), key=f"states_input_{n}")
            new_states = [s.strip() for s in states_str.split(",") if s.strip()]
            if new_states != st.session_state.nodes_config[n]["states"]:
                st.session_state.nodes_config[n]["states"] = new_states
                st.session_state.training_data = pd.DataFrame(columns=get_one_hot_columns())
                st.session_state.cpt_values = {}
            st.divider()

    with st.expander("🔗 Kausalpfade", expanded=True):
        node_opts = {n: st.session_state.nodes_config[n]["name"] for n in ["A", "B", "C", "D"]}
        src_id = st.selectbox("Ursache", options=list(node_opts.keys()), format_func=lambda x: node_opts[x])
        tgt_id = st.selectbox("Wirkung", options=[n for n in node_opts.keys() if n != src_id], format_func=lambda x: node_opts[x])
        
        if st.button("Pfad hinzufügen +"):
            if (src_id, tgt_id) not in st.session_state.edges:
                st.session_state.edges.append((src_id, tgt_id))
                st.session_state.cpt_values = {} 
                st.rerun()
        if st.button("Reset Structure", type="secondary"):
            st.session_state.edges = []
            st.session_state.cpt_values = {}
            st.rerun()

# --- HAUPTBEREICH ---
col_main_left, col_main_right = st.columns([1, 1])

with col_main_left:
    st.subheader("📥 2. Dateneingabe (Soft-Weights)")
    current_cols = get_one_hot_columns()
    if list(st.session_state.training_data.columns) != current_cols:
         st.session_state.training_data = pd.DataFrame(columns=current_cols)
    
    trained_df = st.data_editor(st.session_state.training_data, num_rows="dynamic", use_container_width=True)
    st.session_state.training_data = trained_df

with col_main_right:
    st.subheader("🕸️ Kausalgraph")
    # Der Graph-Code wurde hier explizit eingefügt
    dot = "digraph { rankdir=LR; bgcolor='transparent'; node [style=filled, fillcolor='#1f2937', color='#3b82f6', fontcolor='#ffffff', shape=box, fontname='Arial', fontsize=12]; edge [color='#8b949e', penwidth=2]; "
    for nid, cfg in st.session_state.nodes_config.items():
        dot += f'{nid} [label="{cfg["name"]}\\n({"/".join(cfg["states"])})"]; '
    for s, t in st.session_state.edges:
        dot += f"{s} -> {t}; "
    dot += "}"
    # Zeichnet das Diagramm
    st.graphviz_chart(dot)



# --- 3. TRAINING & CPTs ---
st.divider()
st.subheader("🧠 3. Wahrscheinlichkeitsschätzung (CPTs)")

if st.button("MODELL TRAINING STARTEN"):
    new_cpts = {}
    learning_df = trained_df.fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
    for n in ["A", "B", "C", "D"]:
        parents = [s for s, t in st.session_state.edges if t == n]
        states = st.session_state.nodes_config[n]["states"]
        n_cols = [f"{n}_{s}" for s in states]
        
        if not parents:
            if len(learning_df) > 0:
                weights = learning_df[n_cols].sum()
                vals = (weights / weights.sum() * 100).values if weights.sum() > 0 else np.full(len(states), 100.0/len(states))
            else: vals = np.full(len(states), 100.0/len(states))
            new_cpts[n] = pd.DataFrame([vals], columns=states, index=["Basis (%)"])
        else:
            p_states_list = [st.session_state.nodes_config[p]["states"] for p in parents]
            combinations = list(itertools.product(*p_states_list))
            row_labels = [" | ".join(map(str, combo)) for combo in combinations]
            df_cpt = pd.DataFrame(np.full((len(row_labels), len(states)), 100.0/len(states)), index=row_labels, columns=states)
            if len(learning_df) > 0:
                for i, combo in enumerate(combinations):
                    parent_weight = pd.Series(1.0, index=learning_df.index)
                    for p_idx, p_id in enumerate(parents):
                        parent_weight *= learning_df[f"{p_id}_{combo[p_idx]}"]
                    if parent_weight.sum() > 0:
                        for s_name in states:
                            weighted_sum = (learning_df[f"{n}_{s_name}"] * parent_weight).sum()
                            df_cpt.loc[row_labels[i], s_name] = (weighted_sum / parent_weight.sum()) * 100
            new_cpts[n] = df_cpt
    st.session_state.cpt_values = new_cpts
    st.rerun()

cpt_storage_for_sim = {}
cpt_cols = st.columns(2)
for idx, n in enumerate(["A", "B", "C", "D"]):
    with cpt_cols[idx % 2]:
        cfg = st.session_state.nodes_config[n]
        parents = [s for s, t in st.session_state.edges if t == n]
        p_states_list = [st.session_state.nodes_config[p]["states"] for p in parents]
        p_combs = list(itertools.product(*p_states_list))
        row_labels = [" | ".join(map(str, combo)) for combo in p_combs] if parents else ["Basis (%)"]
        
        if n not in st.session_state.cpt_values or len(st.session_state.cpt_values[n]) != len(row_labels):
            init_data = np.full((len(row_labels), len(cfg["states"])), 100.0 / len(cfg["states"]))
            st.session_state.cpt_values[n] = pd.DataFrame(init_data, index=row_labels, columns=cfg["states"])

        with st.expander(f"⚙️ Tabelle: {cfg['name']}", expanded=False):
            edited_cpt = st.data_editor(st.session_state.cpt_values[n], key=f"editor_{n}", use_container_width=True)
            norm = edited_cpt.div(edited_cpt.sum(axis=1), axis=0).fillna(1.0 / len(cfg["states"]))
            if not parents: cpt_storage_for_sim[n] = norm.iloc[0].to_dict()
            else: cpt_storage_for_sim[n] = {"parents": parents, "lookup": {p_combs[i]: norm.iloc[i].to_dict() for i in range(len(p_combs))}}

# --- 4. INFERENZ ---
st.divider()
st.subheader("🔍 4. Inferenz & Vorhersage")

col_ev, col_target = st.columns(2)
with col_ev:
    st.markdown("<h4 style='color: #8b949e;'>Bekannte Evidenz</h4>", unsafe_allow_html=True)
    evidence = {}
    for n in ["A", "B", "C", "D"]:
        cfg = st.session_state.nodes_config[n]
        sel = st.selectbox(f"{cfg['name']}", ["Keine"] + cfg["states"], key=f"ev_inf_{n}")
        if sel != "Keine": evidence[n] = sel

with col_target:
    st.markdown("<h4 style='color: #8b949e;'>Zielvariablen</h4>", unsafe_allow_html=True)
    avail_ids = [n for n in ["A", "B", "C", "D"] if n not in evidence]
    query_ids = st.multiselect("Variablen wählen", avail_ids, format_func=lambda x: st.session_state.nodes_config[x]["name"], key="query_select")
    target_vals = {}
    for qid in query_ids:
        target_vals[qid] = st.selectbox(f"Wert für {st.session_state.nodes_config[qid]['name']}", st.session_state.nodes_config[qid]["states"], key=f"tval_{qid}")

num_samples = st.select_slider("Stichproben-Präzision", options=[100, 1000, 5000, 10000, 50000], value=5000)

if st.button("INFERENZ STARTEN ⚡"):
    valid_samples = []
    attempts = 0
    with st.spinner("Monte Carlo Simulation läuft..."):
        while len(valid_samples) < num_samples and attempts < num_samples * 100:
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
            st.metric(label=f"Wahrscheinlichkeit: P({t_str} | {e_str if e_str else '∅'})", value=f"{mask.mean():.2%}")
        
        st.divider()
        st.markdown("### 📊 Verteilungen unter der Bedingung")
        res_cols = st.columns(4)
        for i, n in enumerate(["A", "B", "C", "D"]):
            with res_cols[i]:
                st.write(f"**{st.session_state.nodes_config[n]['name']}**")
                dist = df_res[n].value_counts(normalize=True).sort_index()
                st.bar_chart(dist, color="#3b82f6")
    else:
        st.error("Konfiguration ungültig: Die Bedingung ist in diesem Modell unmöglich.")
