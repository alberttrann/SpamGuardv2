# dashboard/app.py 

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import time
import os
import joblib
import numpy as np
import sys
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import io

# --- Definitive PYTHONPATH Fix ---
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DASHBOARD_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from backend.utils import preprocess_tokenizer

# --- Absolute Paths ---
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'backend', 'data')

# --- Global Components for k-NN (Load Once) ---
@st.cache_resource
def load_transformer_model():
    print("UI: Loading sentence-transformer model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(device).eval()
    print("UI: Transformer model loaded.")
    return tokenizer, model, device
TOKENIZER, TRANSFORMER_MODEL, DEVICE = load_transformer_model()

# --- Helper functions ---
def average_pool(states, mask): return (states * mask[..., None]).sum(1) / mask.sum(-1)[..., None]
def get_embeddings(texts: list, prefix: str, batch_size: int = 32):
    all_embeds = [];
    for i in range(0, len(texts), batch_size):
        batch = [f"{prefix}: {text}" for text in texts[i:i + batch_size]]
        tokens = TOKENIZER(batch, max_length=512, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): outputs = TRANSFORMER_MODEL(**tokens)
        embeds = average_pool(outputs.last_hidden_state, tokens['attention_mask'])
        all_embeds.append(F.normalize(embeds, p=2, dim=1).cpu().numpy())
    return np.vstack(all_embeds)
def parse_labeled_data_from_stream(file_stream):
    """
    Robustly parses a file-like object (which could be bytes or string based)
    and correctly handles multi-line messages.
    """
    true_labels = []; messages = []; records_for_retraining = []; errors = []
    
    # Determine if the stream's content is bytes or string and handle accordingly.
    try:
        # Try getting raw bytes first, for uploaded files
        content_raw = file_stream.getvalue()
        if isinstance(content_raw, bytes):
            content = content_raw.decode("utf-8")
        else:
            content = content_raw
    except Exception as e:
        errors.append(f"Could not read the input stream: {e}")
        return [], [], [], errors

    lines = content.splitlines()
    
    current_message_lines = []
    current_label = None

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        is_new_record = False
        potential_label = None
        content_start_index = -1

        if line_stripped.lower().startswith('ham,'):
            is_new_record = True
            potential_label = 'ham'
            content_start_index = 4
        elif line_stripped.lower().startswith('spam,'):
            is_new_record = True
            potential_label = 'spam'
            content_start_index = 5
        
        if is_new_record:
            # Save the previous record before starting a new one
            if current_label and current_message_lines:
                full_message = "\n".join(current_message_lines).strip()
                if full_message:
                    true_labels.append(current_label)
                    messages.append(full_message)
                    records_for_retraining.append({"label": current_label, "message": full_message})
            
            # Start the new record
            current_label = potential_label
            current_message_lines = [line_stripped[content_start_index:].strip()]
        else:
            # It's a continuation line
            if current_label is not None:
                current_message_lines.append(line)

    # Save the very last record after the loop finishes
    if current_label and current_message_lines:
        full_message = "\n".join(current_message_lines).strip()
        if full_message:
            true_labels.append(current_label)
            messages.append(full_message)
            records_for_retraining.append({"label": current_label, "message": full_message})

    if not records_for_retraining and lines:
        errors.append("Could not find any valid records in the format 'ham,<message>' or 'spam,<message>'.")
        
    return true_labels, messages, records_for_retraining, errors

def bulk_classify(messages: list):
    try:
        response = requests.post(f"{API_BASE_URL}/bulk_classify", json={"messages": messages}, timeout=600) # Long timeout for big batches
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error during bulk classification: {e}")
        return None
        
# --- Configuration & Setup ---
API_BASE_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="wide")

# --- Session State Initialization ---
states_to_init = {
    'last_classified_message': None, 'generating': False, 'generation_type': None,
    'evaluation_results': None, 'backend_ready': False, 'generated_data_for_review': [],
    'keep_generated_flags': [], 'explanation': None
}
for key, default_value in states_to_init.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- API Functions ---
def check_backend_status():
    """Polls the backend. MUST NOT BE CACHED."""
    try: response = requests.get(f"{API_BASE_URL}/status", timeout=2); response.raise_for_status(); return response.json()
    except: return None
def get_config():
    """Gets the current config. MUST NOT BE CACHED."""
    try: response = requests.get(f"{API_BASE_URL}/config", timeout=2); response.raise_for_status(); return response.json()
    except: return None
def classify_message(message):
    try: response = requests.post(f"{API_BASE_URL}/classify", json={"text": message}); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}"); return None
def send_feedback(message, correct_label):
    try: response = requests.post(f"{API_BASE_URL}/feedback", json={"message": message, "correct_label": correct_label}); response.raise_for_status(); st.toast(f"✅ Feedback sent!")
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}")
def send_bulk_feedback(payload):
    try: response = requests.post(f"{API_BASE_URL}/bulk_feedback", json=payload); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}"); return None
def retrain_model():
    try: response = requests.post(f"{API_BASE_URL}/retrain"); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}"); return None
def get_model_explanation():
    try: response = requests.get(f"{API_BASE_URL}/explain_model"); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}"); return None
def get_models():
    try: response = requests.get(f"{API_BASE_URL}/models"); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}"); return None
def set_active_model(model_id):
    try: response = requests.post(f"{API_BASE_URL}/models/activate", json={"model_id": model_id}); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}"); return None
def set_config(mode: str, knn_dataset_file: str):
    try: response = requests.post(f"{API_BASE_URL}/config", json={"mode": mode, "knn_dataset_file": knn_dataset_file}); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}"); return None
def get_all_feedback():
    try: response = requests.get(f"{API_BASE_URL}/feedback/all"); response.raise_for_status(); return response.json()
    except: return []
def delete_feedback(ids):
    try: response = requests.post(f"{API_BASE_URL}/feedback/delete", json={"ids": ids}); response.raise_for_status(); return response.json()
    except: return None
def delete_all_feedback():
    try: response = requests.post(f"{API_BASE_URL}/feedback/delete_all"); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}"); return None
def start_llm_generation(payload):
    try: response = requests.post(f"{API_BASE_URL}/llm/start_generation", json=payload); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}"); return None
def stop_llm_generation():
    try: response = requests.post(f"{API_BASE_URL}/llm/stop_generation"); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}"); return None
@st.cache_data(ttl=3)
def get_llm_review_data():
    try: response = requests.get(f"{API_BASE_URL}/llm/review_data"); response.raise_for_status(); return response.json()
    except: return None
def clear_llm_review_data():
    try: response = requests.post(f"{API_BASE_URL}/llm/clear_review_data"); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}"); return None
def get_analytics():
    try: response = requests.get(f"{API_BASE_URL}/analytics"); response.raise_for_status(); return response.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError): return None
def log_evaluation(log_entry):
    try: response = requests.post(f"{API_BASE_URL}/evaluations/log", json=log_entry); response.raise_for_status(); return response.json()
    except: return None
def get_eval_logs():
    try: response = requests.get(f"{API_BASE_URL}/evaluations/logs"); response.raise_for_status(); return response.json()
    except: return []
def delete_eval_log(timestamp):
    try: response = requests.delete(f"{API_BASE_URL}/evaluations/log/{timestamp}"); response.raise_for_status(); return response.json()
    except: return None
def get_datasets():
    """Fetches the data registry from the backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/datasets")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Could not fetch datasets. {e}")
        return {}

# --- Main Application Logic ---
st.title("🛡️ SpamGuard AI: An Adaptive Spam Filtering System")

status_response = check_backend_status()
if not (status_response and status_response.get("is_ready")):
    st.info("⏳ SpamGuard AI engine is warming up..."); st.warning("This page will refresh automatically."); time.sleep(5); st.rerun()
else:
    # --- Persistent Status Indicators ---
    config_response = get_config()
    col1, col2, col3 = st.columns(3)
    if config_response:
        col1.metric("Mode", config_response.get('mode', 'N/A').upper()); col2.metric("Active k-NN Dataset", config_response.get('knn_dataset_file', 'N/A'))
    if status_response and status_response.get("is_loading_new_config"):
        col3.warning(f"⏳ {status_response.get('loading_status_message', 'Applying new config...')}"); time.sleep(5); st.rerun()
    else: col3.success("✅ System Ready")

    # --- UI Sections ---
    # SECTION 1: CLASSIFICATION
    st.header("1. Real-time Classification")
    message_input = st.text_area("Enter a message to analyze:", height=100, key="msg_input")
    if st.button("Classify Message", use_container_width=True):
        if message_input:
            with st.spinner("Analyzing..."): result = classify_message(message_input)
            if result:
                if "error" in result: st.error(f"Backend Error: {result['error']}")
                else: st.session_state.last_classified_message = {"message": message_input, **result}
        else: st.warning("Please enter a message.")
    if st.session_state.last_classified_message:
        res = st.session_state.last_classified_message; st.subheader("Analysis Result")
        pred = res['prediction']; conf = res['confidence']; model = res['model']
        color = "error" if pred == "spam" else "success"; icon = "🚨" if pred == "spam" else "✅"
        st.markdown(f"### <span style='color:{'red' if color=='error' else 'green'};'>{icon} Prediction: **{pred.upper()}**</span>", unsafe_allow_html=True)
        col1a, col2a = st.columns(2); col1a.metric("Confidence", f"{conf:.2%}"); col2a.metric("Model Used", model)
        if res.get('evidence'):
            with st.expander("💡 See Why (Explainable AI)"):
                st.write("Similar messages from the database:"); [st.info(f"**{item['label'].upper()}** (Similarity: {item['similarity_score']:.3f}):\n\n_{item['similar_message']}_") for item in res['evidence']]
        st.subheader("Was this correct?")
        fb_col1, fb_col2, fb_col3 = st.columns(3)
        if fb_col1.button("✔️ Yes, it's correct!", use_container_width=True): send_feedback(res['message'], res['prediction'])
        if fb_col2.button("❌ No, it's wrong!", use_container_width=True): wrong_label = "ham" if pred == "spam" else "spam"; send_feedback(res['message'], wrong_label)
        if fb_col3.button("🗑️ Dismiss (Don't Retrain)", use_container_width=True): st.toast("Result dismissed."); st.session_state.last_classified_message = None; st.rerun()

    st.divider()

    # --- SECTION 2: MODEL MANAGEMENT & REGISTRY (FINAL OVERHAUL) ---
    st.header("2. Model Management & Registry")
    st.write("View and manage trained model versions and classifier operational settings.")

    tab_model_select, tab_config = st.tabs(["⚙️ Model Version", "🛠️ Operational Config"])

    with tab_model_select:
        st.subheader("Model Versions")
        if st.button("Refresh Model List", key="refresh_model_list_button"):
            st.rerun() 
        
        registry_data = get_models()
        if registry_data and registry_data.get("models"):
            models = registry_data["models"]
            active_model_id = registry_data.get("active_model_id")
            
            # Prepare DataFrame for the data editor
            models_df = pd.DataFrame.from_dict(models, orient='index')
            models_df['id'] = models_df.index
            models_df['active'] = models_df['id'] == active_model_id
            if 'note' not in models_df.columns:
                models_df['note'] = ""
            
            # Sort by creation date, newest first
            models_df['creation_date'] = pd.to_datetime(models_df['creation_date'])
            models_df = models_df.sort_values(by="creation_date", ascending=False)

            edited_models_df = st.data_editor(
                models_df[['id', 'creation_date', 'note', 'active']],
                column_order=("active", "id", "note", "creation_date"),
                column_config={
                    "id": st.column_config.TextColumn("Model ID", disabled=True),
                    "creation_date": st.column_config.DatetimeColumn("Created On", disabled=True, format="YYYY-MM-DD HH:mm:ss"),
                    "note": st.column_config.TextColumn("Note", width="large", help="Add a description for this model version."),
                    "active": st.column_config.CheckboxColumn("Active", disabled=True)
                }, use_container_width=True, hide_index=True, key="model_editor"
            )
            if st.button("Save Model Notes", key="save_model_notes_button"):
                notes_to_save = edited_models_df.set_index('id')['note'].to_dict()
                with st.spinner("Saving notes..."):
                    update_model_notes(notes_to_save)
                st.toast("Model notes saved!")
                time.sleep(1)
                st.rerun()
            
            st.subheader("Activate a Different Model")
            inactive_models = edited_models_df[~edited_models_df['active']]['id'].tolist()
            if inactive_models:
                model_to_activate = st.selectbox("Select model to activate:", inactive_models, key="model_selector")
                if st.button("Activate Selected Model", type="primary"):
                    with st.spinner(f"Activating model '{model_to_activate}'..."):
                        response = set_active_model(model_to_activate)
                        if response and response['status'] == 'success':
                            st.success(response['message']); st.info("The new model is now loading in the background."); time.sleep(2); st.rerun()
                        else: st.error("Failed to activate model.")
            else:
                st.info("There are no other inactive models to activate.")
        elif registry_data is None:
            st.error("Could not connect to the backend API.")
        else:
            st.warning("No models found in the registry. Please retrain a model in Section 3.")

    with tab_config:
        st.subheader("Classifier Operational Mode")
        if config_response:
            mode_options = ["hybrid", "nb_only", "knn_only"]
            selected_mode = st.radio("Choose classification mode:", options=mode_options, index=mode_options.index(config_response["mode"]), horizontal=True)

            st.markdown("---")
            st.subheader("k-NN Indexing Dataset")
            st.info("Select which dataset the k-NN/Hybrid models build their knowledge from.")
            
            datasets_reg = get_datasets()
            if datasets_reg:
                datasets_df = pd.DataFrame.from_dict(datasets_reg, orient='index')
                datasets_df['filename'] = datasets_df.index
                datasets_df['active'] = datasets_df['filename'] == config_response.get("knn_dataset_file")
                if 'note' not in datasets_df.columns:
                    datasets_df['note'] = ""
                
                # Use data_editor to DISPLAY the table and EDIT notes
                edited_datasets_df = st.data_editor(
                    datasets_df[['filename', 'note', 'active']],
                    column_order=("active", "filename", "note"),
                    column_config={
                        "filename": st.column_config.TextColumn("Dataset File", disabled=True),
                        "note": st.column_config.TextColumn("Note", width="large", help="Add a description for this dataset."),
                        "active": st.column_config.CheckboxColumn("Active", disabled=True)
                    },
                    use_container_width=True, hide_index=True, key="dataset_editor"
                )
                if st.button("Save Dataset Notes"):
                    notes_to_save = edited_datasets_df.set_index('filename')['note'].to_dict()
                    with st.spinner("Saving notes..."):
                        update_dataset_notes(notes_to_save)
                    st.toast("Dataset notes saved!")
                    time.sleep(1); st.rerun()
                
                st.markdown("---")
                
                # Use a separate selectbox to CHANGE the active dataset
                st.write("**Change Active Dataset**")
                all_dataset_files = edited_datasets_df['filename'].tolist()
                current_active_dataset = config_response.get("knn_dataset_file")
                current_index = all_dataset_files.index(current_active_dataset) if current_active_dataset in all_dataset_files else 0
                
                newly_selected_dataset = st.selectbox(
                    "Select a dataset to make active for k-NN:",
                    all_dataset_files,
                    index=current_index,
                    key="knn_dataset_selector_config"
                )

                # The "Apply" button considers changes from BOTH widgets (radio and selectbox)
                if (selected_mode != config_response["mode"] or newly_selected_dataset != config_response["knn_dataset_file"]):
                    if st.button("Apply Configuration Changes", type="primary"):
                        with st.spinner("Applying new configuration..."):
                            response = set_config(selected_mode, newly_selected_dataset)
                            if response and response['status'] == 'success':
                                st.success(response['message'])
                                st.info("The new configuration is loading in the background. The UI will update when ready.")
                                time.sleep(2); st.rerun()
                            else:
                                st.error("Failed to apply configuration.")
                else:
                    st.info("Current configuration is applied.")
            else:
                st.warning("No datasets found in the data directory.")
        else:
            st.error("Could not fetch current configuration from backend.")

    st.divider()

# --- SECTION 3: MODEL PERFORMANCE & LEARNING (ENHANCED STAGING) ---
    st.header("3. Model Performance & Learning")
    analytics_col, training_col = st.columns([0.4, 0.6])

    with analytics_col:
        st.subheader("Dataset Analytics")
        if st.button("Refresh Analytics"): st.rerun()
        analytics_data = get_analytics()
        if analytics_data:
            if config_response: st.caption(f"Displaying stats for: `{config_response.get('knn_dataset_file')}`")
            total_base = analytics_data['base_ham_count'] + analytics_data['base_spam_count']
            total_new = analytics_data['new_ham_count'] + analytics_data['new_spam_count']
            st.metric("Total Messages in Core Dataset", f"{total_base:,}")
            st.metric("New Messages in Staging Area", f"{total_new:,}", help=f"User: {analytics_data['user_contribution']}, LLM: {analytics_data['llm_contribution']}")
            chart_data = pd.DataFrame({"Type": ["Ham", "Spam"], "Core Dataset": [analytics_data['base_ham_count'], analytics_data['base_spam_count']], "Staging Area": [analytics_data['new_ham_count'], analytics_data['new_spam_count']]}).set_index('Type'); st.bar_chart(chart_data)
        else: st.warning("Could not fetch dataset analytics.")
        st.markdown("---"); st.subheader("🧠 Model Interpretation (XAI)"); st.write("See the top keywords the active Naive Bayes model uses.")
        if st.button("Analyze Model Keywords"):
            with st.spinner("Analyzing model..."): explanation = get_model_explanation()
            if explanation and not explanation.get("error"): st.session_state.explanation = explanation
            elif explanation: st.error(explanation["error"])
        if st.session_state.explanation:
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1: st.success("Top HAM Keywords"); st.table(pd.DataFrame(st.session_state.explanation["top_ham_keywords"], columns=["Keyword"]))
            with exp_col2: st.error("Top SPAM Keywords"); st.table(pd.DataFrame(st.session_state.explanation["top_spam_keywords"], columns=["Keyword"]))

    with training_col:
        st.subheader("Continuous Learning & Data Staging")
        st.write("Review all pending feedback in the staging area. Commit kept data, then retrain.")
        pending_feedback = get_all_feedback()
        if pending_feedback:
            df = pd.DataFrame(pending_feedback)
            
            st.write(f"**{len(df)} messages are in the staging area.**")
            
            search_term = st.text_input("Search messages in staging area:", key="staging_search")
            if search_term:
                df = df[df['message'].str.contains(search_term, case=False, na=False)]
            
            df['keep'] = True
            edited_df = st.data_editor(df, column_order=("keep", "label", "source", "message"),
                column_config={ "id": None, "timestamp": None, "keep": st.column_config.CheckboxColumn("Keep?", default=True), "label": st.column_config.TextColumn("Label", disabled=True), "message": st.column_config.TextColumn("Message", disabled=True, width="large"), "source": st.column_config.TextColumn("Source", disabled=True)},
                use_container_width=True, hide_index=True, key="feedback_editor"
            )
            
            st.markdown("---")
            
            sel_col1, sel_col2, sel_col3, sel_col4 = st.columns(4)
            if sel_col1.button("Select All Visible", use_container_width=True):
                st.session_state.edited_df = edited_df
                st.session_state.edited_df['keep'] = True
                st.warning("Select/Unselect functionality requires a different implementation pattern. For now, manual checking is supported.")

            review_col1, review_col2, review_col3 = st.columns([1.2, 1, 1.5])
            with review_col1:
                if st.button("🗑️ Discard Unchecked Items", use_container_width=True):
                    records_to_discard_ids = edited_df[~edited_df["keep"]]["id"].tolist()
                    if records_to_discard_ids:
                        with st.spinner(f"Discarding {len(records_to_discard_ids)} records..."): delete_feedback(records_to_discard_ids)
                        st.success("Selected records discarded."); time.sleep(1); st.rerun()
                    else: st.warning("No records were selected (unchecked) for discarding.")
            with review_col2:
                if st.button("🔥 Discard ALL Items", use_container_width=True, type="secondary"):
                    with st.spinner("Discarding all staging data..."): response = delete_all_feedback()
                    if response: st.success(response['message']); time.sleep(1); st.rerun()
            with review_col3:
                st.write("Add all **kept** messages and retrain.")
                is_busy = status_response and status_response.get("is_loading_new_config", False)
                if st.button("Retrain with Kept Data", use_container_width=True, type="primary", disabled=is_busy):
                    records_to_discard_ids = edited_df[~edited_df["keep"]]["id"].tolist()
                    if records_to_discard_ids: delete_feedback(records_to_discard_ids)
                    with st.spinner("Sending retrain command..."):
                        retrain_result = retrain_model()
                        if retrain_result and retrain_result['status'] == 'success': st.success(retrain_result['message']); st.info("The new model is loading in the background."); st.balloons(); time.sleep(2); st.rerun()
                        elif retrain_result: st.error(retrain_result.get('message', 'Retraining failed.'))
        else:
            st.info("No new feedback data is in the staging area.")
        with st.expander("📤 Add more data to staging area via bulk upload"):
            enrich_tab1, enrich_tab2 = st.tabs(["Upload File", "Paste Text"])
            with enrich_tab1:
                uploaded_file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"], key="enrich_uploader")
                if uploaded_file and st.button("Add from File", use_container_width=True):
                    string_io = io.StringIO(uploaded_file.getvalue().decode("utf-8")); _, _, records, errors = parse_labeled_data_from_stream(string_io)
                    if errors: [st.warning(error) for error in errors]
                    if records:
                        with st.spinner("Sending bulk data..."): response = send_bulk_feedback(records);
                        if response and response['status'] == 'success': st.success(response['message']); time.sleep(1); st.rerun()
                        else: st.error("Failed to add data.")
            with enrich_tab2:
                bulk_text = st.text_area("Paste labeled messages", height=200, placeholder='"spam","..."\n"ham","..."', key="enrich_text_input")
                if st.button("Add from Paste", use_container_width=True):
                    if bulk_text:
                        string_io = io.StringIO(bulk_text); _, _, records, errors = parse_labeled_data_from_stream(string_io)
                        if errors: [st.warning(error) for error in errors]
                        if records:
                            with st.spinner("Sending bulk data..."): response = send_bulk_feedback(records)
                            if response and response['status'] == 'success': st.success(response['message']); time.sleep(1); st.rerun()
                            else: st.error("Failed to add data.")
                    else: st.warning("Text area is empty.")

    st.divider()

    # SECTION 4: LLM DATA DISTILLATION
    st.header("4. Augment Data with an LLM")
    st.write("Generate new training data and send it to the staging area for review in Section 3.")
    review_data = get_llm_review_data()
    is_generating = review_data.get("is_generating", False) if review_data else False
    status_message = review_data.get("status_message", "Idle") if review_data else "Connecting..."
    generated_items = review_data.get("data", []) if review_data else []
    with st.container():
        use_llm = st.toggle("Enable LLM Data Generation Controls", value=(is_generating or bool(generated_items)))
        if use_llm:
            PROVIDER_MAP = {"Ollama (Local)": "ollama", "LM Studio (Local)": "lmstudio", "OpenRouter (Cloud)": "openrouter"}
            provider_display_name = st.radio("Choose LLM Provider", PROVIDER_MAP.keys(), horizontal=True, key="provider", disabled=is_generating)
            provider_api_name = PROVIDER_MAP[provider_display_name]
            if provider_api_name == "ollama": model_id = st.text_input("Ollama Model ID", "llama3", key="ollama_model", disabled=is_generating); api_key = ""
            elif provider_api_name == "lmstudio": model_id = st.text_input("Model Identifier", "local-model", key="lmstudio_model", disabled=is_generating); api_key = ""
            else: model_id = st.text_input("OpenRouter Model ID", "mistralai/mistral-7b-instruct:free", key="or_model", disabled=is_generating); api_key = st.text_input("OpenRouter API Key", type="password", key="or_api_key", disabled=is_generating)
            st.subheader("Generation Controls")
            if is_generating:
                st.info(f"💡 **Status:** {status_message}")
                if st.button("⏹️ Stop Generating", use_container_width=True, type="primary"):
                    with st.spinner("Sending stop signal..."): stop_llm_generation(); st.toast("Stop signal sent."); time.sleep(1); st.rerun()
            else:
                gen_col1, gen_col2, gen_col3 = st.columns(3)
                payload = {"provider": provider_api_name, "model": model_id, "api_key": api_key, "label_to_generate": None}
                with gen_col1:
                    if st.button("🤖 Start Continuous (Random)", use_container_width=True): start_llm_generation(payload); time.sleep(1); st.rerun()
                with gen_col2:
                    payload["label_to_generate"] = "spam"
                    if st.button("🚨 Start Continuous (SPAM)", use_container_width=True): start_llm_generation(payload); time.sleep(1); st.rerun()
                with gen_col3:
                    payload["label_to_generate"] = "ham"
                    if st.button("✅ Start Continuous (HAM)", use_container_width=True): start_llm_generation(payload); time.sleep(1); st.rerun()
            if generated_items:
                st.subheader("Review Generated Data")
                st.write(f"LLM generated {len(generated_items)} new messages this session.")
                st.dataframe(pd.DataFrame(generated_items), use_container_width=True, hide_index=True)
                review_col1, review_col2 = st.columns(2)
                with review_col1:
                    if st.button("➕ Send All to Staging Area", use_container_width=True):
                        with st.spinner("Adding data to staging area..."):
                            response = send_bulk_feedback(generated_items)
                            if response and response['status'] == 'success': st.success(response['message']); clear_llm_review_data(); st.info("Data is now available for review in Section 3."); time.sleep(2); st.rerun()
                            else: st.error("Failed to add data.")
                with review_col2:
                    if st.button("🗑️ Discard Session Data", use_container_width=True, type="primary"):
                        with st.spinner("Discarding..."): clear_llm_review_data(); st.success("Generated data from this session has been discarded."); time.sleep(1); st.rerun()
            if is_generating:
                time.sleep(3); st.rerun()
    
    st.divider()

# --- SECTION 5: BATCH EVALUATION & TESTING  ---
    st.header("5. Batch Evaluation & Testing")
    st.write("Evaluate the performance of the currently active model and operational configuration.")
    eval_tab1, eval_tab2 = st.tabs(["📤 Upload File", "📝 Paste Text"])
    with eval_tab1:
        uploaded_file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"], key="eval_uploader")
        if uploaded_file: st.session_state.eval_input_stream = uploaded_file
    with eval_tab2:
        pasted_text = st.text_area("Paste labeled messages", height=250, placeholder='"spam","..."\n"ham","..."', key="eval_text_area")
        if pasted_text: st.session_state.eval_input_stream = io.StringIO(pasted_text)

    if st.session_state.get('eval_input_stream'):
        if st.button("Run Batch Evaluation", use_container_width=True, type="primary", key="run_batch_eval_button"):
            st.session_state.evaluation_results = None
            eval_stream = st.session_state.eval_input_stream
            if not isinstance(eval_stream, io.StringIO):
                eval_stream = io.StringIO(eval_stream.getvalue().decode("utf-8"))
            eval_stream.seek(0)
            true_labels, messages_to_eval, records, errors = parse_labeled_data_from_stream(eval_stream)
            
            if errors: [st.warning(error) for error in errors]
            if not messages_to_eval: st.warning("No valid messages found to evaluate.")
            
            if messages_to_eval:
                with st.spinner(f"Classifying {len(messages_to_eval)} messages using the live backend..."):
                    list_of_results = bulk_classify(messages_to_eval)
                if list_of_results:
                    if any("error" in r for r in list_of_results):
                        st.error(f"Backend returned an error: {list_of_results[0].get('error', 'Unknown error')}")
                    else:
                        st.session_state.evaluation_results = {
                            "config_at_eval_time": get_config(),
                            "active_id": get_models().get("active_model_id"),
                            "true_labels": true_labels,
                            "final_results_detailed": list_of_results,
                            "messages": messages_to_eval,
                            "records_for_retraining": records,
                        }
                else: st.error("Failed to get results from the backend.")
            st.session_state.eval_input_stream = None
            st.rerun()

    if st.session_state.evaluation_results:
        eval_data = st.session_state.evaluation_results
        config = eval_data["config_at_eval_time"]
        mode = config.get('mode', 'N/A').upper()
        title = f"Evaluation Results for {mode} Mode"
        if mode == "NB_ONLY": title = f"Evaluation Results for Naive Bayes Model: `{eval_data['active_id']}`"
        elif mode == "KNN_ONLY": title = f"Evaluation Results for k-NN Only Mode (Index: `{config.get('knn_dataset_file')}`)"
        else: title = f"Evaluation Results for Hybrid System (NB: `{eval_data['active_id']}`)"
        st.subheader(title)
        
        true_labels = eval_data["true_labels"]; final_results_detailed = eval_data["final_results_detailed"]; pred_labels = [p['prediction'] for p in final_results_detailed]
        with st.container(border=True):
            st.subheader("Performance Summary"); summary_col1, summary_col2 = st.columns(2)
            with summary_col1:
                accuracy = accuracy_score(true_labels, pred_labels); st.metric("Overall Accuracy", f"{accuracy:.2%}")
                st.text("Classification Report:"); report_df = pd.DataFrame(classification_report(true_labels, pred_labels, labels=["ham", "spam"], output_dict=True, zero_division=0)).transpose(); st.dataframe(report_df)
            with summary_col2:
                cm = confusion_matrix(true_labels, pred_labels, labels=["ham", "spam"]); fig, ax = plt.subplots(figsize=(4, 3)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"]); ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label"); st.pyplot(fig)
            st.markdown("---"); st.subheader("Performance Metrics")
            total_time_ms = sum(r.get('time_ms', 0) for r in final_results_detailed)
            avg_time_ms = total_time_ms / len(true_labels) if true_labels else 0
            perf_col1, perf_col2 = st.columns(2)
            perf_col1.metric("Total Prediction Time", f"{total_time_ms / 1000:.4f} s"); perf_col2.metric("Average Time / Message", f"{avg_time_ms:.2f} ms")
            if config.get('mode') == 'hybrid':
                nb_count = sum(1 for r in final_results_detailed if r['model'] == 'MultinomialNB'); knn_count = len(true_labels) - nb_count
                nb_correct = sum(1 for i, r in enumerate(final_results_detailed) if r['model'] == 'MultinomialNB' and r['prediction'] == true_labels[i])
                knn_correct = sum(1 for i, r in enumerate(final_results_detailed) if r['model'] != 'MultinomialNB' and r['prediction'] == true_labels[i])
                nb_accuracy = (nb_correct / nb_count * 100) if nb_count > 0 else 0; knn_accuracy = (knn_correct / knn_count * 100) if knn_count > 0 else 0
                st.markdown("---"); st.write("**Hybrid System Triage Breakdown**")
                usage_col1, usage_col2 = st.columns(2)
                usage_col1.metric(label="NB Triage Usage", value=f"{nb_count / len(true_labels):.1%}", help=f"Handled {nb_count} messages. Correct: {nb_correct} ({nb_accuracy:.1f}%)")
                usage_col2.metric(label="k-NN Escalation Usage", value=f"{knn_count / len(true_labels):.1%}", help=f"Handled {knn_count} messages. Correct: {knn_correct} ({knn_accuracy:.1f}%)")
            with st.expander(f"⬇️ Click to see detailed breakdown for all {len(true_labels)} messages"):
                df_breakdown = pd.DataFrame({"True Label": true_labels, "Predicted Label": [r.get('prediction', 'N/A') for r in final_results_detailed], "Correct?": ["✅" if t == r.get('prediction') else "❌" for t, r in zip(true_labels, final_results_detailed)], "Model Used": [r.get('model', 'N/A') for r in final_results_detailed], "Confidence": [f"{r.get('confidence', 0):.2%}" for r in final_results_detailed], "Time (ms)": [f"{r.get('time_ms', 0):.2f}" for r in final_results_detailed], "Message": eval_data["messages"]})
                st.dataframe(df_breakdown, use_container_width=True)
            st.subheader("Next Steps")
            action_col1, action_col2, action_col3 = st.columns(3)
            with action_col1:
                if st.button("➕ Add All to Training Data", use_container_width=True):
                    with st.spinner("Adding data..."):
                        response = send_bulk_feedback(eval_data["records_for_retraining"])
                        if response and response['status'] == 'success': st.success(response['message']); st.info("Remember to click 'Retrain Model'!"); st.session_state.evaluation_results = None; st.rerun()
            with action_col2:
                log_note = st.text_input("Add a note before logging this result:")
                if st.button("💾 Save Result to Log", use_container_width=True):
                    log_entry = { "model_id": eval_data["active_id"], "mode": config.get('mode'), "knn_dataset": config.get('knn_dataset_file'), "test_set_name": "Custom Upload", "accuracy": accuracy_score(true_labels, pred_labels), "report": classification_report(true_labels, pred_labels, labels=["ham", "spam"], output_dict=True, zero_division=0), "confusion_matrix": confusion_matrix(true_labels, pred_labels, labels=["ham", "spam"]).tolist(), "note": log_note }
                    with st.spinner("Saving to log..."): response = log_evaluation(log_entry)
                    if response and response['status'] == 'success': st.success(response['message']); st.session_state.evaluation_results = None; st.rerun()
                    else: st.error("Failed to save log.")
            with action_col3:
                if st.button("🗑️ Dismiss Results", use_container_width=True, type="secondary"):
                    st.session_state.evaluation_results = None; st.rerun()
    st.markdown("---"); st.subheader("📊 Evaluation History")
    eval_logs = get_eval_logs()
    if not eval_logs: st.info("No evaluation results have been logged yet.")
    else:
        for i, log in enumerate(eval_logs):
            expander_title = (f"**{datetime.fromisoformat(log['timestamp']).strftime('%Y-%m-%d %H:%M')}** | **Acc: {log['accuracy']:.2%}** | **Model:** `{log['model_id']}` | **Note:** {log.get('note', 'N/A')}")
            with st.expander(expander_title):
                st.write(f"**Mode:** `{log.get('mode', 'N/A').upper()}` | **k-NN Data:** `{log.get('knn_dataset', 'N/A')}`")
                log_col1, log_col2 = st.columns(2)
                with log_col1:
                    st.text("Classification Report:"); st.dataframe(pd.DataFrame(log.get('report', {})).transpose())
                with log_col2:
                    st.text("Confusion Matrix:"); cm = np.array(log.get('confusion_matrix', [[0,0],[0,0]])); fig, ax = plt.subplots(figsize=(4, 3)); sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"]); st.pyplot(fig)
                if st.button("🗑️ Delete this Log Entry", key=f"delete_log_{log['timestamp']}", type="secondary"):
                    with st.spinner("Deleting log..."): delete_eval_log(log['timestamp']); st.rerun()