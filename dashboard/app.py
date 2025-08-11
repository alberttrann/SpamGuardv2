# dashboard/app.py (Final, Definitive, Working Version)

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
    true_labels = []; messages = []; records_for_retraining = []; errors = []
    try:
        reader = csv.reader(file_stream)
        for i, row in enumerate(reader):
            if not row: continue
            if len(row) != 2: errors.append(f"Line {i+1}: Invalid format."); continue
            label, message = row; label = label.strip().lower(); message = message.strip()
            if label not in ['ham', 'spam']: errors.append(f"Line {i+1}: Invalid label '{label}'."); continue
            if not message: errors.append(f"Line {i+1}: Message is empty."); continue
            true_labels.append(label); messages.append(message); records_for_retraining.append({"label": label, "message": message})
    except Exception as e: errors.append(f"A critical error occurred: {e}")
    return true_labels, messages, records_for_retraining, errors

# --- Configuration & Setup ---
API_BASE_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="wide")

# --- Session State Initialization ---
states_to_init = {
    'last_classified_message': None,
    'generating': False,
    'generation_type': None,
    'evaluation_results': None,
    'backend_ready': False,
    'generated_data_for_review': [],
    'keep_generated_flags': [],
    'explanation': None
}
for key, default_value in states_to_init.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- API Functions ---
@st.cache_data(ttl=3)
def check_backend_status():
    try: response = requests.get(f"{API_BASE_URL}/status", timeout=2); response.raise_for_status(); return response.json()
    except: return None
@st.cache_data(ttl=3)
def get_config():
    try: response = requests.get(f"{API_BASE_URL}/config", timeout=2); response.raise_for_status(); return response.json()
    except: return None
def classify_message(message):
    try: response = requests.post(f"{API_BASE_URL}/classify", json={"text": message}); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}"); return None
def bulk_classify(messages: list):
    try:
        response = requests.post(f"{API_BASE_URL}/bulk_classify", json={"messages": messages})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error during bulk classification: {e}")
        return None
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
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e: print(f"Could not fetch analytics: {e}"); return None
def delete_all_feedback():
    """Calls the backend to delete all records from the feedback database."""
    try:
        response = requests.post(f"{API_BASE_URL}/feedback/delete_all")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Could not discard all feedback. {e}")
        return None

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

    # SECTION 2: MODEL MANAGEMENT & REGISTRY
    st.header("2. Model Management & Registry")
    st.write("View and manage trained model versions and classifier operational settings.")
    tab_model_select, tab_config = st.tabs(["⚙️ Model Version", "🛠️ Operational Config"])
    with tab_model_select:
        if st.button("Refresh Model List", key="refresh_model_list_button"): st.rerun() 
        with st.spinner("Fetching model registry..."): registry_data = get_models()
        if registry_data and registry_data.get("models"):
            active_model_id = registry_data.get("active_model_id")
            sorted_models = sorted(registry_data["models"].items(), key=lambda i: i[1]['creation_date'], reverse=True)
            models_list = [{"ID": model_id, "Created On": datetime.fromisoformat(details["creation_date"]).strftime("%Y-%m-%d %H:%M:%S"), "Status": "✅ Active" if model_id == active_model_id else ""} for model_id, details in sorted_models]
            models_df = pd.DataFrame(models_list).set_index("ID"); st.dataframe(models_df, use_container_width=True)
            st.subheader("Activate a Different Model")
            inactive_models = [m["ID"] for m in models_list if m["Status"] == ""]
            if inactive_models:
                model_to_activate = st.selectbox("Select a model version to make active:", inactive_models, key="model_selector")
                if st.button("Activate Selected Model", type="primary"):
                    with st.spinner(f"Activating model '{model_to_activate}'..."):
                        response = set_active_model(model_to_activate)
                        if response and response['status'] == 'success': st.success(response['message']); st.info("The new model is now loading in the background."); time.sleep(2); st.rerun()
                        else: st.error("Failed to activate model.")
            else: st.info("There are no other inactive models to activate.")
        elif registry_data is None: st.error("Could not connect to the backend API.")
        else: st.warning("No models found. Please retrain a model in Section 3.")
    with tab_config:
        st.subheader("Classifier Operational Mode")
        if config_response:
            mode_options = ["hybrid", "nb_only", "knn_only"]
            selected_mode = st.radio("Choose classification mode:", options=mode_options, index=mode_options.index(config_response["mode"]), horizontal=True)
            if selected_mode in ["hybrid", "knn_only"]:
                st.markdown("---"); st.subheader("k-NN Indexing Dataset")
                st.info("Select the dataset for the k-NN Vector Search to build its knowledge base from.")
                available_datasets = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
                try: default_dataset_index = available_datasets.index(config_response["knn_dataset_file"])
                except ValueError: default_dataset_index = 0
                selected_knn_dataset = st.selectbox("Select dataset for k-NN index:", options=available_datasets, index=default_dataset_index)
            else: selected_knn_dataset = config_response["knn_dataset_file"]
            if (selected_mode != config_response["mode"] or selected_knn_dataset != config_response["knn_dataset_file"]):
                if st.button("Apply Configuration Changes", type="primary"):
                    with st.spinner("Applying new configuration..."):
                        response = set_config(selected_mode, selected_knn_dataset)
                        if response and response['status'] == 'success': st.success(response['message']); st.info("The new configuration is loading in the background."); time.sleep(2); st.rerun()
                        else: st.error("Failed to apply configuration.")
            else: st.info("Current configuration is applied.")
        else: st.error("Could not fetch current configuration from backend.")

    st.divider()

# --- SECTION 3: MODEL PERFORMANCE & LEARNING ---
    st.header("3. Model Performance & Learning")
    analytics_col, training_col = st.columns([0.4, 0.6]) # Give more space for the training/review column

    with analytics_col:
        st.subheader("Dataset Analytics")
        st.write("View statistics for the currently active k-NN dataset.")
        if st.button("Refresh Analytics"): st.rerun()
        analytics_data = get_analytics()
        if analytics_data:
            if config_response: st.caption(f"Displaying stats for: `{config_response.get('knn_dataset_file')}`")
            total_base = analytics_data['base_ham_count'] + analytics_data['base_spam_count']
            total_new = analytics_data['new_ham_count'] + analytics_data['new_spam_count']
            st.metric("Total Messages in Core Dataset", f"{total_base:,}")
            st.metric("New Messages in Staging Area", f"{total_new:,}", help=f"User: {analytics_data['user_contribution']}, LLM: {analytics_data['llm_contribution']}")
            chart_data = pd.DataFrame({"Type": ["Ham", "Spam"], "Core Dataset": [analytics_data['base_ham_count'], analytics_data['base_spam_count']], "Staging Area": [analytics_data['new_ham_count'], analytics_data['new_spam_count']]}).set_index('Type')
            st.bar_chart(chart_data)
        else: st.warning("Could not fetch dataset analytics.")
        st.markdown("---")
        st.subheader("🧠 Model Interpretation (XAI)")
        st.write("See the top keywords the active Naive Bayes model uses.")
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
        st.write("Review all pending feedback in the staging area. Commit the data you want to keep, then retrain.")
        pending_feedback = get_all_feedback()
        if pending_feedback:
            df = pd.DataFrame(pending_feedback)
            df['keep'] = True
            st.write(f"**{len(df)} messages are in the staging area.**")
            
            # --- THIS IS THE FIX: Explicitly list the columns to display ---
            # This ensures the 'id' column is available to the logic but not shown to the user.
            edited_df = st.data_editor(
                df, 
                column_order=("keep", "label", "message", "source"),
                column_config={
                    "id": None, # Hide the ID column
                    "timestamp": None, # Hide the timestamp column
                    "keep": st.column_config.CheckboxColumn("Keep?", default=True),
                    "label": st.column_config.TextColumn("Label", disabled=True),
                    "message": st.column_config.TextColumn("Message", disabled=True, width="large"),
                    "source": st.column_config.TextColumn("Source", disabled=True)
                }, 
                use_container_width=True, hide_index=True, key="feedback_editor"
            )
            
            st.markdown("---")
            # --- THIS IS THE FIX: The new button layout ---
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
                    with st.spinner("Discarding all staging data..."):
                        response = delete_all_feedback() # New API function
                    if response: st.success(response['message'])
                    time.sleep(1); st.rerun()

            with review_col3:
                st.write("Add all **kept** messages to the dataset and retrain.")
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

# --- SECTION 5: BATCH EVALUATION & TESTING (FINAL, SIMPLIFIED, AND CORRECT) ---
st.header("5. Batch Evaluation & Testing")
st.write(
    "Evaluate the performance of the **currently active model and operational configuration** on a batch of labeled messages. "
    "This uses the live backend classifier for a true end-to-end test."
)
# NOTE: The k-NN dataset selection has been correctly moved to Section 2 (Model Management).
# This section will now automatically use the globally configured settings.

eval_tab1, eval_tab2 = st.tabs(["📤 Upload File", "📝 Paste Text"])
user_input_text_stream = None
with eval_tab1:
    eval_file = st.file_uploader("Upload a .txt or .csv file for evaluation", type=["txt", "csv"], key="eval_uploader")
    if eval_file: user_input_text_stream = io.StringIO(eval_file.getvalue().decode("utf-8"))
with eval_tab2:
    eval_text_area = st.text_area("Paste labeled messages for evaluation", height=250, placeholder='"spam","..."\n"ham","..."', key="eval_text_area")
    if eval_text_area: user_input_text_stream = io.StringIO(eval_text_area)

if user_input_text_stream:
    if st.button("Run Batch Evaluation", use_container_width=True, type="primary", key="run_batch_eval_button"):
        # Clear any stale results from a previous run
        if 'evaluation_results' in st.session_state:
            del st.session_state['evaluation_results']
            
        user_input_text_stream.seek(0)
        true_labels, messages_to_eval, records, errors = parse_labeled_data_from_stream(user_input_text_stream)
        
        if errors: [st.warning(error) for error in errors]
        
        if messages_to_eval:
            with st.spinner(f"Classifying {len(messages_to_eval)} messages using the live backend..."):
                # --- THIS IS THE SIMPLIFIED AND CORRECT LOGIC ---
                # Call the simple backend endpoint. The backend handles all the complex logic.
                list_of_results = bulk_classify(messages_to_eval)

                if list_of_results:
                    # Check if the backend returned an error (e.g., if it's not ready)
                    if any("error" in r for r in list_of_results):
                        st.error(f"Backend returned an error: {list_of_results[0].get('error', 'Unknown error')}")
                    else:
                        st.session_state.evaluation_results = {
                            "true_labels": true_labels,
                            "final_results_detailed": list_of_results,
                            "messages": messages_to_eval,
                            "records_for_retraining": records,
                        }
                        st.rerun()
                else:
                    st.error("Failed to get results from the backend. The server might be down or busy.")

# --- RESULTS DISPLAY BLOCK ---
if st.session_state.evaluation_results:
    eval_data = st.session_state.evaluation_results
    config = get_config()
    st.subheader(f"Evaluation Results for `{config.get('mode', 'N/A').upper()}` Mode")
    
    true_labels = eval_data["true_labels"]
    final_results_detailed = eval_data["final_results_detailed"]
    pred_labels = [p['prediction'] for p in final_results_detailed]

    # --- Performance Summary ---
    st.subheader("Performance Summary")
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        accuracy = accuracy_score(true_labels, pred_labels)
        st.metric("Overall Accuracy", f"{accuracy:.2%}")
        
        st.text("Classification Report:")
        report_df = pd.DataFrame(classification_report(true_labels, pred_labels, labels=["ham", "spam"], output_dict=True, zero_division=0)).transpose()
        st.dataframe(report_df)

    with summary_col2:
        cm = confusion_matrix(true_labels, pred_labels, labels=["ham", "spam"])
        fig, ax = plt.subplots(figsize=(2, 1))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Performance Metrics")
        # Calculate timing stats from the detailed results
    total_time_ms = sum(r.get('time_ms', 0) for r in final_results_detailed)
    avg_time_ms = total_time_ms / len(true_labels) if true_labels else 0
    
    perf_col1, perf_col2 = st.columns(2)
    perf_col1.metric("Total Prediction Time", f"{total_time_ms / 1000:.4f} s")
    perf_col2.metric("Average Time / Message", f"{avg_time_ms:.2f} ms")

    # Only show hybrid stats if it was a hybrid run
    if config.get('mode') == 'hybrid':
        # Calculate detailed usage and accuracy stats
        nb_results = [res for i, res in enumerate(final_results_detailed) if res['model'] == 'MultinomialNB']
        knn_results = [res for i, res in enumerate(final_results_detailed) if res['model'] != 'MultinomialNB']
        
        nb_indices = [i for i, res in enumerate(final_results_detailed) if res['model'] == 'MultinomialNB']
        knn_indices = [i for i, res in enumerate(final_results_detailed) if res['model'] != 'MultinomialNB']

        nb_count = len(nb_results)
        knn_count = len(knn_results)

        nb_correct = sum(1 for i in nb_indices if final_results_detailed[i]['prediction'] == true_labels[i])
        knn_correct = sum(1 for i in knn_indices if final_results_detailed[i]['prediction'] == true_labels[i])
        
        nb_accuracy = (nb_correct / nb_count * 100) if nb_count > 0 else 0
        knn_accuracy = (knn_correct / knn_count * 100) if knn_count > 0 else 0

        st.markdown("---")
        st.write("**Hybrid System Triage Breakdown**")
        usage_col1, usage_col2 = st.columns(2)
        
        # Display the new, more detailed metrics
        usage_col1.metric(
            label="NB Triage Usage",
            value=f"{nb_count / len(true_labels):.1%}",
            help=f"Handled {nb_count} messages. Correct: {nb_correct} ({nb_accuracy:.1f}%)"
        )
        usage_col2.metric(
            label="k-NN Escalation Usage",
            value=f"{knn_count / len(true_labels):.1%}",
            help=f"Handled {knn_count} messages. Correct: {knn_correct} ({knn_accuracy:.1f}%)"
        )
    
    st.markdown("---")

    # --- Expander for Detailed Breakdown ---
    with st.expander(f"⬇️ Click to see detailed breakdown for all {len(true_labels)} messages"):
        df_breakdown = pd.DataFrame({
            "True Label": true_labels,
            "Predicted Label": [r.get('prediction', 'N/A') for r in final_results_detailed],
            "Correct?": ["✅" if t == r.get('prediction') else "❌" for t, r in zip(true_labels, final_results_detailed)],
            "Model Used": [r.get('model', 'N/A') for r in final_results_detailed],
            "Confidence": [f"{r.get('confidence', 0):.2%}" for r in final_results_detailed],
            "Time (ms)": [f"{r.get('time_ms', 0):.2f}" for r in final_results_detailed],
            "Message": eval_data["messages"]
        })
        st.dataframe(df_breakdown, use_container_width=True)
    
    
    # --- Action Buttons ---
    st.subheader("Next Steps")
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("➕ Add All to Training Data", use_container_width=True):
            with st.spinner("Adding data..."):
                response = send_bulk_feedback(eval_data["records_for_retraining"])
                if response and response['status'] == 'success':
                    st.success(response['message']); st.info("Remember to click 'Retrain Model'!")
                    st.session_state.evaluation_results = None; st.rerun()
    with action_col2:
        if st.button("🗑️ Dismiss Results", use_container_width=True):
            st.session_state.evaluation_results = None; st.rerun()