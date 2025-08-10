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

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DASHBOARD_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from backend.utils import preprocess_tokenizer

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'backend', 'data')

# --- Global Components for k-NN (Load Once) ---
@st.cache_resource
def load_transformer_model():
    print("UI: Loading sentence-transformer model for the first time...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(device).eval()
    print("UI: Transformer model loaded.")
    return tokenizer, model, device

TOKENIZER, TRANSFORMER_MODEL, DEVICE = load_transformer_model()

# --- Helper functions for k-NN ---
def average_pool(states, mask): return (states * mask[..., None]).sum(1) / mask.sum(-1)[..., None]
def get_embeddings(texts: list, prefix: str, batch_size: int = 32) -> np.ndarray:
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = [f"{prefix}: {text}" for text in texts[i:i + batch_size]]
        tokens = TOKENIZER(batch, max_length=512, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): outputs = TRANSFORMER_MODEL(**tokens)
        embeds = average_pool(outputs.last_hidden_state, tokens['attention_mask'])
        all_embeds.append(F.normalize(embeds, p=2, dim=1).cpu().numpy())
    return np.vstack(all_embeds)

# --- Configuration & Setup ---
API_BASE_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="wide")

# --- Session State ---
if 'last_classified_message' not in st.session_state: st.session_state.last_classified_message = None
if 'generating' not in st.session_state: st.session_state.generating = False
if 'generation_type' not in st.session_state: st.session_state.generation_type = None
if 'evaluation_results' not in st.session_state: st.session_state.evaluation_results = None
if 'backend_ready' not in st.session_state: st.session_state.backend_ready = False

# --- API Functions ---
def check_backend_status():
    """Polls the backend to see if the classifier is ready."""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return {"is_ready": False}
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
def get_config():
    try: response = requests.get(f"{API_BASE_URL}/config"); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}"); return None

def set_config(mode, knn_dataset_file):
    try: response = requests.post(f"{API_BASE_URL}/config", json={"mode": mode, "knn_dataset_file": knn_dataset_file}); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}"); return None

# --- Helper Function for Parsing ---
def parse_labeled_data_from_stream(file_stream):
    true_labels = []; messages = []; records_for_retraining = []; errors = []
    try:
        reader = csv.reader(file_stream)
        for i, row in enumerate(reader):
            if not row: continue
            if len(row) != 2: errors.append(f"Line {i+1}: Invalid format."); continue
            label, message = row
            label = label.strip().lower(); message = message.strip()
            if label not in ['ham', 'spam']: errors.append(f"Line {i+1}: Invalid label '{label}'."); continue
            if not message: errors.append(f"Line {i+1}: Message is empty."); continue
            true_labels.append(label); messages.append(message); records_for_retraining.append({"label": label, "message": message})
    except Exception as e: errors.append(f"A critical error occurred during parsing: {e}")
    return true_labels, messages, records_for_retraining, errors

# --- Main Application Logic ---

st.title("🛡️ SpamGuard AI: An Adaptive Spam Filtering System")

if not st.session_state.backend_ready:
    status_response = check_backend_status()
    
    if status_response and status_response.get("is_ready"):
        st.session_state.backend_ready = True
        st.success("✅ SpamGuard AI engine is ready! Loading application...")
        time.sleep(1.5) 
        st.rerun() 
    else:
        st.info("⏳ Waiting for SpamGuard AI engine to be ready... This may take several minutes.")
        st.warning("If this is the first run or after retraining, please run `python -m backend.loader` in a separate terminal.")
        time.sleep(5)
        st.rerun() 
        st.stop() 


# SECTION 1: CLASSIFICATION
st.header("1. Real-time Classification")
message_input = st.text_area("Enter a message to analyze:", height=100, key="msg_input")
if st.button("Classify Message", use_container_width=True):
    if message_input:
        with st.spinner("Analyzing..."): result = classify_message(message_input)
        if result: st.session_state.last_classified_message = {"message": message_input, **result}
    else: st.warning("Please enter a message.")
if st.session_state.last_classified_message:
    res = st.session_state.last_classified_message; st.subheader("Analysis Result")
    pred = res['prediction']; conf = res['confidence']; model = res['model']
    color = "error" if pred == "spam" else "success"; icon = "🚨" if pred == "spam" else "✅"
    st.markdown(f"### <span style='color:{'red' if color=='error' else 'green'};'>{icon} Prediction: **{pred.upper()}**</span>", unsafe_allow_html=True)
    col1, col2 = st.columns(2); col1.metric("Confidence", f"{conf:.2%}"); col2.metric("Model Used", model)
    if res.get('evidence'):
        with st.expander("💡 See Why (Explainable AI)"):
            st.write("Similar messages from the database:"); [st.info(f"**{item['label'].upper()}** (Similarity: {item['similarity_score']:.3f}):\n\n_{item['similar_message']}_") for item in res['evidence']]
    st.subheader("Was this correct?")
    fb_col1, fb_col2, fb_col3 = st.columns(3)
    if fb_col1.button("✔️ Yes, it's correct!", use_container_width=True): send_feedback(res['message'], res['prediction'])
    if fb_col2.button("❌ No, it's wrong!", use_container_width=True): wrong_label = "ham" if pred == "spam" else "spam"; send_feedback(res['message'], wrong_label)
    if fb_col3.button("🗑️ Dismiss (Don't Retrain)", use_container_width=True): st.toast("Result dismissed."); st.session_state.last_classified_message = None; st.rerun()

st.divider()

# --- SECTION 2: MODEL MANAGEMENT & REGISTRY ---
st.header("2. Model Management & Registry")
st.write("View and manage trained model versions and classifier operational settings.")

# Use tabs for Model Selection and Configuration
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
            if st.button("Activate Selected Model", type="primary", key="activate_selected_model_button"):
                with st.spinner(f"Activating model '{model_to_activate}'..."):
                    response = set_active_model(model_to_activate)
                    if response and response['status'] == 'success': st.success(response['message']); st.info("Loader script must be run to load the new model."); time.sleep(2); st.rerun()
                    else: st.error("Failed to activate model.")
        else: st.info("No other inactive models to activate.")
    elif registry_data is None: st.error("Could not connect to the backend API.")
    else: st.warning("No models found. Please retrain a model in Section 3.")

with tab_config:
    st.subheader("Classifier Operational Mode")
    current_config = get_config()
    
    # Mode selection
    mode_options = ["hybrid", "nb_only", "knn_only"]
    selected_mode = st.radio(
        "Choose classification mode:",
        options=mode_options,
        index=mode_options.index(current_config["mode"]),
        horizontal=True,
        key="classifier_mode_radio"
    )

    # k-NN dataset selection (only shown if k-NN is relevant)
    if selected_mode == "hybrid" or selected_mode == "knn_only":
        st.markdown("---")
        st.subheader("k-NN Indexing Dataset")
        st.info(
            "The k-NN Vector Search component builds its index from a specific dataset. "
            "Select the dataset that the Naive Bayes model (if used) was trained on for consistency."
        )
        
        available_datasets = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        try:
            default_dataset_index = available_datasets.index(current_config["knn_dataset_file"])
        except ValueError:
            default_dataset_index = 0 
            
        selected_knn_dataset = st.selectbox(
            "Select dataset for k-NN index:",
            options=available_datasets,
            index=default_dataset_index,
            key="knn_dataset_selector"
        )
    else:
        selected_knn_dataset = current_config["knn_dataset_file"] 

    if (selected_mode != current_config["mode"] or 
        selected_knn_dataset != current_config["knn_dataset_file"]):
        
        if st.button("Apply Configuration Changes", type="primary", key="apply_config_button"):
            with st.spinner("Applying new configuration..."):
                response = set_config(selected_mode, selected_knn_dataset)
                if response and response['status'] == 'success':
                    st.success(response['message'])
                    st.info("Classifier will reload with new settings on the next request. Remember to run loader if needed.")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Failed to apply configuration.")
    else:
        st.info("No pending configuration changes.")

st.divider()

# SECTION 3: MODEL PERFORMANCE & LEARNING
st.header("3. Model Performance & Learning")
analytics_col, training_col = st.columns(2)
with analytics_col:
    st.subheader("Dataset Analytics");
    if st.button("Refresh Analytics"):
        try:
            stats = requests.get(f"{API_BASE_URL}/analytics").json()
            total_base = stats['base_ham_count'] + stats['base_spam_count']; total_new = stats['new_ham_count'] + stats['new_spam_count']
            st.metric("Total Messages in Core Dataset", f"{total_base:,}"); st.metric("New Messages Ready for Training", f"{total_new:,}", help=f"User: {stats['user_contribution']}, LLM: {stats['llm_contribution']}")
            chart_data = pd.DataFrame({"Type": ["Ham", "Spam"], "Core Dataset": [stats['base_ham_count'], stats['base_spam_count']], "New Data": [stats['new_ham_count'], stats['new_spam_count']]}).set_index('Type'); st.bar_chart(chart_data)
        except (requests.exceptions.RequestException, json.JSONDecodeError): st.error("Could not fetch analytics.")
    st.markdown("---"); st.subheader("🧠 Model Interpretation (XAI)"); st.write("See the top keywords the Naive Bayes model uses.")
    if st.button("Analyze Model Keywords"):
        with st.spinner("Analyzing model..."): explanation = get_model_explanation()
        if explanation and not explanation.get("error"): st.session_state.explanation = explanation
        elif explanation: st.error(explanation["error"])
    if 'explanation' in st.session_state:
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1: st.success("Top HAM Keywords"); st.table(pd.DataFrame(st.session_state.explanation["top_ham_keywords"], columns=["Keyword"]))
        with exp_col2: st.error("Top SPAM Keywords"); st.table(pd.DataFrame(st.session_state.explanation["top_spam_keywords"], columns=["Keyword"]))
with training_col:
    st.subheader("Continuous Learning"); st.write("Enrich the dataset with new examples.")
    tab1, tab2 = st.tabs(["📤 Upload File", "📝 Paste Text"])
    with tab1:
        uploaded_file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"], help="Format: label,message")
        if uploaded_file is not None and st.button("Add Data from File", use_container_width=True, key="add_from_file_button"):
            string_io = io.StringIO(uploaded_file.getvalue().decode("utf-8")); _, _, records, errors = parse_labeled_data_from_stream(string_io)
            if errors: [st.warning(error) for error in errors]
            if records:
                with st.spinner("Sending bulk data..."): response = send_bulk_feedback(records)
                if response and response['status'] == 'success': st.success(response['message'])
                else: st.error("Failed to add data.")
    with tab2:
        bulk_text = st.text_area("Paste labeled messages", height=200, placeholder='"spam","..."\n"ham","..."', key="bulk_text_input")
        if st.button("Add Pasted Data", use_container_width=True, key="add_from_paste_button"):
            if bulk_text:
                string_io = io.StringIO(bulk_text); _, _, records, errors = parse_labeled_data_from_stream(string_io)
                if errors: [st.warning(error) for error in errors]
                if records:
                    with st.spinner("Sending bulk data..."): response = send_bulk_feedback(records)
                    if response and response['status'] == 'success': st.success(response['message'])
                    else: st.error("Failed to add data.")
            else: st.warning("Text area is empty.")
    st.markdown("---"); st.write("Add all new data to the main dataset and retrain the models.")
    if st.button("Retrain Model with New Data", use_container_width=True, type="primary"):
        with st.spinner("Retraining in progress... This may take several minutes."): retrain_result = retrain_model()
        if retrain_result:
            if retrain_result['status'] == 'success': st.success(retrain_result['message']); st.info("Please restart the `loader.py` script to activate the new model."); st.balloons()
            else: st.info(retrain_result['message'])

st.divider()

# --- SECTION 4: LLM DATA DISTILLATION  ---
st.header("4. Augment Data with an LLM")
st.write("Automatically generate new training data and review it before adding to the dataset.")

with st.container():
    use_llm = st.toggle("Enable LLM Data Generation Controls")

    if use_llm:
        PROVIDER_MAP = {"Ollama (Local)": "ollama", "LM Studio (Local)": "lmstudio", "OpenRouter (Cloud)": "openrouter"}
        provider_display_name = st.radio("Choose LLM Provider", PROVIDER_MAP.keys(), horizontal=True, key="provider")
        provider_api_name = PROVIDER_MAP[provider_display_name]
        if provider_api_name == "ollama": model_id = st.text_input("Ollama Model ID", "llama3", key="ollama_model"); api_key = ""
        elif provider_api_name == "lmstudio": model_id = st.text_input("Model Identifier", "local-model", key="lmstudio_model"); api_key = ""
        else: model_id = st.text_input("OpenRouter Model ID", "mistralai/mistral-7b-instruct:free", key="or_model"); api_key = st.text_input("OpenRouter API Key", type="password", key="or_api_key")
        st.subheader("Generation Controls")
        gen_col1, gen_col2, gen_col3 = st.columns(3)
        if gen_col1.button("🤖 Start Continuous (Random)", use_container_width=True, disabled=st.session_state.generating): st.session_state.generating = True; st.session_state.generation_type = None; st.rerun()
        if gen_col2.button("🚨 Start Continuous (SPAM)", use_container_width=True, disabled=st.session_state.generating): st.session_state.generating = True; st.session_state.generation_type = "spam"; st.rerun()
        if gen_col3.button("✅ Start Continuous (HAM)", use_container_width=True, disabled=st.session_state.generating): st.session_state.generating = True; st.session_state.generation_type = "ham"; st.rerun()

        # --- Generation Loop & Temporary Storage ---
        if st.session_state.generating:
            st.info("LLM is continuously generating data. Review below. Press 'Stop Generating' to halt.")
            if st.button("⏹️ Stop Generating", use_container_width=True, type="primary"):
                st.session_state.generating = False; st.rerun()

            status_box = st.empty()
            if 'generated_data_for_review' not in st.session_state:
                st.session_state.generated_data_for_review = []

            payload = {"provider": provider_api_name, "model": model_id, "api_key": api_key, "label_to_generate": st.session_state.generation_type}
            
            try:
                with requests.post(f"{API_BASE_URL}/generate_data", json=payload, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                        if not st.session_state.generating: break 
                        if chunk.startswith("data:"):
                            try:
                                data_str = chunk.split("data:", 1)[1].strip()
                                if data_str.startswith('Generated & Saved'): 
                                    generated_item_str = data_str.replace('Generated & Saved: ', '')
                                    generated_item = json.loads(generated_item_str)
                                    st.session_state.generated_data_for_review.append(generated_item)
                                    status_box.write(f"🤖 LLM Generated: {generated_item['label']} - {generated_item['message'][:50]}...")
                                else:
                                    status_box.write(f"🤖 LLM Status: {data_str}")
                            except json.JSONDecodeError:
                                status_box.write(f"🤖 LLM Raw Output: {data_str}") 
                                
                        time.sleep(0.1) # Small delay to keep UI responsive

            except requests.exceptions.RequestException as e:
                status_box.error(f"Error during generation: {e}")
            
            if not st.session_state.generating:
                status_box.success("Generation halted.")
            else:
                status_box.warning("Stream closed unexpectedly.")
            st.session_state.generating = False 
            st.rerun() 

        # --- Review and Action Section (appears after generation stops) ---
        if st.session_state.generated_data_for_review:
            st.subheader("Review Generated Data")
            st.write(f"Review {len(st.session_state.generated_data_for_review)} new messages. Select to keep or discard.")
            
            # Display generated messages with checkboxes
            cols = st.columns([0.1, 0.2, 0.7])
            cols[0].write("**Keep?**")
            cols[1].write("**Label**")
            cols[2].write("**Message**")
            st.markdown("---")
            
            # Create a list of booleans to track which messages to keep
            if 'keep_generated_flags' not in st.session_state or len(st.session_state.keep_generated_flags) != len(st.session_state.generated_data_for_review):
                st.session_state.keep_generated_flags = [True] * len(st.session_state.generated_data_for_review) # Default to keep all

            for i, item in enumerate(st.session_state.generated_data_for_review):
                col_keep, col_label, col_msg = st.columns([0.1, 0.2, 0.7])
                st.session_state.keep_generated_flags[i] = col_keep.checkbox("", value=st.session_state.keep_generated_flags[i], key=f"keep_gen_{i}")
                col_label.write(f"**{item['label'].upper()}**")
                col_msg.write(f"_{item['message']}_")
            st.markdown("---")

            # Action buttons for review
            review_col1, review_col2 = st.columns(2)
            with review_col1:
                if st.button("➕ Add Selected to Training Data", use_container_width=True, key="add_selected_gen_button"):
                    selected_records = [st.session_state.generated_data_for_review[i] for i, keep in enumerate(st.session_state.keep_generated_flags) if keep]
                    if selected_records:
                        with st.spinner(f"Adding {len(selected_records)} selected records..."):
                            response = send_bulk_feedback(selected_records)
                            if response and response['status'] == 'success': st.success(response['message'])
                            else: st.error("Failed to add selected data.")
                    else: st.warning("No records selected to add.")
                    st.session_state.generated_data_for_review = [] # Clear after action
                    st.session_state.keep_generated_flags = []
                    st.rerun()
            with review_col2:
                if st.button("🗑️ Discard All Generated Data", use_container_width=True, key="discard_all_gen_button"):
                    st.session_state.generated_data_for_review = []
                    st.session_state.keep_generated_flags = []
                    st.success("All generated data discarded.")
                    st.rerun()
st.divider()

# SECTION 5: BATCH EVALUATION & TESTING
st.header("5. Batch Evaluation & Testing")
st.write("Evaluate the currently active model's full HYBRID architecture on a batch of labeled messages.")
eval_tab1, eval_tab2 = st.tabs(["📤 Upload File", "📝 Paste Text"])
user_input_text_stream = None
with eval_tab1:
    eval_file = st.file_uploader("Upload a .txt or .csv file for evaluation", type=["txt", "csv"], key="eval_uploader")
    if eval_file: user_input_text_stream = io.StringIO(eval_file.getvalue().decode("utf-8"))
with eval_tab2:
    eval_text_area = st.text_area("Paste labeled messages for evaluation", height=250, placeholder='"spam","..."\n"ham","..."', key="eval_text_area")
    if eval_text_area: user_input_text_stream = io.StringIO(eval_text_area)
if user_input_text_stream:
    st.markdown("---")
    st.subheader("Select k-NN Indexing Dataset")
    st.info(
        "For a correct evaluation, the k-NN Vector Search must be indexed with the same dataset "
        "that the selected Naive Bayes model was trained on. This ensures a consistent model behavior."
    )
    
    # Get all available CSV files in the backend/data directory
    available_datasets = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    # Attempt to pre-select a default based on the current backend config
    current_config = get_config()
    default_dataset_file = current_config.get("knn_dataset_file", "2cls_spam_text_cls.csv")
    
    try:
        default_index = available_datasets.index(default_dataset_file)
    except ValueError:
        default_index = 0 # Fallback to first available if default isn't found
            
    selected_knn_dataset = st.selectbox(
        "Which dataset should be used to build the k-NN index?",
        options=available_datasets,
        index=default_index,
        key="knn_dataset_selector_eval" # Unique key for this selectbox
    )
    training_csv_path = os.path.join(DATA_DIR, selected_knn_dataset)
    
    st.markdown("---")

    # The main button to run the evaluation
    if st.button("Run Full Hybrid Evaluation", use_container_width=True, type="primary", key="run_full_hybrid_eval_button"):
        # --- Clear any old, stale results before starting a new evaluation ---
        if 'evaluation_results' in st.session_state:
            del st.session_state['evaluation_results']
            
        # Reset the stream's cursor to the beginning before parsing, essential for re-reads
        user_input_text_stream.seek(0)
        true_labels, messages_to_eval, records, errors = parse_labeled_data_from_stream(user_input_text_stream)
        
        if errors:
            [st.warning(error) for error in errors]
        
        # Proceed only if messages were successfully parsed
        if messages_to_eval:
            with st.spinner(f"Performing full hybrid evaluation on {len(messages_to_eval)} messages... This may take several minutes."):
                # Get the active model details from the backend
                registry_data = get_models()
                active_id = registry_data.get("active_model_id")
                if not registry_data or not active_id:
                    st.error("Could not determine the active model from the registry. Please ensure a model is trained and active in Section 2.")
                    st.stop() # Halt execution if no active model is found
                
                model_details = registry_data["models"][active_id]
                
                try:
                    # 1. Load the active Naive Bayes model based on registry
                    pipeline_path = os.path.join(MODELS_DIR, model_details["pipeline_file"])
                    encoder_path = os.path.join(MODELS_DIR, model_details["encoder_file"])
                    
                    if not os.path.exists(pipeline_path) or not os.path.exists(encoder_path):
                        st.error(f"FATAL: Model files for '{active_id}' not found at '{MODELS_DIR}'. Please re-run loader or retrain.")
                        st.stop()
                    
                    pipeline = joblib.load(pipeline_path)
                    label_encoder = joblib.load(encoder_path)

                    # 2. Build the FAISS index from the USER-SELECTED training data
                    st.info(f"Building FAISS index from: {os.path.basename(training_csv_path)}...")
                    df_train = pd.read_csv(training_csv_path, quotechar='"', on_bad_lines='skip')
                    df_train.dropna(subset=['Message'], inplace=True)
                    db_messages = df_train["Message"].astype(str).tolist()
                    db_labels = df_train["Category"].tolist()
                    passage_embeddings = get_embeddings(db_messages, "passage")
                    faiss_index = faiss.IndexFlatIP(passage_embeddings.shape[1])
                    faiss_index.add(passage_embeddings.astype('float32'))
                    st.info("FAISS index built.")

                    # 3. Run the full hybrid classification logic and capture ALL metrics
                    final_results_detailed = []; raw_nb_probs_list = []
                    spam_class_index = np.where(label_encoder.classes_ == 'spam')[0][0]
                    start_time_total = time.perf_counter()
                    
                    for message in messages_to_eval:
                        nb_probabilities = pipeline.predict_proba([message])[0]; spam_prob = nb_probabilities[spam_class_index]
                        raw_nb_probs_list.append(spam_prob)
                        if spam_prob > 0.95: prediction_label = "spam"; model_used = "MultinomialNB"
                        elif spam_prob < 0.05: prediction_label = "ham"; model_used = "MultinomialNB"
                        else:
                            model_used = "Vector Search (k-NN)"; q_emb = get_embeddings([message], "query", 1); 
                            if faiss_index and faiss_index.ntotal > 0: # Ensure index is not empty
                                _, indices = faiss_index.search(q_emb.astype('float32'), 5)
                                n_labels = [db_labels[i] for i in indices[0]]; prediction_label = max(set(n_labels), key=n_labels.count)
                            else: # Fallback if k-NN index is empty
                                prediction_label = label_encoder.inverse_transform([np.argmax(nb_probabilities)])[0]; model_used = "MultinomialNB (FAISS unavailable)"
                        final_results_detailed.append({"prediction": prediction_label, "model": model_used})
                    
                    total_time_s = time.perf_counter() - start_time_total

                    st.session_state.evaluation_results = {
                        "model_id": active_id, "true_labels": true_labels, "final_results_detailed": final_results_detailed,
                        "raw_nb_probs": raw_nb_probs_list, "total_time_s": total_time_s,
                        "messages": messages_to_eval, "records_for_retraining": records,
                    }
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred during local model evaluation: {e}")
                    if 'evaluation_results' in st.session_state:
                        del st.session_state['evaluation_results']

        else:
            st.warning("No valid messages found to evaluate. Please check input format.")
if st.session_state.evaluation_results:
    eval_data = st.session_state.evaluation_results
    st.subheader(f"Evaluation Results for Model: `{eval_data['model_id']}`")
    true_labels = eval_data["true_labels"]; final_results_detailed = eval_data["final_results_detailed"]; pred_labels = [p['prediction'] for p in final_results_detailed]
    st.subheader("Performance Summary")
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        accuracy = accuracy_score(true_labels, pred_labels); st.metric("Overall Accuracy", f"{accuracy:.2%}")
        st.text("Performance Metrics:")
        total_time_s = eval_data['total_time_s']; avg_time_ms = (total_time_s * 1000) / len(true_labels)
        st.write(f"**Total Prediction Time:** {total_time_s:.4f} seconds"); st.write(f"**Average Prediction Time:** {avg_time_ms:.2f} ms/message")
        nb_count = sum(1 for r in final_results_detailed if r['model'] == 'MultinomialNB'); knn_count = len(true_labels) - nb_count
        st.write(f"**NB Triage Usage:** {nb_count / len(true_labels):.1%}"); st.write(f"**k-NN Escalation Usage:** {knn_count / len(true_labels):.1%}")
    with summary_col2:
        st.text("Classification Report:"); report_df = pd.DataFrame(classification_report(true_labels, pred_labels, labels=["ham", "spam"], output_dict=True, zero_division=0)).transpose(); st.dataframe(report_df)
        cm = confusion_matrix(true_labels, pred_labels, labels=["ham", "spam"]); fig, ax = plt.subplots(figsize=(4, 3)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
        ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label"); st.pyplot(fig)
    with st.expander(f"⬇️ Click to see detailed breakdown for all {len(true_labels)} messages"):
        df = pd.DataFrame({"True Label": true_labels, "Predicted Label": [r['prediction'] for r in final_results_detailed], "Correct?": ["✅" if t == r['prediction'] else "❌" for t, r in zip(true_labels, final_results_detailed)], "Model Used": [r['model'] for r in final_results_detailed], "Message": eval_data["messages"]})
        st.dataframe(df, use_container_width=True)
    st.subheader("🔬 Interactive Threshold Simulation")
    st.write("Simulate the hybrid system's performance using the raw probabilities from the active Naive Bayes model.")
    confidence_threshold_percent = st.slider("Required Confidence Threshold (%)", min_value=51, max_value=99, value=85, step=1)
    confidence_threshold = confidence_threshold_percent / 100.0
    raw_nb_probs = eval_data["raw_nb_probs"]
    simulated_predictions = []
    for i, spam_prob in enumerate(raw_nb_probs):
        if spam_prob > confidence_threshold: simulated_predictions.append("spam")
        elif spam_prob < (1 - confidence_threshold): simulated_predictions.append("ham")
        else: simulated_predictions.append(pred_labels[i])
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_accuracy = accuracy_score(true_labels, simulated_predictions); st.metric("Simulated Accuracy", f"{sim_accuracy:.2%}")
    with sim_col2:
        nb_usage_count = sum(1 for p in raw_nb_probs if p > confidence_threshold or p < (1 - confidence_threshold)); st.metric("NB Triage Usage", f"{nb_usage_count / len(true_labels):.1%}")
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