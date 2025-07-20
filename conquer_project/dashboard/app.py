import streamlit as st
import requests
import json
import time
import pandas as pd

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"

# --- Page Setup ---
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="wide")
st.title("🛡️ SpamGuard AI: An Adaptive Spam Filtering System")

# --- Initialize Session State ---
if 'last_classified_message' not in st.session_state:
    st.session_state.last_classified_message = None
if 'generating' not in st.session_state:
    st.session_state.generating = False
if 'generation_type' not in st.session_state:
    st.session_state.generation_type = None # Will store 'spam', 'ham', or None

# --- API Functions ---
def classify_message(message):
    try:
        response = requests.post(f"{API_BASE_URL}/classify", json={"text": message})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Could not classify message. Details: {e}")
        return None

def send_feedback(message, correct_label):
    try:
        payload = {"message": message, "correct_label": correct_label}
        response = requests.post(f"{API_BASE_URL}/feedback", json=payload)
        response.raise_for_status()
        st.toast(f"✅ Feedback sent! Model will learn from this.")
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Could not send feedback. Details: {e}")

def retrain_model():
    try:
        response = requests.post(f"{API_BASE_URL}/retrain")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Could not trigger retraining. Details: {e}")
        return None

# --- UI Sections ---

# SECTION 1: CLASSIFICATION
st.header("1. Real-time Classification")
message_input = st.text_area("Enter a message to analyze:", height=100, key="msg_input")

if st.button("Classify Message", use_container_width=True):
    if message_input:
        with st.spinner("Analyzing..."):
            result = classify_message(message_input)
        if result:
            st.session_state.last_classified_message = {"message": message_input, **result}
    else:
        st.warning("Please enter a message.")

# Display classification result and feedback options
if st.session_state.last_classified_message:
    res = st.session_state.last_classified_message
    st.subheader("Analysis Result")
    
    pred = res['prediction']
    conf = res['confidence']
    model = res['model']
    
    color = "error" if pred == "spam" else "success"
    icon = "🚨" if pred == "spam" else "✅"
    st.markdown(f"### <span style='color:{'red' if color=='error' else 'green'};'>{icon} Prediction: **{pred.upper()}**</span>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    col1.metric("Confidence", f"{conf:.2%}")
    col2.metric("Model Used", model)

    if res.get('evidence'):
        with st.expander("💡 See Why (Explainable AI)"):
            st.write("The model found these messages in its database to be most similar:")
            for item in res['evidence']:
                st.info(f"**{item['label'].upper()}** (Similarity: {item['similarity_score']:.3f}):\n\n_{item['similar_message']}_")

    st.subheader("Was this correct?")
    fb_col1, fb_col2, fb_col3 = st.columns([1,1,5])
    if fb_col1.button("✔️ Yes, it's correct!"):
        send_feedback(res['message'], res['prediction'])
    if fb_col2.button("❌ No, it's wrong!"):
        wrong_label = "ham" if pred == "spam" else "spam"
        send_feedback(res['message'], wrong_label)

st.divider()

# SECTION 2: MODEL TRAINING & ANALYTICS
st.header("2. Model Performance & Learning")
analytics_col, training_col = st.columns(2)

with analytics_col:
    st.subheader("Dataset Analytics")
    if st.button("Refresh Analytics"):
        try:
            stats = requests.get(f"{API_BASE_URL}/analytics").json()
            total_base = stats['base_ham_count'] + stats['base_spam_count']
            total_new = stats['new_ham_count'] + stats['new_spam_count']

            st.metric("Total Messages in Core Dataset", f"{total_base:,}")
            st.metric("New Messages Ready for Training", f"{total_new:,}", help=f"User: {stats['user_contribution']}, LLM: {stats['llm_contribution']}")
            
            chart_data = pd.DataFrame({
                "Type": ["Ham", "Spam"],
                "Core Dataset": [stats['base_ham_count'], stats['base_spam_count']],
                "New Data": [stats['new_ham_count'], stats['new_spam_count']]
            }).set_index('Type')
            st.bar_chart(chart_data)

        except (requests.exceptions.RequestException, json.JSONDecodeError):
            st.error("Could not fetch analytics.")

with training_col:
    st.subheader("Continuous Learning")
    st.write("Add the new data from user feedback and LLM generation to the main dataset and retrain the models.")
    if st.button("Retrain Model with New Data", use_container_width=True, type="primary"):
        with st.spinner("Retraining in progress... This may take a few minutes."):
            retrain_result = retrain_model()
            if retrain_result:
                if retrain_result['status'] == 'success':
                    st.success(retrain_result['message'])
                    st.balloons()
                else:
                    st.info(retrain_result['message'])

st.divider()


# SECTION 3: LLM DATA DISTILLATION (Final Version)
st.header("3. Augment Data with an LLM")
st.write("Automatically generate new training data to improve the model's performance and help it adapt to new types of messages.")

with st.container():
    use_llm = st.toggle("Enable LLM Data Generation Controls")

    if use_llm:
        provider = st.radio("Choose LLM Provider", ["Ollama (Local)", "OpenRouter (Cloud)"], horizontal=True, key="provider")
        
        if provider == "Ollama (Local)":
            model_id = st.text_input("Ollama Model ID", "llama3", key="ollama_model")
            api_key = ""
        else: # OpenRouter
            model_id = st.text_input("OpenRouter Model ID", "mistralai/mistral-7b-instruct", key="or_model")
            api_key = st.text_input("OpenRouter API Key", type="password", key="or_api_key")

        st.subheader("Generation Controls")
        gen_col1, gen_col2, gen_col3 = st.columns(3)

        if gen_col1.button("🤖 Start Continuous Generation (Random)", use_container_width=True, disabled=st.session_state.generating):
            st.session_state.generating = True
            st.session_state.generation_type = None
            st.rerun()

        if gen_col2.button("🚨 Start Continuous Generation (SPAM)", use_container_width=True, disabled=st.session_state.generating):
            st.session_state.generating = True
            st.session_state.generation_type = "spam"
            st.rerun()
        
        if gen_col3.button("✅ Start Continuous Generation (HAM)", use_container_width=True, disabled=st.session_state.generating):
            st.session_state.generating = True
            st.session_state.generation_type = "ham"
            st.rerun()

        # This block now handles the actual streaming and display
        if st.session_state.generating:
            st.info("LLM is continuously generating data. Press 'Stop Generating' to halt the process.")
            if st.button("⏹️ Stop Generating", use_container_width=True, type="primary"):
                st.session_state.generating = False
                st.session_state.generation_type = None
                st.rerun()

            # Create a placeholder for status updates.
            status_box = st.empty()
            
            payload = {
                "provider": provider.split(" ")[0].lower(),
                "model": model_id,
                "api_key": api_key,
                "label_to_generate": st.session_state.generation_type
            }

            try:
                # This `with` block establishes ONE connection that the backend will keep alive.
                with requests.post(f"{API_BASE_URL}/generate_data", json=payload, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    # Iterate over the continuous stream of chunks from the backend.
                    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                        # If the user clicks "Stop", this session state changes and we break the loop.
                        if not st.session_state.generating:
                            status_box.warning("Generation stopped by user.")
                            break
                        
                        if chunk.startswith("data:"):
                            status_message = chunk.split("data:", 1)[1].strip()
                            # Display the live status message from the backend.
                            status_box.write(f"🤖 LLM Status: {status_message}")

            except requests.exceptions.RequestException as e:
                status_box.error(f"Connection lost or error during generation: {e}")
            
            # This part will be reached only when the stream is broken (by stopping or error).
            if not st.session_state.generating:
                st.success("Process halted.")
            else:
                # If we get here but generating is still true, it means the stream broke unexpectedly.
                st.session_state.generating = False
                st.warning("Stream closed unexpectedly. Please start generation again.")