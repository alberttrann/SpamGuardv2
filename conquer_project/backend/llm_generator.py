import requests
import json
import asyncio

# --- HELPER FUNCTION (No changes) ---
def get_llm_prompt(label_to_generate: str | None = None) -> str:
    """Constructs the appropriate prompt for the LLM based on the desired label."""
    base_prompt = "You are a data generation assistant. Your task is to create a realistic, short SMS text message."
    
    if label_to_generate == "spam":
        instruction = "The message MUST be 'spam' (e.g., an advertisement, scam, phishing attempt, or unwanted promotion)."
    elif label_to_generate == "ham":
        instruction = "The message MUST be 'ham' (a normal, personal, everyday message between friends or family)."
    else: # Random
        instruction = "It can be either a 'spam' (advertisement, scam) or 'ham' (a normal, personal message)."

    return f"""
    {base_prompt}
    {instruction}
    The message should be creative and diverse.
    Return ONLY a single, valid JSON object with two keys: "message" and "label".
    The "label" key must be either "spam" or "ham".
    """

# --- OLLAMA GENERATOR (Final, Correct Version for Modern Ollama) ---
async def generate_with_ollama(model: str, label_to_generate: str | None = None):
    prompt = get_llm_prompt(label_to_generate)
    payload = {
        "model": model.strip(),
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    print(f"Ollama payload: {payload}")

    # Try /api/chat first
    OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
    OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
    try:
        yield "Thinking... 🧠 (chat)"
        response = requests.post(OLLAMA_CHAT_URL, json=payload)
        response.raise_for_status()
        json_response = response.json()
        full_response_str = json_response.get('message', {}).get('content', '')
        print(f"Ollama /api/chat raw response: {full_response_str}")
        if not full_response_str.strip():
            # If chat response is empty, try /api/generate
            yield "No response from /api/chat, trying /api/generate..."
            response = requests.post(OLLAMA_GEN_URL, json=payload)
            response.raise_for_status()
            json_response = response.json()
            full_response_str = json_response.get('response', '')
            print(f"Ollama /api/generate raw response: {full_response_str}")

        # Robustly find and parse the JSON from the full text response
        start_index = full_response_str.find('{')
        end_index = full_response_str.rfind('}') + 1
        if start_index != -1 and end_index != 0:
            json_str = full_response_str[start_index:end_index]
            generated_json = json.loads(json_str)
        else:
            generated_json = {}

        if "message" in generated_json and "label" in generated_json:
            yield f"Generated: {generated_json}"
        else:
            yield f"Error: LLM returned an invalid response or no JSON. Full response: {full_response_str}"
    except requests.exceptions.RequestException as e:
        error_msg = f"Error: Could not connect to Ollama. Details: {e}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.text
                error_msg += f"\nOllama response: {error_details}"
            except Exception:
                pass
        yield error_msg
    except (json.JSONDecodeError, KeyError):
        yield "Error: Failed to parse the LLM's response. It may not have returned valid JSON."


# --- OPENROUTER GENERATOR (No changes needed) ---
async def generate_with_openrouter(model: str, api_key: str, label_to_generate: str | None = None):
    # This function is correct and remains the same.
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    prompt = get_llm_prompt(label_to_generate)
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    try:
        yield "Sending request to OpenRouter... ☁️"
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']
        generated_json = json.loads(content)
        if "message" in generated_json and "label" in generated_json:
            yield f"Generated: {generated_json}"
        else:
            yield "Error: LLM returned invalid JSON. Retrying..."
    except requests.exceptions.RequestException as e:
        yield f"Error: Could not connect to OpenRouter. Check API key and model name. Details: {e}"
    except json.JSONDecodeError:
        yield "Error: Failed to parse LLM response. Retrying..."