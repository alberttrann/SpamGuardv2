import requests
import json
import asyncio
import random


# --- OLLAMA GENERATOR (Corrected Version) ---
def get_llm_prompt(label_to_generate: str) -> str:
    """
    Constructs a clearer, more focused prompt for a small model.
    It asks the model to focus on ONE scenario at a time.
    """
    base_prompt = "You are a data generation assistant. Create a single, short, realistic SMS text message."
    
    # --- CHANGE: Drastically improved prompts to prevent repetition and literalism ---
    if label_to_generate == "spam":
        scenarios = [
            # --- Financial & Account Security Scams ---
            "a fake security alert from a bank (e.g., 'Unusual login from Vietnam'), urging you to click a link to secure your account.",
            "a message claiming your account (e.g., Netflix, Amazon, iCloud, PayPal) has been suspended and requires immediate action.",
            "a fake notification about a failed payment for a bill (e.g., mobile phone, electricity), threatening service disconnection.",
            "a warning that your device has been infected with viruses and you must install a provided 'security app' immediately.",
            "a notification that you have been selected for an exclusive credit card or a pre-approved loan with an unbelievably low interest rate.",
            "an alert that your 'Wallet' or payment service has been restricted and you need to verify your identity.",
            "a fake invoice for a high-value item or subscription (e.g., 'Thank you for your $500 purchase at...'), hoping you'll click to dispute it.",
            "an 'urgent' message about a crypto wallet transaction or a new token airdrop that requires connecting your wallet to a scam site.",
            "a tax-related scam, such as a notice of an overdue tax payment or a pending tax refund from a fake government agency.",
            "a notification about an expiring subscription for a service you don't use, urging you to 'log in to cancel'.",

            # --- Prize, Lottery & Giveaway Scams ---
            "a prize, lottery, or giveaway scam (e.g., 'You've won an iPhone 15!') that requires a 'small shipping fee' or personal details to claim.",
            "a message claiming you've won a gift card for a popular store (e.g., Walmart, Shein, Target) and need to click to claim it.",
            "a notification that you are the 'visitor of the day' on a website and have won a prize.",
            "a fake 'Raffle Winner!' notification for a raffle you never entered.",
            "a message saying a major brand (e.g., 'Amazon', 'Costco') is giving away free products as part of a loyalty program.",

            # --- Impersonation & Social Engineering Scams ---
            "an urgent message pretending to be from a relative or friend who is in trouble and needs money or gift card codes immediately.",
            "a cryptic message like 'Is this you in this photo?' or 'Someone has uploaded a video of you' with a shortened, suspicious link.",
            "an impersonation of a well-known figure or celebrity 'endorsing' a get-rich-quick scheme.",
            "a message claiming to be from your mobile provider offering a 'free upgrade' or 'bonus data' if you log in via their link.",
            "a fake 'missed call' or 'new voicemail' notification with a link to listen to it.",

            # --- Job, Product & Service Scams ---
            "an unsolicited 'secret shopper' or 'work from home' job offer promising high pay for little work.",
            "a limited-time offer for a dubious product (e.g., diet pills, 'male enhancement' supplements, teeth whitening kits).",
            "a message about your car's 'extended warranty' being about to expire.",
            "a fake shipping or delivery notification (e.g., 'Your FedEx package has a delivery issue') with a link to 'reschedule' or 'track'.",
            "a message offering a free or heavily discounted trial for a psychic reading or astrology service.",
            "a deal on a software key or subscription (e.g., 'Windows 10 Pro for $5') that is almost certainly pirated or fake.",
            "a political survey that promises a reward upon completion but leads to a phishing site.",
            "a fake alert from a food delivery service saying 'Your order is delayed' with a link to a fake login page.",

            # --- Curiosity & Bait Scams ---
            "a fake confirmation for an appointment or reservation you never made, with a link to 'view details' or 'cancel'.",
            "a message that simply says 'Hey, I've been trying to reach you, please get back to me' from an unknown number.",
            "a message from a dating bot, often starting with a generic compliment like 'I saw your profile and you look very handsome/beautiful'.",
            "a 'Class Action Lawsuit' notification, claiming you may be eligible for a settlement if you provide your information.",
            "a fake 'lost item' message, such as 'Did you lose a wallet? I found one with your contact info' trying to lure you into a conversation.",
            "a stock tip for a 'guaranteed' high-return penny stock, which is a classic 'pump and dump' scheme."
]
        instruction = f"The message MUST be 'spam'. It should be about: **{random.choice(scenarios)}**"
    
    else: # Ham
        scenarios = [
            # --- Planning & Logistics ---
            "making plans to meet up for coffee, dinner, or a movie tonight.",
            "confirming you are on your way, but might be a few minutes late (e.g., 'running 5 mins behind').",
            "asking a quick question about a shared task or errand (e.g., 'did you remember to buy milk?').",
            "coordinating details for a gathering or event (e.g., 'what time should I come over?').",
            "confirming an appointment or reservation (e.g., 'See you at 3pm for our meeting').",
            "checking availability for a call or video chat (e.g., 'Are you free for a quick chat around 2?').",
            "asking about public transport or travel logistics (e.g., 'Is the bus still running on schedule?').",
            "discussing plans for the weekend or an upcoming trip.",
            "asking for directions or clarifying a meeting point.",
            "confirming receipt of an item or document (e.g., 'Got the file, thanks!').",

            # --- Social & Casual Chat ---
            "sharing a simple, funny observation from your day (e.g., 'just saw a dog on a skateboard').",
            "checking in on a friend you haven't talked to in a while (e.g., 'How have you been?').",
            "sending a simple greeting or saying good morning/night.",
            "commenting on a photo or social media post you saw (e.g., 'Loved your beach pics!').",
            "asking for a recommendation (e.g., for a restaurant, movie, book, or service).",
            "sending a congratulatory message for an achievement or event (e.g., 'Congrats on the new job!').",
            "sending a get-well-soon message or thinking of you message.",
            "sharing a quick update about your day or what you're doing.",
            "making a casual observation about the weather or current events.",
            "responding to a previous message with a quick 'okay' or 'got it'.",
            "asking how someone's day or week is going.",
            "sharing exciting news (e.g., 'You won't believe what happened today!').",
            "sending a message of encouragement or support.",
            "thanking someone for something small (e.g., 'Thanks for the ride!').",

            # --- Work & Academic Related (Normal) ---
            "asking a colleague a work-related question (e.g., 'Can you send me that report?').",
            "confirming completion of a task (e.g., 'Project is done, sending it over now.').",
            "briefly discussing a meeting or project detail.",
            "asking for clarification on an assignment or task.",
            "informing a team member about a minor update or change.",

            # --- Personal & Domestic ---
            "asking about something related to home or family (e.g., 'Did you feed the cat?').",
            "discussing groceries or household chores.",
            "reminding someone about a family event or commitment.",
            "sending a message to a family member about daily routines.",
            "sharing a cute photo or video of a pet or child.",
            "discussing plans for dinner at home."
        ]
        instruction = f"The message MUST be a normal, personal message ('ham'). It should be about: **{random.choice(scenarios)}**"

    return f"""
    {base_prompt}
    {instruction}

    IMPORTANT:
    1. Return ONLY the raw text of the message itself. Do NOT add JSON or explanations.
    2. Do NOT use the literal words 'ham' or 'spam' in the message text itself.
    """

# (The rest of the file remains the same as the previous correct version)
async def generate_with_ollama(model: str, label_to_generate: str | None = None):
    actual_label = label_to_generate if label_to_generate else random.choice(['spam', 'ham'])
    prompt = get_llm_prompt(actual_label)
    
    payload = {"model": model.strip(), "prompt": prompt, "stream": False}
    OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
    try:
        yield "Thinking... 🧠"
        response = requests.post(OLLAMA_GEN_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        json_response = response.json()
        message_text = json_response.get('response', '').strip().replace('"', '') # Clean up quotes
        
        if message_text:
            generated_data = {"message": message_text, "label": actual_label}
            yield generated_data
        else:
            yield "Error: LLM returned an empty response."

    except requests.exceptions.RequestException as e:
        yield f"Error: Could not connect to Ollama. Details: {e}"
    except Exception as e:
        yield f"An unexpected error occurred: {e}"

# --- NEW: Added generator for LM Studio ---
async def generate_with_lmstudio(model: str, label_to_generate: str | None = None):
    """Generates data using an LM Studio local server."""
    actual_label = label_to_generate if label_to_generate else random.choice(['spam', 'ham'])
    prompt = get_llm_prompt(actual_label)
    
    # LM Studio uses an OpenAI-compatible endpoint.
    LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
    
    # The payload is also OpenAI-compatible.
    # The 'model' parameter is often ignored if a model is pre-loaded in the UI.
    payload = {
        "model": model, 
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7, # A common default
    }

    try:
        yield "Sending request to LM Studio... 💻"
        # API key is not needed for the local server.
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
        response.raise_for_status()

        response_json = response.json()
        message_text = response_json['choices'][0]['message']['content'].strip().replace('"', '')

        if message_text:
            generated_data = {"message": message_text, "label": actual_label}
            yield generated_data
        else:
            yield "Error: LLM returned an empty response."

    except requests.exceptions.RequestException as e:
        yield f"Error: Could not connect to LM Studio. Is the server running? Details: {e}"
    except (KeyError, IndexError) as e:
        yield f"Error: Unexpected response format from LM Studio. Details: {e}"
    except Exception as e:
        yield f"An unexpected error occurred: {e}"

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
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
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