import os
import json
import urllib.request
import urllib.error
from langchain_core.prompts import PromptTemplate
import db
import model

# Cloud & Local LLM Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

_ollama_client = None

def get_ollama():
    """Lazily load Ollama client to prevent hang/error on serverless cold starts."""
    global _ollama_client
    if _ollama_client is None:
        try:
            from langchain_community.llms import Ollama
            _ollama_client = Ollama(model="llama3.2:1b", base_url=OLLAMA_HOST, timeout=5)
        except Exception:
            _ollama_client = False
    return _ollama_client if _ollama_client is not False else None


# Recommendation Prompt
recommendation_prompt = PromptTemplate(
    input_variables=["plant_name", "disease_name"],
    template=(
        "You are an expert agricultural AI assistant. A farmer has uploaded an image of a {plant_name} plant "
        "and it has been diagnosed with {disease_name}. "
        "Provide a short, personalized, and actionable recommendation (3-4 sentences) to the farmer on how to treat and prevent this disease."
    )
)

def _call_cloud_llm(prompt: str) -> str:
    """Invokes Groq or OpenAI compatible cloud API if key is available."""
    api_key = GROQ_API_KEY or OPENAI_API_KEY
    if not api_key:
        return None

    url = "https://api.groq.com/openai/v1/chat/completions" if GROQ_API_KEY else "https://api.openai.com/v1/chat/completions"
    model_name = "llama-3.1-8b-instant" if GROQ_API_KEY else "gpt-4o-mini"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.5
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Cloud LLM call failed: {e}")
        return None


def get_recommendation(plant_name, disease_name):
    """Generates a personalized recommendation based on the diagnosed plant disease."""
    if "Healthy" in disease_name or "healthy" in disease_name:
        return "Your plant appears to be healthy! Maintain appropriate soil moisture, balanced nutrition, and regular scouting for early pest detection."

    # 1. Try Cloud LLM if configured
    prompt_text = recommendation_prompt.format(plant_name=plant_name, disease_name=disease_name)
    cloud_resp = _call_cloud_llm(prompt_text)
    if cloud_resp:
        return cloud_resp

    # 2. Try Local Ollama if available
    ollama = get_ollama()
    if ollama:
        try:
            resp = ollama.invoke(prompt_text)
            if resp:
                return resp.strip()
        except Exception:
            pass

    # 3. Robust Knowledge-Base Fallback (Always reliable on Vercel)
    for class_key, info in model.DISEASE_KNOWLEDGE_BASE.items():
        if plant_name.lower() in class_key.lower() and disease_name.lower() in class_key.lower():
            return (
                f"Pathology Alert ({info['severity']} severity): {info['description']} "
                f"Actionable Treatment: {info['treatment']} "
                f"Ensure proper plant spacing and sanitation to prevent recurring spores."
            )

    return (
        f"For {plant_name} showing symptoms of {disease_name}, remove and isolate affected foliage immediately. "
        "Avoid overhead irrigation to keep leaves dry, and apply an appropriate organic or chemical fungicide recommended by your local agricultural extension service."
    )


# Chat Prompt
chat_prompt = PromptTemplate(
    input_variables=["history", "human_input"],
    template=(
        "You are an expert agricultural AI assistant called FloraScan AI. "
        "You help farmers diagnose plant diseases and provide treatment advice. "
        "Keep your answers concise, helpful, and friendly.\n\n"
        "{history}\n"
        "Farmer: {human_input}\n"
        "FloraScan AI:"
    )
)

def chat_with_bot(session_id, user_message, user_id=None):
    """Processes a user message and returns the AI's response while saving to DB."""
    # Retrieve history from DB
    history = db.get_chat_history(session_id, limit=5, user_id=user_id)

    # Format history string
    history_str = ""
    for msg in history:
        prefix = "Farmer" if msg["role"] == "user" else "FloraScan AI"
        history_str += f"{prefix}: {msg['content']}\n"

    prompt = chat_prompt.format(history=history_str, human_input=user_message)
    response = None

    # 1. Try Cloud LLM
    response = _call_cloud_llm(prompt)

    # 2. Try Local Ollama
    if not response:
        ollama = get_ollama()
        if ollama:
            try:
                response = ollama.invoke(prompt).strip()
            except Exception:
                pass

    # 3. Built-in Agronomist Assistant Fallback
    if not response:
        lower_msg = user_message.lower()
        matched = False
        for class_key, info in model.DISEASE_KNOWLEDGE_BASE.items():
            plant, disease = class_key.split("___", 1) if "___" in class_key else ("", class_key)
            if (plant.lower() in lower_msg or disease.lower().replace("_", " ") in lower_msg) and not matched:
                response = (
                    f"Regarding {plant.replace('_', ' ')} ({disease.replace('_', ' ')}): "
                    f"{info['description']} Treatment recommendation: {info['treatment']}"
                )
                matched = True
                break

        if not response:
            response = (
                "Hello! I am FloraScan AI, your crop diagnostics assistant. "
                "You can upload an image of a crop leaf to identify diseases like blight, rust, scab, or mildew, "
                "or ask me specific questions about crop treatments and organic fungicides!"
            )

    # Save messages to DB
    db.save_chat_message(session_id, "user", user_message, user_id=user_id)
    db.save_chat_message(session_id, "ai", response, user_id=user_id)

    return response
