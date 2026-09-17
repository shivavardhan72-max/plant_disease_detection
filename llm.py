import os
import json
import urllib.request
import urllib.error
from langchain_core.prompts import PromptTemplate
import db
import model

from dotenv import load_dotenv
load_dotenv()

# Hugging Face Configuration (Qwen Model on Serverless Inference Router)
HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HF_TOKEN")
HF_MODEL = os.environ.get("HUGGINGFACE_MODEL", "Qwen/Qwen2.5-72B-Instruct")
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

# Additional Optional Providers
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

_ollama_client = None

def get_ollama():
    """Lazily load Ollama client if available locally."""
    global _ollama_client
    if _ollama_client is None:
        try:
            from langchain_community.llms import Ollama
            _ollama_client = Ollama(model="llama3.2:1b", base_url=OLLAMA_HOST, timeout=3)
        except Exception:
            _ollama_client = False
    return _ollama_client if _ollama_client is not False else None


def _call_hf_qwen(messages: list, max_tokens: int = 350, temperature: float = 0.6) -> str:
    """Invokes Hugging Face Serverless Router using Qwen model."""
    token = os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HF_TOKEN") or HF_TOKEN
    model_name = os.environ.get("HUGGINGFACE_MODEL") or HF_MODEL
    if not token:
        return None

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    req = urllib.request.Request(
        HF_ROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                return content.strip()
    except Exception as e:
        print(f"Hugging Face Qwen call error: {e}")
        return None


def _call_cloud_llm(prompt: str) -> str:
    """Fallback to Groq or OpenAI if configured."""
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
        print(f"Secondary Cloud LLM call failed: {e}")
        return None


def get_recommendation(plant_name: str, disease_name: str) -> str:
    """Generates an agronomic recommendation using Hugging Face Qwen or fallbacks."""
    if "Healthy" in disease_name or "healthy" in disease_name:
        return "Your plant appears to be healthy! Maintain appropriate soil moisture, balanced nutrition, and regular scouting for early pest detection."

    # 1. Primary: Hugging Face Qwen Model
    messages = [
        {
            "role": "system",
            "content": (
                "You are FloraScan AI, an expert agricultural pathologist and agronomist. "
                "Provide a concise, practical, and actionable recommendation (3-4 sentences) "
                "explaining how to treat and prevent this plant disease."
            )
        },
        {
            "role": "user",
            "content": f"A farmer's {plant_name} plant has been diagnosed with {disease_name}. What is your treatment and prevention plan?"
        }
    ]

    hf_response = _call_hf_qwen(messages, max_tokens=250, temperature=0.5)
    if hf_response:
        return hf_response

    # 2. Secondary: Other Cloud LLM (Groq / OpenAI)
    prompt_text = (
        f"You are an expert agricultural assistant. A farmer has a {plant_name} with {disease_name}. "
        "Provide a short, actionable recommendation (3-4 sentences) on how to treat and prevent this disease."
    )
    cloud_resp = _call_cloud_llm(prompt_text)
    if cloud_resp:
        return cloud_resp

    # 3. Tertiary: Local Ollama (if running on local machine)
    ollama = get_ollama()
    if ollama:
        try:
            resp = ollama.invoke(prompt_text)
            if resp:
                return resp.strip()
        except Exception:
            pass

    # 4. Built-in Scientific Knowledge Base Fallback
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


def chat_with_bot(session_id: str, user_message: str, user_id=None) -> str:
    """Processes a user message and returns the AI's response while saving to DB."""
    # Retrieve conversation history from DB
    history = db.get_chat_history(session_id, limit=6, user_id=user_id)

    # Build standard OpenAI-compatible messages payload for Qwen
    messages = [
        {
            "role": "system",
            "content": (
                "You are FloraScan AI, an expert agricultural pathologist and crop diagnostic assistant. "
                "You help farmers diagnose crop diseases, identify foliar symptoms, and provide clear, "
                "actionable organic and chemical treatment advice. Keep your answers friendly, practical, "
                "and concise."
            )
        }
    ]

    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    response = None

    # 1. Primary: Hugging Face Qwen Model
    response = _call_hf_qwen(messages, max_tokens=350, temperature=0.6)

    # 2. Secondary: Other Cloud LLM (Groq / OpenAI)
    if not response:
        history_str = ""
        for msg in history:
            prefix = "Farmer" if msg["role"] == "user" else "FloraScan AI"
            history_str += f"{prefix}: {msg['content']}\n"
        fallback_prompt = f"Chat history:\n{history_str}\nFarmer: {user_message}\nFloraScan AI:"
        response = _call_cloud_llm(fallback_prompt)

    # 3. Tertiary: Local Ollama
    if not response:
        ollama = get_ollama()
        if ollama:
            try:
                response = ollama.invoke(fallback_prompt).strip()
            except Exception:
                pass

    # 4. Built-in Agronomist Assistant Fallback
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
                "Hello! I am FloraScan AI, your crop diagnostics assistant powered by Qwen. "
                "You can upload an image of a crop leaf to identify diseases, "
                "or ask me specific questions about crop treatments, organic fungicides, and soil health!"
            )

    # Save messages to DB
    try:
        db.save_chat_message(session_id, "user", user_message, user_id=user_id)
        db.save_chat_message(session_id, "ai", response, user_id=user_id)
    except Exception as e:
        print(f"Notice: Failed saving chat message to DB: {e}")

    return response
