import os
from openai import OpenAI

# Ollama OpenAI-compatible endpoint
OLLAMA_BASE_URL = "http://localhost:11434/v1"


def _call_ollama(*, model: str, system: str, user: str):
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return {
        "text": resp.choices[0].message.content,
        "usage": getattr(resp, "usage", None),
    }


def _call_gemini(*, model: str, api_key: str, system: str, user: str):
    # official Gemini SDK: google-genai
    from google import genai

    client = genai.Client(api_key=api_key)
    prompt = f"{system}\n\n{user}"
    resp = client.models.generate_content(model=model, contents=prompt)

    # google-genai doesn't always provide token usage in the same way
    return {
        "text": resp.text,
        "usage": None,
    }


def generate_llm_response(*, config: dict, model: str, persona: str, task: str, context: str, user_query: str):
    system = persona + "\n\n" + task
    user = f"""
[CONTEXT]
{context}

[USER_QUESTION]
{user_query}
""".strip()

    # Gemini-hosted models (Gemini + hosted Gemma)
    if model.startswith("gemini-") or model.startswith("gemma-"):
        api_key = config.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {
                "text": "I don't know from the knowledge graph. Reasons:\n- GEMINI_API_KEY is missing in config.txt/environment.",
                "usage": None,
            }
        return _call_gemini(model=model, api_key=api_key, system=system, user=user)

    # Otherwise: treat as local Ollama model
    return _call_ollama(model=model, system=system, user=user)
