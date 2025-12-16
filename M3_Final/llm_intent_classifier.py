import json
import re
from openai import OpenAI

ALLOWED_INTENTS = [
    "player_stats",
    "top_players",
    "best_by_position",
    "compare_players",
    "team_fixtures",
    "gameweek_fixtures",
    "team_clean_sheets",
    "player_form",
    "player_discipline",
    "goalkeeper_stats",
    "recommendation",
    "general_query",
]

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def _safe_json_load(s: str):
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def llm_classify_intent(user_query: str, model: str = "llama3.1:8b") -> str:
    system = (
    "You classify the user's intent for an FPL Neo4j QA system. "
    "Return ONLY valid JSON, no commentary. "
    "Tie-break rules: "
    "If a team name is present and the user asks about fixtures/schedule -> team_fixtures unless a specific GW is stated, then gameweek_fixtures. "
    "If the user compares two players -> compare_players. "
    "If best/top + a position is explicitly mentioned -> best_by_position. "
    "If question asks about a single player's cards -> player_discipline. "
    )


    user = f"""
Pick exactly ONE intent from this list:
{ALLOWED_INTENTS}

User query:
{user_query}

Return JSON ONLY:
{{"intent":"..."}}
"""

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )

    raw = resp.choices[0].message.content
    data = _safe_json_load(raw) or {}
    intent = data.get("intent", "general_query")

    if intent not in ALLOWED_INTENTS:
        return "general_query"
    return intent
