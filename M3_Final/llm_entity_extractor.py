import json
import re
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def _safe_json_load(s: str):
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _normalize_season(s: str) -> str:
    s = str(s).strip()
    # 2022-2023 -> 2022-23
    m = re.match(r"^(20\d{2})\s*[-/–]\s*(20\d{2})$", s)
    if m:
        y1 = int(m.group(1)); y2 = int(m.group(2))
        return f"{y1}-{str(y2)[-2:]}"
    # 2022-23
    m2 = re.match(r"^(20\d{2})\s*[-/–]\s*(\d{2})$", s)
    if m2:
        return f"{m2.group(1)}-{m2.group(2)}"
    return s


def _normalize_team_key(key: str) -> str:
    """
    Light normalization BEFORE vocab lookup.
    Keeps things simple: we still only accept teams that exist in vocab maps.
    """
    k = key.strip().lower()
    k = re.sub(r"[^\w\s'&]", " ", k)      # drop punctuation
    k = re.sub(r"\s+", " ", k).strip()    # normalize spaces

    # Common EPL aliases (safe; will still be validated by vocab)
    alias = {
        "man city": "manchester city",
        "man utd": "manchester united",
        "spurs": "tottenham hotspur",
        "wolves": "wolverhampton wanderers",
        "newcastle": "newcastle united",
        "nottm forest": "nott'm forest",
        "nottingham forest": "nott'm forest",
        "brighton": "brighton and hove albion",
    }
    return alias.get(k, k)


def _to_text(x):
    # Accept strings OR dicts like {"name": "..."} OR {"value": "..."}
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("name", "value", "text", "label"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return str(x)



def llm_extract_entities(user_query: str, vocab: dict, model: str = "llama3.1:8b") -> dict:
    """
    vocab:
      players_lower_map: lower->canonical
      teams_lower_map: lower->canonical
      seasons_set: set
    """
    notes = []

    system = (
        "You extract entities for an FPL Neo4j QA system. "
        "Return ONLY valid JSON, no commentary."
    )

    user = f"""
Extract entities from the user query.

User query:
{user_query}

Return JSON ONLY with this structure:
{{
  "players": [],
  "teams": [],
  "seasons": [],
  "positions": [],
  "gameweeks": []
}}

Rules:
- positions must be one of ["GKP","DEF","MID","FWD"] (convert from gk/goalkeeper/defender/etc.)
- gameweeks must be numbers (like 7)
- seasons must be like "2022-23" or "2022-2023"
"""

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )

    raw = resp.choices[0].message.content
    data = _safe_json_load(raw) or {}

    players_in = [_to_text(x) for x in (data.get("players", []) or [])]
    teams_in = [_to_text(x) for x in (data.get("teams", []) or [])]
    seasons_in = [_to_text(x) for x in (data.get("seasons", []) or [])]
    positions_in = [_to_text(x) for x in (data.get("positions", []) or [])]
    gws_in = [_to_text(x) for x in (data.get("gameweeks", []) or [])]


    # Seasons
    seasons_out = []
    for s in seasons_in:
        s2 = _normalize_season(s)
        if s2 in vocab["seasons_set"]:
            seasons_out.append(s2)
        else:
            notes.append(f"Dropped unknown season '{s}'.")

    # Positions
    pos_alias = {
        "GK": "GKP", "GOALKEEPER": "GKP", "KEEPER": "GKP",
        "DEFENDER": "DEF", "DEF": "DEF",
        "MIDFIELDER": "MID", "MID": "MID",
        "FORWARD": "FWD", "STRIKER": "FWD", "FWD": "FWD",
        "GKP": "GKP",
    }
    positions_out = []
    for p in positions_in:
        p2 = pos_alias.get(str(p).strip().upper(), str(p).strip().upper())
        if p2 in {"GKP", "DEF", "MID", "FWD"}:
            positions_out.append(p2)
        else:
            notes.append(f"Dropped unknown position '{p}'.")

    # Gameweeks
    gws_out = []
    for g in gws_in:
        m = re.search(r"\d+", str(g))
        if m:
            gws_out.append(str(int(m.group(0))))
        else:
            notes.append(f"Dropped invalid gameweek '{g}'.")

    # Canonical maps (anti-hallucination)
    pl_map = vocab["players_lower_map"]
    tm_map = vocab["teams_lower_map"]

    players_out = []
    for p in players_in:
        key = str(p).strip().lower()

        # 1) Exact full-name match
        if key in pl_map:
            players_out.append(pl_map[key])
            continue

        # 2) Unique first-name match (safe fallback)
        candidates = [
            full_name for lname, full_name in pl_map.items()
            if lname.split()[0] == key
        ]
        if len(candidates) == 1:
            players_out.append(candidates[0])
            notes.append(f"Resolved '{p}' to '{candidates[0]}' by unique first-name match.")
            continue

        # 3) Fail safely
        notes.append(f"Unrecognized or ambiguous player '{p}'.")


    teams_out = []
    for t in teams_in:
        key = _normalize_team_key(str(t))
        if key in tm_map:
            teams_out.append(tm_map[key])
        else:
            notes.append(f"Unrecognized team '{t}' (not in dataset).")

    def dedupe(xs):
        return list(dict.fromkeys(xs))

    return {
        "players": dedupe(players_out),
        "teams": dedupe(teams_out),
        "seasons": dedupe(seasons_out),
        "positions": dedupe(positions_out),
        "gameweeks": dedupe(gws_out),
        "statistics": [],
        "normalization_notes": notes,
    }
