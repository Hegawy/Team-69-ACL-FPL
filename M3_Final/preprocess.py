import re
from difflib import get_close_matches

TEAM_ALIASES = {
    "man city": "manchester city",
    "man utd": "manchester united",
    "spurs": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "west ham": "west ham united",
    "newcastle": "newcastle united",
    "brighton": "brighton and hove albion",
    "forest": "nottingham forest",
}

POSITION_ALIASES = {
    "gk": "goalkeeper",
    "goalie": "goalkeeper",
    "keeper": "goalkeeper",
    "defs": "defenders",
    "mids": "midfielders",
    "fwds": "forwards",
    "def": "defender",
    "mid": "midfielder",
    "fwd": "forward",
}

def normalize_text(q: str) -> str:
    q = q.strip().lower()
    q = q.replace("’", "'").replace("–", "-").replace("—", "-")
    q = re.sub(r"\s+", " ", q)
    return q

def apply_aliases(q: str) -> tuple[str, list[str]]:
    notes = []
    for a, full in TEAM_ALIASES.items():
        if re.search(r"\b" + re.escape(a) + r"\b", q):
            q = re.sub(r"\b" + re.escape(a) + r"\b", full, q)
            notes.append(f"Interpreting '{a}' as '{full}'")
    for a, full in POSITION_ALIASES.items():
        if re.search(r"\b" + re.escape(a) + r"\b", q):
            q = re.sub(r"\b" + re.escape(a) + r"\b", full, q)
            notes.append(f"Interpreting '{a}' as '{full}'")
    return q, notes

def extract_phrases(q: str):
    tokens = re.findall(r"[a-z']+", q)
    phrases = set(tokens)
    for i in range(len(tokens) - 1):
        phrases.add(tokens[i] + " " + tokens[i + 1])
    for i in range(len(tokens) - 2):
        phrases.add(tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2])
    return sorted(phrases, key=len, reverse=True)

def fuzzy_pick(term: str, candidates: list[str], cutoff=0.84):
    hits = get_close_matches(term, candidates, n=1, cutoff=cutoff)
    return hits[0] if hits else None
