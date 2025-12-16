import re

def classify_intent(q: str) -> str:
    q = q.lower()

    # Compare
    if "compare" in q or re.search(r"\bvs\b", q):
        return "compare_players"

    # Best by position MUST come early (it can include GW)
    if any(k in q for k in ["best", "top"]) and any(
        k in q for k in ["defender", "defenders", "midfielder", "midfielders",
                         "forward", "forwards", "striker", "strikers",
                         "goalkeeper", "goalkeepers", "gk", "def", "mid", "fwd"]
    ):
        return "best_by_position"

    # Top players overall
    if re.search(r"\btop\s*\d+\b", q) or "top players" in q or "best players" in q:
        return "top_players"
    if "top ten" in q or "top five" in q:
        return "top_players"

    # Gameweek fixtures (ONLY when user is asking for fixtures)
    if ("gameweek" in q or re.search(r"\bgw\b", q)) and any(
        k in q for k in ["fixture", "fixtures", "match", "matches", "schedule"]
    ):
        return "gameweek_fixtures"

    # Team fixtures
    if any(k in q for k in ["fixture", "fixtures", "schedule", "matches", "next match"]):
        return "team_fixtures"

    if "clean sheet" in q and ("team" in q or "club" in q):
        return "team_clean_sheets"

    if "form" in q:
        return "player_form"

    if any(k in q for k in ["cards", "yellow", "red", "discipline"]):
        return "player_discipline"

    if any(k in q for k in ["saves", "penalties saved", "penalty saved"]):
        return "goalkeeper_stats"

    if any(k in q for k in ["stats", "points", "goals", "assists"]):
        return "player_stats"

    if any(k in q for k in ["recommend", "suggest", "pick", "captain"]):
        return "recommendation"

    # If they ONLY said GW without saying fixtures, treat as general_query
    # (or you can still return gameweek_fixtures; but this avoids wrong routing)
    return "general_query"
