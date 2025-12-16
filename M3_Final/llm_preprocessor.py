import re
from llm_intent_classifier import llm_classify_intent
from llm_entity_extractor import llm_extract_entities


def llm_preprocess(user_query: str, vocab: dict, model: str = "llama3.1:8b"):
    """
    LLM-first preprocessing + deterministic alignment rules to prevent intent mixups.
    """
    q = user_query.lower().strip()

    intent = llm_classify_intent(user_query, model=model)
    entities = llm_extract_entities(user_query, vocab=vocab, model=model)

    players = entities.get("players", []) or []
    teams = entities.get("teams", []) or []
    seasons = entities.get("seasons", []) or []
    positions = entities.get("positions", []) or []
    gameweeks = entities.get("gameweeks", []) or []

    # -----------------------------
    # Keyword detectors (regex-safe)
    # -----------------------------
    has_vs = bool(re.search(r"\bvs\b", q)) or ("compare" in q)
    has_gw = bool(re.search(r"\bgw\b", q)) or ("gameweek" in q)
    has_fixture_words = any(k in q for k in ["fixture", "fixtures", "schedule", "match", "matches", "game", "games", "next match"])
    has_top_best = any(k in q for k in ["top", "best"])
    has_clean_sheet = "clean sheet" in q or "clean sheets" in q
    has_form = "form" in q
    has_discipline = any(k in q for k in ["cards", "card", "yellow", "red", "discipline"])
    has_keeper_stats = any(k in q for k in ["saves", "penalties saved", "penalty saved", "penalties_saved", "penalties_saved"])
    has_reco = any(k in q for k in ["recommend", "suggest", "pick", "captain"])

    explicit_pos_words = [
        "defender", "defenders",
        "midfielder", "midfielders",
        "forward", "forwards",
        "striker", "strikers",
        "goalkeeper", "goalkeepers",
        "gk"
    ]

    # -----------------------------
    # HARD OVERRIDES (prevent mixups)
    # -----------------------------

    # A) Compare: if user clearly compares and we have >=2 players, force compare_players
    if has_vs and len(players) >= 2:
        intent = "compare_players"

    # B) Team fixtures vs gameweek fixtures (main bug fix)
    # - If GW is specified -> gameweek_fixtures (optionally filtered by team in Cypher)
    # - Else if team + fixture-ish words -> team_fixtures
    # Priority: GW beats team_fixtures
    if (gameweeks or has_gw) and has_fixture_words:
        intent = "gameweek_fixtures"
    elif teams and has_fixture_words:
        intent = "team_fixtures"

    


    # C) Best/top routing:
    # - If a position is explicitly requested -> best_by_position
    # - Otherwise -> top_players
    pos_requested = any(w in q for w in explicit_pos_words) or bool(positions)

    if has_top_best:
        if pos_requested and positions:
            intent = "best_by_position"
        else:
            intent = "top_players"

    # If the query does NOT mention any position words, ignore any extracted positions.
    if has_top_best and not any(w in q for w in explicit_pos_words):
        entities["positions"] = []
        positions = []
        if intent == "best_by_position":
            intent = "top_players"




    # Clean sheets → PLAYER clean sheets (requires a player)
    if has_clean_sheet and players:
        intent = "player_clean_sheets"


    # E) Player form requires a player (otherwise it's ambiguous)
    if has_form and players:
        intent = "player_form"
    elif intent == "player_form" and not players:
        intent = "general_query"

    # F) Player discipline requires a player
    if has_discipline and players:
        intent = "player_discipline"
    elif intent == "player_discipline" and not players:
        intent = "general_query"

    # G) Goalkeeper stats requires a player (we don't assume team -> keeper)
    if has_keeper_stats and players:
        intent = "goalkeeper_stats"
    elif intent == "goalkeeper_stats" and not players:
        intent = "general_query"

    # H) Recommendation: if recommend-like phrasing, keep recommendation
    # If also position exists, great. If not, still allow recommendation (LLM answer should say missing position).
    if has_reco:
        intent = "recommendation"

    # I) Player stats: if question is clearly about points/goals/assists/stats and has a player
    if any(k in q for k in ["stats", "points", "goals", "assists", "total points"]) and players:
        # don't override compare/team fixtures
        if intent not in ["compare_players", "team_fixtures", "gameweek_fixtures"]:
            intent = "player_stats"

    return intent, entities
