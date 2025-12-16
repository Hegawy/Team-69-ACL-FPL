import os
import re
import pandas as pd

from preprocess import normalize_text, apply_aliases, extract_phrases, fuzzy_pick

# -----------------------------
# Load CSV (used for candidates)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "fpl_two_seasons.csv")

df = pd.read_csv(CSV_PATH)

# Prefer node-like naming; fall back to common CSV naming
PLAYER_COL = "player_name" if "player_name" in df.columns else ("name" if "name" in df.columns else None)
SEASON_COL = "season_name" if "season_name" in df.columns else ("season" if "season" in df.columns else None)
GW_COL = "GW_number" if "GW_number" in df.columns else ("GW" if "GW" in df.columns else None)

if PLAYER_COL is None or SEASON_COL is None:
    raise ValueError("CSV must contain player_name/name AND season_name/season columns.")

TEAM_COLUMNS = [
    c for c in ["home_team", "away_team", "home_team_name", "away_team_name", "team", "team_name"]
    if c in df.columns
]

PLAYERS = set(df[PLAYER_COL].dropna().unique())
SEASONS = set(df[SEASON_COL].dropna().unique())

TEAMS = set()
for c in TEAM_COLUMNS:
    TEAMS |= set(df[c].dropna().unique())

GAMEWEEKS = set(df[GW_COL].dropna().astype(str).unique()) if GW_COL and GW_COL in df.columns else set()

POSITIONS = {"GKP", "DEF", "MID", "FWD"}

STAT_KEYWORDS = {
    "assists", "bonus", "bps", "clean sheets", "clean_sheets",
    "creativity", "form", "goals conceded", "goals_conceded",
    "goals scored", "goals_scored", "ict index", "ict_index",
    "influence", "minutes", "own goals", "own_goals",
    "penalties missed", "penalties_missed", "penalties saved", "penalties_saved",
    "red cards", "red_cards", "saves", "threat",
    "total points", "total_points", "yellow cards", "yellow_cards",
    "points", "goals", "stats"
}


# -----------------------------
# Helpers: season parsing
# -----------------------------
def _to_short_season(y1: int, y2: int) -> str:
    return f"{y1}-{str(y2)[-2:]}"

def _extract_season(q: str):
    found = []

    # 2022-23 / 2022/23 / 2022–23
    for y1, y2 in re.findall(r"\b(20\d{2})\s*[-/–]\s*(\d{2})\b", q):
        found.append(f"{y1}-{y2}")

    # 2022-2023
    for y1, y2 in re.findall(r"\b(20\d{2})\s*[-/–]\s*(20\d{2})\b", q):
        found.append(_to_short_season(int(y1), int(y2)))

    # "2023 season" (pick 2023-24 if exists else 2022-23 if exists)
    if "season" in q and not found:
        years = re.findall(r"\b(20\d{2})\b", q)
        if years:
            y = int(years[0])
            cand1 = _to_short_season(y, y + 1)
            cand2 = _to_short_season(y - 1, y)
            if cand1 in SEASONS:
                found.append(cand1)
            elif cand2 in SEASONS:
                found.append(cand2)
            else:
                found.append(cand1)

    # keep only existing seasons (if known)
    found = [s for s in dict.fromkeys(found) if (not SEASONS or s in SEASONS)]
    return found


# -----------------------------
# Helpers: "A vs B" parsing
# -----------------------------
def extract_two_player_candidates(q: str):
    """
    Extract raw left/right strings in:
      - "A vs B"
      - "A v B"
      - "compare A and B"
      - "compare A vs B"
    """
    m = re.search(r"(?:compare\s+)?(.+?)\s+(?:vs|v|versus|and)\s+(.+)", q)
    if not m:
        return None, None

    left = m.group(1).strip()
    right = m.group(2).strip()

    # Trim common trailing stuff from right side (season/gw words)
    right = re.split(r"\b(season|gw|gameweek|20\d{2})\b", right, maxsplit=1)[0].strip()

    # Avoid very short junk
    if len(left) < 3 or len(right) < 3:
        return None, None

    return left, right


# -----------------------------
# Main extraction
# -----------------------------
def extract_entities(user_query: str):
    q = normalize_text(user_query)
    q, notes = apply_aliases(q)

    # Lowercase maps for fuzzy picking
    player_lower = {p.lower(): p for p in PLAYERS}
    team_lower = {t.lower(): t for t in TEAMS}

    found_players = []
    found_teams = []

    # -------------------------
    # 1) Positions (NL + codes)
    # -------------------------
    pos_map = {
        "defender": "DEF", "defenders": "DEF",
        "midfielder": "MID", "midfielders": "MID",
        "forward": "FWD", "forwards": "FWD",
        "striker": "FWD", "strikers": "FWD",
        "goalkeeper": "GKP", "goalkeepers": "GKP",
    }

    found_positions = []
    for k, v in pos_map.items():
        if re.search(r"\b" + re.escape(k) + r"\b", q):
            found_positions.append(v)

    for code in POSITIONS:
        if re.search(r"\b" + re.escape(code.lower()) + r"\b", q):
            found_positions.append(code)

    found_positions = list(dict.fromkeys(found_positions))

    # -------------
    # 2) Seasons
    # -------------
    found_seasons = _extract_season(q)

    # -----------------
    # 3) Gameweeks
    # -----------------
    # Matches: gw5, gw 5, GW05, gameweek 10, gameweek10
    gws = re.findall(r"\b(?:gw|gameweek)\s*0*(\d+)\b", q)
    found_gameweeks = []
    for gw in gws:
        if not GAMEWEEKS or gw in GAMEWEEKS:
            found_gameweeks.append(gw)
    found_gameweeks = list(dict.fromkeys(found_gameweeks))

    # -----------------
    # 4) Stats keywords
    # -----------------
    found_stats = []
    for s in STAT_KEYWORDS:
        if " " in s and s in q:
            found_stats.append(s)
        elif " " not in s and re.search(r"\b" + re.escape(s) + r"\b", q):
            found_stats.append(s)
    found_stats = list(dict.fromkeys(found_stats))

    # -----------------------------------------
    # 5) Player extraction (support comparisons)
    # -----------------------------------------
    left, right = extract_two_player_candidates(q)
    if left and right:
        # fuzzy match both
        p1_key = fuzzy_pick(left, list(player_lower.keys()), cutoff=0.80)
        p2_key = fuzzy_pick(right, list(player_lower.keys()), cutoff=0.80)
        if p1_key and p2_key:
            found_players = [player_lower[p1_key], player_lower[p2_key]]
            if left != p1_key:
                notes.append(f"Interpreting '{left}' as player '{player_lower[p1_key]}'")
            if right != p2_key:
                notes.append(f"Interpreting '{right}' as player '{player_lower[p2_key]}'")

    # If not comparison or didn't find two, fallback to single-player logic
    if not found_players:
        # Exact full-name substring match first
        exact_matches = [p for p in PLAYERS if p.lower() in q]
        if exact_matches:
            found_players = [max(exact_matches, key=len)]
        else:
            # Fuzzy by phrase overlap
            phrases = extract_phrases(q)
            for ph in phrases:
                hit = fuzzy_pick(ph, list(player_lower.keys()), cutoff=0.86)
                if hit:
                    found_players = [player_lower[hit]]
                    if ph != hit:
                        notes.append(f"Interpreting '{ph}' as player '{player_lower[hit]}'")
                    break

    # --------------------------
    # 6) Team extraction (fuzzy)
    # --------------------------
    if TEAMS:
        exact_t = [t for t in TEAMS if t.lower() in q]
        if exact_t:
            found_teams = [max(exact_t, key=len)]
        else:
            phrases = extract_phrases(q)
            for ph in phrases:
                hit = fuzzy_pick(ph, list(team_lower.keys()), cutoff=0.84)
                if hit:
                    found_teams = [team_lower[hit]]
                    if ph != hit:
                        notes.append(f"Interpreting '{ph}' as team '{team_lower[hit]}'")
                    break

    return {
        "players": found_players,
        "teams": found_teams,
        "seasons": found_seasons,
        "positions": found_positions,
        "gameweeks": found_gameweeks,
        "statistics": found_stats,
        "normalization_notes": notes,
    }
