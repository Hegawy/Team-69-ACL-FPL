#!/usr/bin/env python3
"""
LLM evaluation runner for Milestone 3 (FPL KG RAG)
--------------------------------------------------
Runs a fixed set of questions across:
- Retrieval: Baseline / Embedding / Hybrid
- LLMs: local Ollama + Gemini models (as configured in your project)

It computes:
- PASS/FAIL (simple automatic checks vs CSV-derived ground truth)
- latency + LLM latency
- token usage (if provided by the provider)
- retrieval timing (baseline_s / embedding_s)

Usage:
  (venv) python eval_llms.py

Requirements:
- Your project files are in the same folder (main.py, config.txt, etc.)
- Ollama is running (for llama3.1:8b)
- Gemini API key is in config.txt (if you test Gemini/Gemma)
"""

import re
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ---- import your pipeline ----
from main import answer_query, DEFAULT_LLMS  # uses your existing config + drivers


CSV_PATH = Path("fpl_two_seasons.csv")  # put CSV next to this script
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


# -----------------------------
# Ground truth from CSV
# -----------------------------
def load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["season"] = df["season"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["position"] = df["position"].astype(str).str.strip()
    return df


def player_agg(df: pd.DataFrame, *, name: str, season: str) -> Dict[str, int]:
    d = df[(df["name"] == name) & (df["season"] == season)]
    if d.empty:
        return {}
    return {
        "total_points": int(d["total_points"].sum()),
        "goals": int(d["goals_scored"].sum()),
        "assists": int(d["assists"].sum()),
        "clean_sheets": int(d["clean_sheets"].sum()),
        "saves": int(d["saves"].sum()),
        "penalties_saved": int(d["penalties_saved"].sum()),
        "yellow": int(d["yellow_cards"].sum()),
        "red": int(d["red_cards"].sum()),
    }


def top_players(df: pd.DataFrame, *, season: str, n: int = 10) -> List[Tuple[str, int]]:
    d = (
        df[df["season"] == season]
        .groupby("name")["total_points"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )
    return list(zip(d.index.tolist(), d.astype(int).tolist()))


def fixtures_for_team(df: pd.DataFrame, *, team: str, season: str) -> pd.DataFrame:
    fx = (
        df[df["season"] == season][["GW", "home_team", "away_team", "kickoff_time", "fixture"]]
        .drop_duplicates()
    )
    return fx[(fx["home_team"] == team) | (fx["away_team"] == team)].sort_values(["GW", "kickoff_time"])


def fixtures_for_gw(df: pd.DataFrame, *, season: str, gw: int) -> pd.DataFrame:
    fx = (
        df[df["season"] == season][["GW", "home_team", "away_team", "kickoff_time", "fixture"]]
        .drop_duplicates()
    )
    return fx[fx["GW"] == gw].sort_values(["kickoff_time"])


# -----------------------------
# Simple evaluators
# -----------------------------
def contains_all(text: str, items: List[str]) -> bool:
    t = (text or "").lower()
    return all(str(x).lower() in t for x in items)


def says_dont_know(text: str) -> bool:
    return "i don't know from the knowledge graph" in (text or "").lower()


def eval_player_stats(df: pd.DataFrame, answer: str, *, name: str, season: str) -> bool:
    gt = player_agg(df, name=name, season=season)
    if not gt:
        return says_dont_know(answer)
    return contains_all(answer, [name, str(gt["total_points"]), str(gt["goals"]), str(gt["assists"])])


def eval_top_players(df: pd.DataFrame, answer: str, *, season: str) -> bool:
    top = top_players(df, season=season, n=10)
    if not top:
        return says_dont_know(answer)
    n1, p1 = top[0]
    return contains_all(answer, [n1, str(p1)])


def eval_best_by_position(df: pd.DataFrame, answer: str, *, season: str, pos: str) -> bool:
    d = (
        df[(df["season"] == season) & (df["position"] == pos)]
        .groupby("name")["total_points"]
        .sum()
        .sort_values(ascending=False)
    )
    if d.empty:
        return says_dont_know(answer)
    n1, p1 = d.index[0], int(d.iloc[0])
    return contains_all(answer, [n1, str(p1)])


def eval_compare_players(df: pd.DataFrame, answer: str, *, season: str, p1: str, p2: str) -> bool:
    g1 = player_agg(df, name=p1, season=season)
    g2 = player_agg(df, name=p2, season=season)
    if not g1 or not g2:
        return says_dont_know(answer)
    return contains_all(answer, [p1, p2]) and (str(g1["total_points"]) in answer or str(g2["total_points"]) in answer)


def eval_team_fixtures(df: pd.DataFrame, answer: str, *, team: str, season: str) -> bool:
    fx = fixtures_for_team(df, team=team, season=season)
    if fx.empty:
        return says_dont_know(answer)
    row0 = fx.iloc[0]
    opponent = row0["away_team"] if row0["home_team"] == team else row0["home_team"]
    return contains_all(answer, [team, opponent])


def eval_gameweek_fixtures(df: pd.DataFrame, answer: str, *, season: str, gw: int) -> bool:
    fx = fixtures_for_gw(df, season=season, gw=gw)
    if fx.empty:
        return says_dont_know(answer)
    r0 = fx.iloc[0]
    return contains_all(answer, [str(gw), r0["home_team"], r0["away_team"]])


def eval_player_clean_sheets(df: pd.DataFrame, answer: str, *, name: str, season: str) -> bool:
    gt = player_agg(df, name=name, season=season)
    if not gt:
        return says_dont_know(answer)
    return contains_all(answer, [name, str(gt["clean_sheets"])])


def eval_player_form(df: pd.DataFrame, answer: str, *, name: str, season: str) -> bool:
    gt = player_agg(df, name=name, season=season)
    if not gt:
        return says_dont_know(answer)
    return name.lower() in (answer or "").lower() and bool(re.search(r"\d+(\.\d+)?", answer or ""))


def eval_player_discipline(df: pd.DataFrame, answer: str, *, name: str, season: str) -> bool:
    gt = player_agg(df, name=name, season=season)
    if not gt:
        return says_dont_know(answer)
    return contains_all(answer, [name, str(gt["yellow"]), str(gt["red"])])


def eval_goalkeeper_stats(df: pd.DataFrame, answer: str, *, name: str, season: str) -> bool:
    gt = player_agg(df, name=name, season=season)
    if not gt:
        return says_dont_know(answer)
    return contains_all(answer, [name, str(gt["saves"])])


# -----------------------------
# Test suite aligned with intents
# -----------------------------
def build_tests(df: pd.DataFrame):
    season = "2022-23"

    return [
        # 1) player_stats
        {
            "name": "player_stats_working",
            "query": "Show Mohamed Salah’s stats in the 2022-23 season.",
            "expect": lambda ans: eval_player_stats(df, ans, name="Mohamed Salah", season=season),
        },
        {
            "name": "player_stats_nonworking",
            "query": "Show Lionel Messi’s stats in the 2022-23 season.",
            "expect": lambda ans: says_dont_know(ans),
        },

        # 2) top_players
        {
            "name": "top_players_working",
            "query": "Who are the top 10 players in the 2022-23 season?",
            "expect": lambda ans: eval_top_players(df, ans, season=season),
        },
        {
            "name": "top_players_nonworking",
            "query": "Who are the top 10 players in the 2018-19 season?",
            "expect": lambda ans: says_dont_know(ans),
        },

        # 3) best_by_position
        {
            "name": "best_by_position_working",
            "query": "Who are the best defenders in the 2022-23 season?",
            "expect": lambda ans: eval_best_by_position(df, ans, season=season, pos="DEF"),
        },
        {
            "name": "best_by_position_nonworking",
            "query": "Who are the best wing-backs in the 2022-23 season?",
            "expect": lambda ans: says_dont_know(ans),
        },

        # 4) compare_players
        {
            "name": "compare_players_working",
            "query": "Compare Mohamed Salah vs Kevin De Bruyne in the 2022-23 season.",
            "expect": lambda ans: eval_compare_players(df, ans, season=season, p1="Mohamed Salah", p2="Kevin De Bruyne"),
        },
        {
            "name": "compare_players_nonworking",
            "query": "Compare Mohamed Salah vs Cristiano Ronaldo in the 2022-23 season.",
            "expect": lambda ans: says_dont_know(ans),
        },

        # 5) team_fixtures
        {
            "name": "team_fixtures_working",
            "query": "Show Liverpool’s fixtures in the 2022-23 season.",
            "expect": lambda ans: eval_team_fixtures(df, ans, team="Liverpool", season=season),
        },
        {
            "name": "team_fixtures_nonworking",
            "query": "Show Barcelona’s fixtures in the 2022-23 season.",
            "expect": lambda ans: says_dont_know(ans),
        },

        # 6) gameweek_fixtures
        {
            "name": "gameweek_fixtures_working",
            "query": "Show fixtures in GW 8 of the 2022-23 season.",
            "expect": lambda ans: eval_gameweek_fixtures(df, ans, season=season, gw=8),
        },
        {
            "name": "gameweek_fixtures_nonworking",
            "query": "Show fixtures in GW 50 of the 2022-23 season.",
            "expect": lambda ans: says_dont_know(ans),
        },

        # 7) player_clean_sheets
        {
            "name": "player_clean_sheets_working",
            "query": "How many clean sheets did Alisson Becker keep in the 2022-23 season?",
            "expect": lambda ans: eval_player_clean_sheets(df, ans, name="Alisson Becker", season=season),
        },
        {
            "name": "player_clean_sheets_nonworking",
            "query": "How many clean sheets did Lionel Messi keep in the 2022-23 season?",
            "expect": lambda ans: says_dont_know(ans),
        },

        # 8) player_form
        {
            "name": "player_form_working",
            "query": "What was Mohamed Salah’s form in the 2022-23 season?",
            "expect": lambda ans: eval_player_form(df, ans, name="Mohamed Salah", season=season),
        },
        {
            "name": "player_form_nonworking",
            "query": "What was Mohamed Salah’s form in the 2019-20 season?",
            "expect": lambda ans: says_dont_know(ans),
        },

        # 9) player_discipline
        {
            "name": "player_discipline_working",
            "query": "How many yellow and red cards did Mateo Kovacic receive in 2022-23?",
            "expect": lambda ans: eval_player_discipline(df, ans, name="Mateo Kovacic", season=season),
        },
        {
            "name": "player_discipline_nonworking",
            "query": "How many fouls did Ben Davies commit in 2022-23?",
            "expect": lambda ans: says_dont_know(ans),
        },

        # 10) goalkeeper_stats
        {
            "name": "goalkeeper_stats_working",
            "query": "How many saves did Alisson Becker make in the 2022-23 season?",
            "expect": lambda ans: eval_goalkeeper_stats(df, ans, name="Alisson Becker", season=season),
        },
        {
            "name": "goalkeeper_stats_nonworking",
            "query": "How many saves did Mohamed Salah make in the 2022-23 season?",
            "expect": lambda ans: says_dont_know(ans),
        },
    ]


# -----------------------------
# Runner
# -----------------------------
def run_one(query: str, *, retrieval: str, llm_model: str, embedding_model: str) -> Dict[str, Any]:
    return answer_query(
        query,
        retrieval_type=retrieval,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )


def main():
    df = load_df()
    tests = build_tests(df)

    llms = DEFAULT_LLMS
    retrievals = ["Baseline", "Embedding", "Hybrid"]
    embedding_models = ["mpnet", "minilm"]

    rows = []

    for retrieval in retrievals:
        for emb_model in embedding_models:
            for llm in llms:
                for t in tests:
                    q = t["query"]

                    t0 = time.time()
                    out = run_one(q, retrieval=retrieval, llm_model=llm, embedding_model=emb_model)
                    dt = round(time.time() - t0, 3)

                    ans = out.get("answer", "") or ""
                    passed = bool(t["expect"](ans))

                    usage = out.get("usage")
                    if usage and not isinstance(usage, dict):
                        usage = {
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                            "total_tokens": getattr(usage, "total_tokens", None),
                        }

                    debug = out.get("debug", {}) or {}
                    timing = (debug.get("timing") or {})

                    rows.append({
                        "test": t["name"],
                        "query": q,
                        "retrieval": retrieval,
                        "embedding_model": emb_model,
                        "llm_model": llm,
                        "pass": passed,
                        "total_latency_s": out.get("latency_s", dt),
                        "llm_s": out.get("llm_s"),
                        "baseline_s": timing.get("baseline_s"),
                        "embedding_s": timing.get("embedding_s"),
                        "prompt_tokens": (usage or {}).get("prompt_tokens"),
                        "completion_tokens": (usage or {}).get("completion_tokens"),
                        "total_tokens": (usage or {}).get("total_tokens"),
                        "answer_preview": (ans[:180] + "...") if len(ans) > 180 else ans,
                    })

                    print(f"[{retrieval} | {emb_model} | {llm}] {t['name']}: {'PASS' if passed else 'FAIL'}")

    res = pd.DataFrame(rows)

    summary = (
        res.groupby(["retrieval", "embedding_model", "llm_model"])["pass"]
        .mean()
        .reset_index()
        .rename(columns={"pass": "accuracy"})
        .sort_values(["retrieval", "embedding_model", "accuracy"], ascending=[True, True, False])
    )

    out_csv = REPORT_DIR / "llm_eval_detailed.csv"
    out_sum = REPORT_DIR / "llm_eval_summary.csv"
    res.to_csv(out_csv, index=False)
    summary.to_csv(out_sum, index=False)

    print("\nSaved:")
    print(f"- {out_csv}")
    print(f"- {out_sum}")


if __name__ == "__main__":
    main()
