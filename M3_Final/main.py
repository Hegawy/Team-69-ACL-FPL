import json
import time
from config_loader import load_config
from preprocess import normalize_text
from intent_classifier import classify_intent
from entity_extractor import extract_entities
from cypher_templates import CypherQueryBuilder
from retriever_baseline import BaselineRetriever
from retriever_embeddings import EmbeddingRetriever
from llm_pipeline import generate_llm_response

CONFIG = load_config("config.txt")

# init neo4j drivers once
BaselineRetriever.init_driver(CONFIG)
EmbeddingRetriever.init_driver(CONFIG)

PERSONA = (
    "You are a professional Fantasy Premier League (FPL) assistant. "
    "You answer using ONLY the provided Neo4j Knowledge Graph context. "
    "Use FPL terms like Gameweek, total points, goals, assists, clean sheets."
)

TASK = (
    "Answer the user's question using ONLY the provided CONTEXT.\n"
    "Rules:\n"
    "1) If CONTEXT contains baseline/embedding results relevant to the question, summarize them directly.\n"
    "2) If CONTEXT does NOT contain enough information or the results are empty, reply: "
    "\"I don't know from the knowledge graph.\" Then list the reason(s) briefly.\n"
    "3) Do NOT suggest or write Cypher queries.\n"
    "4) Do NOT ask the user follow-up questions.\n"
    "5) Do NOT invent any players, fixtures, seasons, or stats.\n"
    "6) Do NOT describe form as a percentage, rating, or 'out of 10'\n"
)





DEFAULT_LLMS = [
    "llama3.1:8b",        # local Ollama
    "gemma-3-4b-it",         # hosted via Gemini API key
    "gemini-2.5-flash",   # hosted via Gemini API key
]


def answer_query(user_query: str, *, retrieval_type: str, llm_model: str, embedding_model: str):
    t0 = time.time()
    q_norm = normalize_text(user_query)

    from llm_preprocessor import llm_preprocess
    from entity_extractor import PLAYERS, TEAMS, SEASONS

    VOCAB = {
        "players_lower_map": {p.lower(): p for p in PLAYERS},
        "teams_lower_map": {t.lower(): t for t in TEAMS},
        "seasons_set": set(SEASONS),
    }

    intent, entities = llm_preprocess(user_query, vocab=VOCAB, model="llama3.1:8b")


    cypher, params = CypherQueryBuilder.build(intent, entities)

    # baseline
    t_retr0 = time.time()
    baseline_rows = []
    if retrieval_type in ("Baseline", "Hybrid"):
        baseline_rows = BaselineRetriever.run_query(cypher, params)
    baseline_s = round(time.time() - t_retr0, 3)

    # embeddings
    t_emb0 = time.time()
    embedding_rows = []
    emb_s = 0.0
    if retrieval_type in ("Embedding", "Hybrid"):
        embedding_rows = EmbeddingRetriever.search(user_query, embedding_model=embedding_model, top_k=5)
        emb_s = round(time.time() - t_emb0, 3)

    combined = {
        "intent": intent,
        "entities": entities,
        "cypher": {"query": cypher, "params": params},
        "baseline": baseline_rows,
        "embedding": embedding_rows,
        "timing": {"baseline_s": baseline_s, "embedding_s": emb_s},
    }

    # # LLM (Groq OpenAI-compatible)
    # api_key = CONFIG.get("GROQ_API_KEY", "")
    # if not api_key:
    #     raise RuntimeError("Missing GROQ_API_KEY in config.txt")

    t_llm0 = time.time()
    llm = generate_llm_response(
        config=CONFIG,
        model=llm_model,
        persona=PERSONA,
        task=TASK,
        context=json.dumps(combined, indent=2),
        user_query=user_query,
    )

    llm_s = round(time.time() - t_llm0, 3)

    return {
        "answer": llm["text"],
        "usage": llm["usage"],
        "debug": combined,
        "latency_s": round(time.time() - t0, 3),
        "llm_s": llm_s,
    }
