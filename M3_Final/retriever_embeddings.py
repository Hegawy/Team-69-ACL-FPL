from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from functools import lru_cache


class EmbeddingRetriever:
    driver = None
    database = "neo4j"

    MODEL_INFO = {
        "mpnet": {"hf": "all-mpnet-base-v2", "index": "player_index_mpnet"},
        "minilm": {"hf": "all-MiniLM-L6-v2", "index": "player_index_minilm"},
    }

    @staticmethod
    def init_driver(config: dict):
        EmbeddingRetriever.driver = GraphDatabase.driver(
            config["NEO4J_URI"],
            auth=(config["NEO4J_USERNAME"], config["NEO4J_PASSWORD"])
        )
        EmbeddingRetriever.database = config.get("NEO4J_DATABASE", "neo4j")

    @staticmethod
    @lru_cache(maxsize=4)
    def _model(key: str):
        if key not in EmbeddingRetriever.MODEL_INFO:
            raise ValueError(f"Unknown embedding model '{key}'. Use mpnet/minilm.")
        return SentenceTransformer(EmbeddingRetriever.MODEL_INFO[key]["hf"])

    @staticmethod
    def search(user_query: str, embedding_model: str = "mpnet", top_k: int = 5):
        if EmbeddingRetriever.driver is None:
            raise RuntimeError("EmbeddingRetriever not initialized. Call init_driver(config).")

        info = EmbeddingRetriever.MODEL_INFO[embedding_model]
        index_name = info["index"]
        model = EmbeddingRetriever._model(embedding_model)

        query_vec = model.encode(user_query).astype(float).tolist()

        with EmbeddingRetriever.driver.session(database=EmbeddingRetriever.database) as session:
            res = session.run(
                f"""
                CALL db.index.vector.queryNodes('{index_name}', $k, $embedding)
                YIELD node, score
                OPTIONAL MATCH (node)-[:PLAYS_AS]->(pos:Position)
                RETURN node.player_name AS player_name,
                       coalesce(pos.name,"") AS position,
                       node.embedding_text AS embedding_text,
                       score
                ORDER BY score DESC
                """,
                k=top_k,
                embedding=query_vec
            )
            return res.data()
