from neo4j import GraphDatabase

class BaselineRetriever:
    driver = None
    database = "neo4j"

    @staticmethod
    def init_driver(config: dict):
        BaselineRetriever.driver = GraphDatabase.driver(
            config["NEO4J_URI"],
            auth=(config["NEO4J_USERNAME"], config["NEO4J_PASSWORD"])
        )
        BaselineRetriever.database = config.get("NEO4J_DATABASE", "neo4j")

    @staticmethod
    def run_query(query: str, params: dict):
        with BaselineRetriever.driver.session(database=BaselineRetriever.database) as session:
            res = session.run(query, params)
            return [r.data() for r in res]
