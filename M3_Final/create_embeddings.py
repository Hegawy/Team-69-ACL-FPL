from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from config_loader import load_config


def _safe_num(x, default=0):
    return default if x is None else x


def _round(x, nd=2):
    if x is None:
        return 0
    try:
        return round(float(x), nd)
    except Exception:
        return 0


def main():
    cfg = load_config("config.txt")
    uri = cfg["NEO4J_URI"]
    user = cfg["NEO4J_USERNAME"]
    pwd = cfg["NEO4J_PASSWORD"]
    db = cfg.get("NEO4J_DATABASE", "neo4j")

    print("Using Neo4j database:", db)
    driver = GraphDatabase.driver(uri, auth=(user, pwd))

    models = {
        "mpnet": SentenceTransformer("all-mpnet-base-v2"),   # 768 dims
        "minilm": SentenceTransformer("all-MiniLM-L6-v2"),   # 384 dims
    }

    with driver.session(database=db) as session:
        # 1) Pull all player aggregates (all important FPL stats)
        players = session.run("""
            MATCH (p:Player)
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
            OPTIONAL MATCH (p)-[r:PLAYED_IN]->(:Fixture)
            WITH p, pos,
                 sum(coalesce(r.total_points,0))        AS total_points,
                 sum(coalesce(r.goals_scored,0))        AS goals_scored,
                 sum(coalesce(r.assists,0))             AS assists,
                 sum(coalesce(r.clean_sheets,0))        AS clean_sheets,
                 sum(coalesce(r.saves,0))               AS saves,
                 sum(coalesce(r.penalties_saved,0))     AS penalties_saved,
                 sum(coalesce(r.penalties_missed,0))    AS penalties_missed,
                 sum(coalesce(r.goals_conceded,0))      AS goals_conceded,
                 sum(coalesce(r.own_goals,0))           AS own_goals,
                 sum(coalesce(r.yellow_cards,0))        AS yellow_cards,
                 sum(coalesce(r.red_cards,0))           AS red_cards,
                 sum(coalesce(r.bonus,0))               AS bonus,
                 avg(coalesce(r.bps,0))                 AS avg_bps,
                 avg(coalesce(r.ict_index,0))           AS avg_ict,
                 avg(coalesce(r.influence,0))           AS avg_influence,
                 avg(coalesce(r.creativity,0))          AS avg_creativity,
                 avg(coalesce(r.threat,0))              AS avg_threat,
                 avg(coalesce(r.minutes,0))             AS avg_minutes,
                 avg(coalesce(r.form,0))                AS avg_form
            RETURN id(p) AS id,
                   p.player_name AS name,
                   coalesce(pos.name,"") AS position,
                   total_points, goals_scored, assists,
                   clean_sheets, saves, penalties_saved, penalties_missed,
                   goals_conceded, own_goals, yellow_cards, red_cards,
                   bonus, avg_bps, avg_ict, avg_influence, avg_creativity, avg_threat,
                   avg_minutes, avg_form
        """).data()

        print("Found Player nodes:", len(players))

        # 2) Build embedding text per player (POSITION-AWARE)
        for row in players:
            pos = (row.get("position") or "").upper().strip()

            # common stats
            total_points = _safe_num(row.get("total_points"))
            bonus = _safe_num(row.get("bonus"))
            avg_minutes = _round(row.get("avg_minutes"))
            avg_bps = _round(row.get("avg_bps"))
            avg_ict = _round(row.get("avg_ict"))
            avg_infl = _round(row.get("avg_influence"))
            avg_crea = _round(row.get("avg_creativity"))
            avg_thr = _round(row.get("avg_threat"))

            # form sometimes is 0..1 in your data; keep it readable
            avg_form = row.get("avg_form")
            if avg_form is None:
                avg_form_out = 0
            else:
                try:
                    avg_form_f = float(avg_form)
                    avg_form_out = _round(avg_form_f * 100, 2) if 0 <= avg_form_f <= 1 else _round(avg_form_f, 2)
                except Exception:
                    avg_form_out = 0

            common = (
                f"Player {row['name']}, position {pos}, "
                f"total points {total_points}, "
                f"avg minutes {avg_minutes}, form {avg_form_out}, "
                f"bonus {bonus}, avg bps {avg_bps}, avg ict {avg_ict}. "
            )

            goals = _safe_num(row.get("goals_scored"))
            assists = _safe_num(row.get("assists"))
            cs = _safe_num(row.get("clean_sheets"))
            saves = _safe_num(row.get("saves"))
            pens_saved = _safe_num(row.get("penalties_saved"))
            pens_missed = _safe_num(row.get("penalties_missed"))
            conceded = _safe_num(row.get("goals_conceded"))
            own_goals = _safe_num(row.get("own_goals"))
            yc = _safe_num(row.get("yellow_cards"))
            rc = _safe_num(row.get("red_cards"))

            # NOTE: your dataset uses GK not GKP, but keep both safe.
            if pos in ["GK", "GKP"]:
                extra = (
                    f"goalkeeper stats: saves {saves}, clean sheets {cs}, "
                    f"penalties saved {pens_saved}, goals conceded {conceded}, "
                    f"discipline: yellow cards {yc}, red cards {rc}. "
                    f"attacking: goals {goals}, assists {assists}."
                )
            elif pos == "DEF":
                extra = (
                    f"defender stats: clean sheets {cs}, goals conceded {conceded}, "
                    f"attacking returns: goals {goals}, assists {assists}, "
                    f"discipline: yellow cards {yc}, red cards {rc}, own goals {own_goals}. "
                    f"creativity {avg_crea}, threat {avg_thr}, influence {avg_infl}."
                )
            else:  # MID / FWD / unknown
                extra = (
                    f"attacking stats: goals {goals}, assists {assists}, "
                    f"creativity {avg_crea}, threat {avg_thr}, influence {avg_infl}, "
                    f"penalties missed {pens_missed}, discipline: yellow cards {yc}, red cards {rc}. "
                    f"defensive: clean sheets {cs}, goals conceded {conceded}."
                )

            row["embedding_text"] = common + extra

        # 3) Store embeddings for BOTH models (correct property per model)
        for key, model in models.items():
            prop = f"embedding_{key}"
            print(f"Embedding with {key} -> writing p.{prop}")

            for row in players:
                emb = model.encode(row["embedding_text"]).astype(float).tolist()
                session.run(
                    f"""
                    MATCH (p) WHERE id(p) = $id
                    SET p.{prop} = $embedding,
                        p.embedding_text = $text
                    """,
                    id=row["id"],
                    embedding=emb,
                    text=row["embedding_text"],
                )

        # 4) Create vector indexes (Neo4j 5.x+)
        session.run("""
        CREATE VECTOR INDEX player_index_mpnet IF NOT EXISTS
        FOR (p:Player)
        ON (p.embedding_mpnet)
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: 768,
            `vector.similarity_function`: 'cosine'
          }
        }
        """)

        session.run("""
        CREATE VECTOR INDEX player_index_minilm IF NOT EXISTS
        FOR (p:Player)
        ON (p.embedding_minilm)
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: 384,
            `vector.similarity_function`: 'cosine'
          }
        }
        """)

        # 5) Quick verification counts
        counts = session.run("""
        MATCH (p:Player)
        RETURN count(p) AS total,
               count(p.embedding_mpnet) AS mpnet_count,
               count(p.embedding_minilm) AS minilm_count,
               count(p.embedding_text) AS text_count
        """).single()

        print("Total players:", counts["total"])
        print("Players with mpnet embeddings:", counts["mpnet_count"])
        print("Players with minilm embeddings:", counts["minilm_count"])
        print("Players with embedding_text:", counts["text_count"])

        # Optional: show indexes
        idx = session.run("SHOW INDEXES YIELD name, type RETURN name, type ORDER BY name").data()
        print("Indexes (player_index*):")
        for r in idx:
            if "player_index" in r["name"]:
                print("-", r["name"], r["type"])

    driver.close()
    print("Done ✅ embeddings stored + vector indexes created.")


if __name__ == "__main__":
    main()
