class CypherQueryBuilder:
    @staticmethod
    def build(intent: str, entities: dict):
        p = entities.get("players", [])
        t = entities.get("teams", [])
        s = entities.get("seasons", [])
        pos = entities.get("positions", [])
        gw = entities.get("gameweeks", [])

        params = {}

        def season_path():
            if s:
                params["season_name"] = s[0]
                return "MATCH (se:Season {season_name:$season_name})-[:HAS_GW]->(g:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)\n"
            return "MATCH (g:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)\n"

        def gw_filter():
            if gw:
                params["gw_number"] = int(gw[0])
                return "WHERE g.GW_number = $gw_number\n"
            return ""

        # 1) player_stats
        if intent == "player_stats" and p:
            params["p_name"] = p[0]
            q = (
                season_path() + gw_filter() +
                """
                MATCH (pl:Player {player_name:$p_name})-[r:PLAYED_IN]->(f)
                RETURN pl.player_name AS player,
                       SUM(COALESCE(r.total_points, r.points, r.event_points, 0)) AS total_points,
                       SUM(COALESCE(r.goals_scored,0)) AS goals_scored,
                       SUM(COALESCE(r.assists,0)) AS assists,
                       AVG(COALESCE(r.minutes,0)) AS avg_minutes,
                       AVG(COALESCE(r.form,0)) AS avg_form
                """
            )
            return q, params

        # 2) top_players (can later adjust LIMIT in code from "top 5")
        if intent == "top_players":
            q = (
                season_path() + gw_filter() +
                """
                MATCH (pl:Player)-[r:PLAYED_IN]->(f)
                RETURN pl.player_name AS player,
                       SUM(COALESCE(r.total_points, r.points, r.event_points, 0)) AS total_points
                ORDER BY total_points DESC
                LIMIT 10
                """
            )
            return q, params

        # 3) best_by_position
        if intent == "best_by_position" and pos:
            params["pos"] = pos[0]
            q = (
                season_path() + gw_filter() +
                """
                MATCH (pl:Player)-[:PLAYS_AS]->(po:Position {name:$pos})
                MATCH (pl)-[r:PLAYED_IN]->(f)
                RETURN pl.player_name AS player, po.name AS position,
                       SUM(COALESCE(r.total_points, r.points, r.event_points, 0)) AS total_points
                ORDER BY total_points DESC
                LIMIT 10
                """
            )
            return q, params

        # 4) compare_players (expects 2 players)
        if intent == "compare_players" and len(p) >= 2:
            params["p1"] = p[0]
            params["p2"] = p[1]
            q = (
                season_path() + gw_filter() +
                """
                MATCH (pl:Player)-[r:PLAYED_IN]->(f)
                WHERE pl.player_name IN [$p1,$p2]
                RETURN pl.player_name AS player,
                       SUM(COALESCE(r.total_points, r.points, r.event_points, 0)) AS total_points,
                       SUM(COALESCE(r.goals_scored,0)) AS goals_scored,
                       SUM(COALESCE(r.assists,0)) AS assists
                ORDER BY total_points DESC
                """
            )
            return q, params

        # 5) team_fixtures
        if intent == "team_fixtures" and t:
            params["team_name"] = t[0]
            q = (
                season_path() + gw_filter() +
                """
                MATCH (f)-[:HAS_HOME_TEAM]->(home:Team),
                      (f)-[:HAS_AWAY_TEAM]->(away:Team)
                WHERE home.name = $team_name OR away.name = $team_name
                RETURN g.GW_number AS gameweek,
                       home.name AS home_team,
                       away.name AS away_team,
                       f.kickoff_time AS kickoff_time,
                       f.fixture_number AS fixture_number
                ORDER BY g.GW_number ASC, f.kickoff_time ASC
                LIMIT 40
                """
            )
            return q, params

        # 6) gameweek_fixtures
        if intent == "gameweek_fixtures":
            q = (
                season_path() + gw_filter() +
                """
                MATCH (f)-[:HAS_HOME_TEAM]->(home:Team),
                      (f)-[:HAS_AWAY_TEAM]->(away:Team)
                RETURN g.GW_number AS gameweek,
                       home.name AS home_team,
                       away.name AS away_team,
                       f.kickoff_time AS kickoff_time,
                       f.fixture_number AS fixture_number
                ORDER BY f.kickoff_time ASC
                LIMIT 40
                """
            )
            return q, params

        # 7) Player clean sheets
        if intent == "player_clean_sheets" and p:
            params["p_name"] = p[0]
            q = (
                season_path() + gw_filter() +
                """
                MATCH (pl:Player {player_name:$p_name})-[r:PLAYED_IN]->(f)
                RETURN pl.player_name AS player,
                    SUM(COALESCE(r.clean_sheets,0)) AS clean_sheets
                """
            )
            return q, params


        # 8) player_form
        if intent == "player_form" and p:
            params["p_name"] = p[0]
            q = (
                season_path() + gw_filter() +
                """
                MATCH (pl:Player {player_name:$p_name})-[r:PLAYED_IN]->(f)
                RETURN pl.player_name AS player,
                       AVG(COALESCE(r.form,0)) AS avg_form,
                       SUM(COALESCE(r.total_points, r.points, r.event_points, 0)) AS total_points
                """
            )
            return q, params

        # 9) player_discipline
        if intent == "player_discipline" and p:
            params["p_name"] = p[0]
            q = (
                season_path() + gw_filter() +
                """
                MATCH (pl:Player {player_name:$p_name})-[r:PLAYED_IN]->(f)
                RETURN pl.player_name AS player,
                       SUM(COALESCE(r.yellow_cards,0)) AS yellow_cards,
                       SUM(COALESCE(r.red_cards,0)) AS red_cards
                """
            )
            return q, params

        # 10) goalkeeper_stats (saves / pens saved) — MUST be GK
        if intent == "goalkeeper_stats" and p:
            params["p_name"] = p[0]
            q = (
                season_path() + gw_filter() +
                """
                MATCH (pl:Player {player_name:$p_name})-[:PLAYS_AS]->(:Position {name:'GK'})
                MATCH (pl)-[r:PLAYED_IN]->(f)
                RETURN pl.player_name AS player,
                    SUM(COALESCE(r.saves,0)) AS saves,
                    SUM(COALESCE(r.penalties_saved,0)) AS penalties_saved
                """
            )
            return q, params


        if intent == "recommendation" and pos:
            params["pos"] = pos[0]
            q = (
                season_path() + gw_filter() +
                """
                MATCH (pl:Player)-[:PLAYS_AS]->(po:Position {name:$pos})
                MATCH (pl)-[r:PLAYED_IN]->(f)
                RETURN pl.player_name AS player,
                    po.name AS position,
                    AVG(COALESCE(r.form,0)) AS avg_form,
                    SUM(COALESCE(r.total_points, r.points, r.event_points, 0)) AS total_points,
                    AVG(COALESCE(r.minutes,0)) AS avg_minutes
                ORDER BY avg_form DESC, total_points DESC
                LIMIT 5
                """
            )
            return q, params

        # fallback
        return "MATCH (n) RETURN n LIMIT 10", {}
