from agent_adaptive_pro_v2 import Agent as AdaptiveProV2


class Agent(AdaptiveProV2):
    team_id = "MetricOptimizedV1"

    def _strategic_mode(self, ctx):
        mode, target = super()._strategic_mode(ctx)
        if mode in ("escape", "hunt", "box_farm", "item_race"):
            return mode, target

        if not ctx["enemies"]:
            return mode, target

        my_stats = ctx["my_stats"]
        my_kills = int(my_stats.get("kills", 0))
        my_boxes = int(my_stats.get("boxes", 0))
        strongest = max(
            ctx["enemies"],
            key=lambda enemy: (
                int(enemy["stats"].get("kills", 0)),
                int(enemy["stats"].get("boxes", 0)),
                enemy["radius"],
            ),
        )
        enemy_kills = int(strongest["stats"].get("kills", 0))
        enemy_boxes = int(strongest["stats"].get("boxes", 0))

        if self.turn >= 260 and enemy_kills >= my_kills and enemy_boxes >= my_boxes:
            return "hunt", strongest["id"]
        if self.turn >= 210 and ctx["boxes_remaining"] > 0 and my_boxes + 3 < enemy_boxes:
            return "box_farm", None
        return mode, target

    def _engagement_deficit(self, ctx):
        stats = ctx["telemetry"]
        engagement = (
            int(stats.get("own_kills", 0)) * 7
            + int(stats.get("own_boxes", 0)) * 1.25
            + int(stats.get("own_items", 0)) * 0.9
            + int(stats.get("own_bombs", 0)) * 0.18
        )
        expected = max(2.0, self.turn / 50.0)
        return max(0.0, expected - engagement)

    def _bomb_threshold(self, bomb):
        threshold = super()._bomb_threshold(bomb)
        tempo = bomb.get("tempo", "balanced")
        if bomb["enemy_hits"]:
            threshold -= 0.55
        if bomb["trap_score"] >= 1.5:
            threshold -= 0.65
        elif bomb["trap_score"] >= 1.0:
            threshold -= 0.25
        if tempo in ("hunt", "pressure", "tiebreak", "duel"):
            threshold -= 0.25
        if bomb["boxes"] >= 3 and tempo not in ("escape", "duel", "survive_duel"):
            threshold -= 0.25
        if bomb.get("slack", 7) < 3 or bomb.get("escape_area", 10) <= 4:
            threshold += 0.35
        return threshold

    def _score_bomb_at(self, ctx, pos):
        score = super()._score_bomb_at(ctx, pos)
        if not score["legal"]:
            return score

        tempo = ctx["tempo"]
        if tempo in ("hunt", "pressure", "tiebreak", "duel"):
            score["score"] += score["enemy_hits"] * 1.6 + score["trap_score"] * 0.9
        if score["boxes"] >= 2 and tempo in ("balanced", "economy", "box_farm", "pressure", "tiebreak"):
            score["score"] += min(score["boxes"], 4) * 0.35
        if score["escape_area"] <= 5:
            score["score"] -= 0.6
        return score

    def _item_value(self, ctx, cell):
        value = super()._item_value(ctx, cell)
        if self.turn >= 330 and ctx["tempo"] not in ("escape", "item_race"):
            value *= 0.75
        return value
