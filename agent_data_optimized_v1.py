from agent_adaptive_pro_v2 import Agent as AdaptiveProV2


class Agent(AdaptiveProV2):
    team_id = "DataOptimizedV1"

    def _strategic_mode(self, ctx):
        my_stats = ctx["my_stats"]
        enemies = ctx["enemies"]
        if not enemies:
            return "adaptive", None

        my_kills = int(my_stats.get("kills", 0))
        my_boxes = int(my_stats.get("boxes", 0))
        strongest = max(
            enemies,
            key=lambda enemy: (
                int(enemy["stats"].get("kills", 0)),
                int(enemy["stats"].get("boxes", 0)),
                enemy["radius"],
                enemy["bombs_left"],
            ),
        )
        lead_enemy_kills = int(strongest["stats"].get("kills", 0))
        lead_enemy_boxes = int(strongest["stats"].get("boxes", 0))

        if ctx["phase"] == "end":
            if lead_enemy_kills > my_kills:
                return "hunt", strongest["id"]
            if lead_enemy_kills == my_kills and lead_enemy_boxes >= my_boxes:
                return "hunt", strongest["id"]
            return "escape", None

        if my_kills >= 2 and my_kills > lead_enemy_kills:
            return "escape", None

        if self.turn >= 115 and my_kills <= lead_enemy_kills:
            return "hunt", strongest["id"]

        if ctx["boxes_remaining"] > 0 and my_boxes + 2 < lead_enemy_boxes:
            return "box_farm", None

        return "adaptive", None

    def _tempo_mode(self, ctx):
        strategic = ctx.get("strategic_mode", "adaptive")
        if strategic == "escape":
            return "escape"
        if strategic == "hunt":
            return "hunt"
        if strategic == "box_farm":
            return "box_farm"

        power = ctx["my_radius"] + max(ctx["my_bombs_left"], ctx["max_bombs_left_seen"])
        if ctx["phase"] == "end":
            return "duel" if power >= 4 else "survive_duel"
        if self.turn >= 330:
            return "tiebreak"
        if self.turn < 105 and (ctx["my_bonus"] < 2 or ctx["max_bombs_left_seen"] < 2):
            return "economy"
        if power >= 4 and self.turn >= 80:
            return "pressure"
        return "balanced"

    def _engagement_deficit(self, ctx):
        stats = ctx["telemetry"]
        engagement = (
            int(stats.get("own_kills", 0)) * 9.0
            + int(stats.get("own_boxes", 0)) * 1.35
            + int(stats.get("own_items", 0)) * 0.45
            + int(stats.get("own_bombs", 0)) * 0.08
        )
        expected = max(3.0, self.turn / 42.0)
        return max(0.0, expected - engagement)

    def _bomb_threshold(self, bomb):
        threshold = super()._bomb_threshold(bomb)
        tempo = bomb.get("tempo", "balanced")
        if bomb["enemy_hits"] or bomb["trap_score"] >= 1.0:
            threshold -= 1.15
        if bomb["trap_score"] >= 1.8:
            threshold -= 0.9
        if tempo in ("hunt", "pressure", "tiebreak", "duel"):
            threshold -= 0.55
        if tempo == "economy" and not bomb["enemy_hits"]:
            threshold += 0.35
        if bomb["boxes"] >= 2 and tempo not in ("escape", "duel", "survive_duel"):
            threshold -= 0.35
        if bomb.get("escape_area", 0) <= 5:
            threshold += 0.7
        return threshold

    def _score_bomb_at(self, ctx, pos):
        score = super()._score_bomb_at(ctx, pos)
        if not score["legal"]:
            return score

        tempo = ctx["tempo"]
        kill_pressure = score["enemy_hits"] * 3.6 + score["trap_score"] * 2.2
        if tempo in ("hunt", "pressure", "tiebreak", "duel"):
            score["score"] += kill_pressure
        else:
            score["score"] += kill_pressure * 0.45

        if score["boxes"] >= 2 and tempo not in ("escape", "duel", "survive_duel"):
            score["score"] += score["boxes"] * 0.75

        if tempo in ("duel", "survive_duel") and score["boxes"] and not score["enemy_hits"]:
            score["score"] -= 1.5

        if score["escape_area"] < 6:
            score["score"] -= 1.2
        return score

    def _choose_objective(self, ctx, started):
        reach = self._temporal_reach_map(ctx, self.HORIZON)

        if ctx["tempo"] in ("hunt", "pressure", "tiebreak", "duel"):
            enemy_choice = self._best_enemy_pressure_move(ctx, reach)
            if enemy_choice is not None and enemy_choice[0] >= 1.4:
                return enemy_choice[1]

        box_choice = None
        if ctx["tempo"] in ("box_farm", "balanced", "pressure", "tiebreak", "economy"):
            box_choice = self._best_box_position_move(ctx, reach)

        item_choice = self._best_item_move(ctx, reach)
        if item_choice is not None and ctx["tempo"] in ("economy", "escape"):
            if box_choice is None or item_choice[0] >= box_choice[0] + 0.8:
                return item_choice[1]

        if box_choice is not None:
            return box_choice[1]

        enemy_choice = self._best_enemy_pressure_move(ctx, reach)
        if enemy_choice is not None:
            return enemy_choice[1]

        if item_choice is not None:
            return item_choice[1]
        return None

    def _item_value(self, ctx, cell):
        value = super()._item_value(ctx, cell)
        if ctx["tempo"] in ("pressure", "hunt", "tiebreak", "duel"):
            value *= 0.45
        elif self.turn >= 220:
            value *= 0.65
        return value

    def _best_enemy_pressure_move(self, ctx, reach):
        choice = super()._best_enemy_pressure_move(ctx, reach)
        if choice is None:
            return None
        score, action = choice
        if ctx["tempo"] in ("hunt", "pressure", "tiebreak", "duel"):
            score += 1.0
        if ctx["engagement_deficit"] >= 2:
            score += min(ctx["engagement_deficit"], 5) * 0.22
        return score, action
