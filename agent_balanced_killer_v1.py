from agent_killer import AdaptiveProV2Base, Agent as KillerAgent


class Agent(KillerAgent):
    team_id = "BalancedKillerV1"

    def _strategic_mode(self, ctx):
        if not ctx["enemies"]:
            self.focus_enemy_id = None
            return "adaptive", None

        base_mode, base_target = AdaptiveProV2Base._strategic_mode(self, ctx)
        my_stats = ctx["my_stats"]
        my_score = (
            int(my_stats.get("kills", 0)) * 100
            + int(my_stats.get("boxes", 0)) * 4
            + int(my_stats.get("items", 0)) * 3
            + int(my_stats.get("bombs", 0))
        )
        enemy_best = -1
        for enemy in ctx["enemies"]:
            stats = ctx["estimated_stats"][enemy["id"]]
            score = (
                int(stats.get("kills", 0)) * 100
                + int(stats.get("boxes", 0)) * 4
                + int(stats.get("items", 0)) * 3
                + int(stats.get("bombs", 0))
            )
            enemy_best = max(enemy_best, score)

        behind = my_score + 8 < enemy_best
        late = self.turn >= 260
        duel = ctx["alive_count"] <= 2
        high_power = ctx["my_radius"] >= 3 and max(ctx["my_bombs_left"], ctx.get("max_bombs_left_seen", 1)) >= 2

        if duel or (late and high_power) or (behind and high_power):
            return KillerAgent._strategic_mode(self, ctx)

        self.focus_enemy_id = base_target
        return base_mode, base_target

    def _tempo_mode(self, ctx):
        if ctx.get("strategic_mode") == "hunt":
            return KillerAgent._tempo_mode(self, ctx)
        return AdaptiveProV2Base._tempo_mode(self, ctx)

    def _engagement_deficit(self, ctx):
        if ctx.get("strategic_mode") == "hunt":
            return KillerAgent._engagement_deficit(self, ctx)
        return AdaptiveProV2Base._engagement_deficit(self, ctx)

    def _bomb_threshold(self, bomb):
        threshold = super()._bomb_threshold(bomb)
        if bomb.get("tempo") not in ("hunt", "duel"):
            if bomb.get("boxes", 0) >= 2 and bomb.get("enemy_hits", 0) == 0:
                threshold -= 0.35
            if bomb.get("enemy_hits", 0) > 0 and bomb.get("trap_score", 0.0) >= 1.0:
                threshold -= 0.35
        return threshold
