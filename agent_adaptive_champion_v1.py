from collections import deque

from agent_adaptive_pro_v2 import Agent as AdaptiveProV2


class Agent(AdaptiveProV2):
    team_id = "AdaptiveChampionV1"

    def parse_observation(self, obs):
        self._update_bomb_memory(obs)
        return self._build_context(obs)

    def compute_blast_map(self, grid, bombs):
        return {
            bomb["pos"]: self._blast_tiles(grid, bomb["pos"][0], bomb["pos"][1], bomb["radius"])
            for bomb in bombs
        }

    def compute_danger_map(self, grid, bombs):
        return self._compute_schedule(grid, bombs)

    def bfs_safe_positions(self, ctx, start=None, start_time=0, horizon=None):
        if start is None:
            start = ctx["my_pos"]
        if horizon is None:
            horizon = self.HORIZON
        q = deque([(start, start_time)])
        seen = {(start, start_time)}
        safe = {}
        while q:
            pos, t = q.popleft()
            if self._is_future_safe(pos, t, ctx["danger_at"]):
                safe.setdefault(pos, t)
            if t - start_time >= horizon:
                continue
            for action in self.SEARCH_ACTIONS:
                npos = self._next_pos(pos, action)
                nt = t + 1
                if action != self.STOP:
                    if not self._passable(ctx["grid"], npos[0], npos[1]):
                        continue
                    if self._bomb_blocks(pos, npos, nt, ctx["bombs"]):
                        continue
                if self._is_deadly(npos, nt, ctx["danger_at"]):
                    continue
                state = (npos, nt)
                if state in seen:
                    continue
                seen.add(state)
                q.append((npos, nt))
        return safe

    def action_filter(self, ctx):
        filtered = []
        for action in ctx["valid_actions"]:
            if action == self.BOMB:
                if self._bomb_survives(ctx):
                    filtered.append(action)
                continue

            npos = self._next_pos(ctx["my_pos"], action)
            if action != self.STOP:
                if not self._passable(ctx["grid"], npos[0], npos[1]):
                    continue
                if self._bomb_blocks(ctx["my_pos"], npos, 1, ctx["bombs"]):
                    continue
            if self._is_deadly(npos, 1, ctx["danger_at"]):
                continue
            hit = self._next_danger_time(npos, 1, ctx["danger_at"])
            if hit is None:
                filtered.append(action)
                continue
            if self._find_escape(
                ctx["grid"], npos, 1, ctx["bombs"], ctx["danger_at"], horizon=self.HORIZON
            ) is not None:
                filtered.append(action)

        if filtered:
            return filtered
        return [a for a in ctx["valid_actions"] if a != self.BOMB] or [self.STOP]

    def _build_context(self, obs):
        ctx = super()._build_context(obs)
        ctx["danger_map"] = {tile: min(times) for tile, times in ctx["danger_at"].items() if times}
        ctx["blast_map"] = self.compute_blast_map(ctx["grid"], ctx["bombs"])
        ctx["enemy_risk"] = self._expanded_enemy_risk(
            ctx["grid"], ctx["enemies"], ctx["bombs"], ctx["bomb_positions"]
        )
        ctx["safe_positions"] = self.bfs_safe_positions(ctx, horizon=9)
        ctx["filtered_actions"] = self.action_filter(ctx)
        return ctx

    def _expanded_enemy_risk(self, grid, enemies, bombs, bomb_positions):
        risk = set()
        for enemy in enemies:
            soon_refill = any(
                bomb["owner"] == enemy["id"]
                and int(bomb.get("det_time", bomb.get("timer", 7))) <= 2
                for bomb in bombs
            )
            if enemy["bombs_left"] <= 0 and not soon_refill:
                continue
            depth = 1 if not soon_refill else 2
            for origin in self._static_reachable_origins(grid, enemy["pos"], depth, bomb_positions):
                risk.update(self._blast_tiles(grid, origin[0], origin[1], enemy["radius"]))
        return risk

    def _static_reachable_origins(self, grid, start, depth, bomb_positions):
        q = deque([(start, 0)])
        seen = {start}
        while q:
            pos, d = q.popleft()
            if d >= depth:
                continue
            for action in self.DIRS:
                npos = self._next_pos(pos, action)
                if npos in seen or npos in bomb_positions:
                    continue
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                seen.add(npos)
                q.append((npos, d + 1))
        return seen

    def _bomb_survives(self, ctx):
        if ctx["my_bombs_left"] <= 0 or ctx["my_pos"] in ctx["bomb_positions"]:
            return False
        hypo_bombs, hypo_danger, _ = self._hypothetical_schedule(
            ctx, ctx["my_pos"], ctx["my_radius"]
        )
        if self._is_deadly(ctx["my_pos"], 1, hypo_danger):
            return False
        escape = self._find_escape(
            ctx["grid"], ctx["my_pos"], 1, hypo_bombs, hypo_danger, horizon=self.HORIZON
        )
        if escape is None:
            return False
        _, t, safe_pos = escape
        area = self._reachable_count(ctx, safe_pos, t, hypo_bombs, hypo_danger, 8)
        return area >= (3 if ctx["phase"] == "end" else 4)

    def _decide(self, ctx, started):
        proposal = super()._decide(ctx, started)
        if proposal in ctx.get("filtered_actions", ctx["valid_actions"]):
            if proposal != self.BOMB or self._score_bomb_at(ctx, ctx["my_pos"])["legal"]:
                return proposal
        if self._next_danger_time(ctx["my_pos"], 1, ctx["danger_at"]) is not None:
            action = self._best_escape_action(ctx)
            if action is not None:
                return action
        objective = self._choose_objective(ctx, started)
        if objective in ctx.get("filtered_actions", ctx["valid_actions"]):
            return objective
        return self._fallback_action(ctx)

    def _score_bomb_at(self, ctx, pos):
        score = super()._score_bomb_at(ctx, pos)
        if not score.get("legal"):
            return score
        if self.BOMB not in ctx.get("filtered_actions", ctx["valid_actions"]):
            return self._illegal_bomb_score()

        hypo_bombs, hypo_danger, _ = self._hypothetical_schedule(ctx, pos, ctx["my_radius"])
        escape = self._find_escape(
            ctx["grid"], pos, 1, hypo_bombs, hypo_danger, horizon=self.HORIZON
        )
        if escape is None:
            return self._illegal_bomb_score()
        first, t, safe_pos = escape
        area = self._reachable_count(ctx, safe_pos, t, hypo_bombs, hypo_danger, 9)
        first_pos = self._next_pos(pos, first)
        if area <= 2:
            return self._illegal_bomb_score()
        if first_pos in ctx["enemy_risk"] and score.get("trap_score", 0.0) < 1.4 and score.get("boxes", 0) < 2:
            score["score"] -= 2.0
        if safe_pos in ctx["enemy_risk"] and area < 7:
            score["score"] -= 1.5
        if score.get("enemy_hits", 0) > 0 and score.get("trap_score", 0.0) < 1.0:
            score["score"] -= 0.9
        if score.get("boxes", 0) >= 2 and score.get("enemy_hits", 0) == 0:
            score["score"] += 0.4
        if ctx["tempo"] in ("tiebreak", "pressure") and score.get("boxes", 0) >= 1:
            score["score"] += 0.3
        return score

    def _bomb_threshold(self, bomb):
        base = super()._bomb_threshold(bomb)
        if bomb.get("trap_score", 0.0) >= 2.0 or bomb.get("enemy_hits", 0) >= 1:
            return base - 0.4
        if bomb.get("boxes", 0) >= 2:
            return base - 0.35
        return base

    def _best_escape_action(self, ctx):
        candidates = []
        for action in ctx.get("filtered_actions", ctx["valid_actions"]):
            if action == self.BOMB:
                continue
            npos = self._next_pos(ctx["my_pos"], action)
            if action != self.STOP:
                if not self._passable(ctx["grid"], npos[0], npos[1]):
                    continue
                if self._bomb_blocks(ctx["my_pos"], npos, 1, ctx["bombs"]):
                    continue
            if self._is_deadly(npos, 1, ctx["danger_at"]):
                continue
            area = self._reachable_count(ctx, npos, 1, ctx["bombs"], ctx["danger_at"], 9)
            future = self._next_danger_time(npos, 1, ctx["danger_at"])
            score = min(area, 16) * 0.8
            score += 9.0 if future is None else min(future, 7) * 0.7
            score += self._open_neighbors(ctx["grid"], npos, ctx["bomb_positions"]) * 0.8
            risky = npos in ctx["enemy_risk"]
            if risky:
                score -= 2.0
            if action == self.STOP:
                score -= 0.4
            candidates.append((risky, score, action))
        if not candidates:
            return None
        if any(not risky for risky, _, _ in candidates):
            candidates = [c for c in candidates if not c[0]]
        return max(candidates, key=lambda item: item[1])[2]

    def _fallback_action(self, ctx):
        best = None
        for action in ctx.get("filtered_actions", ctx["valid_actions"]):
            if action == self.BOMB:
                continue
            npos = self._next_pos(ctx["my_pos"], action)
            if action != self.STOP:
                if not self._passable(ctx["grid"], npos[0], npos[1]):
                    continue
                if self._bomb_blocks(ctx["my_pos"], npos, 1, ctx["bombs"]):
                    continue
            if self._is_deadly(npos, 1, ctx["danger_at"]):
                continue
            area = self._reachable_count(ctx, npos, 1, ctx["bombs"], ctx["danger_at"], 8)
            future = self._next_danger_time(npos, 1, ctx["danger_at"])
            score = area + (8.0 if future is None else min(future, 7) * 0.5)
            if npos in ctx["enemy_risk"]:
                score -= 1.3
            if action == self.STOP:
                score -= 0.25
            if best is None or score > best[0]:
                best = (score, action)
        return best[1] if best is not None else self.STOP

    def _temporal_reach_map(self, ctx, horizon):
        grid = ctx["grid"]
        start = ctx["my_pos"]
        first_actions = [a for a in ctx.get("filtered_actions", ctx["valid_actions"]) if a != self.BOMB]
        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        reach = {start: (0, self.STOP)}
        while q:
            pos, t, first = q.popleft()
            if t >= horizon:
                continue
            actions = first_actions if t == 0 else self.SEARCH_ACTIONS
            for action in actions:
                npos = self._next_pos(pos, action)
                nt = t + 1
                if action != self.STOP:
                    if not self._passable(grid, npos[0], npos[1]):
                        continue
                    if self._bomb_blocks(pos, npos, nt, ctx["bombs"]):
                        continue
                if self._is_deadly(npos, nt, ctx["danger_at"]):
                    continue
                state = (npos, nt)
                if state in seen:
                    continue
                seen.add(state)
                first_action = action if first is None else first
                if npos not in reach or nt < reach[npos][0]:
                    reach[npos] = (nt, first_action)
                q.append((npos, nt, first_action))
        return reach
