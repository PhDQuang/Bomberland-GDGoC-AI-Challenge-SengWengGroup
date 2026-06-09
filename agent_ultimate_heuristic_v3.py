from collections import deque

from agent_ultimate_heuristic_v2 import Agent as UltimateHeuristicV2


class Agent(UltimateHeuristicV2):
    team_id = "UltimateHeuristicV3"

    # ------------------------------------------------------------------
    # Explicit rule-core modules.
    # ------------------------------------------------------------------

    def parse_observation(self, obs):
        self._update_bomb_memory(obs)
        return self._build_context(obs)

    def compute_blast_map(self, grid, bombs):
        blast_map = {}
        for bomb in bombs:
            blast_map[bomb["pos"]] = self._blast_tiles(
                grid, bomb["pos"][0], bomb["pos"][1], bomb["radius"]
            )
        return blast_map

    def compute_danger_map(self, grid, bombs):
        return self._compute_schedule(grid, bombs)

    def bfs_safe_positions(self, ctx, start=None, start_time=0, horizon=None):
        if start is None:
            start = ctx["my_pos"]
        if horizon is None:
            horizon = self.HORIZON

        grid = ctx["grid"]
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
                q.append((npos, nt))
        return safe

    def action_filter(self, ctx):
        filtered = []
        for action in ctx["valid_actions"]:
            if action == self.BOMB:
                if self._bomb_action_survives(ctx):
                    filtered.append(action)
                continue

            npos = self._next_pos(ctx["my_pos"], action)
            if action != self.STOP and not self._movement_action_legal(ctx, ctx["my_pos"], npos, 1):
                continue
            if self._is_deadly(npos, 1, ctx["danger_at"]):
                continue
            future = self._next_danger_time(npos, 1, ctx["danger_at"])
            if future is None:
                filtered.append(action)
                continue
            escape = self._find_escape(
                ctx["grid"],
                npos,
                1,
                ctx["bombs"],
                ctx["danger_at"],
                self.HORIZON,
                enemies=ctx.get("enemies"),
            )
            if escape is not None:
                filtered.append(action)

        if filtered:
            return filtered

        fallback = []
        for action in ctx["valid_actions"]:
            if action == self.BOMB:
                continue
            npos = self._next_pos(ctx["my_pos"], action)
            if action != self.STOP and not self._movement_action_legal(ctx, ctx["my_pos"], npos, 1):
                continue
            if not self._is_deadly(npos, 1, ctx["danger_at"]):
                fallback.append(action)
        return fallback if fallback else [self.STOP]

    # ------------------------------------------------------------------
    # Context enrichment.
    # ------------------------------------------------------------------

    def _build_context(self, obs):
        ctx = super()._build_context(obs)
        ctx["danger_map"] = {
            tile: min(times) for tile, times in ctx["danger_at"].items() if times
        }
        ctx["blast_map"] = self.compute_blast_map(ctx["grid"], ctx["bombs"])
        ctx["enemy_risk"] = self._expanded_enemy_risk(
            ctx["grid"], ctx["enemies"], ctx["bombs"], ctx["bomb_positions"]
        )
        ctx["safe_positions"] = self.bfs_safe_positions(ctx, horizon=10)
        ctx["filtered_actions"] = self.action_filter(ctx)
        return ctx

    def _expanded_enemy_risk(self, grid, enemies, bombs, bomb_positions):
        risk = set()
        for enemy in enemies:
            soon_refill = any(
                bomb["owner"] == enemy["id"]
                and int(bomb.get("det_time", bomb.get("timer", 7))) <= 3
                for bomb in bombs
            )
            if enemy["bombs_left"] <= 0 and not soon_refill:
                continue
            if enemy["pos"] in bomb_positions and not soon_refill:
                continue
            depth = 2 if soon_refill else 1
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

    # ------------------------------------------------------------------
    # Safer bomb policy with stronger lobby awareness.
    # ------------------------------------------------------------------

    def _bomb_action_survives(self, ctx):
        if ctx["my_bombs_left"] <= 0 or ctx["my_pos"] in ctx["bomb_positions"]:
            return False
        hypo_bombs, hypo_danger, _ = self._hypothetical_schedule(
            ctx, ctx["my_pos"], ctx["my_radius"]
        )
        if self._is_deadly(ctx["my_pos"], 1, hypo_danger):
            return False
        escape = self._find_escape(
            ctx["grid"],
            ctx["my_pos"],
            1,
            hypo_bombs,
            hypo_danger,
            self.HORIZON,
            enemies=ctx.get("enemies"),
        )
        if escape is None:
            return False
        return self._escape_has_margin(ctx, escape, hypo_bombs, hypo_danger, ctx["my_pos"])

    def _escape_has_margin(self, ctx, escape, bombs, danger_at, start_pos):
        first, t, safe_pos = escape
        area = self._reachable_count(ctx, safe_pos, t, bombs, danger_at, 9)
        min_area = 3 if ctx["phase"] == "end" else 5
        if area < min_area:
            return False
        if safe_pos in ctx["enemy_risk"] and area < 8:
            return False
        return True

    def _score_bomb_at(self, ctx, pos):
        score = super()._score_bomb_at(ctx, pos)
        if not score.get("legal"):
            return score
        if self.BOMB not in ctx.get("filtered_actions", ctx["valid_actions"]):
            return self._illegal_bomb_score()

        hypo_bombs, hypo_danger, _ = self._hypothetical_schedule(ctx, pos, ctx["my_radius"])
        escape = self._find_escape(
            ctx["grid"],
            pos,
            1,
            hypo_bombs,
            hypo_danger,
            self.HORIZON,
            enemies=ctx.get("enemies"),
        )
        if escape is None:
            return self._illegal_bomb_score()

        if not self._escape_has_margin(ctx, escape, hypo_bombs, hypo_danger, pos):
            # Direct, high-confidence traps or strong box hits are still allowed.
            strong_box_hit = score.get("boxes", 0) >= 2
            strong_trap = score.get("enemy_hits", 0) > 0 and score.get("trap_score", 0.0) >= 1.8
            if not (strong_box_hit or strong_trap):
                return self._illegal_bomb_score()
            score["score"] -= 2.5

        first_pos = self._next_pos(pos, escape[0])
        if first_pos in ctx["enemy_risk"]:
            score["score"] -= 1.0
        if score.get("enemy_hits", 0) > 0 and score.get("trap_score", 0.0) < 1.0:
            score["score"] -= 1.5
        return score

    def _bomb_threshold(self, bomb):
        base = super()._bomb_threshold(bomb)
        if bomb.get("mode") == "combat" and bomb.get("trap_score", 0.0) >= 2.0:
            return base - 0.7
        if bomb.get("enemy_hits", 0) > 0 and bomb.get("trap_score", 0.0) < 1.0:
            return base + 1.2
        if bomb.get("boxes", 0) >= 2 and bomb.get("enemy_hits", 0) == 0:
            return base - 0.4
        return base

    # ------------------------------------------------------------------
    # Filtered decision/search.
    # ------------------------------------------------------------------

    def _decide(self, ctx, started):
        my_pos = ctx["my_pos"]
        next_hit = self._next_danger_time(my_pos, 1, ctx["danger_at"])

        if self.escape_mode and self._is_future_safe(my_pos, 1, ctx["danger_at"]):
            if self._reachable_count(ctx, my_pos, 0, ctx["bombs"], ctx["danger_at"], 9) >= 8:
                self.escape_mode = False

        if next_hit is not None and next_hit <= 2:
            action = self._best_escape_action(ctx)
            return action if action is not None else self._fallback_action(ctx)

        if ctx["mode"] == "escape":
            action = self._best_escape_action(ctx)
            if action is not None:
                return action

        current_bomb = self._score_bomb_at(ctx, my_pos)
        if current_bomb["legal"] and current_bomb["score"] >= self._bomb_threshold(current_bomb):
            self.escape_mode = True
            return self.BOMB

        # Minimax is useful, but never let it bypass the filter.
        if ctx["alive_count"] == 2:
            mm_action = self._minimax_decide(ctx, started)
            if mm_action is not None and mm_action in ctx["filtered_actions"]:
                if mm_action != self.BOMB or self._score_bomb_at(ctx, my_pos)["legal"]:
                    if mm_action == self.BOMB:
                        self.escape_mode = True
                    return mm_action

        if next_hit is not None and next_hit <= 4:
            action = self._best_escape_action(ctx)
            if action is not None:
                return action

        objective = self._choose_objective(ctx, started)
        if objective is not None and objective in ctx["filtered_actions"]:
            return objective
        return self._fallback_action(ctx)

    def _best_escape_action(self, ctx):
        candidates = []
        for action in ctx.get("filtered_actions", ctx["valid_actions"]):
            if action == self.BOMB:
                continue
            npos = self._next_pos(ctx["my_pos"], action)
            if action != self.STOP and not self._movement_action_legal(ctx, ctx["my_pos"], npos, 1):
                continue
            if self._is_deadly(npos, 1, ctx["danger_at"]):
                continue

            future = self._next_danger_time(npos, 1, ctx["danger_at"])
            area = self._reachable_count(ctx, npos, 1, ctx["bombs"], ctx["danger_at"], 10)
            enemy_dist = min(
                [self._manhattan(npos, enemy["pos"]) for enemy in ctx["enemies"]] or [8]
            )
            score = min(area, 18) * 0.8 + min(enemy_dist, 8) * 0.8
            score += 11.0 if future is None else min(future, 8) * 0.75
            score += self._open_neighbors(ctx["grid"], npos, ctx["bomb_positions"]) * 0.9
            risky = npos in ctx["enemy_risk"]
            if risky:
                score -= 3.0
            if npos in ctx["enemy_set"]:
                score -= 1.2
            if action == self.STOP:
                score -= 0.6
            candidates.append((risky, score, action))
        if not candidates:
            return None
        if any(not risky for risky, _, _ in candidates):
            candidates = [item for item in candidates if not item[0]]
        return max(candidates, key=lambda item: item[1])[2]

    def _fallback_action(self, ctx):
        best = None
        for action in ctx.get("filtered_actions", ctx["valid_actions"]):
            if action == self.BOMB:
                continue
            npos = self._next_pos(ctx["my_pos"], action)
            if action != self.STOP and not self._movement_action_legal(ctx, ctx["my_pos"], npos, 1):
                continue
            if self._is_deadly(npos, 1, ctx["danger_at"]):
                continue
            future = self._next_danger_time(npos, 1, ctx["danger_at"])
            score = self._reachable_count(ctx, npos, 1, ctx["bombs"], ctx["danger_at"], 8)
            score += self._open_neighbors(ctx["grid"], npos, ctx["bomb_positions"]) * 1.3
            score += 9.0 if future is None else min(future, 7) * 0.55
            cell = int(ctx["grid"][npos[0], npos[1]]) if self._passable(ctx["grid"], npos[0], npos[1]) else -1
            if cell == self.ITEM_CAPACITY:
                score += 1.0
            elif cell == self.ITEM_RADIUS:
                score += 0.8
            if npos in ctx["enemy_risk"]:
                score -= 1.8
            if action == self.STOP:
                score -= 0.3
            if best is None or score > best[0]:
                best = (score, action)
        return best[1] if best is not None else self.STOP

    def _quick_safe_action(self, ctx):
        return self._fallback_action(ctx)

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
