from collections import deque
import time


class Agent:
    team_id = "SkynetPruneV1"

    GRASS = 0
    WALL = 1
    BOX = 2
    ITEM_RADIUS = 3
    ITEM_CAPACITY = 4

    STOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    BOMB = 5

    MOVES = {
        STOP: (0, 0),
        LEFT: (-1, 0),
        RIGHT: (1, 0),
        UP: (0, -1),
        DOWN: (0, 1),
    }
    DIRS = (LEFT, RIGHT, UP, DOWN)
    SEARCH_ACTIONS = (LEFT, RIGHT, UP, DOWN, STOP)

    MAX_RADIUS = 5
    HORIZON = 15
    TIME_BUDGET_S = 0.087

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.turn = 0
        self.last_action = self.STOP
        self.last_bonuses = None
        self.bomb_radii = {}
        self.prev_grid = None
        self.prev_players = None
        self.prev_bombs = []
        self.seen_bombs = set()
        self.stats = None
        self.max_capacity_seen = 1

    def act(self, obs: dict) -> int:
        started = time.perf_counter()
        try:
            if self._looks_like_new_game(obs):
                self._reset_state()

            self._observe_transition(obs)
            self.turn += 1

            players = obs["players"]
            if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
                self._snapshot(obs)
                return self.STOP

            self._update_bomb_memory(obs)
            ctx = self._build_context(obs)
            action = self._choose_action(ctx, started)

            if action not in (0, 1, 2, 3, 4, 5):
                action = self._fallback_action(ctx)

            self.last_action = int(action)
            self._snapshot(obs)
            return int(action)
        except Exception:
            return int(self._emergency_action(obs))

    # ------------------------------------------------------------------
    # State tracking
    # ------------------------------------------------------------------

    def _reset_state(self):
        self.turn = 0
        self.last_action = self.STOP
        self.last_bonuses = None
        self.bomb_radii = {}
        self.prev_grid = None
        self.prev_players = None
        self.prev_bombs = []
        self.seen_bombs = set()
        self.stats = [
            {"bombs": 0, "boxes": 0, "items": 0, "kills": 0, "deaths": 0}
            for _ in range(4)
        ]
        self.max_capacity_seen = 1

    def _looks_like_new_game(self, obs):
        if self.turn <= 0:
            return False
        bombs = obs.get("bombs", [])
        if len(bombs) != 0:
            return False
        players = obs.get("players")
        grid = obs.get("map")
        if players is None or grid is None or len(players) < 4:
            return False
        h, w = grid.shape
        starts = {(1, 1), (h - 2, w - 2), (1, w - 2), (h - 2, 1)}
        seen = set()
        for p in players[:4]:
            if int(p[2]) != 1:
                return False
            seen.add((int(p[0]), int(p[1])))
        return seen == starts

    def _snapshot(self, obs):
        players = obs.get("players")
        if players is not None:
            self.last_bonuses = [int(p[4]) for p in players]
            self.prev_players = players.copy()
        grid = obs.get("map")
        if grid is not None:
            self.prev_grid = grid.copy()
        self.prev_bombs = self._bombs_from_obs(obs)

    def _observe_transition(self, obs):
        if self.stats is None:
            self._reset_state()

        players = obs.get("players")
        grid = obs.get("map")
        if players is None or grid is None or self.agent_id >= len(players):
            return
        while len(self.stats) < len(players):
            self.stats.append({"bombs": 0, "boxes": 0, "items": 0, "kills": 0, "deaths": 0})

        current_bombs = set()
        for row in obs.get("bombs", []):
            bx, by, timer, owner = self._parse_bomb_row(row)
            if timer <= 0:
                continue
            key = (bx, by, owner)
            current_bombs.add(key)
            if key not in self.seen_bombs and 0 <= owner < len(self.stats):
                self.stats[owner]["bombs"] += 1
            self.seen_bombs.add(key)
        for key in list(self.seen_bombs):
            if key not in current_bombs:
                self.seen_bombs.discard(key)

        if self.prev_players is not None:
            for i, p in enumerate(players):
                if i >= len(self.prev_players) or i >= len(self.stats):
                    continue
                prev = self.prev_players[i]
                if int(prev[2]) == 1 and int(p[2]) == 1:
                    radius_gain = max(0, int(p[4]) - int(prev[4]))
                    if radius_gain:
                        self.stats[i]["items"] += radius_gain
                if int(prev[2]) == 1 and int(p[2]) == 0:
                    self.stats[i]["deaths"] += 1

        own = players[self.agent_id]
        active_own_bombs = sum(1 for row in obs.get("bombs", []) if int(row[3]) == self.agent_id)
        possible_capacity = int(own[3]) + active_own_bombs
        if possible_capacity > self.max_capacity_seen:
            self.stats[self.agent_id]["items"] += possible_capacity - self.max_capacity_seen
            self.max_capacity_seen = possible_capacity

        if self.prev_grid is None or self.prev_players is None:
            return

        current_keys = {
            (int(row[0]), int(row[1]), int(row[3]))
            for row in obs.get("bombs", [])
            if len(row) >= 4
        }
        exploded = []
        for bomb in self.prev_bombs:
            key = (bomb["pos"][0], bomb["pos"][1], bomb["owner"])
            if key not in current_keys:
                exploded.append(bomb)

        credited_boxes = set()
        for bomb in exploded:
            owner = bomb["owner"]
            if not (0 <= owner < len(self.stats)):
                continue
            blast = self._blast_tiles(self.prev_grid, bomb["pos"][0], bomb["pos"][1], bomb["radius"])
            for x, y in blast:
                tile_key = (owner, x, y)
                if tile_key in credited_boxes:
                    continue
                if int(self.prev_grid[x, y]) == self.BOX and int(grid[x, y]) != self.BOX:
                    credited_boxes.add(tile_key)
                    self.stats[owner]["boxes"] += 1
            for i, prev_p in enumerate(self.prev_players):
                if i == owner or i >= len(players):
                    continue
                if int(prev_p[2]) == 1 and int(players[i][2]) == 0:
                    if (int(prev_p[0]), int(prev_p[1])) in blast:
                        self.stats[owner]["kills"] += 1

    def _update_bomb_memory(self, obs):
        players = obs["players"]
        observed = set()
        for row in obs.get("bombs", []):
            bx, by, timer, owner = self._parse_bomb_row(row)
            if timer <= 0:
                continue
            key = (bx, by, owner)
            observed.add(key)
            if key in self.bomb_radii:
                continue
            if self.last_bonuses is not None and 0 <= owner < len(self.last_bonuses):
                bonus = self.last_bonuses[owner]
            elif 0 <= owner < len(players):
                bonus = int(players[owner][4])
            else:
                bonus = 0
            self.bomb_radii[key] = self._clamp_radius(1 + bonus)
        for key in list(self.bomb_radii):
            if key not in observed:
                del self.bomb_radii[key]

    # ------------------------------------------------------------------
    # Context and decision policy
    # ------------------------------------------------------------------

    def _build_context(self, obs):
        grid = obs["map"]
        players = obs["players"]
        my = players[self.agent_id]
        my_pos = (int(my[0]), int(my[1]))
        my_radius = self._clamp_radius(1 + int(my[4]))

        enemies = []
        for i, p in enumerate(players):
            if i == self.agent_id or int(p[2]) != 1:
                continue
            enemies.append(
                {
                    "id": i,
                    "pos": (int(p[0]), int(p[1])),
                    "bombs_left": int(p[3]),
                    "radius": self._clamp_radius(1 + int(p[4])),
                    "stats": dict(self.stats[i]) if self.stats and i < len(self.stats) else {},
                }
            )

        bombs = self._bombs_from_obs(obs)
        schedule = self._compute_schedule(grid, bombs)
        ctx = {
            "grid": grid,
            "players": players,
            "my_pos": my_pos,
            "my_bombs_left": int(my[3]),
            "my_bonus": int(my[4]),
            "my_radius": my_radius,
            "enemies": enemies,
            "alive_count": 1 + len(enemies),
            "bombs": schedule["bombs"],
            "danger": schedule["danger"],
            "det_by_pos": schedule["det_by_pos"],
            "bomb_positions": {bomb["pos"] for bomb in schedule["bombs"]},
            "enemy_risk": self._enemy_potential_blasts(grid, enemies, schedule["bombs"]),
            "phase": "end" if len(enemies) <= 1 else ("mid" if len(enemies) == 2 else "early"),
            "my_stats": dict(self.stats[self.agent_id]) if self.stats else {},
        }
        ctx["legal_actions"] = self._legal_actions(ctx)
        ctx["filtered_actions"] = self._filtered_actions(ctx)
        return ctx

    def _choose_action(self, ctx, started):
        my_pos = ctx["my_pos"]
        imminent = self._soonest_hit(my_pos, 1, ctx["danger"])
        if imminent is not None and imminent <= 2:
            escape = self._best_escape_action(ctx)
            if escape is not None:
                return escape

        bomb_score = self._score_bomb_at(ctx, my_pos)
        if self.BOMB in ctx["filtered_actions"] and bomb_score["score"] >= self._bomb_threshold(ctx, bomb_score):
            return self.BOMB

        if imminent is not None and imminent <= 4:
            escape = self._best_escape_action(ctx)
            if escape is not None:
                return escape

        if time.perf_counter() - started < self.TIME_BUDGET_S:
            objective = self._objective_action(ctx, started)
            if objective is not None:
                return objective

        return self._fallback_action(ctx)

    def _bomb_threshold(self, ctx, bomb_score):
        boxes = bomb_score["boxes"]
        enemy_hits = bomb_score["enemy_hits"]
        trap = bomb_score["trap_score"]
        phase = ctx["phase"]
        if enemy_hits or trap >= 1.4:
            if enemy_hits and trap < 1.0:
                return 4.8 if phase == "end" else 6.1
            return 3.0 if phase == "end" else 4.2
        if boxes >= 3:
            return 4.8 if phase != "end" else 6.4
        if boxes == 2:
            return 4.1 if phase != "end" else 6.0
        if boxes == 1:
            if self.turn < 190:
                return 2.8
            return 3.7 if phase != "end" else 6.2
        return 5.8 if phase == "end" else 7.0

    # ------------------------------------------------------------------
    # Action pruning adapted from Pommerman action filtering
    # ------------------------------------------------------------------

    def _filtered_actions(self, ctx):
        safe = []
        for action in ctx["legal_actions"]:
            if action == self.BOMB:
                hypo = self._schedule_with_hypothetical(ctx, ctx["my_pos"], ctx["my_radius"], 7)
                if self._is_deadly(ctx["my_pos"], 1, hypo["danger"]):
                    continue
                escape = self._find_escape(ctx["grid"], ctx["my_pos"], 1, hypo["bombs"], hypo["danger"])
                if escape and self._escape_is_roomy(ctx, escape, hypo, ctx["my_pos"]):
                    safe.append(action)
                continue

            npos = self._next_pos(ctx["my_pos"], action)
            if self._is_deadly(npos, 1, ctx["danger"]):
                continue
            if not self._covered_after(npos, 1, ctx["danger"], self.HORIZON):
                safe.append(action)
                continue
            if self._find_escape(ctx["grid"], npos, 1, ctx["bombs"], ctx["danger"]):
                safe.append(action)

        if safe:
            return safe

        # Last-resort pruning: keep actions that do not die on the next tick.
        fallback = []
        for action in ctx["legal_actions"]:
            if action == self.BOMB:
                continue
            npos = self._next_pos(ctx["my_pos"], action)
            if not self._is_deadly(npos, 1, ctx["danger"]):
                fallback.append(action)
        return fallback if fallback else [self.STOP]

    def _find_escape(self, grid, start, start_t, bombs, danger, horizon=None):
        if horizon is None:
            horizon = self.HORIZON
        q = deque([(start, start_t, None)])
        seen = {(start, start_t)}

        while q:
            pos, t, first = q.popleft()
            if not self._is_deadly(pos, t, danger) and not self._covered_after(pos, t, danger, horizon):
                return (first if first is not None else self.STOP, t, pos)
            if t - start_t >= horizon:
                continue

            for action in self.SEARCH_ACTIONS:
                npos = self._next_pos(pos, action)
                nt = t + 1
                if action != self.STOP:
                    if not self._passable(grid, npos[0], npos[1]):
                        continue
                    if self._bomb_blocks(pos, npos, nt, bombs):
                        continue
                if self._is_deadly(npos, nt, danger):
                    continue
                state = (npos, nt)
                if state in seen:
                    continue
                seen.add(state)
                q.append((npos, nt, action if first is None else first))
        return None

    # ------------------------------------------------------------------
    # Objectives
    # ------------------------------------------------------------------

    def _objective_action(self, ctx, started):
        reach = self._temporal_reach_map(ctx, self.HORIZON)
        best = None

        item = self._best_item_move(ctx, reach)
        if item is not None:
            best = item

        if time.perf_counter() - started < self.TIME_BUDGET_S:
            attack = self._best_attack_spot(ctx, reach)
            if attack is not None and (best is None or attack[0] > best[0]):
                best = attack

        if time.perf_counter() - started < self.TIME_BUDGET_S:
            box = self._best_box_spot(ctx, reach)
            if box is not None and (best is None or box[0] > best[0]):
                best = box

        if best is not None and best[1] in ctx["filtered_actions"]:
            return best[1]
        return None

    def _best_item_move(self, ctx, reach):
        grid = ctx["grid"]
        best = None
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                cell = int(grid[x, y])
                if cell not in (self.ITEM_RADIUS, self.ITEM_CAPACITY):
                    continue
                path = reach.get((x, y))
                if path is None:
                    continue
                dist, action = path
                value = self._item_value(ctx, cell)
                enemy_dist = self._nearest_enemy_distance(ctx, (x, y))
                if enemy_dist is not None and enemy_dist <= dist:
                    value -= 1.4
                if (x, y) in ctx["enemy_risk"]:
                    value -= 0.8
                score = value - 0.45 * dist
                if best is None or score > best[0]:
                    best = (score, action)
        return best

    def _best_attack_spot(self, ctx, reach):
        if not ctx["enemies"]:
            return None
        grid = ctx["grid"]
        candidates = []
        for x in range(1, grid.shape[0] - 1):
            for y in range(1, grid.shape[1] - 1):
                pos = (x, y)
                if pos not in reach or pos in ctx["bomb_positions"]:
                    continue
                if not self._passable(grid, x, y):
                    continue
                blast = self._blast_tiles(grid, x, y, ctx["my_radius"])
                hit_enemies = [enemy for enemy in ctx["enemies"] if enemy["pos"] in blast]
                if not hit_enemies:
                    continue
                dist, action = reach[pos]
                if not self._can_escape_after_future_bomb(ctx, pos, dist):
                    continue
                choke = sum(self._dead_zone_score(ctx, enemy["pos"]) for enemy in hit_enemies)
                value = 7.2 * len(hit_enemies) + 1.8 * choke - 0.55 * dist
                if ctx["phase"] == "end":
                    value += 2.3 * len(hit_enemies)
                if pos in ctx["enemy_risk"]:
                    value -= 1.0
                candidates.append((value, action))
        return max(candidates, default=None)

    def _best_box_spot(self, ctx, reach):
        grid = ctx["grid"]
        best = None
        late_penalty = 1.4 if ctx["phase"] == "end" else 0.0
        for x in range(1, grid.shape[0] - 1):
            for y in range(1, grid.shape[1] - 1):
                pos = (x, y)
                if pos not in reach or pos in ctx["bomb_positions"]:
                    continue
                if not self._passable(grid, x, y):
                    continue
                blast = self._blast_tiles(grid, x, y, ctx["my_radius"])
                boxes = sum(1 for tx, ty in blast if int(grid[tx, ty]) == self.BOX)
                if boxes <= 0:
                    continue
                dist, action = reach[pos]
                if dist > 10 and boxes == 1:
                    continue
                if not self._can_escape_after_future_bomb(ctx, pos, dist):
                    continue
                open_n = self._open_neighbors(grid, pos, ctx["bomb_positions"])
                score = boxes * 3.5 + max(0, boxes - 1) * 1.2 + open_n * 0.25
                score -= dist * 0.48 + late_penalty
                if pos in ctx["enemy_risk"]:
                    score -= 0.8
                if best is None or score > best[0]:
                    best = (score, action)
        return best

    def _score_bomb_at(self, ctx, pos):
        if self.BOMB not in ctx["legal_actions"]:
            return self._illegal_bomb_score()
        if self.BOMB not in ctx["filtered_actions"]:
            return self._illegal_bomb_score()

        grid = ctx["grid"]
        blast = self._blast_tiles(grid, pos[0], pos[1], ctx["my_radius"])
        boxes = sum(1 for tile in blast if int(grid[tile[0], tile[1]]) == self.BOX)
        hit_enemies = [enemy for enemy in ctx["enemies"] if enemy["pos"] in blast]

        hypo = self._schedule_with_hypothetical(ctx, pos, ctx["my_radius"], 7)
        escape = self._find_escape(grid, pos, 1, hypo["bombs"], hypo["danger"])
        if escape is None:
            return self._illegal_bomb_score()
        _, escape_t, escape_pos = escape
        det_time = hypo["det_by_pos"].get(pos, 7)
        slack = det_time - escape_t
        escape_area = self._reachable_count(grid, escape_pos, escape_t, hypo["bombs"], hypo["danger"], 14)

        trap_score = 0.0
        for enemy in ctx["enemies"]:
            epos = enemy["pos"]
            if epos not in blast:
                if self._manhattan(pos, epos) <= ctx["my_radius"] + 1:
                    trap_score += 0.15
                continue
            enemy_escape = self._find_escape(grid, epos, 0, hypo["bombs"], hypo["danger"], horizon=8)
            if enemy_escape is None:
                trap_score += 2.6
            else:
                _, et, e_safe = enemy_escape
                area = self._reachable_count(grid, e_safe, et, hypo["bombs"], hypo["danger"], 8)
                if area <= 2:
                    trap_score += 1.3
                elif area <= 5:
                    trap_score += 0.7
                else:
                    trap_score += 0.2
            trap_score += 0.35 * self._dead_zone_score(ctx, epos)

        if not hit_enemies:
            min_area = 5 if ctx["phase"] == "end" else 7
            if escape_area < min_area:
                return self._illegal_bomb_score()
            if escape_pos in ctx["enemy_risk"] and escape_area < 11:
                return self._illegal_bomb_score()
        first_escape_pos = self._next_pos(pos, escape[0])
        risky_escape = first_escape_pos in ctx["enemy_risk"] or escape_pos in ctx["enemy_risk"]

        score = boxes * 3.2 + max(0, boxes - 1) * 1.0
        score += len(hit_enemies) * 5.7 + trap_score * 2.1
        score += min(escape_area, 12) * 0.10
        if ctx["phase"] == "end":
            score += len(hit_enemies) * 2.0
            if boxes and not hit_enemies:
                score -= 2.4
        if slack < 3:
            score -= (3 - slack) * 1.7
        if escape_area <= 3:
            score -= 2.2
        if boxes == 0 and not hit_enemies and trap_score < 1.0:
            score -= 4.0
        if pos in ctx["enemy_risk"]:
            score -= 0.8
        if escape_pos in ctx["enemy_risk"]:
            score -= 2.4
        if risky_escape and trap_score < 1.2:
            if boxes == 0:
                return self._illegal_bomb_score()
            score -= 3.0

        return {
            "score": score,
            "boxes": boxes,
            "enemy_hits": len(hit_enemies),
            "trap_score": trap_score,
            "slack": slack,
            "escape_area": escape_area,
        }

    def _illegal_bomb_score(self):
        return {
            "score": -10**9,
            "boxes": 0,
            "enemy_hits": 0,
            "trap_score": 0.0,
            "slack": -99,
            "escape_area": 0,
        }

    # ------------------------------------------------------------------
    # Pathing and scoring helpers
    # ------------------------------------------------------------------

    def _temporal_reach_map(self, ctx, horizon):
        grid = ctx["grid"]
        start = ctx["my_pos"]
        allowed_first = {a for a in ctx["filtered_actions"] if a != self.BOMB}
        q = deque([(start, 0, self.STOP)])
        seen = {(start, 0)}
        reach = {start: (0, self.STOP)}

        while q:
            pos, t, first = q.popleft()
            if t >= horizon:
                continue
            actions = allowed_first if t == 0 else self.SEARCH_ACTIONS
            for action in actions:
                npos = self._next_pos(pos, action)
                nt = t + 1
                if action != self.STOP:
                    if not self._passable(grid, npos[0], npos[1]):
                        continue
                    if self._bomb_blocks(pos, npos, nt, ctx["bombs"]):
                        continue
                if self._is_deadly(npos, nt, ctx["danger"]):
                    continue
                state = (npos, nt)
                if state in seen:
                    continue
                seen.add(state)
                first_action = action if t == 0 else first
                if npos not in reach:
                    reach[npos] = (nt, first_action)
                q.append((npos, nt, first_action))
        return reach

    def _best_escape_action(self, ctx):
        candidates = []
        for action in ctx["filtered_actions"]:
            if action == self.BOMB:
                continue
            npos = self._next_pos(ctx["my_pos"], action)
            if self._is_deadly(npos, 1, ctx["danger"]):
                continue
            area = self._reachable_count(ctx["grid"], npos, 1, ctx["bombs"], ctx["danger"], 18)
            next_hit = self._soonest_hit(npos, 1, ctx["danger"])
            slack = 20 if next_hit is None else next_hit - 1
            score = area * 0.7 + min(slack, 8) * 1.1 + self._open_neighbors(ctx["grid"], npos, ctx["bomb_positions"])
            score += self._distance_from_bombs(npos, ctx["bombs"]) * 0.12
            if npos in ctx["enemy_risk"]:
                score -= 2.2
            if action == self.STOP:
                score -= 0.4
            candidates.append((npos in ctx["enemy_risk"], score, action))
        if not candidates:
            return None
        if any(not risky for risky, _, _ in candidates):
            candidates = [item for item in candidates if not item[0]]
        return max(candidates, key=lambda item: item[1])[2]

    def _fallback_action(self, ctx):
        best = None
        for action in ctx["filtered_actions"]:
            if action == self.BOMB:
                continue
            npos = self._next_pos(ctx["my_pos"], action)
            if self._is_deadly(npos, 1, ctx["danger"]):
                continue
            area = self._reachable_count(ctx["grid"], npos, 1, ctx["bombs"], ctx["danger"], 16)
            item_bonus = 0.0
            if self._passable(ctx["grid"], npos[0], npos[1]):
                cell = int(ctx["grid"][npos[0], npos[1]])
                if cell in (self.ITEM_RADIUS, self.ITEM_CAPACITY):
                    item_bonus = self._item_value(ctx, cell)
            enemy_dist = self._nearest_enemy_distance(ctx, npos)
            enemy_score = 0.0 if enemy_dist is None else -0.05 * enemy_dist
            danger_time = self._soonest_hit(npos, 1, ctx["danger"])
            safety = 7.0 if danger_time is None else max(0.0, danger_time - 1) * 0.8
            score = area * 0.45 + safety + item_bonus + enemy_score
            score += self._open_neighbors(ctx["grid"], npos, ctx["bomb_positions"]) * 0.25
            if npos in ctx["enemy_risk"]:
                score -= 1.8
            if action == self.STOP:
                score -= 0.25
            if action == self._opposite(self.last_action):
                score -= 0.05
            tie = ((self.turn * 17 + self.agent_id * 31 + action * 7) % 19) * 0.001
            score += tie
            if best is None or score > best[0]:
                best = (score, action)
        if best is not None:
            return best[1]
        for action in ctx["legal_actions"]:
            if action != self.BOMB:
                return action
        return self.STOP

    def _can_escape_after_future_bomb(self, ctx, pos, dist):
        if ctx["my_bombs_left"] <= 0 or pos in ctx["bomb_positions"]:
            return False
        det_time = dist + 7
        start_t = dist + 1
        hypo = self._schedule_with_hypothetical(ctx, pos, ctx["my_radius"], det_time)
        if self._is_deadly(pos, start_t, hypo["danger"]):
            return False
        escape = self._find_escape(ctx["grid"], pos, start_t, hypo["bombs"], hypo["danger"])
        return bool(escape and self._escape_is_roomy(ctx, escape, hypo, pos))

    def _escape_is_roomy(self, ctx, escape, schedule, start_pos=None):
        first_action, escape_t, escape_pos = escape
        area = self._reachable_count(ctx["grid"], escape_pos, escape_t, schedule["bombs"], schedule["danger"], 14)
        min_area = 4 if ctx["phase"] == "end" else 6
        if area < min_area:
            return False
        if start_pos is not None:
            first_pos = self._next_pos(start_pos, first_action)
            if first_pos in ctx["enemy_risk"] and area < 12:
                return False
        if escape_pos in ctx["enemy_risk"] and area < 10:
            return False
        return True

    def _reachable_count(self, grid, start, start_t, bombs, danger, limit, horizon=10):
        if self._is_deadly(start, start_t, danger):
            return 0
        q = deque([(start, start_t)])
        seen = {(start, start_t)}
        positions = {start}
        while q and len(positions) < limit:
            pos, t = q.popleft()
            if t - start_t >= horizon:
                continue
            for action in self.SEARCH_ACTIONS:
                npos = self._next_pos(pos, action)
                nt = t + 1
                if action != self.STOP:
                    if not self._passable(grid, npos[0], npos[1]):
                        continue
                    if self._bomb_blocks(pos, npos, nt, bombs):
                        continue
                if self._is_deadly(npos, nt, danger):
                    continue
                state = (npos, nt)
                if state in seen:
                    continue
                seen.add(state)
                positions.add(npos)
                q.append((npos, nt))
        return len(positions)

    # ------------------------------------------------------------------
    # Bomb schedule and map primitives
    # ------------------------------------------------------------------

    def _bombs_from_obs(self, obs):
        players = obs.get("players", [])
        bombs = []
        for row in obs.get("bombs", []):
            bx, by, timer, owner = self._parse_bomb_row(row)
            key = (bx, by, owner)
            radius = self.bomb_radii.get(key)
            if radius is None:
                if self.last_bonuses is not None and 0 <= owner < len(self.last_bonuses):
                    radius = self._clamp_radius(1 + self.last_bonuses[owner])
                elif 0 <= owner < len(players):
                    radius = self._clamp_radius(1 + int(players[owner][4]))
                else:
                    radius = 1
            bombs.append({"pos": (bx, by), "timer": max(1, timer), "owner": owner, "radius": radius})
        return bombs

    def _compute_schedule(self, grid, bombs):
        scheduled = [dict(bomb) for bomb in bombs if int(bomb.get("timer", 0)) > 0]
        det = [max(1, int(bomb["timer"])) for bomb in scheduled]
        blasts = [
            self._blast_tiles(grid, bomb["pos"][0], bomb["pos"][1], bomb["radius"])
            for bomb in scheduled
        ]

        changed = True
        while changed:
            changed = False
            for i, blast in enumerate(blasts):
                for j, other in enumerate(scheduled):
                    if i == j:
                        continue
                    if other["pos"] in blast and det[i] < det[j]:
                        det[j] = det[i]
                        changed = True

        danger = {}
        det_by_pos = {}
        for bomb, det_time, blast in zip(scheduled, det, blasts):
            bomb["det"] = det_time
            old = det_by_pos.get(bomb["pos"])
            det_by_pos[bomb["pos"]] = det_time if old is None else min(old, det_time)
            for tile in blast:
                danger.setdefault(tile, set()).add(det_time)
        return {"bombs": scheduled, "danger": danger, "det_by_pos": det_by_pos}

    def _schedule_with_hypothetical(self, ctx, pos, radius, timer):
        bombs = [dict(bomb) for bomb in ctx["bombs"]]
        bombs.append(
            {
                "pos": pos,
                "timer": max(1, int(timer)),
                "owner": self.agent_id,
                "radius": self._clamp_radius(radius),
                "hypothetical": True,
            }
        )
        return self._compute_schedule(ctx["grid"], bombs)

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(int(bx), int(by))}
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for step in range(1, int(radius) + 1):
                x = int(bx) + dx * step
                y = int(by) + dy * step
                if not self._in_bounds(grid, x, y):
                    break
                cell = int(grid[x, y])
                if cell == self.WALL:
                    break
                tiles.add((x, y))
                if cell == self.BOX:
                    break
        return tiles

    def _enemy_potential_blasts(self, grid, enemies, bombs):
        risk = set()
        for enemy in enemies:
            soon_refill = any(
                bomb["owner"] == enemy["id"] and int(bomb.get("det", bomb.get("timer", 7))) <= 3
                for bomb in bombs
            )
            if enemy["bombs_left"] <= 0 and not soon_refill:
                continue
            depth = 2 if soon_refill else 1
            origins = self._static_reachable_origins(grid, enemy["pos"], depth)
            for origin in origins:
                risk.update(self._blast_tiles(grid, origin[0], origin[1], enemy["radius"]))
        return risk

    def _static_reachable_origins(self, grid, start, depth):
        q = deque([(start, 0)])
        seen = {start}
        while q:
            pos, d = q.popleft()
            if d >= depth:
                continue
            for action in self.DIRS:
                npos = self._next_pos(pos, action)
                if npos in seen:
                    continue
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                seen.add(npos)
                q.append((npos, d + 1))
        return seen

    def _legal_actions(self, ctx):
        grid = ctx["grid"]
        pos = ctx["my_pos"]
        actions = [self.STOP]
        for action in self.DIRS:
            npos = self._next_pos(pos, action)
            if self._passable(grid, npos[0], npos[1]) and npos not in ctx["bomb_positions"]:
                actions.append(action)
        if ctx["my_bombs_left"] > 0 and pos not in ctx["bomb_positions"]:
            actions.append(self.BOMB)
        return actions

    def _bomb_blocks(self, old_pos, new_pos, nt, bombs):
        if new_pos == old_pos:
            return False
        for bomb in bombs:
            if bomb["pos"] == new_pos and nt <= int(bomb.get("det", bomb.get("timer", 7))):
                return True
        return False

    def _is_deadly(self, pos, t, danger):
        return int(t) in danger.get(pos, ())

    def _covered_after(self, pos, t, danger, horizon):
        for hit_t in danger.get(pos, ()):
            if int(t) <= int(hit_t) <= int(t) + int(horizon):
                return True
        return False

    def _soonest_hit(self, pos, t, danger):
        future = [int(hit_t) for hit_t in danger.get(pos, ()) if int(hit_t) >= int(t)]
        return min(future) if future else None

    def _next_pos(self, pos, action):
        dx, dy = self.MOVES.get(action, (0, 0))
        return (pos[0] + dx, pos[1] + dy)

    def _in_bounds(self, grid, x, y):
        return 0 <= int(x) < grid.shape[0] and 0 <= int(y) < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and int(grid[x, y]) in (
            self.GRASS,
            self.ITEM_RADIUS,
            self.ITEM_CAPACITY,
        )

    def _parse_bomb_row(self, row):
        return int(row[0]), int(row[1]), int(row[2]), int(row[3])

    def _clamp_radius(self, radius):
        return max(1, min(self.MAX_RADIUS, int(radius)))

    def _open_neighbors(self, grid, pos, blocked):
        count = 0
        for action in self.DIRS:
            npos = self._next_pos(pos, action)
            if self._passable(grid, npos[0], npos[1]) and npos not in blocked:
                count += 1
        return count

    def _dead_zone_score(self, ctx, pos):
        exits = self._open_neighbors(ctx["grid"], pos, ctx["bomb_positions"])
        if exits <= 1:
            return 2.5
        if exits == 2:
            return 1.0
        if exits == 3:
            return 0.25
        return 0.0

    def _item_value(self, ctx, cell):
        if cell == self.ITEM_CAPACITY:
            if ctx["my_bombs_left"] <= 1:
                return 7.4
            return 4.2
        if cell == self.ITEM_RADIUS:
            if ctx["my_bonus"] <= 1:
                return 7.0
            if ctx["my_bonus"] < self.MAX_RADIUS - 1:
                return 3.6
        return 0.0

    def _nearest_enemy_distance(self, ctx, pos):
        if not ctx["enemies"]:
            return None
        return min(self._manhattan(pos, enemy["pos"]) for enemy in ctx["enemies"])

    def _distance_from_bombs(self, pos, bombs):
        if not bombs:
            return 8
        return min(self._manhattan(pos, bomb["pos"]) for bomb in bombs)

    def _manhattan(self, a, b):
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    def _opposite(self, action):
        return {
            self.LEFT: self.RIGHT,
            self.RIGHT: self.LEFT,
            self.UP: self.DOWN,
            self.DOWN: self.UP,
        }.get(action, self.STOP)

    def _emergency_action(self, obs):
        try:
            grid = obs["map"]
            players = obs["players"]
            bombs = obs["bombs"]
            if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
                return self.STOP
            pos = (int(players[self.agent_id][0]), int(players[self.agent_id][1]))
            bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
            danger = set()
            for row in bombs:
                bx, by, timer, owner = self._parse_bomb_row(row)
                if timer <= 1:
                    radius = 1
                    if 0 <= owner < len(players):
                        radius = self._clamp_radius(1 + int(players[owner][4]))
                    danger.update(self._blast_tiles(grid, bx, by, radius))
            for action in self.DIRS:
                npos = self._next_pos(pos, action)
                if self._passable(grid, npos[0], npos[1]) and npos not in bomb_positions and npos not in danger:
                    return action
        except Exception:
            pass
        return self.STOP
