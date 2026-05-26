from collections import deque
import time


class Agent:
    team_id = "TemporalHybridAgent"

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
    DIRS = [LEFT, RIGHT, UP, DOWN]
    SEARCH_ACTIONS = [LEFT, RIGHT, UP, DOWN, STOP]
    MAX_RADIUS = 5
    MAX_CAPACITY = 5
    HORIZON = 15

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.bomb_radii = {}
        self.last_bonuses = None
        self.turn = 0
        self.escape_mode = False
        self.last_action = 0

    def act(self, obs: dict) -> int:
        started = time.perf_counter()
        try:
            if self._looks_like_new_game(obs):
                self._reset_state()

            self.turn += 1
            grid = obs["map"]
            players = obs["players"]
            if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
                self._remember_player_bonuses(players)
                return 0

            self._update_bomb_memory(obs)
            ctx = self._build_context(obs)
            action = self._decide(ctx, started)

            if action not in (0, 1, 2, 3, 4, 5):
                action = self._fallback_action(ctx)
            self.last_action = int(action)
            self._remember_player_bonuses(players)
            return int(action)
        except Exception:
            return int(self._emergency_action(obs))

    # ------------------------------------------------------------------
    # State and observation parsing
    # ------------------------------------------------------------------

    def _reset_state(self):
        self.bomb_radii.clear()
        self.last_bonuses = None
        self.turn = 0
        self.escape_mode = False
        self.last_action = 0

    def _looks_like_new_game(self, obs):
        if self.turn == 0:
            return False
        bombs = obs.get("bombs", [])
        try:
            if len(bombs) != 0:
                return False
        except TypeError:
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

    def _remember_player_bonuses(self, players):
        self.last_bonuses = [int(p[4]) for p in players]

    def _update_bomb_memory(self, obs):
        players = obs["players"]
        observed = set()
        for row in obs["bombs"]:
            bx, by, timer, owner = self._parse_bomb(row)
            if timer <= 0:
                continue
            key = (bx, by, owner)
            observed.add(key)
            if key not in self.bomb_radii:
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

    def _build_context(self, obs):
        grid = obs["map"]
        players = obs["players"]
        my = players[self.agent_id]
        my_pos = (int(my[0]), int(my[1]))
        my_bombs_left = int(my[3])
        my_radius = self._clamp_radius(1 + int(my[4]))

        enemies = []
        enemy_set = set()
        for i, p in enumerate(players):
            if i == self.agent_id or int(p[2]) != 1:
                continue
            pos = (int(p[0]), int(p[1]))
            enemies.append(
                {
                    "id": i,
                    "pos": pos,
                    "bombs_left": int(p[3]),
                    "radius": self._clamp_radius(1 + int(p[4])),
                }
            )
            enemy_set.add(pos)

        bombs = []
        bomb_positions = set()
        for row in obs["bombs"]:
            bx, by, timer, owner = self._parse_bomb(row)
            if timer <= 0:
                continue
            key = (bx, by, owner)
            radius = self.bomb_radii.get(key)
            if radius is None:
                if 0 <= owner < len(players):
                    radius = self._clamp_radius(1 + int(players[owner][4]))
                else:
                    radius = 2
            bomb = {
                "pos": (bx, by),
                "timer": max(1, int(timer)),
                "owner": owner,
                "radius": self._clamp_radius(radius),
                "hypothetical": False,
            }
            bombs.append(bomb)
            bomb_positions.add((bx, by))

        schedule = self._compute_schedule(grid, bombs)
        enemy_risk = self._enemy_potential_blasts(grid, enemies, bomb_positions)

        ctx = {
            "grid": grid,
            "players": players,
            "my_pos": my_pos,
            "my_bombs_left": my_bombs_left,
            "my_radius": my_radius,
            "my_bonus": int(my[4]),
            "enemies": enemies,
            "enemy_set": enemy_set,
            "bombs": schedule["bombs"],
            "bomb_positions": bomb_positions,
            "danger_at": schedule["danger_at"],
            "earliest": schedule["earliest"],
            "enemy_risk": enemy_risk,
            "valid_actions": None,
        }
        ctx["valid_actions"] = self._valid_actions(ctx)
        return ctx

    def _parse_bomb(self, row):
        bx = int(row[0])
        by = int(row[1])
        timer = int(row[2]) if len(row) > 2 else 7
        owner = int(row[3]) if len(row) > 3 else -1
        return bx, by, timer, owner

    def _clamp_radius(self, radius):
        return max(1, min(self.MAX_RADIUS, int(radius)))

    # ------------------------------------------------------------------
    # Decision policy
    # ------------------------------------------------------------------

    def _decide(self, ctx, started):
        my_pos = ctx["my_pos"]
        danger_at = ctx["danger_at"]
        next_hit = self._next_danger_time(my_pos, 1, danger_at)

        if self.escape_mode and self._is_future_safe(my_pos, 1, danger_at):
            if self._reachable_count(ctx, my_pos, 0, ctx["bombs"], danger_at, 9) >= 8:
                self.escape_mode = False

        if next_hit == 1 or (self.escape_mode and next_hit is not None):
            action = self._best_escape_action(ctx)
            if action is not None:
                return action
            return self._fallback_action(ctx)

        current_bomb = self._score_bomb_at(ctx, my_pos)
        if current_bomb["legal"] and current_bomb["score"] >= self._bomb_threshold(current_bomb):
            self.escape_mode = True
            return self.BOMB
        if time.perf_counter() - started > 0.08:
            return self._quick_safe_action(ctx)

        if next_hit is not None and next_hit <= 3:
            action = self._best_escape_action(ctx)
            if action is not None:
                return action

        objective = self._choose_objective(ctx, started)
        if objective is not None:
            return objective

        return self._fallback_action(ctx)

    def _bomb_threshold(self, bomb):
        if bomb["enemy_hits"] or bomb["trap_score"] >= 1.5:
            return 5.5 if self.turn < 120 else 4.2
        if bomb["boxes"] >= 3:
            return 5.0
        if bomb["boxes"] == 2:
            return 4.1
        if bomb["boxes"] == 1:
            return 3.2 if self.turn < 180 else 3.7
        return 6.0

    def _choose_objective(self, ctx, started):
        best = None

        if time.perf_counter() - started > 0.08:
            return None

        reach = self._temporal_reach_map(ctx, self.HORIZON)

        item_choice = self._best_item_move(ctx, reach)
        if item_choice is not None:
            best = item_choice

        if time.perf_counter() - started < 0.075:
            box_choice = self._best_box_position_move(ctx, reach)
            if box_choice is not None and (best is None or box_choice[0] > best[0]):
                best = box_choice

        if time.perf_counter() - started < 0.08:
            enemy_choice = self._best_enemy_pressure_move(ctx, reach)
            if enemy_choice is not None and (best is None or enemy_choice[0] > best[0]):
                best = enemy_choice

        if best is None:
            return None
        return best[1]

    # ------------------------------------------------------------------
    # Bomb scoring and objectives
    # ------------------------------------------------------------------

    def _score_bomb_at(self, ctx, pos):
        if ctx["my_bombs_left"] <= 0 or pos in ctx["bomb_positions"]:
            return self._illegal_bomb_score()

        grid = ctx["grid"]
        radius = ctx["my_radius"]
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        boxes = sum(1 for tile in blast if int(grid[tile[0], tile[1]]) == self.BOX)
        enemy_hits = [enemy for enemy in ctx["enemies"] if enemy["pos"] in blast]

        hypo_bombs, hypo_danger, hypo_earliest = self._hypothetical_schedule(ctx, pos, radius)
        if self._is_deadly(pos, 1, hypo_danger):
            return self._illegal_bomb_score()

        escape = self._find_escape(
            grid,
            pos,
            1,
            hypo_bombs,
            hypo_danger,
            horizon=self.HORIZON,
        )
        if escape is None:
            return self._illegal_bomb_score()

        escape_action, escape_time, escape_pos = escape
        own_det = 7
        slack = own_det - escape_time
        escape_area = self._reachable_count(
            ctx, escape_pos, escape_time, hypo_bombs, hypo_danger, 8
        )

        trap_score = 0.0
        for enemy in ctx["enemies"]:
            epos = enemy["pos"]
            if epos not in blast:
                if self._manhattan(pos, epos) <= radius + 1:
                    trap_score += 0.25
                continue
            enemy_escape = self._find_escape(
                grid,
                epos,
                0,
                hypo_bombs,
                hypo_danger,
                horizon=7,
                ignore_start_bomb=True,
            )
            if enemy_escape is None:
                trap_score += 2.8
            else:
                _, etime, e_safe = enemy_escape
                enemy_area = self._reachable_count(
                    ctx, e_safe, etime, hypo_bombs, hypo_danger, 5
                )
                if enemy_area <= 2:
                    trap_score += 1.4
                elif enemy_area <= 5:
                    trap_score += 0.8
                else:
                    trap_score += 0.25

        item_value = boxes * 0.9
        multi_bonus = max(0, boxes - 1) * 1.2
        score = boxes * 3.0 + multi_bonus + item_value
        score += len(enemy_hits) * 6.5 + trap_score * 3.0
        score += min(escape_area, 12) * 0.12
        if slack < 3:
            score -= (3 - slack) * 1.6
        if escape_area <= 3:
            score -= 2.0
        if pos in ctx["enemy_risk"]:
            score -= 1.0
        if boxes == 0 and not enemy_hits and trap_score < 1.0:
            score -= 4.0

        return {
            "legal": True,
            "score": score,
            "boxes": boxes,
            "enemy_hits": len(enemy_hits),
            "trap_score": trap_score,
            "slack": slack,
            "escape_area": escape_area,
            "escape_action": escape_action,
        }

    def _illegal_bomb_score(self):
        return {
            "legal": False,
            "score": -10**9,
            "boxes": 0,
            "enemy_hits": 0,
            "trap_score": 0.0,
            "slack": -99,
            "escape_area": 0,
            "escape_action": None,
        }

    def _best_item_move(self, ctx, reach):
        grid = ctx["grid"]
        targets = []
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                cell = int(grid[x, y])
                if cell not in (self.ITEM_RADIUS, self.ITEM_CAPACITY):
                    continue
                value = self._item_value(ctx, cell)
                if value <= 0:
                    continue
                targets.append(((x, y), value))

        best = None
        for target, value in targets:
            path = reach.get(target)
            if path is None:
                continue
            dist, action = path
            score = value - dist * 0.55
            if target in ctx["enemy_risk"]:
                score -= 1.2
            if self._adjacent_enemy(ctx, target):
                score -= 1.0
            if best is None or score > best[0]:
                best = (score, action)
        return best

    def _item_value(self, ctx, cell):
        if cell == self.ITEM_CAPACITY:
            if ctx["my_bombs_left"] <= 1:
                return 7.0
            return 4.0
        if cell == self.ITEM_RADIUS:
            if ctx["my_bonus"] <= 1:
                return 6.5
            if ctx["my_bonus"] < self.MAX_RADIUS - 1:
                return 3.5
        return 0.0

    def _best_box_position_move(self, ctx, reach):
        grid = ctx["grid"]
        candidates = []
        for x in range(1, grid.shape[0] - 1):
            for y in range(1, grid.shape[1] - 1):
                pos = (x, y)
                if not self._passable(grid, x, y):
                    continue
                if pos in ctx["bomb_positions"]:
                    continue
                blast = self._blast_tiles(grid, x, y, ctx["my_radius"])
                boxes = sum(1 for tx, ty in blast if int(grid[tx, ty]) == self.BOX)
                if boxes <= 0:
                    continue
                open_n = self._open_neighbors(grid, pos, ctx["bomb_positions"])
                value = boxes * 3.0 + max(0, boxes - 1) * 1.3 + open_n * 0.25
                candidates.append((value, pos, boxes))

        candidates.sort(reverse=True)
        best = None
        for value, pos, boxes in candidates[:24]:
            path = reach.get(pos)
            if path is None:
                continue
            dist, action = path
            score = value - dist * 0.42
            if pos in ctx["enemy_risk"]:
                score -= 0.8
            if boxes >= 2:
                score += 0.6
            if best is None or score > best[0]:
                best = (score, action)
        return best

    def _best_enemy_pressure_move(self, ctx, reach):
        if ctx["my_bombs_left"] <= 0 or not ctx["enemies"]:
            return None
        grid = ctx["grid"]
        candidates = []
        for enemy in ctx["enemies"]:
            epos = enemy["pos"]
            for x in range(1, grid.shape[0] - 1):
                for y in range(1, grid.shape[1] - 1):
                    pos = (x, y)
                    if pos in ctx["bomb_positions"] or not self._passable(grid, x, y):
                        continue
                    if self._line_blast_hits(grid, pos, epos, ctx["my_radius"]):
                        value = 5.2 - self._manhattan(pos, epos) * 0.12
                    elif self._manhattan(pos, epos) <= 3:
                        value = 2.0 - self._manhattan(pos, epos) * 0.25
                    else:
                        continue
                    candidates.append((value, pos))

        candidates.sort(reverse=True)
        best = None
        for value, pos in candidates[:20]:
            path = reach.get(pos)
            if path is None:
                continue
            dist, action = path
            score = value - dist * 0.35
            if pos in ctx["enemy_risk"]:
                score -= 0.9
            if best is None or score > best[0]:
                best = (score, action)
        return best

    # ------------------------------------------------------------------
    # Safety, escape, and path finding
    # ------------------------------------------------------------------

    def _best_escape_action(self, ctx):
        best_action = None
        best_score = -10**9
        for action in ctx["valid_actions"]:
            if action == self.BOMB:
                continue
            npos = self._next_pos(ctx["my_pos"], action)
            if not self._movement_action_legal(ctx, ctx["my_pos"], npos, 1):
                continue
            if self._is_deadly(npos, 1, ctx["danger_at"]):
                continue

            score = 0.0
            if self._is_future_safe(npos, 1, ctx["danger_at"]):
                score += 18.0
            else:
                escape = self._find_escape(
                    ctx["grid"], npos, 1, ctx["bombs"], ctx["danger_at"], self.HORIZON
                )
                if escape is None:
                    score -= 30.0
                else:
                    _, dist, safe_pos = escape
                    score += 13.0 - dist * 0.9
                    score += self._open_neighbors(ctx["grid"], safe_pos, ctx["bomb_positions"]) * 0.6

            score += self._open_neighbors(ctx["grid"], npos, ctx["bomb_positions"]) * 1.3
            score += min(
                self._reachable_count(ctx, npos, 1, ctx["bombs"], ctx["danger_at"], 8),
                16,
            ) * 0.25
            if npos in ctx["enemy_risk"]:
                score -= 1.5
            if npos in ctx["enemy_set"]:
                score -= 0.8
            if action == self.STOP and self._next_danger_time(ctx["my_pos"], 1, ctx["danger_at"]) is not None:
                score -= 4.0

            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _path_to_targets(self, ctx, targets, horizon):
        grid = ctx["grid"]
        start = ctx["my_pos"]
        if start in targets:
            return (self.STOP, 0, start)

        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        while q:
            pos, t, first = q.popleft()
            if t >= horizon:
                continue
            ordered = self._ordered_actions_toward(pos, targets)
            for action in ordered:
                npos = self._next_pos(pos, action)
                nt = t + 1
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
                if npos in targets:
                    return (first_action, nt, npos)
                q.append((npos, nt, first_action))
        return None

    def _temporal_reach_map(self, ctx, horizon):
        grid = ctx["grid"]
        start = ctx["my_pos"]
        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        reach = {start: (0, self.STOP)}

        while q:
            pos, t, first = q.popleft()
            if t >= horizon:
                continue
            for action in self.SEARCH_ACTIONS:
                npos = self._next_pos(pos, action)
                nt = t + 1
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

    def _find_escape(
        self,
        grid,
        start,
        start_time,
        bombs,
        danger_at,
        horizon,
        ignore_start_bomb=False,
    ):
        q = deque([(start, start_time, None)])
        seen = {(start, start_time)}
        while q:
            pos, t, first = q.popleft()
            if t > start_time and self._is_future_safe(pos, t, danger_at):
                return (first if first is not None else self.STOP, t, pos)
            if t - start_time >= horizon:
                continue

            for action in self.SEARCH_ACTIONS:
                npos = self._next_pos(pos, action)
                nt = t + 1
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                if self._bomb_blocks(pos, npos, nt, bombs):
                    if not (ignore_start_bomb and npos == start):
                        continue
                if self._is_deadly(npos, nt, danger_at):
                    continue
                state = (npos, nt)
                if state in seen:
                    continue
                seen.add(state)
                first_action = action if first is None else first
                q.append((npos, nt, first_action))
        return None

    def _reachable_count(self, ctx, start, start_time, bombs, danger_at, horizon):
        grid = ctx["grid"]
        q = deque([(start, start_time)])
        seen = {(start, start_time)}
        positions = set()
        while q:
            pos, t = q.popleft()
            if self._is_future_safe(pos, t, danger_at):
                positions.add(pos)
            if t - start_time >= horizon:
                continue
            for action in self.SEARCH_ACTIONS:
                npos = self._next_pos(pos, action)
                nt = t + 1
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                if self._bomb_blocks(pos, npos, nt, bombs):
                    continue
                if self._is_deadly(npos, nt, danger_at):
                    continue
                state = (npos, nt)
                if state in seen:
                    continue
                seen.add(state)
                q.append((npos, nt))
        return len(positions)

    def _fallback_action(self, ctx):
        best = None
        for action in ctx["valid_actions"]:
            if action == self.BOMB:
                continue
            npos = self._next_pos(ctx["my_pos"], action)
            if not self._movement_action_legal(ctx, ctx["my_pos"], npos, 1):
                continue
            if self._is_deadly(npos, 1, ctx["danger_at"]):
                continue
            score = self._open_neighbors(ctx["grid"], npos, ctx["bomb_positions"]) * 1.5
            score += self._reachable_count(ctx, npos, 1, ctx["bombs"], ctx["danger_at"], 8)
            future = self._next_danger_time(npos, 1, ctx["danger_at"])
            if future is None:
                score += 10
            else:
                score += min(future, 7) * 0.4
            if npos in ctx["enemy_risk"]:
                score -= 1.0
            if action == self.STOP:
                score -= 0.25
            if best is None or score > best[0]:
                best = (score, action)
        return best[1] if best is not None else 0

    def _quick_safe_action(self, ctx):
        best = None
        for action in ctx["valid_actions"]:
            if action == self.BOMB:
                continue
            npos = self._next_pos(ctx["my_pos"], action)
            if not self._movement_action_legal(ctx, ctx["my_pos"], npos, 1):
                continue
            if self._is_deadly(npos, 1, ctx["danger_at"]):
                continue
            future = self._next_danger_time(npos, 1, ctx["danger_at"])
            score = self._open_neighbors(ctx["grid"], npos, ctx["bomb_positions"]) * 2.0
            score += 8.0 if future is None else min(future, 7) * 0.6
            if npos in ctx["enemy_risk"]:
                score -= 1.0
            if action == self.STOP:
                score -= 0.2
            if best is None or score > best[0]:
                best = (score, action)
        return best[1] if best is not None else 0

    # ------------------------------------------------------------------
    # Schedule and danger maps
    # ------------------------------------------------------------------

    def _hypothetical_schedule(self, ctx, pos, radius):
        bombs = [dict(b) for b in ctx["bombs"]]
        bombs.append(
            {
                "pos": pos,
                "timer": 7,
                "owner": self.agent_id,
                "radius": self._clamp_radius(radius),
                "hypothetical": True,
            }
        )
        schedule = self._compute_schedule(ctx["grid"], bombs)
        return schedule["bombs"], schedule["danger_at"], schedule["earliest"]

    def _compute_schedule(self, grid, bombs):
        bombs = [dict(b) for b in bombs if int(b.get("timer", 0)) > 0]
        n = len(bombs)
        if n == 0:
            return {"bombs": [], "danger_at": {}, "earliest": {}}

        det = [max(1, int(b["timer"])) for b in bombs]
        processed = [False] * n
        affected_by_time = {}
        sim_grid = grid.copy()
        max_timer = max(max(det), 7)

        for t in range(1, max_timer + 1):
            exploding = [i for i in range(n) if not processed[i] and det[i] == t]
            if not exploding:
                continue
            affected = set()
            while exploding:
                i = exploding.pop()
                if processed[i]:
                    continue
                processed[i] = True
                bx, by = bombs[i]["pos"]
                blast = self._blast_tiles(sim_grid, bx, by, bombs[i]["radius"])
                bombs[i]["blast"] = blast
                affected |= blast
                for j in range(n):
                    if processed[j] or det[j] <= t:
                        continue
                    if bombs[j]["pos"] in blast:
                        det[j] = t
                        exploding.append(j)

            affected_by_time.setdefault(t, set()).update(affected)
            for x, y in affected:
                if int(sim_grid[x, y]) == self.BOX:
                    sim_grid[x, y] = self.GRASS

        danger_at = {}
        earliest = {}
        for t, tiles in affected_by_time.items():
            for tile in tiles:
                danger_at.setdefault(tile, set()).add(t)
                if tile not in earliest or t < earliest[tile]:
                    earliest[tile] = t

        for i, bomb in enumerate(bombs):
            bomb["det_time"] = det[i]
            if "blast" not in bomb:
                bx, by = bomb["pos"]
                bomb["blast"] = self._blast_tiles(grid, bx, by, bomb["radius"])
        return {"bombs": bombs, "danger_at": danger_at, "earliest": earliest}

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(int(bx), int(by))}
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for r in range(1, int(radius) + 1):
                x = int(bx) + dx * r
                y = int(by) + dy * r
                if not self._in_bounds(grid, x, y):
                    break
                cell = int(grid[x, y])
                if cell == self.WALL:
                    break
                tiles.add((x, y))
                if cell == self.BOX:
                    break
        return tiles

    def _enemy_potential_blasts(self, grid, enemies, bomb_positions):
        risk = set()
        for enemy in enemies:
            if enemy["bombs_left"] <= 0 or enemy["pos"] in bomb_positions:
                continue
            ex, ey = enemy["pos"]
            risk |= self._blast_tiles(grid, ex, ey, enemy["radius"])
        return risk

    # ------------------------------------------------------------------
    # Movement and danger predicates
    # ------------------------------------------------------------------

    def _valid_actions(self, ctx):
        actions = [self.STOP]
        my_pos = ctx["my_pos"]
        for action in self.DIRS:
            npos = self._next_pos(my_pos, action)
            if self._movement_action_legal(ctx, my_pos, npos, 1):
                actions.append(action)
        if ctx["my_bombs_left"] > 0 and my_pos not in ctx["bomb_positions"]:
            actions.append(self.BOMB)
        return actions

    def _movement_action_legal(self, ctx, pos, npos, arrival_time):
        if not self._passable(ctx["grid"], npos[0], npos[1]):
            return False
        return not self._bomb_blocks(pos, npos, arrival_time, ctx["bombs"])

    def _bomb_blocks(self, from_pos, to_pos, arrival_time, bombs):
        if to_pos == from_pos:
            return False
        for bomb in bombs:
            if bomb["pos"] == to_pos and int(bomb.get("det_time", bomb["timer"])) >= arrival_time:
                return True
        return False

    def _is_deadly(self, pos, time_step, danger_at):
        return time_step in danger_at.get(pos, ())

    def _is_future_safe(self, pos, time_step, danger_at):
        for t in danger_at.get(pos, ()):
            if t >= time_step:
                return False
        return True

    def _next_danger_time(self, pos, time_step, danger_at):
        best = None
        for t in danger_at.get(pos, ()):
            if t >= time_step and (best is None or t < best):
                best = t
        return best

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and int(grid[x, y]) in (
            self.GRASS,
            self.ITEM_RADIUS,
            self.ITEM_CAPACITY,
        )

    def _in_bounds(self, grid, x, y):
        return 0 <= int(x) < grid.shape[0] and 0 <= int(y) < grid.shape[1]

    def _next_pos(self, pos, action):
        dx, dy = self.MOVES.get(action, (0, 0))
        return (pos[0] + dx, pos[1] + dy)

    def _open_neighbors(self, grid, pos, bomb_positions):
        count = 0
        for action in self.DIRS:
            nx, ny = self._next_pos(pos, action)
            if self._passable(grid, nx, ny) and (nx, ny) not in bomb_positions:
                count += 1
        return count

    def _ordered_actions_toward(self, pos, targets):
        if not targets:
            return self.SEARCH_ACTIONS
        tx, ty = min(targets, key=lambda t: abs(pos[0] - t[0]) + abs(pos[1] - t[1]))
        actions = self.DIRS[:]
        actions.sort(
            key=lambda a: abs(self._next_pos(pos, a)[0] - tx)
            + abs(self._next_pos(pos, a)[1] - ty)
        )
        actions.append(self.STOP)
        return actions

    # ------------------------------------------------------------------
    # Tactical geometry
    # ------------------------------------------------------------------

    def _line_blast_hits(self, grid, origin, target, radius):
        ox, oy = origin
        tx, ty = target
        if ox == tx and abs(ty - oy) <= radius:
            step = 1 if ty > oy else -1
            for y in range(oy + step, ty, step):
                if int(grid[ox, y]) in (self.WALL, self.BOX):
                    return False
            return True
        if oy == ty and abs(tx - ox) <= radius:
            step = 1 if tx > ox else -1
            for x in range(ox + step, tx, step):
                if int(grid[x, oy]) in (self.WALL, self.BOX):
                    return False
            return True
        return False

    def _adjacent_enemy(self, ctx, pos):
        for enemy in ctx["enemies"]:
            if self._manhattan(enemy["pos"], pos) <= 1:
                return True
        return False

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ------------------------------------------------------------------
    # Last-ditch fallback
    # ------------------------------------------------------------------

    def _emergency_action(self, obs):
        try:
            grid = obs["map"]
            players = obs["players"]
            bombs = obs["bombs"]
            if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
                return 0
            my_pos = (int(players[self.agent_id][0]), int(players[self.agent_id][1]))
            danger_now = set()
            bomb_positions = set()
            for b in bombs:
                bx, by, timer, owner = self._parse_bomb(b)
                bomb_positions.add((bx, by))
                if timer <= 1:
                    radius = self._clamp_radius(1 + int(players[owner][4])) if 0 <= owner < len(players) else 2
                    danger_now |= self._blast_tiles(grid, bx, by, radius)
            for action in self.DIRS:
                npos = self._next_pos(my_pos, action)
                if self._passable(grid, npos[0], npos[1]) and npos not in bomb_positions and npos not in danger_now:
                    return action
            return 0
        except Exception:
            return 0
