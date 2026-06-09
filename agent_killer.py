from collections import deque
from heapq import heappop, heappush
import time


class AdaptiveProV2Base:
    team_id = "AdaptiveProV2"

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
        self.prev_grid = None
        self.prev_players = None
        self.prev_bombs = []
        self.seen_bombs = set()
        self.telemetry = {}
        self.max_bombs_left_seen = 1
        self.player_stats = None
        self.max_bombs_left_seen_by_player = None

    def act(self, obs: dict) -> int:
        started = time.perf_counter()
        try:
            if self._looks_like_new_game(obs):
                self._reset_state()

            self._observe_transition(obs)
            self.turn += 1
            grid = obs["map"]
            players = obs["players"]
            if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
                self._remember_player_bonuses(players)
                self._snapshot_obs(obs)
                return 0

            self._update_bomb_memory(obs)
            ctx = self._build_context(obs)
            action = self._decide(ctx, started)

            if action not in (0, 1, 2, 3, 4, 5):
                action = self._fallback_action(ctx)
            self.last_action = int(action)
            self._remember_player_bonuses(players)
            self._snapshot_obs(obs)
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
        self.prev_grid = None
        self.prev_players = None
        self.prev_bombs = []
        self.seen_bombs = set()
        self.telemetry = {
            "own_bombs": 0,
            "own_boxes": 0,
            "own_items": 0,
            "own_kills": 0,
            "enemy_deaths": 0,
            "last_progress_turn": 0,
        }
        self.max_bombs_left_seen = 1
        self.player_stats = [
            {"bombs": 0, "boxes": 0, "items": 0, "kills": 0, "deaths": 0}
            for _ in range(4)
        ]
        self.max_bombs_left_seen_by_player = [1, 1, 1, 1]

    def _snapshot_obs(self, obs):
        self.prev_grid = obs["map"].copy()
        self.prev_players = obs["players"].copy()
        self.prev_bombs = [self._parse_bomb(row) for row in obs["bombs"]]

    def _observe_transition(self, obs):
        if not self.telemetry:
            self._reset_state()
        players = obs.get("players")
        grid = obs.get("map")
        if players is None or grid is None or self.agent_id >= len(players):
            return
        if self.player_stats is None or len(self.player_stats) < len(players):
            self.player_stats = [
                {"bombs": 0, "boxes": 0, "items": 0, "kills": 0, "deaths": 0}
                for _ in range(len(players))
            ]
        if self.max_bombs_left_seen_by_player is None or len(self.max_bombs_left_seen_by_player) < len(players):
            self.max_bombs_left_seen_by_player = [1 for _ in range(len(players))]

        observed_bombs = set()
        for row in obs.get("bombs", []):
            bx, by, timer, owner = self._parse_bomb(row)
            if timer <= 0:
                continue
            stable_key = (bx, by, owner)
            observed_bombs.add(stable_key)
            if 0 <= owner < len(self.player_stats) and stable_key not in self.seen_bombs:
                self.player_stats[owner]["bombs"] += 1
                if owner == self.agent_id:
                    self.telemetry["own_bombs"] += 1
                    self.telemetry["last_progress_turn"] = self.turn
            self.seen_bombs.add(stable_key)

        for key in list(self.seen_bombs):
            if key not in observed_bombs:
                self.seen_bombs.discard(key)

        my = players[self.agent_id]
        prev_max_bombs_left = self.max_bombs_left_seen
        self.max_bombs_left_seen = max(self.max_bombs_left_seen, int(my[3]))
        if self.prev_players is not None:
            for i, p in enumerate(players):
                if i >= len(self.prev_players) or i >= len(self.player_stats):
                    continue
                prev_p = self.prev_players[i]
                if int(p[2]) == 1 and int(prev_p[2]) == 1:
                    radius_gain = max(0, int(p[4]) - int(prev_p[4]))
                    if radius_gain:
                        self.player_stats[i]["items"] += radius_gain
                    old_cap = self.max_bombs_left_seen_by_player[i]
                    new_cap = max(old_cap, int(p[3]))
                    if new_cap > old_cap:
                        self.player_stats[i]["items"] += new_cap - old_cap
                    self.max_bombs_left_seen_by_player[i] = new_cap

            own_items = self.player_stats[self.agent_id]["items"]
            if own_items > self.telemetry["own_items"]:
                self.telemetry["own_items"] = own_items
                self.telemetry["last_progress_turn"] = self.turn
            if int(my[3]) > prev_max_bombs_left:
                self.telemetry["last_progress_turn"] = self.turn

        if self.prev_grid is None or self.prev_players is None:
            return

        exploded_blasts = []
        current_bomb_keys = {(self._parse_bomb(row)[0], self._parse_bomb(row)[1], self._parse_bomb(row)[3]) for row in obs.get("bombs", [])}
        for bx, by, timer, owner in self.prev_bombs:
            if (bx, by, owner) in current_bomb_keys:
                continue
            radius = self.bomb_radii.get((bx, by, owner))
            if radius is None and 0 <= owner < len(self.prev_players):
                radius = self._clamp_radius(1 + int(self.prev_players[owner][4]))
            elif radius is None:
                radius = 1
            exploded_blasts.append((owner, self._blast_tiles(self.prev_grid, bx, by, radius)))

        if exploded_blasts:
            credited_boxes = set()
            for owner, blast in exploded_blasts:
                if not (0 <= owner < len(self.player_stats)):
                    continue
                boxes = 0
                for x, y in blast:
                    tile = (owner, x, y)
                    if tile in credited_boxes:
                        continue
                    if int(self.prev_grid[x, y]) == self.BOX and int(grid[x, y]) != self.BOX:
                        credited_boxes.add(tile)
                        boxes += 1
                if boxes:
                    self.player_stats[owner]["boxes"] += boxes

                for i, prev_p in enumerate(self.prev_players):
                    if i == owner or i >= len(players) or int(prev_p[2]) != 1:
                        continue
                    if int(players[i][2]) == 0 and (int(prev_p[0]), int(prev_p[1])) in blast:
                        self.player_stats[owner]["kills"] += 1

            for i, prev_p in enumerate(self.prev_players):
                if i < len(players) and i < len(self.player_stats):
                    if int(prev_p[2]) == 1 and int(players[i][2]) == 0:
                        self.player_stats[i]["deaths"] += 1

            own = self.player_stats[self.agent_id]
            if own["boxes"] > self.telemetry["own_boxes"]:
                self.telemetry["own_boxes"] = own["boxes"]
                self.telemetry["last_progress_turn"] = self.turn
            if own["kills"] > self.telemetry["own_kills"]:
                self.telemetry["own_kills"] = own["kills"]
                self.telemetry["enemy_deaths"] = own["kills"]
                self.telemetry["last_progress_turn"] = self.turn

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
        estimated_stats = [dict(s) for s in (self.player_stats or [])]
        while len(estimated_stats) < len(players):
            estimated_stats.append({"bombs": 0, "boxes": 0, "items": 0, "kills": 0, "deaths": 0})
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
                    "stats": dict(estimated_stats[i]),
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
        alive_count = 1 + len(enemies)
        phase = "end" if alive_count <= 2 else ("mid" if alive_count == 3 else "early")

        ctx = {
            "grid": grid,
            "players": players,
            "my_pos": my_pos,
            "my_bombs_left": my_bombs_left,
            "my_radius": my_radius,
            "my_bonus": int(my[4]),
            "max_bombs_left_seen": self.max_bombs_left_seen,
            "telemetry": dict(self.telemetry),
            "estimated_stats": estimated_stats,
            "my_stats": dict(estimated_stats[self.agent_id]),
            "boxes_remaining": self._boxes_remaining(grid),
            "enemies": enemies,
            "enemy_set": enemy_set,
            "bombs": schedule["bombs"],
            "bomb_positions": bomb_positions,
            "danger_at": schedule["danger_at"],
            "earliest": schedule["earliest"],
            "enemy_risk": enemy_risk,
            "alive_count": alive_count,
            "phase": phase,
            "valid_actions": None,
        }
        strategic_mode, target_enemy_id = self._strategic_mode(ctx)
        ctx["strategic_mode"] = strategic_mode
        ctx["target_enemy_id"] = target_enemy_id
        ctx["tempo"] = self._tempo_mode(ctx)
        ctx["engagement_deficit"] = self._engagement_deficit(ctx)
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

    def _boxes_remaining(self, grid):
        total = 0
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if int(grid[x, y]) == self.BOX:
                    total += 1
        return total

    def _strategic_mode(self, ctx):
        my_stats = ctx["my_stats"]
        alive_enemy_ids = {enemy["id"] for enemy in ctx["enemies"]}
        enemy_stats = [
            (i, stats)
            for i, stats in enumerate(ctx["estimated_stats"])
            if i != self.agent_id and i in alive_enemy_ids
        ]

        if int(my_stats.get("kills", 0)) >= 2:
            return "escape", None

        killer_enemies = [(i, s) for i, s in enemy_stats if int(s.get("kills", 0)) >= 2]
        if killer_enemies:
            target = max(killer_enemies, key=lambda item: (int(item[1].get("kills", 0)), int(item[1].get("boxes", 0))))[0]
            return "hunt", target

        if ctx["alive_count"] == 2 and ctx["enemies"]:
            enemy = ctx["enemies"][0]
            enemy_id = enemy["id"]
            enemy_s = ctx["estimated_stats"][enemy_id]
            my_kills = int(my_stats.get("kills", 0))
            enemy_kills = int(enemy_s.get("kills", 0))
            my_boxes = int(my_stats.get("boxes", 0))
            enemy_boxes = int(enemy_s.get("boxes", 0))
            if enemy_kills > my_kills:
                return "hunt", enemy_id
            if enemy_kills == my_kills and enemy_boxes == my_boxes:
                my_items = int(my_stats.get("items", 0))
                enemy_items = int(enemy_s.get("items", 0))
                item_gap = my_items - enemy_items
                steps_left = max(0, 500 - self.turn)
                if abs(item_gap) * 10 <= steps_left:
                    return "item_race", enemy_id
                if item_gap > 0:
                    return "escape", None
                return "hunt", enemy_id
            if my_kills > enemy_kills or my_boxes > enemy_boxes:
                return "escape", None

        if ctx["boxes_remaining"] == 0:
            my_boxes = int(my_stats.get("boxes", 0))
            max_boxes = max(int(stats.get("boxes", 0)) for stats in ctx["estimated_stats"])
            if my_boxes < max_boxes and ctx["enemies"]:
                target = max(ctx["enemies"], key=lambda e: (int(e["stats"].get("boxes", 0)), int(e["stats"].get("kills", 0))))["id"]
                return "hunt", target

        if ctx["boxes_remaining"] > 0:
            my_boxes = int(my_stats.get("boxes", 0))
            max_boxes = max(int(stats.get("boxes", 0)) for stats in ctx["estimated_stats"])
            if my_boxes + 2 < max_boxes and ctx["phase"] != "end":
                return "box_farm", None

        return "adaptive", None

    def _tempo_mode(self, ctx):
        strategic = ctx.get("strategic_mode", "adaptive")
        if strategic == "escape":
            return "escape"
        if strategic == "hunt":
            return "hunt"
        if strategic == "item_race":
            return "item_race"
        if strategic == "box_farm":
            return "box_farm"
        stats = ctx["telemetry"]
        progress_age = self.turn - int(stats.get("last_progress_turn", 0))
        power = ctx["my_radius"] + max(ctx["my_bombs_left"], ctx["max_bombs_left_seen"])
        if ctx["phase"] == "end":
            return "duel" if power >= 4 else "survive_duel"
        if self.turn >= 360:
            return "tiebreak"
        if progress_age >= 75 and ctx["alive_count"] >= 3:
            return "unstuck"
        if ctx["phase"] == "early" and self.turn < 220:
            if ctx["my_bonus"] <= 1 or ctx["max_bombs_left_seen"] <= 1:
                return "economy"
        if power >= 5 and self.turn >= 90:
            return "pressure"
        return "balanced"

    def _engagement_deficit(self, ctx):
        stats = ctx["telemetry"]
        engagement = (
            int(stats.get("own_kills", 0)) * 5
            + int(stats.get("own_boxes", 0))
            + int(stats.get("own_items", 0)) * 2
            + int(stats.get("own_bombs", 0))
        )
        expected = max(2, self.turn // 55)
        return max(0, expected - engagement)

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

        if ctx["strategic_mode"] == "escape":
            action = self._best_survival_action(ctx)
            if action is not None:
                return action

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
        phase = bomb.get("phase", "mid")
        tempo = bomb.get("tempo", "balanced")
        if tempo == "escape":
            return 10**6
        safety_tax = 0.0
        if bomb.get("slack", 7) < 3:
            safety_tax += 2.0
        if bomb.get("escape_area", 10) <= 4:
            safety_tax += 1.0
        if tempo in ("pressure", "tiebreak", "duel") and (bomb["enemy_hits"] or bomb["trap_score"] >= 1.0):
            safety_tax -= 0.7
        if tempo == "economy" and bomb["enemy_hits"] == 0:
            safety_tax -= min(0.8, 0.25 * bomb["boxes"])
        if tempo == "unstuck" and bomb["boxes"] > 0:
            safety_tax -= 0.9
        if tempo == "hunt" and (bomb["enemy_hits"] or bomb["trap_score"] >= 0.8):
            safety_tax -= 1.4
        if tempo == "box_farm" and bomb["boxes"] > 0:
            safety_tax -= min(1.6, 0.45 * bomb["boxes"])
        if tempo == "item_race":
            safety_tax += 0.8
        if bomb["enemy_hits"] or bomb["trap_score"] >= 1.5:
            return (3.6 if phase == "end" else 5.5 if self.turn < 120 else 4.2) + safety_tax
        if bomb["boxes"] >= 3:
            return (5.6 if phase == "end" else 5.0) + safety_tax
        if bomb["boxes"] == 2:
            return (5.4 if phase == "end" else 4.1) + safety_tax
        if bomb["boxes"] == 1:
            return (6.0 if phase == "end" else 3.2 if self.turn < 180 else 3.7) + safety_tax
        return (4.2 if phase == "end" else 6.0) + safety_tax

    def _choose_objective(self, ctx, started):
        best = None

        if time.perf_counter() - started > 0.08:
            return None

        reach = self._temporal_reach_map(ctx, self.HORIZON)

        if ctx["strategic_mode"] == "hunt":
            enemy_choice = self._best_enemy_pressure_move(ctx, reach)
            if enemy_choice is not None:
                best = enemy_choice
            if time.perf_counter() - started < 0.085:
                chase_choice = self._best_chase_space_move(ctx, reach)
                if chase_choice is not None and (best is None or chase_choice[0] > best[0]):
                    best = chase_choice
            return best[1] if best is not None else None

        item_choice = self._best_item_move(ctx, reach)
        if item_choice is not None:
            best = item_choice

        if ctx["strategic_mode"] == "item_race" and best is not None:
            return best[1]

        if time.perf_counter() - started < 0.075:
            box_choice = self._best_box_position_move(ctx, reach)
            if box_choice is not None and (best is None or box_choice[0] > best[0]):
                best = box_choice

        if ctx["strategic_mode"] == "box_farm" and best is not None:
            return best[1]

        if time.perf_counter() - started < 0.08:
            enemy_choice = self._best_enemy_pressure_move(ctx, reach)
            if enemy_choice is not None and (best is None or enemy_choice[0] > best[0]):
                best = enemy_choice

        should_chase = ctx["tempo"] == "tiebreak" and ctx["engagement_deficit"] >= 2
        should_chase = should_chase or (ctx["tempo"] == "duel" and best is None)
        if time.perf_counter() - started < 0.085 and should_chase:
            chase_choice = self._best_chase_space_move(ctx, reach)
            if chase_choice is not None and (best is None or chase_choice[0] > best[0]):
                best = chase_choice

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

        endgame_trap_bonus = 0.0
        if ctx["phase"] == "end" and enemy_hits:
            det_time = self._hypothetical_det_time(hypo_bombs, pos)
            for enemy in enemy_hits:
                profile = self._enemy_escape_profile(
                    ctx, enemy["pos"], hypo_bombs, hypo_danger, min(det_time + 1, 8)
                )
                deadness = self._dead_zone_score(ctx, enemy["pos"])
                if profile["routes"] <= 1:
                    endgame_trap_bonus += 3.0
                elif profile["routes"] == 2:
                    endgame_trap_bonus += 1.0
                else:
                    endgame_trap_bonus -= 0.8
                if profile["area"] <= 3:
                    endgame_trap_bonus += 2.2
                elif profile["area"] <= 6:
                    endgame_trap_bonus += 0.9
                else:
                    endgame_trap_bonus -= 0.5
                endgame_trap_bonus += deadness * 0.8

        item_value = boxes * 0.9
        multi_bonus = max(0, boxes - 1) * 1.2
        score = boxes * 3.0 + multi_bonus + item_value
        score += len(enemy_hits) * 6.5 + trap_score * 3.0
        score += endgame_trap_bonus
        tempo = ctx["tempo"]
        if tempo == "economy":
            score += min(boxes, 3) * 0.7
            if enemy_hits and trap_score < 1.8:
                score -= 1.6
        elif tempo == "box_farm":
            score += boxes * 1.7 + max(0, boxes - 2) * 0.8
            if boxes == 0 and not enemy_hits:
                score -= 2.0
        elif tempo == "item_race":
            score -= 0.6
            if boxes:
                score += min(boxes, 2) * 0.2
            if enemy_hits:
                score += 0.5
        elif tempo == "hunt":
            score += len(enemy_hits) * 3.2 + trap_score * 1.8
            if boxes and not enemy_hits:
                score -= 1.8
            target_id = ctx.get("target_enemy_id")
            if target_id is not None:
                target_hits = [enemy for enemy in enemy_hits if enemy["id"] == target_id]
                if target_hits:
                    score += 4.0
        elif tempo == "unstuck":
            score += boxes * 0.9 + min(escape_area, 10) * 0.08
        elif tempo in ("pressure", "tiebreak"):
            score += len(enemy_hits) * 0.9 + trap_score * 0.45
            if boxes and not enemy_hits:
                score -= 0.25
        elif tempo == "duel":
            score += len(enemy_hits) * 1.4 + trap_score * 0.75
            if boxes and not enemy_hits:
                score -= 1.2
        elif tempo == "survive_duel":
            score += len(enemy_hits) * 1.2 + trap_score * 0.6
            if escape_area <= 6:
                score -= 1.2
        elif tempo == "escape":
            score -= 10.0
        if ctx["phase"] == "end" and boxes and not enemy_hits:
            score -= 2.0
        if ctx["engagement_deficit"] and boxes:
            score += min(ctx["engagement_deficit"], 4) * 0.35
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
            "phase": ctx["phase"],
            "tempo": ctx["tempo"],
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

    def _hypothetical_det_time(self, bombs, pos):
        best = 7
        for bomb in bombs:
            if bomb.get("hypothetical") and bomb["pos"] == pos:
                best = min(best, int(bomb.get("det_time", bomb.get("timer", 7))))
        return max(1, best)

    def _enemy_escape_profile(self, ctx, start, bombs, danger_at, horizon):
        grid = ctx["grid"]
        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        routes = set()
        safe_positions = set()
        while q:
            pos, t, first = q.popleft()
            if t > 0 and self._is_future_safe(pos, t, danger_at):
                routes.add(first if first is not None else self.STOP)
                safe_positions.add(pos)
            if t >= horizon:
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
                q.append((npos, nt, action if first is None else first))
        return {"routes": len(routes), "area": len(safe_positions)}

    def _dead_zone_score(self, ctx, pos):
        grid = ctx["grid"]
        if not self._passable(grid, pos[0], pos[1]):
            return 0.0
        exits = 0
        actions = []
        for action in self.DIRS:
            npos = self._next_pos(pos, action)
            if self._passable(grid, npos[0], npos[1]) and npos not in ctx["bomb_positions"]:
                exits += 1
                actions.append(action)
        if exits <= 1:
            score = 2.5
        elif exits == 2:
            opposite = set(actions) in ({self.LEFT, self.RIGHT}, {self.UP, self.DOWN})
            score = 0.8 if opposite else 1.35
        elif exits == 3:
            score = 0.25
        else:
            score = 0.0
        if self._static_local_area(ctx, pos, 4) <= 6:
            score += 0.6
        return score

    def _static_local_area(self, ctx, start, depth):
        grid = ctx["grid"]
        q = deque([(start, 0)])
        seen = {start}
        while q:
            pos, d = q.popleft()
            if d >= depth:
                continue
            for action in self.DIRS:
                npos = self._next_pos(pos, action)
                if npos in seen or npos in ctx["bomb_positions"]:
                    continue
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                seen.add(npos)
                q.append((npos, d + 1))
        return len(seen)

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
        if ctx["tempo"] == "item_race":
            return 10.5 if cell == self.ITEM_CAPACITY else 9.8
        if ctx["tempo"] == "escape":
            return 2.5 if cell == self.ITEM_CAPACITY else 2.0
        if cell == self.ITEM_CAPACITY:
            if ctx["my_bombs_left"] <= 1:
                return 8.0 if ctx["tempo"] == "economy" else 6.2 if ctx["phase"] == "end" else 7.0
            return 5.0 if ctx["tempo"] == "economy" else 3.6 if ctx["phase"] == "end" else 4.0
        if cell == self.ITEM_RADIUS:
            if ctx["my_bonus"] <= 1:
                return 7.8 if ctx["tempo"] == "economy" else 5.8 if ctx["phase"] == "end" else 6.5
            if ctx["my_bonus"] < self.MAX_RADIUS - 1:
                return 4.2 if ctx["tempo"] == "economy" else 3.0 if ctx["phase"] == "end" else 3.5
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
                if ctx["tempo"] in ("economy", "unstuck", "box_farm"):
                    value += boxes * 0.9
                    if ctx["tempo"] == "box_farm":
                        value += boxes * 1.4 + max(0, boxes - 2) * 1.2
                elif ctx["tempo"] in ("duel", "survive_duel"):
                    value *= 0.45
                elif ctx["tempo"] in ("pressure", "tiebreak", "hunt"):
                    value *= 0.75
                elif ctx["tempo"] in ("escape", "item_race"):
                    value *= 0.35
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
        target_enemy_id = ctx.get("target_enemy_id")
        for enemy in ctx["enemies"]:
            if target_enemy_id is not None and enemy["id"] != target_enemy_id and ctx["tempo"] == "hunt":
                continue
            epos = enemy["pos"]
            for x in range(1, grid.shape[0] - 1):
                for y in range(1, grid.shape[1] - 1):
                    pos = (x, y)
                    if pos in ctx["bomb_positions"] or not self._passable(grid, x, y):
                        continue
                    if self._line_blast_hits(grid, pos, epos, ctx["my_radius"]):
                        value = 5.2 - self._manhattan(pos, epos) * 0.12
                        if ctx["phase"] == "end":
                            value += self._dead_zone_score(ctx, epos) * 1.1 + 1.8
                    elif self._manhattan(pos, epos) <= 3:
                        value = 2.0 - self._manhattan(pos, epos) * 0.25
                    else:
                        continue
                    if ctx["tempo"] in ("pressure", "tiebreak"):
                        value += 0.7
                    elif ctx["tempo"] == "duel":
                        value += 1.2
                    elif ctx["tempo"] == "survive_duel":
                        value += 0.7
                    elif ctx["tempo"] == "economy":
                        value -= 1.5
                    elif ctx["tempo"] == "hunt":
                        value += 2.6
                        if target_enemy_id == enemy["id"]:
                            value += 2.2 + enemy["stats"].get("kills", 0) * 0.8
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

    def _best_chase_space_move(self, ctx, reach):
        if not ctx["enemies"]:
            return None
        grid = ctx["grid"]
        center = (grid.shape[0] // 2, grid.shape[1] // 2)
        targets = ctx["enemies"]
        if ctx.get("target_enemy_id") is not None:
            narrowed = [e for e in ctx["enemies"] if e["id"] == ctx["target_enemy_id"]]
            if narrowed:
                targets = narrowed
        best = None
        for pos, path in reach.items():
            dist, action = path
            if dist == 0:
                continue
            if pos in ctx["bomb_positions"] or pos in ctx["enemy_risk"]:
                continue
            nearest_enemy_dist = min(self._manhattan(pos, e["pos"]) for e in targets)
            if nearest_enemy_dist > 6 and ctx["tempo"] not in ("tiebreak", "hunt"):
                continue
            score = 3.2 - nearest_enemy_dist * 0.38 - dist * 0.18
            score += self._open_neighbors(grid, pos, ctx["bomb_positions"]) * 0.25
            score -= self._manhattan(pos, center) * 0.05
            if ctx["tempo"] == "duel":
                score += 1.0
            elif ctx["tempo"] == "survive_duel":
                score += min(
                    self._reachable_count(ctx, pos, dist, ctx["bombs"], ctx["danger_at"], 6),
                    12,
                ) * 0.08
                score -= max(0, 3 - nearest_enemy_dist) * 0.4
            elif ctx["tempo"] == "tiebreak":
                score += max(0, 5 - nearest_enemy_dist) * 0.25
                score += min(ctx["engagement_deficit"], 5) * 0.18
            elif ctx["tempo"] == "hunt":
                score += 2.8 + max(0, 6 - nearest_enemy_dist) * 0.45
                score -= dist * 0.1
            if best is None or score > best[0]:
                best = (score, action)
        return best

    # ------------------------------------------------------------------
    # Safety, escape, and path finding
    # ------------------------------------------------------------------

    def _best_survival_action(self, ctx):
        best = None
        for action in ctx["valid_actions"]:
            if action == self.BOMB:
                continue
            npos = self._next_pos(ctx["my_pos"], action)
            if not self._movement_action_legal(ctx, ctx["my_pos"], npos, 1):
                continue
            if self._is_deadly(npos, 1, ctx["danger_at"]):
                continue

            area = self._reachable_count(ctx, npos, 1, ctx["bombs"], ctx["danger_at"], 10)
            future = self._next_danger_time(npos, 1, ctx["danger_at"])
            enemy_dist = 8
            if ctx["enemies"]:
                enemy_dist = min(self._manhattan(npos, enemy["pos"]) for enemy in ctx["enemies"])
            score = min(area, 18) * 0.8 + min(enemy_dist, 8) * 0.9
            score += 10.0 if future is None else min(future, 7) * 0.6
            score += self._open_neighbors(ctx["grid"], npos, ctx["bomb_positions"]) * 0.8
            if npos in ctx["enemy_risk"]:
                score -= 3.0
            if npos in ctx["enemy_set"]:
                score -= 2.5
            if action == self.STOP:
                score -= 0.5
            if best is None or score > best[0]:
                best = (score, action)
        return best[1] if best is not None else None

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


    def _evaluate_state(self, grid, my_pos, my_alive, enemies, bombs, danger_at):
        if not my_alive: return -999999
        score = -len(enemies) * 10000 
        
        # Space evaluation
        escape = self._find_escape(grid, my_pos, 0, [], danger_at, 10)
        e_space = 10 if escape is not None else 0
        if e_space == 0: score -= 50000
        else: score += e_space * 10
        
        score -= self._manhattan(my_pos, (grid.shape[0]//2, grid.shape[1]//2)) * 0.5
        
        for e_pos in enemies:
            trap_escape = self._find_escape(grid, e_pos, 0, [], danger_at, 5)
            trap_score = 5 if trap_escape is not None else 0
            if trap_score == 0:
                score += 10000 
            
            exits = 0
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(e_pos, a)
                if self._passable(grid, nx, ny): exits += 1
            if exits <= 1: score += 250
            elif exits == 2: score += 100
                
        # cascade
        true_timers = {b: b_info['timer'] for b, b_info in bombs.items()}
        changed = True
        while changed:
            changed = False
            for b1, i1 in bombs.items():
                t1 = true_timers[b1]
                blast1 = self._blast_tiles(grid, b1[0], b1[1], i1['radius'])
                for b2, i2 in bombs.items():
                    if b1 != b2 and b2 in blast1 and true_timers[b2] > t1:
                        true_timers[b2] = t1
                        changed = True
        for b_pos, info in bombs.items():
            t = true_timers[b_pos]
            blast = self._blast_tiles(grid, b_pos[0], b_pos[1], info['radius'])
            if t <= 5:
                for e_pos in enemies:
                    if e_pos in blast: score += 5000 * (6 - t)
            if my_pos in blast:
                score -= (10 - min(t, 9)) * 5000
        return score

    def _minimax(self, grid, my_pos, my_alive, enemies, bombs, my_bomb_radius, depth, start_time, time_limit, danger_at, tt, alpha=-float('inf'), beta=float('inf')):
        import time
        if not my_alive: return -999999, None
        if not enemies: return self._evaluate_state(grid, my_pos, my_alive, enemies, bombs, danger_at), None
        if depth == 0: return self._evaluate_state(grid, my_pos, my_alive, enemies, bombs, danger_at), None
        
        state_key = (my_pos, tuple(enemies), tuple((b, v['timer']) for b, v in bombs.items()))
        if state_key in tt and tt[state_key]['type'] == 'EXACT':
            return tt[state_key]['value'], tt[state_key]['action']
            
        valid_actions = [0, 1, 2, 3, 4, 5]
            
        max_score = -float('inf')
        best_action = None
        
        for a in valid_actions:
            n_my_pos = my_pos
            n_bombs = {}
            for b_pos, b_info in bombs.items():
                t = b_info['timer'] - 1
                n_bombs[b_pos] = {'timer': t, 'radius': b_info['radius']}
                
            if a == 5:
                if my_pos not in n_bombs:
                    n_bombs[my_pos] = {'timer': 9, 'radius': my_bomb_radius}
            elif a != 0:
                nx, ny = self._next_pos(my_pos, a)
                if self._passable(grid, nx, ny):
                    n_my_pos = (nx, ny)
                    
            n_my_alive = True
            for b_pos, info in n_bombs.items():
                if info['timer'] <= 0:
                    blast = self._blast_tiles(grid, b_pos[0], b_pos[1], info['radius'])
                    if n_my_pos in blast: n_my_alive = False
                    
            n_bombs = {k: v for k, v in n_bombs.items() if v['timer'] > 0}
            
            min_score = float('inf')
            
            n_enemies = []
            for e_pos in enemies:
                best_e_dist = 999
                best_e_pos = e_pos
                for ea in [0, 1, 2, 3, 4]:
                    ex, ey = self._next_pos(e_pos, ea)
                    if self._passable(grid, ex, ey):
                        dist = self._manhattan((ex, ey), n_my_pos)
                        if dist < best_e_dist:
                            best_e_dist = dist
                            best_e_pos = (ex, ey)
                n_enemies.append(best_e_pos)
                
            n_enemies_alive = []
            for e_pos in n_enemies:
                e_alive = True
                for b_pos, info in n_bombs.items():
                    if info['timer'] <= 0:
                        blast = self._blast_tiles(grid, b_pos[0], b_pos[1], info['radius'])
                        if e_pos in blast: e_alive = False
                if e_alive: n_enemies_alive.append(e_pos)
                
            score, _ = self._minimax(grid, n_my_pos, n_my_alive, n_enemies_alive, n_bombs, my_bomb_radius, depth - 1, start_time, time_limit, danger_at, tt, alpha, min(beta, min_score))
            
            if score < min_score: min_score = score
                
            if min_score > max_score:
                max_score = min_score
                best_action = a
                
            if time.perf_counter() - start_time > time_limit: break
            if max_score >= beta: break
            if max_score > alpha: alpha = max_score
            
        tt[state_key] = {'value': max_score, 'action': best_action, 'type': 'EXACT'}
        return max_score, best_action

class Agent(AdaptiveProV2Base):
    team_id = "AgentKiller"

    HUNT_HORIZON = 17
    MICRO_TIME_LIMIT = 0.088

    def __init__(self, agent_id: int):
        super().__init__(agent_id)
        self.focus_enemy_id = None

    # ------------------------------------------------------------------
    # Hunt-first strategy
    # ------------------------------------------------------------------

    def _strategic_mode(self, ctx):
        if not ctx["enemies"]:
            self.focus_enemy_id = None
            return "adaptive", None

        target = self._nearest_enemy(ctx)
        if target is None:
            self.focus_enemy_id = None
            return "adaptive", None

        self.focus_enemy_id = target["id"]
        return "hunt", target["id"]

    def _tempo_mode(self, ctx):
        if ctx.get("strategic_mode") == "hunt":
            if ctx["alive_count"] <= 2:
                return "duel"
            return "hunt"
        return super()._tempo_mode(ctx)

    def _engagement_deficit(self, ctx):
        if ctx.get("strategic_mode") == "hunt":
            own_kills = int(ctx["my_stats"].get("kills", 0))
            expected = max(1.0, self.turn / 70.0)
            return max(0.0, expected - own_kills * 1.8)
        return super()._engagement_deficit(ctx)

    def _nearest_enemy(self, ctx):
        enemies = ctx["enemies"]
        if not enemies:
            return None

        my_pos = ctx["my_pos"]
        scored = []
        for enemy in enemies:
            epos = enemy["pos"]
            dist = self._static_distance(ctx, my_pos, epos, limit=28)
            if dist is None:
                dist = self._manhattan(my_pos, epos) + 12
            deadness = self._dead_zone_score(ctx, epos)
            power = enemy["radius"] + max(1, enemy["bombs_left"])
            score = dist + self._manhattan(my_pos, epos) * 0.08
            score -= min(deadness, 3.0) * 0.9
            score += max(0, power - (ctx["my_radius"] + max(1, ctx["my_bombs_left"]))) * 0.2
            scored.append((score, enemy))

        scored.sort(key=lambda item: (item[0], item[1]["id"]))
        best = scored[0][1]

        if self.focus_enemy_id is not None:
            sticky = [enemy for _, enemy in scored if enemy["id"] == self.focus_enemy_id]
            if sticky:
                sticky_enemy = sticky[0]
                sticky_score = next(score for score, enemy in scored if enemy["id"] == sticky_enemy["id"])
                if sticky_score <= scored[0][0] + 2.2:
                    return sticky_enemy
        return best

    def _get_target_enemy(self, ctx):
        target_id = ctx.get("target_enemy_id", self.focus_enemy_id)
        if target_id is not None:
            for enemy in ctx["enemies"]:
                if enemy["id"] == target_id:
                    return enemy
        return self._nearest_enemy(ctx)

    # ------------------------------------------------------------------
    # Decision layer
    # ------------------------------------------------------------------

    def _choose_objective(self, ctx, started):
        if ctx.get("strategic_mode") != "hunt":
            return super()._choose_objective(ctx, started)

        if time.perf_counter() - started > 0.078:
            return self._quick_safe_action(ctx)

        reach = self._temporal_reach_map(ctx, self.HUNT_HORIZON)
        low_power = (
            ctx["my_radius"] < 3
            or max(ctx["my_bombs_left"], int(ctx.get("max_bombs_left_seen", 1))) < 2
        )

        if low_power and self.turn < 150:
            power_item = self._best_hunt_item_move(ctx, reach)
            if power_item is not None and power_item[0] >= 1.0:
                return power_item[1]

        kill_move = self._best_kill_position_move(ctx, reach, started)
        if kill_move is not None:
            return kill_move[1]

        if time.perf_counter() - started < 0.078:
            plan_move = self._micro_minimax_move(ctx, reach, started)
            if plan_move is not None:
                return plan_move

        if time.perf_counter() - started < 0.082:
            power_item = self._best_hunt_item_move(ctx, reach)
            if power_item is not None and power_item[0] >= 1.4:
                return power_item[1]

        if time.perf_counter() - started < 0.085:
            opener = self._best_path_opening_box_move(ctx, reach)
            opener_threshold = 0.55 if low_power and self.turn < 150 else 1.1
            if opener is not None and opener[0] >= opener_threshold:
                return opener[1]

        if time.perf_counter() - started < 0.086 and (self.turn >= 180 or low_power):
            farm = super()._best_box_position_move(ctx, reach)
            if farm is not None and farm[0] >= (2.6 if self.turn >= 180 else 3.4):
                return farm[1]

        astar = self._a_star_chase_action(ctx)
        if astar is not None:
            return astar

        chase = self._best_chase_space_move(ctx, reach)
        if chase is not None:
            return chase[1]

        return None

    def _bomb_threshold(self, bomb):
        threshold = super()._bomb_threshold(bomb)
        tempo = bomb.get("tempo", "balanced")
        if tempo in ("hunt", "duel"):
            if bomb.get("target_hits", 0):
                threshold -= 2.2
            if bomb.get("target_trap", 0.0) >= 2.2:
                threshold -= 2.8
            elif bomb.get("target_trap", 0.0) >= 1.2:
                threshold -= 1.5
            if bomb.get("enemy_hits", 0) or bomb.get("trap_score", 0.0) >= 1.0:
                threshold -= 0.9
            if bomb.get("boxes", 0) and not bomb.get("enemy_hits", 0):
                low_power = bomb.get("my_radius", 5) < 3 or bomb.get("my_capacity_seen", 1) < 2
                if low_power and self.turn < 150:
                    threshold = max(threshold + 0.7, 8.7)
                    if bomb.get("boxes", 0) >= 2:
                        threshold -= 0.8
                else:
                    threshold = max(threshold + 2.4, 12.0)
                    if bomb.get("path_opening", 0.0) < 1.8:
                        threshold += 1.6
            if bomb.get("path_opening", 0.0) > 0:
                threshold -= min(1.0, bomb["path_opening"] * 0.55)
            if bomb.get("escape_area", 0) <= 5:
                threshold += 1.2
            if bomb.get("slack", 7) < 3:
                threshold += 1.4
            if bomb.get("my_capacity_seen", 1) >= 2 and bomb.get("target_trap", 0.0) >= 0.8:
                threshold -= 0.45
        return threshold

    def _score_bomb_at(self, ctx, pos):
        score = super()._score_bomb_at(ctx, pos)
        if not score["legal"]:
            return score

        score["target_hits"] = 0
        score["target_trap"] = 0.0
        score["path_opening"] = 0.0
        score["my_capacity_seen"] = max(1, int(ctx.get("max_bombs_left_seen", 1)))
        score["my_radius"] = ctx["my_radius"]

        target = self._get_target_enemy(ctx)
        if target is None:
            return score

        grid = ctx["grid"]
        radius = ctx["my_radius"]
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        hypo_bombs, hypo_danger, _ = self._hypothetical_schedule(ctx, pos, radius)

        killer_bonus = 0.0
        target_blast_hit = target["pos"] in blast
        if target_blast_hit:
            score["target_hits"] = 1

        for enemy in ctx["enemies"]:
            epos = enemy["pos"]
            if epos not in blast:
                continue
            profile = self._enemy_escape_profile(ctx, epos, hypo_bombs, hypo_danger, horizon=8)
            deadness = self._dead_zone_score(ctx, epos)
            cutoff = self._cutoff_score(ctx, epos, blast)
            trap = deadness * 0.95 + cutoff

            if profile["routes"] == 0:
                trap += 3.8
            elif profile["routes"] == 1:
                trap += 2.2
            elif profile["routes"] == 2:
                trap += 1.0
            else:
                trap -= 0.7

            if profile["area"] <= 2:
                trap += 2.4
            elif profile["area"] <= 5:
                trap += 1.25
            elif profile["area"] >= 12:
                trap -= 0.8

            if enemy["id"] == target["id"]:
                score["target_trap"] = trap
                killer_bonus += 5.8 + trap * 2.6
            else:
                killer_bonus += 2.8 + trap * 1.25

        if not target_blast_hit:
            dist_to_target = self._manhattan(pos, target["pos"])
            exit_cut = self._cutoff_score(ctx, target["pos"], blast)
            if dist_to_target <= radius + 1:
                killer_bonus += max(0.0, 1.6 - 0.35 * dist_to_target)
                if exit_cut >= 1.4:
                    cutoff_trap = exit_cut * (0.9 if score["my_capacity_seen"] >= 2 else 0.55)
                    score["target_trap"] = max(score["target_trap"], cutoff_trap)
                    killer_bonus += exit_cut * (1.65 if score["my_capacity_seen"] >= 2 else 0.9)
            if score["boxes"] > 0:
                score["path_opening"] = self._path_opening_value(ctx, pos, target)
                killer_bonus += score["path_opening"] * 1.15
            if score["boxes"] > 0 and score["path_opening"] <= 0.3:
                killer_bonus -= min(3.5, 1.4 * score["boxes"])

        if pos in ctx["enemy_risk"] and not target_blast_hit:
            killer_bonus -= 0.8
        if score["escape_area"] <= 4:
            killer_bonus -= 1.7
        elif score["escape_area"] >= 10:
            killer_bonus += 0.45

        score["score"] += killer_bonus
        return score

    # ------------------------------------------------------------------
    # Hunt objectives
    # ------------------------------------------------------------------

    def _best_kill_position_move(self, ctx, reach, started):
        target = self._get_target_enemy(ctx)
        if target is None or ctx["my_bombs_left"] <= 0:
            return None

        candidates = []
        grid = ctx["grid"]
        radius = ctx["my_radius"]
        target_pos = target["pos"]

        for pos, path in reach.items():
            if time.perf_counter() - started > 0.083:
                break
            dist, first_action = path
            if dist == 0:
                continue
            if dist > 12 or pos in ctx["bomb_positions"]:
                continue
            if not self._passable(grid, pos[0], pos[1]):
                continue

            line_hit = self._line_blast_hits(grid, pos, target_pos, radius)
            close = self._manhattan(pos, target_pos) <= max(2, radius + 1)
            if not line_hit and not close:
                continue

            static_value = 0.0
            if line_hit:
                blast = self._blast_tiles(grid, pos[0], pos[1], radius)
                static_value += 8.0 + self._dead_zone_score(ctx, target_pos) * 1.4
                static_value += self._cutoff_score(ctx, target_pos, blast)
            else:
                blast = self._blast_tiles(grid, pos[0], pos[1], radius)
                static_value += 2.6 - self._manhattan(pos, target_pos) * 0.35
                static_value += self._cutoff_score(ctx, target_pos, blast) * 0.8

            bomb_score = self._score_bomb_at(ctx, pos)
            if bomb_score["legal"]:
                static_value += min(28.0, bomb_score["score"]) * 0.32
                static_value += bomb_score.get("target_trap", 0.0) * 0.9
                if bomb_score.get("escape_area", 0) <= 4:
                    static_value -= 2.4
            else:
                static_value -= 4.0

            static_value -= dist * 0.42
            if pos in ctx["enemy_risk"]:
                static_value -= 1.0
            if first_action == self.STOP:
                static_value -= 0.4
            candidates.append((static_value, first_action, pos))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        best = candidates[0]
        if best[0] < 2.0:
            return None
        return best

    def _best_hunt_item_move(self, ctx, reach):
        grid = ctx["grid"]
        target = self._get_target_enemy(ctx)
        if target is None:
            return None

        best = None
        power = ctx["my_radius"] + max(ctx["my_bombs_left"], ctx.get("max_bombs_left_seen", 1))
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                cell = int(grid[x, y])
                if cell not in (self.ITEM_RADIUS, self.ITEM_CAPACITY):
                    continue
                path = reach.get((x, y))
                if path is None:
                    continue
                dist, action = path
                if dist > 7 and power >= 5:
                    continue

                value = self._item_value(ctx, cell)
                if power >= 6:
                    value *= 0.45
                target_dist_after = self._manhattan((x, y), target["pos"])
                score = value - dist * 0.62 - target_dist_after * 0.08
                if (x, y) in ctx["enemy_risk"]:
                    score -= 1.5
                if best is None or score > best[0]:
                    best = (score, action)
        return best

    def _best_path_opening_box_move(self, ctx, reach):
        target = self._get_target_enemy(ctx)
        if target is None or ctx["my_bombs_left"] <= 0:
            return None

        grid = ctx["grid"]
        best = None
        target_pos = target["pos"]
        direct_dist = self._static_distance(ctx, ctx["my_pos"], target_pos, limit=40)
        if direct_dist is not None and direct_dist <= 5:
            return None

        for pos, path in reach.items():
            dist, action = path
            if dist > 9 or pos in ctx["bomb_positions"]:
                continue
            if not self._passable(grid, pos[0], pos[1]):
                continue
            boxes = [tile for tile in self._blast_tiles(grid, pos[0], pos[1], ctx["my_radius"]) if int(grid[tile[0], tile[1]]) == self.BOX]
            if not boxes:
                continue
            opening = self._path_opening_value(ctx, pos, target)
            if opening <= 0:
                continue
            bomb_score = self._score_bomb_at(ctx, pos)
            if not bomb_score["legal"]:
                continue
            score = opening * 2.2 + min(len(boxes), 3) * 0.65 - dist * 0.45
            score += min(bomb_score.get("escape_area", 0), 10) * 0.05
            if pos in ctx["enemy_risk"]:
                score -= 0.9
            if best is None or score > best[0]:
                best = (score, action)
        return best

    def _best_enemy_pressure_move(self, ctx, reach):
        if ctx.get("strategic_mode") != "hunt":
            return super()._best_enemy_pressure_move(ctx, reach)

        target = self._get_target_enemy(ctx)
        if target is None:
            return None

        grid = ctx["grid"]
        radius = ctx["my_radius"]
        target_pos = target["pos"]
        best = None

        for pos, path in reach.items():
            dist, action = path
            if dist == 0 or dist > 13:
                continue
            if pos in ctx["bomb_positions"] or not self._passable(grid, pos[0], pos[1]):
                continue
            line_hit = self._line_blast_hits(grid, pos, target_pos, radius)
            near = self._manhattan(pos, target_pos) <= 3
            if not line_hit and not near:
                continue
            score = 2.0 - dist * 0.28
            if line_hit:
                blast = self._blast_tiles(grid, pos[0], pos[1], radius)
                score += 5.4 + self._cutoff_score(ctx, target_pos, blast)
                score += self._dead_zone_score(ctx, target_pos) * 0.9
            else:
                score += max(0.0, 2.5 - self._manhattan(pos, target_pos) * 0.45)
            if pos in ctx["enemy_risk"]:
                score -= 0.8
            if best is None or score > best[0]:
                best = (score, action)
        return best

    def _best_chase_space_move(self, ctx, reach):
        if ctx.get("strategic_mode") != "hunt":
            return super()._best_chase_space_move(ctx, reach)

        target = self._get_target_enemy(ctx)
        if target is None:
            return None

        target_pos = target["pos"]
        best = None
        for pos, path in reach.items():
            dist, action = path
            if dist == 0:
                continue
            enemy_dist = self._manhattan(pos, target_pos)
            if enemy_dist > 8:
                continue
            area = self._reachable_count(ctx, pos, dist, ctx["bombs"], ctx["danger_at"], 6)
            score = 5.5 - enemy_dist * 0.65 - dist * 0.22
            score += min(area, 12) * 0.08
            score += self._open_neighbors(ctx["grid"], pos, ctx["bomb_positions"]) * 0.18
            if self._line_blast_hits(ctx["grid"], pos, target_pos, ctx["my_radius"]):
                score += 1.8
            if pos in ctx["enemy_risk"]:
                score -= 1.1
            if action == self.STOP:
                score -= 0.3
            if best is None or score > best[0]:
                best = (score, action)
        return best

    # ------------------------------------------------------------------
    # A*, micro lookahead, and geometry
    # ------------------------------------------------------------------

    def _a_star_chase_action(self, ctx):
        target = self._get_target_enemy(ctx)
        if target is None:
            return None

        grid = ctx["grid"]
        target_pos = target["pos"]
        attack_targets = set()
        for x in range(1, grid.shape[0] - 1):
            for y in range(1, grid.shape[1] - 1):
                pos = (x, y)
                if not self._passable(grid, x, y) or pos in ctx["bomb_positions"]:
                    continue
                if self._line_blast_hits(grid, pos, target_pos, ctx["my_radius"]):
                    attack_targets.add(pos)
                elif self._manhattan(pos, target_pos) <= 2:
                    attack_targets.add(pos)
        if not attack_targets:
            attack_targets.add(target_pos)

        return self._a_star_action_to_targets(ctx, attack_targets, horizon=self.HUNT_HORIZON)

    def _a_star_action_to_targets(self, ctx, targets, horizon):
        start = ctx["my_pos"]
        if start in targets:
            return self.STOP

        counter = 0
        heap = []
        best_seen = {(start, 0): 0.0}
        first_action = None
        h0 = self._target_heuristic(start, targets)
        heappush(heap, (h0, 0.0, counter, start, 0, first_action))

        while heap:
            _, cost, _, pos, t, first = heappop(heap)
            if t >= horizon:
                continue
            for action in self.SEARCH_ACTIONS:
                npos = self._next_pos(pos, action)
                nt = t + 1
                if not self._passable(ctx["grid"], npos[0], npos[1]):
                    continue
                if self._bomb_blocks(pos, npos, nt, ctx["bombs"]):
                    continue
                if self._is_deadly(npos, nt, ctx["danger_at"]):
                    continue

                step_cost = 1.0
                if npos in ctx["enemy_risk"]:
                    step_cost += 1.2
                if action == self.STOP:
                    step_cost += 0.15
                new_cost = cost + step_cost
                state = (npos, nt)
                if state in best_seen and best_seen[state] <= new_cost:
                    continue
                best_seen[state] = new_cost
                next_first = action if first is None else first
                if npos in targets:
                    return next_first
                counter += 1
                priority = new_cost + self._target_heuristic(npos, targets) * 1.05
                heappush(heap, (priority, new_cost, counter, npos, nt, next_first))
        return None

    def _target_heuristic(self, pos, targets):
        return min(self._manhattan(pos, target) for target in targets) if targets else 0

    def _micro_minimax_move(self, ctx, reach, started):
        target = self._get_target_enemy(ctx)
        if target is None:
            return None

        best = None
        for action in ctx["valid_actions"]:
            if time.perf_counter() - started > self.MICRO_TIME_LIMIT:
                break
            if action == self.BOMB:
                bomb = self._score_bomb_at(ctx, ctx["my_pos"])
                if not bomb["legal"]:
                    continue
                score = bomb["score"] - self._bomb_threshold(bomb)
                score += bomb.get("target_trap", 0.0) * 0.8
            else:
                npos = self._next_pos(ctx["my_pos"], action)
                if not self._movement_action_legal(ctx, ctx["my_pos"], npos, 1):
                    continue
                if self._is_deadly(npos, 1, ctx["danger_at"]):
                    continue
                score = self._micro_state_score(ctx, npos, target)
                score -= self._adversarial_escape_bonus(ctx, npos, target) * 0.35
            if best is None or score > best[0]:
                best = (score, action)

        if best is None or best[0] < 0.65:
            return None
        return best[1]

    def _micro_state_score(self, ctx, pos, target):
        target_pos = target["pos"]
        area = self._reachable_count(ctx, pos, 1, ctx["bombs"], ctx["danger_at"], 7)
        future = self._next_danger_time(pos, 1, ctx["danger_at"])
        dist = self._manhattan(pos, target_pos)
        score = 5.0 - dist * 0.72
        score += min(area, 14) * 0.13
        if future is None:
            score += 1.4
        else:
            score += min(future, 7) * 0.18
        if ctx["my_bombs_left"] > 0 and self._line_blast_hits(ctx["grid"], pos, target_pos, ctx["my_radius"]):
            score += 2.2 + self._dead_zone_score(ctx, target_pos) * 0.7
        if pos in ctx["enemy_risk"]:
            score -= 1.5
        if pos in ctx["enemy_set"]:
            score -= 0.3
        return score

    def _adversarial_escape_bonus(self, ctx, my_pos, target):
        target_pos = target["pos"]
        best_enemy_space = 0.0
        for action in self.SEARCH_ACTIONS:
            npos = self._next_pos(target_pos, action)
            if not self._passable(ctx["grid"], npos[0], npos[1]):
                continue
            if self._bomb_blocks(target_pos, npos, 1, ctx["bombs"]):
                continue
            if self._is_deadly(npos, 1, ctx["danger_at"]):
                continue
            enemy_space = self._static_local_area(ctx, npos, 4)
            distance_gain = self._manhattan(npos, my_pos)
            best_enemy_space = max(best_enemy_space, enemy_space * 0.18 + distance_gain * 0.35)
        return best_enemy_space

    def _cutoff_score(self, ctx, enemy_pos, blast):
        exits = []
        covered = 0
        for action in self.DIRS:
            npos = self._next_pos(enemy_pos, action)
            if not self._passable(ctx["grid"], npos[0], npos[1]):
                continue
            if npos in ctx["bomb_positions"]:
                continue
            exits.append(npos)
            if npos in blast:
                covered += 1
        if not exits:
            return 3.0
        score = covered * 0.85
        if len(exits) == 1:
            score += 1.8
        elif len(exits) == 2:
            score += 0.65
        score += max(0, len(exits) - covered) * -0.18
        return score

    def _path_opening_value(self, ctx, bomb_pos, target):
        grid = ctx["grid"]
        target_pos = target["pos"]
        blast = self._blast_tiles(grid, bomb_pos[0], bomb_pos[1], ctx["my_radius"])
        destroyed = [tile for tile in blast if int(grid[tile[0], tile[1]]) == self.BOX]
        if not destroyed:
            return 0.0

        before = self._static_distance(ctx, ctx["my_pos"], target_pos, limit=45)
        best_value = 0.0
        for box in destroyed:
            distance_to_target = self._manhattan(box, target_pos)
            distance_from_me = self._manhattan(ctx["my_pos"], box)
            value = max(0.0, 5.0 - distance_to_target * 0.38 - distance_from_me * 0.08)
            if before is None:
                value += 1.5
            best_value = max(best_value, value)
        return best_value

    def _static_distance(self, ctx, start, goal, limit=30):
        if start == goal:
            return 0
        queue = [(start, 0)]
        seen = {start}
        index = 0
        while index < len(queue):
            pos, dist = queue[index]
            index += 1
            if dist >= limit:
                continue
            for action in self.DIRS:
                npos = self._next_pos(pos, action)
                if npos in seen:
                    continue
                if not self._passable(ctx["grid"], npos[0], npos[1]):
                    continue
                if npos in ctx["bomb_positions"]:
                    continue
                if npos == goal:
                    return dist + 1
                seen.add(npos)
                queue.append((npos, dist + 1))
        return None

    def _item_value(self, ctx, cell):
        if ctx.get("strategic_mode") != "hunt":
            return super()._item_value(ctx, cell)

        capacity_seen = max(1, int(ctx.get("max_bombs_left_seen", 1)))
        if cell == self.ITEM_CAPACITY:
            if capacity_seen < 2:
                return 6.8
            if capacity_seen < self.MAX_CAPACITY and self.turn < 260:
                return 3.6
            return 1.1
        if cell == self.ITEM_RADIUS:
            if ctx["my_radius"] < 3:
                return 6.2
            if ctx["my_radius"] < self.MAX_RADIUS and self.turn < 260:
                return 3.0
            return 0.9
        return 0.0
