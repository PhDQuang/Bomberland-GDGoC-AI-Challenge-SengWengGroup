from collections import deque
import time


class Agent:
    team_id = "UltimateHeuristicV2"

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
    MAX_BOMBS = 5
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
        except Exception as e:
            import traceback
            traceback.print_exc()
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
        mode, aggression, target_enemy_id = self._analyze_macro_state(ctx)
        ctx["mode"] = mode
        ctx["aggression"] = aggression
        ctx["target_enemy_id"] = target_enemy_id
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

    def _analyze_macro_state(self, ctx):
        my_stats = ctx["my_stats"]
        alive_count = ctx["alive_count"]
        power = ctx["my_radius"] + max(ctx["my_bombs_left"], ctx["max_bombs_left_seen"])
        
        stats = ctx["telemetry"]
        progress_age = self.turn - int(stats.get("last_progress_turn", 0))
        if progress_age >= 75 and alive_count >= 3:
            return "unstuck", 0.0, None
            
        def get_score(s):
            return int(s.get("kills", 0)) * 1000 + int(s.get("boxes", 0)) * 10 + int(s.get("items", 0))
            
        my_score = get_score(my_stats)
        is_winning = True
        enemy_stats = []
        alive_enemy_ids = {enemy["id"] for enemy in ctx["enemies"]}
        for enemy in ctx["enemies"]:
            s = ctx["estimated_stats"][enemy["id"]]
            enemy_stats.append((enemy["id"], s))
            if get_score(s) > my_score:
                is_winning = False

        my_kills = int(my_stats.get("kills", 0))

        if my_kills >= 2:
            return "escape", 0.0, None

        killer_enemies = [(i, s) for i, s in enemy_stats if int(s.get("kills", 0)) >= 2]
        if killer_enemies:
            target = max(killer_enemies, key=lambda item: (int(item[1].get("kills", 0)), int(item[1].get("boxes", 0))))[0]
            return "combat", 0.9, target

        if alive_count == 2 and ctx["enemies"]:
            enemy_id = ctx["enemies"][0]["id"]
            enemy_s = ctx["estimated_stats"][enemy_id]
            enemy_kills = int(enemy_s.get("kills", 0))
            my_boxes = int(my_stats.get("boxes", 0))
            enemy_boxes = int(enemy_s.get("boxes", 0))
            
            if enemy_kills > my_kills:
                return "combat", 0.9, enemy_id
            if enemy_kills == my_kills and enemy_boxes == my_boxes:
                my_items = int(my_stats.get("items", 0))
                enemy_items = int(enemy_s.get("items", 0))
                item_gap = my_items - enemy_items
                steps_left = max(0, 500 - self.turn)
                if abs(item_gap) * 10 <= steps_left:
                    return "economy", 0.4, None
                if item_gap > 0:
                    return "escape", 0.0, None
                return "combat", 0.8, enemy_id
            if my_kills > enemy_kills or my_boxes > enemy_boxes:
                return "escape", 0.0, None

        if ctx["boxes_remaining"] == 0:
            my_boxes = int(my_stats.get("boxes", 0))
            max_boxes = max([int(s.get("boxes", 0)) for i, s in enemy_stats] + [0])
            if my_boxes < max_boxes and ctx["enemies"]:
                target = max(ctx["enemies"], key=lambda e: (int(ctx["estimated_stats"][e["id"]].get("boxes", 0)), int(ctx["estimated_stats"][e["id"]].get("kills", 0))))["id"]
                return "combat", 0.8, target

        if alive_count == 4:
            return "economy", 0.3, None
        elif alive_count == 3:
            if is_winning:
                return "economy", 0.2, None
            else:
                if power >= 3:
                    target = max(ctx["enemies"], key=lambda e: get_score(ctx["estimated_stats"][e["id"]]))["id"]
                    return "combat", 0.7, target
                else:
                    return "economy", 0.3, None
                    
        return "economy", 0.3, None

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

        if ctx["mode"] == "escape":
            action = self._best_escape_action(ctx)
            if action is not None:
                return action

        # === V3: Minimax for 1v1 endgame ===
        if ctx["alive_count"] == 2 and time.perf_counter() - started < 0.06:
            mm_action = self._minimax_decide(ctx, started)
            if mm_action is not None:
                if mm_action == self.BOMB:
                    self.escape_mode = True
                return mm_action

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

    def _minimax_decide(self, ctx, started):
        """V3: Use minimax in 1v1 endgame for optimal play."""
        try:
            grid = ctx["grid"]
            my_pos = ctx["my_pos"]
            enemies_pos = [e["pos"] for e in ctx["enemies"]]
            if not enemies_pos:
                return None
            bombs_dict = {}
            for b in ctx["bombs"]:
                bombs_dict[b["pos"]] = {"timer": b["timer"], "radius": b["radius"]}
            score, action = self._minimax(
                grid, my_pos, True, enemies_pos, bombs_dict,
                ctx["my_radius"], depth=3, start_time=started,
                time_limit=0.07, danger_at=ctx["danger_at"], tt={}
            )
            if action is not None and score > -500000:
                return action
        except Exception:
            pass
        return None

    def _bomb_threshold(self, bomb):
        phase = bomb.get("phase", "mid")
        mode = bomb.get("mode", "economy")
        aggression = bomb.get("aggression", 0.5)

        if mode == "escape":
            return 10**6

        safety_tax = 0.0
        if bomb.get("slack", 7) < 3:
            safety_tax += 2.0
        if bomb.get("escape_area", 10) <= 4:
            safety_tax += 1.0

        if mode == "combat":
            if bomb["enemy_hits"] or bomb["trap_score"] >= 1.0:
                safety_tax -= 0.7 + aggression * 0.7
        elif mode == "economy":
            if bomb["enemy_hits"] > 0:
                safety_tax -= 1.8
            elif bomb["enemy_hits"] == 0:
                safety_tax -= min(0.8 + (1.0 - aggression)*0.8, (0.25 + (1.0-aggression)*0.2) * bomb["boxes"])
        elif mode == "unstuck" and bomb["boxes"] > 0:
            safety_tax -= 0.9

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
        import time
        mode = ctx.get("mode", "economy")
        aggression = ctx.get("aggression", 0.5)
        reach = self._temporal_reach_map(ctx, 12)

        if mode == "escape":
            return self._best_escape_action(ctx)
        if mode == "unstuck":
            return self._unstuck_action(ctx, started) if hasattr(self, '_unstuck_action') else None

        best = None

        if mode == "economy" or (mode == "combat" and aggression < 0.8):
            item_choice = self._best_item_move(ctx, reach)
            if item_choice is not None:
                best = item_choice

            box_choice = self._best_box_position_move(ctx, reach)
            if box_choice is not None and (best is None or box_choice[0] > best[0]):
                best = box_choice

        if mode == "combat" or (mode == "economy" and aggression >= 0.3):
            if time.perf_counter() - started < 0.08:
                enemy_choice = self._best_enemy_pressure_move(ctx, reach)
                if enemy_choice is not None and (best is None or enemy_choice[0] > best[0]):
                    best = enemy_choice

            if mode == "combat" and time.perf_counter() - started < 0.082:
                trap_choice = self._best_corridor_trap_move(ctx, reach)
                if trap_choice is not None and (best is None or trap_choice[0] > best[0]):
                    best = trap_choice

            if mode == "combat" and aggression >= 0.5 and time.perf_counter() - started < 0.085:
                chase_choice = self._best_chase_space_move(ctx, reach)
                if chase_choice is not None and (best is None or chase_choice[0] > best[0]):
                    best = chase_choice

        return best[1] if best is not None else None

    def _best_corridor_trap_move(self, ctx, reach):
        """V3: Find positions where placing a bomb would cut off enemy escape routes."""
        if ctx["my_bombs_left"] <= 0 or not ctx["enemies"]:
            return None
        grid = ctx["grid"]
        best = None
        for enemy in ctx["enemies"]:
            epos = enemy["pos"]
            # Find enemy's escape routes
            enemy_exits = []
            for action in self.DIRS:
                npos = self._next_pos(epos, action)
                if self._passable(grid, npos[0], npos[1]) and npos not in ctx["bomb_positions"]:
                    enemy_exits.append(npos)
            if len(enemy_exits) <= 2:  # Enemy already has limited exits
                for exit_pos in enemy_exits:
                    # Can I place a bomb that blocks this exit?
                    blocking_positions = []
                    # The exit itself
                    if self._passable(grid, exit_pos[0], exit_pos[1]):
                        blocking_positions.append(exit_pos)
                    # Positions whose blast covers the exit
                    for bx in range(max(1, exit_pos[0] - ctx["my_radius"]), min(grid.shape[0] - 1, exit_pos[0] + ctx["my_radius"] + 1)):
                        if self._passable(grid, bx, exit_pos[1]) and (bx, exit_pos[1]) not in ctx["bomb_positions"]:
                            if self._line_blast_hits(grid, (bx, exit_pos[1]), exit_pos, ctx["my_radius"]):
                                blocking_positions.append((bx, exit_pos[1]))
                    for by in range(max(1, exit_pos[1] - ctx["my_radius"]), min(grid.shape[1] - 1, exit_pos[1] + ctx["my_radius"] + 1)):
                        if self._passable(grid, exit_pos[0], by) and (exit_pos[0], by) not in ctx["bomb_positions"]:
                            if self._line_blast_hits(grid, (exit_pos[0], by), exit_pos, ctx["my_radius"]):
                                blocking_positions.append((exit_pos[0], by))
                    for bpos in blocking_positions:
                        path = reach.get(bpos)
                        if path is None:
                            continue
                        dist, action = path
                        if dist > 8:
                            continue
                        # Score: higher when enemy has fewer exits and we're closer
                        value = 8.0 - len(enemy_exits) * 2.0 - dist * 0.3
                        # Bonus if this also directly threatens the enemy
                        if self._line_blast_hits(grid, bpos, epos, ctx["my_radius"]):
                            value += 4.0
                        # Make sure we can escape after placing bomb here
                        if bpos in ctx["enemy_risk"]:
                            value -= 1.5
                        if best is None or value > best[0]:
                            best = (value, action)
        return best

    # ------------------------------------------------------------------
    # Bomb scoring and objectives
    # ------------------------------------------------------------------

    def _score_bomb_at(self, ctx, pos):
        grid = ctx["grid"]
        bombs = ctx["bombs"]
        danger_at = ctx["danger_at"]
        my_pos = ctx["my_pos"]
        radius = ctx["my_radius"]

        if grid[pos] in (self.WALL, self.BOX) or any(
            b["pos"] == pos for b in bombs
        ):
            return {"legal": False, "score": 0.0, "boxes": 0, "enemy_hits": []}

        if ctx["my_bombs_left"] <= 0 or pos in ctx["bomb_positions"]:
            return self._illegal_bomb_score()

        hypo_bombs = [dict(b) for b in bombs]
        hypo_bombs.append(
            {
                "pos": pos,
                "timer": 7,
                "owner": self.agent_id,
                "radius": radius,
                "hypothetical": True,
            }
        )
        schedule = self._compute_schedule(grid, hypo_bombs)
        hypo_danger = schedule["danger_at"]

        if self._is_deadly(pos, 1, hypo_danger):
            return self._illegal_bomb_score()

        my_escape = self._find_escape(grid, pos, 1, hypo_bombs, hypo_danger, 8, enemies=ctx.get("enemies"))
        if my_escape is None:
            return self._illegal_bomb_score()

        _, safe_t, safe_pos = my_escape
        escape_area = self._reachable_count(
            ctx, safe_pos, safe_t, hypo_bombs, hypo_danger, 15
        )
        slack = max(0, 7 - safe_t)

        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        boxes = sum(1 for tx, ty in blast if grid[tx, ty] == self.BOX)
        enemy_hits = [
            e for e in ctx["enemies"] if e["pos"] in blast and self._is_future_safe(e["pos"], 1, danger_at)
        ]

        trap_score = 0.0
        for enemy in ctx["enemies"]:
            epos = enemy["pos"]
            if epos not in blast:
                if self._manhattan(pos, epos) <= radius + 1:
                    trap_score += 0.25
                continue
            enemy_escape = self._find_escape(
                grid, epos, 0, hypo_bombs, hypo_danger, horizon=7, ignore_start_bomb=True
            )
            if enemy_escape is None:
                trap_score += 2.8
            else:
                _, etime, e_safe = enemy_escape
                enemy_area = self._reachable_count(ctx, e_safe, etime, hypo_bombs, hypo_danger, 5)
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
                profile = self._enemy_escape_profile(ctx, enemy["pos"], hypo_bombs, hypo_danger, min(det_time + 1, 8))
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
        score += endgame_trap_bonus
        
        mode = ctx.get("mode", "economy")
        aggression = ctx.get("aggression", 0.5)

        if mode == "escape":
            score -= 10.0
        elif mode == "unstuck":
            score += boxes * 0.9 + min(escape_area, 10) * 0.08
        else:
            enemy_weight = 0.5 + aggression * 2.7
            trap_weight = 0.1 + aggression * 1.7
            score += len(enemy_hits) * enemy_weight + trap_score * trap_weight

            target_id = ctx.get("target_enemy_id")
            if target_id is not None:
                target_hits = [enemy for enemy in enemy_hits if enemy["id"] == target_id]
                if target_hits:
                    score += aggression * 4.0

            if mode == "economy":
                box_bonus = max(0.0, 1.7 - aggression * 4.0) 
                score += boxes * box_bonus
                if boxes == 0 and not enemy_hits:
                    score -= 1.0 + (1.0 - aggression)
                if enemy_hits and trap_score < 1.8:
                    score -= 1.6
            elif mode == "combat":
                if boxes and not enemy_hits:
                    score -= 0.25 + aggression * 1.55
                if escape_area <= 6 and aggression < 0.6:
                    score -= 1.2

        if ctx["phase"] == "end" and boxes and not enemy_hits:
            score -= 2.0
        if ctx.get("engagement_deficit") and boxes:
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
            "slack": slack,
            "escape_area": escape_area,
            "phase": ctx["phase"],
            "mode": mode,
            "aggression": aggression,
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
        mode = ctx.get("mode", "economy")
        aggression = ctx.get("aggression", 0.5)

        if mode == "escape":
            return 0.0

        base = 8.0 - aggression * 3.0 if mode == "economy" else 6.0 - aggression * 4.0
        
        my_rad = ctx["my_radius"]
        my_bombs = ctx["my_bombs_left"]

        if cell == self.ITEM_RADIUS:
            if my_rad >= self.MAX_RADIUS:
                return 0.0
            return base * (1.2 if my_rad < 3 else 1.0)
        elif cell == self.ITEM_CAPACITY:
            max_b = ctx["max_bombs_left_seen"]
            if my_bombs >= self.MAX_BOMBS:
                return 0.0
            bonus = 1.5 if my_bombs == 1 else (1.2 if my_bombs < max_b else 1.0)
            return base * bonus
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
                if ctx["mode"] in ("economy", "unstuck", "box_farm"):
                    value += boxes * 0.9
                    if ctx["mode"] == "box_farm":
                        value += boxes * 1.4 + max(0, boxes - 2) * 1.2
                elif ctx["mode"] in ("duel", "survive_duel"):
                    value *= 0.45
                elif ctx["mode"] in ("pressure", "tiebreak", "hunt"):
                    value *= 0.75
                elif ctx["mode"] in ("escape", "item_race"):
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
                best = (score, pos, action)
        
        if best is not None:
            best_score, best_pos, best_action = best
            b_score = self._score_bomb_at(ctx, best_pos)
            if b_score["legal"] and b_score["score"] > self._bomb_threshold(b_score):
                if best_pos == ctx["my_pos"]:
                    return (best_score, self.BOMB)
                return (best_score, best_action)
        return None

    def _best_enemy_pressure_move(self, ctx, reach):
        if ctx["my_bombs_left"] <= 0 or not ctx["enemies"]:
            return None
        grid = ctx["grid"]
        candidates = []
        target_enemy_id = ctx.get("target_enemy_id")
        for enemy in ctx["enemies"]:
            if target_enemy_id is not None and enemy["id"] != target_enemy_id and ctx["mode"] == "hunt":
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
                    if ctx["mode"] in ("pressure", "tiebreak"):
                        value += 0.7
                    elif ctx["mode"] == "duel":
                        value += 1.2
                    elif ctx["mode"] == "survive_duel":
                        value += 0.7
                    elif ctx["mode"] == "economy":
                        value -= 1.5
                    elif ctx["mode"] == "hunt":
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
                best = (score, pos, action)
        
        if best is not None:
            best_score, best_pos, best_action = best
            b_score = self._score_bomb_at(ctx, best_pos)
            if b_score["legal"] and b_score["score"] > self._bomb_threshold(b_score):
                if best_pos == ctx["my_pos"]:
                    return (best_score, self.BOMB)
                return (best_score, best_action)
        return None

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
            if nearest_enemy_dist > 6 and ctx["mode"] not in ("tiebreak", "hunt"):
                continue
            score = 3.2 - nearest_enemy_dist * 0.38 - dist * 0.18
            score += self._open_neighbors(grid, pos, ctx["bomb_positions"]) * 0.25
            score -= self._manhattan(pos, center) * 0.05
            if ctx["mode"] == "duel":
                score += 1.0
            elif ctx["mode"] == "survive_duel":
                score += min(
                    self._reachable_count(ctx, pos, dist, ctx["bombs"], ctx["danger_at"], 6),
                    12,
                ) * 0.08
                score -= max(0, 3 - nearest_enemy_dist) * 0.4
            elif ctx["mode"] == "tiebreak":
                score += max(0, 5 - nearest_enemy_dist) * 0.25
                score += min(ctx["engagement_deficit"], 5) * 0.18
            elif ctx["mode"] == "hunt":
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
                    ctx["grid"], npos, 1, ctx["bombs"], ctx["danger_at"], self.HORIZON, enemies=ctx.get("enemies")
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
        enemies=None,
        ignore_start_bomb=False,
    ):
        enemy_risk = set()
        if enemies:
            for e in enemies:
                ex, ey = e["pos"] if isinstance(e, dict) else e
                enemy_risk.add((ex, ey))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    enemy_risk.add((ex + dx, ey + dy))

        def search(avoid_enemies):
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
                    if avoid_enemies and npos in enemy_risk and npos != start:
                        continue
                    state = (npos, nt)
                    if state in seen:
                        continue
                    seen.add(state)
                    first_action = action if first is None else first
                    q.append((npos, nt, first_action))
            return None

        res = search(avoid_enemies=True) if enemies else None
        if res is not None:
            return res
        return search(avoid_enemies=False)

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
        escape = self._find_escape(grid, my_pos, 0, [], danger_at, 10, enemies=enemies)
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
