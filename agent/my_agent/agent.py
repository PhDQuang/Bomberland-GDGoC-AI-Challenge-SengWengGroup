import random
import sys
import time
from pathlib import Path
import heapq
from collections import deque
import copy
import itertools

MOVES = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}

# ---------------------------------------------------------
# 1. Movement / Map
# ---------------------------------------------------------

def _next_pos(pos, action):
    dx, dy = MOVES.get(action, (0, 0))
    return pos[0] + dx, pos[1] + dy

def _in_bounds(grid, x, y):
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

def _passable(grid, x, y):
    # 0: grass, 3: item range, 4: item capacity
    return _in_bounds(grid, x, y) and grid[x, y] in [0, 3, 4]

def _valid_actions(grid, pos, blocked_set):
    actions = [0]
    for a in [1, 2, 3, 4]:
        nx, ny = _next_pos(pos, a)
        if _passable(grid, nx, ny) and (nx, ny) not in blocked_set:
            actions.append(a)
    return actions

# ---------------------------------------------------------
# 2. BFS / Pathfinding (A*)
# ---------------------------------------------------------

def heuristic_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def _shortest_path_length(grid, start, target, blocked_set):
    if start == target:
        return 0
        
    pq = []
    heapq.heappush(pq, (0, 0, start))
    seen = {start: 0}
    
    while pq:
        f, g, pos = heapq.heappop(pq)
        
        if pos == target:
            return g
            
        if g > seen.get(pos, float('inf')):
            continue
            
        for a in [1, 2, 3, 4]:
            nx, ny = _next_pos(pos, a)
            npos = (nx, ny)
            if _passable(grid, nx, ny) and npos not in blocked_set:
                new_g = g + 1
                if new_g < seen.get(npos, float('inf')):
                    seen[npos] = new_g
                    f_val = new_g + heuristic_dist(npos, target)
                    heapq.heappush(pq, (f_val, new_g, npos))
    return float('inf')

def _astar_to_targets(grid, start, targets, blocked_set, danger_tiles):
    if start in targets:
        return 0
    if not targets:
        return None
        
    pq = [(0, 0, start[0], start[1], None)]
    seen = {start: 0}
    
    while pq:
        f_val, g, x, y, first_action = heapq.heappop(pq)
        pos = (x, y)
        
        if pos in targets and first_action is not None:
            return first_action
            
        if g > seen.get(pos, float('inf')):
            continue
            
        for a in [1, 2, 3, 4]:
            nx, ny = _next_pos(pos, a)
            npos = (nx, ny)
            if not _passable(grid, nx, ny): continue
            if npos in blocked_set: continue
            if npos in danger_tiles: continue
                
            new_g = g + 1
            if new_g < seen.get(npos, float('inf')):
                seen[npos] = new_g
                h = min(abs(nx - tx) + abs(ny - ty) for tx, ty in targets)
                heapq.heappush(pq, (new_g + h, new_g, nx, ny, a if first_action is None else first_action))
                
    return None

# ---------------------------------------------------------
# 3. Bomb / Explosion / Chain Reaction
# ---------------------------------------------------------

def _blast_tiles(grid, bx, by, radius):
    tiles = {(bx, by)}
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for r in range(1, radius + 1):
            x, y = bx + dx * r, by + dy * r
            if not _in_bounds(grid, x, y):
                break
            cell = grid[x, y]
            if cell == 1:
                break
            tiles.add((x, y))
            if cell == 2:
                break
    return tiles

def _chain_bomb_prediction(grid, bombs, players, default_radius=2):
    bomb_dict = {}
    for b in bombs:
        pos = (int(b[0]), int(b[1]))
        timer = int(b[2])
        owner_id = int(b[3]) if len(b) > 3 else -1
        
        radius = default_radius
        if 0 <= owner_id < len(players):
            radius = max(1, int(players[owner_id][4]) + 1)
            
        bomb_dict[pos] = {'timer': timer, 'radius': radius}
        
    changed = True
    while changed:
        changed = False
        for b_pos, b_info in bomb_dict.items():
            blast = _blast_tiles(grid, b_pos[0], b_pos[1], b_info['radius'])
            for other_pos in blast:
                if other_pos in bomb_dict and other_pos != b_pos:
                    if bomb_dict[other_pos]['timer'] > b_info['timer']:
                        bomb_dict[other_pos]['timer'] = b_info['timer']
                        changed = True
    return bomb_dict

def _danger_tiles(grid, bombs, players):
    bomb_timers = _chain_bomb_prediction(grid, bombs, players)
    
    danger_soon = set()
    danger_now = set()
    
    for b_pos, b_info in bomb_timers.items():
        if b_info['timer'] <= 0:
            continue
        blast = _blast_tiles(grid, b_pos[0], b_pos[1], b_info['radius'])
        danger_soon |= blast
        if b_info['timer'] <= 1:
            danger_now |= blast
            
    return danger_soon, danger_now

def _build_danger_timeline(grid, bombs, players):
    bomb_timers = _chain_bomb_prediction(grid, bombs, players)
    danger_timeline = {}
    for b_pos, b_info in bomb_timers.items():
        t = b_info['timer']
        if t <= 0: continue
        blast = _blast_tiles(grid, b_pos[0], b_pos[1], b_info['radius'])
        for p in blast:
            if p not in danger_timeline:
                danger_timeline[p] = set()
            danger_timeline[p].add(t)
            danger_timeline[p].add(t + 1)
            danger_timeline[p].add(t + 2)
    return danger_timeline

# ---------------------------------------------------------
# 4. Survival & Anti-Suicide
# ---------------------------------------------------------

def _move_to_safe(grid, start, blocked_set, danger_tiles):
    if start not in danger_tiles and start not in blocked_set:
        return 0
        
    q = deque([(start, None)])
    seen = {start}
    
    while q:
        pos, first_action = q.popleft()
        
        if pos not in danger_tiles and pos not in blocked_set and first_action is not None:
            return first_action
            
        for a in [1, 2, 3, 4]:
            nx, ny = _next_pos(pos, a)
            npos = (nx, ny)
            if npos in seen:
                continue
            if not _passable(grid, nx, ny):
                continue
            if npos in blocked_set:
                continue
                
            seen.add(npos)
            q.append((npos, a if first_action is None else first_action))
            
    return None

def _spacetime_move_to_safe(grid, start, blocked_set, danger_timeline):
    pq = [(0, start[0], start[1], None)]
    seen = {(start[0], start[1], 0)}
    
    while pq:
        t, x, y, first_action = heapq.heappop(pq)
        pos = (x, y)
        
        is_safe_forever = True
        if pos in danger_timeline:
            if any(dt >= t for dt in danger_timeline[pos]):
                is_safe_forever = False
                
        if is_safe_forever and first_action is not None and pos not in blocked_set:
            return first_action
            
        if t > 25: 
            continue
            
        for a in [0, 1, 2, 3, 4]:
            nx, ny = _next_pos(pos, a)
            npos = (nx, ny)
            nt = t + 1
            if not _passable(grid, nx, ny): continue
            if npos in blocked_set and npos != start: continue
            if npos in danger_timeline and nt in danger_timeline[npos]: continue
                
            state = (nx, ny, nt)
            if state not in seen:
                seen.add(state)
                heapq.heappush(pq, (nt, nx, ny, a if first_action is None else first_action))
    return None

def _can_escape_after_placing(grid, my_pos, blocked_set, existing_danger, bomb_radius):
    my_blast = _blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
    combined_danger = set(existing_danger) | my_blast
    
    temp_blocked = set(blocked_set) | {my_pos}
    
    q = deque([(my_pos, 0)])
    seen = {my_pos}
    
    while q:
        pos, dist = q.popleft()
        
        if pos not in combined_danger:
            return True
            
        if dist > 8: 
            continue
            
        for a in [1, 2, 3, 4]:
            nx, ny = _next_pos(pos, a)
            npos = (nx, ny)
            if not _passable(grid, nx, ny):
                continue
            if npos in temp_blocked:
                continue
            if npos in seen:
                continue
                
            seen.add(npos)
            q.append((npos, dist + 1))
            
    return False

# ---------------------------------------------------------
# 5. Territory & Space Analysis
# ---------------------------------------------------------

def _escape_space_score(grid, start, blocked_set, max_depth=10):
    q = deque([(start, 0)])
    seen = {start}
    space = 0
    
    while q:
        pos, d = q.popleft()
        space += 1
        
        if d >= max_depth:
            continue
            
        for a in [1, 2, 3, 4]:
            nx, ny = _next_pos(pos, a)
            npos = (nx, ny)
            if not _passable(grid, nx, ny) or npos in blocked_set or npos in seen:
                continue
            seen.add(npos)
            q.append((npos, d + 1))
            
    return space

def _is_dead_end(grid, start, blocked_set):
    return _escape_space_score(grid, start, blocked_set, max_depth=6) <= 4

def _territory_score(grid, my_pos, enemies, blocked_set):
    q = deque()
    q.append((my_pos, 0, 0))
    for e in enemies:
        q.append((e, 1, 0))
    owner = {}
    while q:
        pos, o_id, dist = q.popleft()
        if pos in owner: continue
        owner[pos] = (o_id, dist)
        for a in [1, 2, 3, 4]:
            nx, ny = _next_pos(pos, a)
            npos = (nx, ny)
            if _passable(grid, nx, ny) and npos not in blocked_set:
                if npos not in owner:
                    q.append((npos, o_id, dist + 1))
    return sum(1 for v in owner.values() if v[0] == 0)

# ---------------------------------------------------------
# 6. Farming & Item Logic
# ---------------------------------------------------------

def _count_boxes_in_blast(grid, bx, by, radius):
    blast = _blast_tiles(grid, bx, by, radius)
    return sum(1 for x, y in blast if grid[x, y] == 2)

def _box_bomb_spots(grid, blocked_set):
    spots = set()
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 2:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if _passable(grid, nx, ny) and (nx, ny) not in blocked_set:
                        spots.add((nx, ny))
    return spots

def _item_tiles(grid, prefer_capacity=False, prefer_radius=False):
    preferred = set()
    if prefer_radius: preferred.add(3)
    if prefer_capacity: preferred.add(4)
    
    preferred_tiles = {(x, y) for x in range(grid.shape[0]) for y in range(grid.shape[1]) if grid[x, y] in preferred}
    if preferred_tiles: return preferred_tiles
    
    return {(x, y) for x in range(grid.shape[0]) for y in range(grid.shape[1]) if grid[x, y] in [3, 4]}

# ---------------------------------------------------------
# 7. Enemy Logic
# ---------------------------------------------------------

def _enemy_trapped_score(grid, enemy_pos, blocked_set):
    return _escape_space_score(grid, enemy_pos, blocked_set, max_depth=5)

# ---------------------------------------------------------
# 8. MINIMAX (Paranoid 1v2) Fast Simulator
# ---------------------------------------------------------

def _evaluate_cascade_trap(grid, bombs, enemies):
    if not bombs or not enemies: return 0
    
    true_timers = {b_pos: b_info['timer'] for b_pos, b_info in bombs.items()}
    changed = True
    while changed:
        changed = False
        for b1, info1 in bombs.items():
            t1 = true_timers[b1]
            blast1 = _blast_tiles(grid, b1[0], b1[1], info1['radius'])
            for b2, info2 in bombs.items():
                if b1 != b2 and b2 in blast1:
                    if true_timers[b2] > t1:
                        true_timers[b2] = t1
                        changed = True
                        
    score = 0
    for b_pos, info in bombs.items():
        t = true_timers[b_pos]
        if t <= 5: # Fast explosion
            blast = _blast_tiles(grid, b_pos[0], b_pos[1], info['radius'])
            for e_pos in enemies:
                if e_pos in blast:
                    score += 5000 * (6 - t)
    return score

def _evaluate_state(grid, my_pos, my_alive, enemies, bombs, blocked_set):
    if not my_alive:
        return -999999
    score = 0
    score -= len(enemies) * 10000 
    
    e_space = _escape_space_score(grid, my_pos, blocked_set, max_depth=6)
    score += e_space * 10
    
    center = (grid.shape[0]//2, grid.shape[1]//2)
    my_dist_to_center = heuristic_dist(my_pos, center)
    score -= my_dist_to_center * 0.5 # Soft center control
    
    for e_pos in enemies:
        trap_score = _enemy_trapped_score(grid, e_pos, blocked_set)
        if trap_score <= 4:
            score += (10 - trap_score) * 100
            
    cascade_score = _evaluate_cascade_trap(grid, bombs, enemies)
    score += cascade_score
            
    return score

def _fast_simulate_step(grid, my_pos, my_action, enemies, enemy_actions, bombs, my_bomb_radius):
    new_bombs = {}
    for b_pos, b_info in bombs.items():
        t = b_info['timer'] - 1
        new_bombs[b_pos] = {'timer': t, 'radius': b_info['radius']}
        
    new_my_pos = my_pos
    if my_action == 5:
        if my_pos not in new_bombs:
            new_bombs[my_pos] = {'timer': 40, 'radius': my_bomb_radius}
    elif my_action != 0:
        nx, ny = _next_pos(my_pos, my_action)
        if _passable(grid, nx, ny):
            new_my_pos = (nx, ny)
            
    new_enemies = []
    for i, e_pos in enumerate(enemies):
        try:
            e_action = enemy_actions[i]
        except IndexError:
            print(f"CRASH: len(enemies)={len(enemies)}, len(enemy_actions)={len(enemy_actions)}")
            print(f"enemies={enemies}")
            print(f"enemy_actions={enemy_actions}")
            e_action = 0
        n_epos = e_pos
        if e_action == 5:
            if e_pos not in new_bombs:
                new_bombs[e_pos] = {'timer': 40, 'radius': 2}
        elif e_action != 0:
            nx, ny = _next_pos(e_pos, e_action)
            if _passable(grid, nx, ny):
                n_epos = (nx, ny)
        new_enemies.append(n_epos)
        
    changed = True
    while changed:
        changed = False
        for b_pos, b_info in list(new_bombs.items()):
            if b_info['timer'] <= 0:
                blast = _blast_tiles(grid, b_pos[0], b_pos[1], b_info['radius'])
                for other_pos in blast:
                    if other_pos in new_bombs and other_pos != b_pos:
                        if new_bombs[other_pos]['timer'] > 0:
                            new_bombs[other_pos]['timer'] = 0
                            changed = True
                            
    my_alive = True
    for b_pos, b_info in new_bombs.items():
        if b_info['timer'] <= 0:
            blast = _blast_tiles(grid, b_pos[0], b_pos[1], b_info['radius'])
            if new_my_pos in blast:
                my_alive = False
            new_enemies = [e for e in new_enemies if e not in blast]
            
    new_bombs = {k: v for k, v in new_bombs.items() if v['timer'] > 0}
    return new_my_pos, my_alive, new_enemies, new_bombs

def _get_enemy_joint_actions(grid, enemies, bombs, my_pos, top_k=None):
    e_actions = []
    for e_pos in enemies:
        acts = [0]
        for a in [1, 2, 3, 4]:
            nx, ny = _next_pos(e_pos, a)
            if _passable(grid, nx, ny):
                acts.append(a)
        # CHỈ cho phép địch thả bom nếu địch ở đủ gần ta (để tránh nổ Branching Factor)
        if heuristic_dist(e_pos, my_pos) <= 3:
            acts.append(5)
        e_actions.append(acts)
    return list(itertools.product(*e_actions))

def _minimax(grid, my_pos, my_alive, enemies, bombs, my_bomb_radius, depth, start_time, time_limit, blocked_set, tt, alpha=-float('inf'), beta=float('inf')):
    if not my_alive:
        return -999999, None
        
    state_key = (my_pos, tuple(sorted(enemies)), tuple(sorted((k, v['timer'], v['radius']) for k, v in bombs.items())), depth)
    if state_key in tt:
        entry = tt[state_key]
        if entry['type'] == 'EXACT': return entry['value'], entry['action']
        if entry['type'] == 'LOWER' and entry['value'] >= beta: return entry['value'], entry['action']
        if entry['type'] == 'UPPER' and entry['value'] <= alpha: return entry['value'], entry['action']
        
    if depth == 0 or not enemies or (time.perf_counter() - start_time) > time_limit:
        blocked = set(enemies) | set(bombs.keys()) | blocked_set
        val = _evaluate_state(grid, my_pos, my_alive, enemies, bombs, blocked)
        tt[state_key] = {'value': val, 'action': None, 'type': 'EXACT'}
        return val, None
        
    best_action = None
    max_score = -float('inf')
    alpha_orig = alpha
    
    # MOVE ORDERING
    my_valid = []
    dist_to_closest = min([heuristic_dist(my_pos, e) for e in enemies]) if enemies else 99
    if dist_to_closest <= 2:
        my_valid.append(5)
        
    moves = []
    for a in [1, 2, 3, 4]:
        nx, ny = _next_pos(my_pos, a)
        if _passable(grid, nx, ny):
            moves.append((a, min([heuristic_dist((nx, ny), e) for e in enemies]) if enemies else 0))
    moves.sort(key=lambda x: x[1])
    my_valid.extend([x[0] for x in moves])
    
    my_valid.append(0)
    
    if dist_to_closest > 2:
        my_valid.append(5)
        
    joint_enemy_actions = _get_enemy_joint_actions(grid, enemies, bombs, my_pos)
    
    for a in my_valid:
        min_score = float('inf')
        for e_actions in joint_enemy_actions:
            n_my_pos, n_my_alive, n_enemies, n_bombs = _fast_simulate_step(
                grid, my_pos, a, enemies, e_actions, bombs, my_bomb_radius
            )
            score, _ = _minimax(grid, n_my_pos, n_my_alive, n_enemies, n_bombs, my_bomb_radius, depth - 1, start_time, time_limit, blocked_set, tt, alpha, min(beta, min_score))
            
            if score < min_score:
                min_score = score
            if time.perf_counter() - start_time > time_limit:
                break
                
            if min_score <= alpha:
                break # Cắt tỉa Alpha
                
        if min_score > max_score:
            max_score = min_score
            best_action = a
            
        if time.perf_counter() - start_time > time_limit:
            break
            
        if max_score >= beta:
            break # Cắt tỉa Beta
        if max_score > alpha:
            alpha = max_score
            
    if time.perf_counter() - start_time <= time_limit:
        flag = 'EXACT'
        if max_score <= alpha_orig:
            flag = 'UPPER'
        elif max_score >= beta:
            flag = 'LOWER'
        tt[state_key] = {'value': max_score, 'action': best_action, 'type': flag}
            
    return max_score, best_action


# =========================================================
# BOT LÕI
# =========================================================
class Agent:
    team_id = "AdvancedHeuristicBot_V2"
    
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)

    def act(self, obs: dict) -> int:
        import traceback
        try:
            return self._act_internal(obs)
        except Exception as e:
            traceback.print_exc()
            return 0
            
    def _act_internal(self, obs: dict) -> int:
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        # Thông tin bản thân
        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        bomb_radius = max(1, int(bomb_bonus) + 1)
        
        # Vật cản
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        occupied = {(int(p[0]), int(p[1])) for i, p in enumerate(players) if i != self.agent_id and p[2] == 1}
        blocked = set(occupied) | bomb_positions
        blocked.discard(my_pos)

        # Tính toán nguy hiểm
        danger_soon, danger_now = _danger_tiles(grid, bombs, players)
        valid_actions = _valid_actions(grid, my_pos, blocked)

        # ========================================================
        # RULE 1: SURVIVAL (Sinh tồn ưu tiên số 1)
        # ========================================================
        if my_pos in danger_now or my_pos in danger_soon:
            escape_action = _move_to_safe(grid, my_pos, blocked, danger_soon)
            if escape_action is not None:
                return escape_action
                
            # FALLBACK: Nếu BFS thông thường không tìm được lối thoát, ta dùng Space-Time A*
            danger_timeline = _build_danger_timeline(grid, bombs, players)
            st_escape = _spacetime_move_to_safe(grid, my_pos, blocked, danger_timeline)
            if st_escape is not None:
                return st_escape
                
            # Không tìm được đường ra khỏi danger_soon, cố né danger_now
            safe_now = [a for a in valid_actions if _next_pos(my_pos, a) not in danger_now]
            return random.choice(safe_now) if safe_now else 0

        # ========================================================
        # Phân tích kẻ địch gần nhất
        # ========================================================
        closest_enemy_pos = None
        min_dist = float('inf')
        for i, p in enumerate(players):
            if i != self.agent_id and p[2] == 1:
                e_pos = (int(p[0]), int(p[1]))
                dist = _shortest_path_length(grid, my_pos, e_pos, blocked)
                if dist < min_dist:
                    min_dist = dist
                    closest_enemy_pos = e_pos

        # ========================================================
        # RULE 2: MINIMAX PARANOID 1v2
        # ========================================================
        # Chỉ kích hoạt Minimax nếu có địch ở gần để đảm bảo tốc độ
        if closest_enemy_pos and min_dist <= 3:
            start_time = time.perf_counter()
            time_limit = 0.097 # 97ms timeout (sát nút 100ms)
            best_minimax_action = None
            max_depth = 3 # Cho phép Minimax chìm sâu hết cỡ nếu còn thời gian
            
            nearby_enemies = []
            for p in players:
                if p[2] == 1:
                    epos = (int(p[0]), int(p[1]))
                    if epos != my_pos and heuristic_dist(my_pos, epos) <= 4:
                        nearby_enemies.append(epos)
            
            if nearby_enemies:
                current_bombs_dict = {}
                for b in bombs:
                    owner_id = int(b[3]) if len(b) > 3 else -1
                    rad = 2
                    if 0 <= owner_id < len(players):
                        rad = max(1, int(players[owner_id][4]) + 1)
                    current_bombs_dict[(int(b[0]), int(b[1]))] = {'timer': int(b[2]), 'radius': rad}
                    
                best_score = -float('inf')
                tt = {}
                reached_depth = 0
                for d in range(1, max_depth + 1):
                    score, action = _minimax(grid, my_pos, True, nearby_enemies, current_bombs_dict, bomb_radius, d, start_time, time_limit, occupied, tt)
                    
                    if time.perf_counter() - start_time > time_limit:
                        break
                        
                    reached_depth = d
                    if action is not None:
                        best_minimax_action = action
                        best_score = score
                        
                elapsed = time.perf_counter() - start_time
                if elapsed > 0.08:
                    print(f"[TIMING] Depth {reached_depth}/{max_depth} completed in {elapsed*1000:.2f}ms")
                        
                if best_minimax_action is not None:
                    # Nếu đường nào cũng chết (score rất âm), Fallback về Heuristic
                    if best_score > -10000:
                        # Kiểm tra an toàn trước khi hành động (tránh đặt bom tự sát)
                        if best_minimax_action == 5:
                            if _can_escape_after_placing(grid, my_pos, blocked, danger_now, bomb_radius):
                                return 5
                        elif best_minimax_action == 0:
                            if my_pos not in danger_soon:
                                return 0
                        else:
                            # Đảm bảo bước đi an toàn
                            nx, ny = _next_pos(my_pos, best_minimax_action)
                            if (nx, ny) not in danger_now:
                                return best_minimax_action

        # ========================================================
        # RULE 2.2: OPPORTUNISTIC TRAPPING (Kẹp chả)
        # ========================================================
        if bombs_left > 0 and len(bombs) > 0 and closest_enemy_pos and min_dist <= 4:
            e_space_current = _escape_space_score(grid, closest_enemy_pos, blocked | danger_soon, max_depth=6)
            if e_space_current > 0 and e_space_current <= 6:
                virtual_danger_my = set(danger_now) | _blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
                combined_danger_all = set(danger_soon) | virtual_danger_my
                e_space_after = _escape_space_score(grid, closest_enemy_pos, blocked | combined_danger_all, max_depth=6)
                if e_space_after == 0:
                    if _can_escape_after_placing(grid, my_pos, blocked, danger_now, bomb_radius):
                        return 5

        # ========================================================
        # RULE 2.5: DOUBLE-BOMB TRAP (Bẫy bom kép)
        # ========================================================
        if bombs_left >= 2 and closest_enemy_pos and min_dist <= 3:
            # Thử thả bom 1 ở vị trí hiện tại
            virtual_danger_1 = set(danger_now) | _blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
            e_space_1 = _escape_space_score(grid, closest_enemy_pos, blocked | {my_pos}, max_depth=5)
            
            # Nếu thả bom 1 chưa đủ ép chết địch, nhưng không gian bị thu hẹp đáng kể
            if e_space_1 > 0 and e_space_1 <= 6:
                for a in valid_actions:
                    if a != 0:
                        npos = _next_pos(my_pos, a)
                        # Giả định bước tới npos và thả quả thứ 2
                        virtual_danger_2 = virtual_danger_1 | _blast_tiles(grid, npos[0], npos[1], bomb_radius)
                        e_space_2 = _escape_space_score(grid, closest_enemy_pos, blocked | {my_pos, npos}, max_depth=5)
                        if e_space_2 == 0:
                            # Địch hoàn toàn hết lối thoát
                            if _can_escape_after_placing(grid, my_pos, blocked, danger_now, bomb_radius):
                                return 5

        # ========================================================
        # RULE 3: AGGRESSIVE TRAPPING (Áp sát & Đặt bom ép góc thường)
        # ========================================================
        if bombs_left > 0 and closest_enemy_pos and min_dist <= 3:
            e_space = _enemy_trapped_score(grid, closest_enemy_pos, blocked)
            if e_space <= 6:
                if _can_escape_after_placing(grid, my_pos, blocked, danger_now, bomb_radius):
                    return 5

        # ========================================================
        # RULE 4: FARMING ITEMS & BOXES
        # ========================================================
        items = _item_tiles(grid, prefer_capacity=(bombs_left <= 1), prefer_radius=(bomb_bonus <= 1))
        if items:
            # Tránh lấy item ở ngõ cụt sâu (không gian thoát < 3)
            safe_items = {item for item in items if _escape_space_score(grid, item, blocked, max_depth=4) >= 3}
            if not safe_items:
                safe_items = set(items)
            move = _astar_to_targets(grid, my_pos, safe_items, blocked, danger_soon)
            if move is not None:
                return move

        boxes_here = _count_boxes_in_blast(grid, my_pos[0], my_pos[1], bomb_radius)
        if bombs_left > 0 and boxes_here > 0:
            if _can_escape_after_placing(grid, my_pos, blocked, danger_now, bomb_radius):
                return 5

        box_spots = _box_bomb_spots(grid, blocked)
        if box_spots:
            # Tránh đặt bom ở ngõ cụt để lấy hộp, rất dễ bị kẹp chết
            safe_spots = {spot for spot in box_spots if _escape_space_score(grid, spot, blocked, max_depth=4) >= 3}
            if not safe_spots:
                safe_spots = set(box_spots)
            move = _astar_to_targets(grid, my_pos, safe_spots, blocked, danger_soon)
            if move is not None:
                return move

        # ========================================================
        # RULE 5: HUNT ENEMY OR WANDER
        # ========================================================
        if closest_enemy_pos:
            move = _astar_to_targets(grid, my_pos, {closest_enemy_pos}, blocked, danger_soon)
            if move is not None:
                return move

        safe_moves = [a for a in valid_actions if _next_pos(my_pos, a) not in danger_soon]
        if safe_moves:
            # Chọn nước đi hướng ra không gian rộng rãi (tránh ngõ cụt)
            best_move = safe_moves[0]
            best_space = -1
            for a in safe_moves:
                nx, ny = _next_pos(my_pos, a)
                space = _escape_space_score(grid, (nx, ny), blocked | set(danger_soon), max_depth=10)
                if space > best_space:
                    best_space = space
                    best_move = a
            return best_move
        return 0
