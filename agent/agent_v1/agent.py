import random
import sys
import heapq
from collections import deque
import copy

MOVES = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}

def _next_pos(pos, action):
    dx, dy = MOVES.get(action, (0, 0))
    return pos[0] + dx, pos[1] + dy

def _in_bounds(grid, x, y):
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

def _passable(grid, x, y):
    return _in_bounds(grid, x, y) and grid[x, y] in [0, 3, 4]

def _valid_actions(grid, pos, blocked_set):
    actions = [0]
    for a in [1, 2, 3, 4]:
        nx, ny = _next_pos(pos, a)
        if _passable(grid, nx, ny) and (nx, ny) not in blocked_set:
            actions.append(a)
    return actions

def heuristic_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def _shortest_path_length(grid, start, target, blocked_set):
    if start == target: return 0
    pq = []
    heapq.heappush(pq, (0, 0, start))
    seen = {start: 0}
    while pq:
        f, g, pos = heapq.heappop(pq)
        if pos == target: return g
        if g > seen.get(pos, float('inf')): continue
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

def _move_to_targets(grid, start, targets, blocked_set, danger_tiles):
    if start in targets: return 0
    if not targets: return None
    q = deque([(start, None)])
    seen = {start}
    while q:
        pos, first_action = q.popleft()
        if pos in targets and first_action is not None:
            return first_action
        for a in [1, 2, 3, 4]:
            nx, ny = _next_pos(pos, a)
            npos = (nx, ny)
            if npos in seen: continue
            if not _passable(grid, nx, ny): continue
            if npos in blocked_set: continue
            if npos in danger_tiles: continue
            seen.add(npos)
            q.append((npos, a if first_action is None else first_action))
    return None

def _blast_tiles(grid, bx, by, radius):
    tiles = {(bx, by)}
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for r in range(1, radius + 1):
            x, y = bx + dx * r, by + dy * r
            if not _in_bounds(grid, x, y): break
            cell = grid[x, y]
            if cell == 1: break
            tiles.add((x, y))
            if cell == 2: break
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
        if b_info['timer'] <= 0: continue
        blast = _blast_tiles(grid, b_pos[0], b_pos[1], b_info['radius'])
        danger_soon |= blast
        if b_info['timer'] <= 1:
            danger_now |= blast
    return danger_soon, danger_now

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
            if npos in seen: continue
            if not _passable(grid, nx, ny): continue
            if npos in blocked_set: continue
            seen.add(npos)
            q.append((npos, a if first_action is None else first_action))
    return None

def _can_escape_after_placing(grid, my_pos, blocked_set, existing_danger, bomb_radius):
    my_blast = _blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
    combined_danger = set(existing_danger) | my_blast
    temp_blocked = set(blocked_set) | {my_pos}
    q = deque([(my_pos, 0)])
    seen = {my_pos}
    while q:
        pos, dist = q.popleft()
        if pos not in combined_danger: return True
        if dist > 8: continue
        for a in [1, 2, 3, 4]:
            nx, ny = _next_pos(pos, a)
            npos = (nx, ny)
            if not _passable(grid, nx, ny): continue
            if npos in temp_blocked: continue
            if npos in seen: continue
            seen.add(npos)
            q.append((npos, dist + 1))
    return False

def _escape_space_score(grid, start, blocked_set, max_depth=10):
    q = deque([(start, 0)])
    seen = {start}
    space = 0
    while q:
        pos, d = q.popleft()
        space += 1
        if d >= max_depth: continue
        for a in [1, 2, 3, 4]:
            nx, ny = _next_pos(pos, a)
            npos = (nx, ny)
            if not _passable(grid, nx, ny) or npos in blocked_set or npos in seen: continue
            seen.add(npos)
            q.append((npos, d + 1))
    return space

def _is_dead_end(grid, start, blocked_set):
    return _escape_space_score(grid, start, blocked_set, max_depth=6) <= 4

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

def _enemy_trapped_score(grid, enemy_pos, blocked_set):
    return _escape_space_score(grid, enemy_pos, blocked_set, max_depth=5)

class Agent:
    team_id = "AdvancedHeuristicBot_V1"
    
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)

    def act(self, obs: dict) -> int:
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        bomb_radius = max(1, int(bomb_bonus) + 1)
        
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        occupied = {(int(p[0]), int(p[1])) for i, p in enumerate(players) if i != self.agent_id and p[2] == 1}
        blocked = set(occupied) | bomb_positions
        blocked.discard(my_pos)

        danger_soon, danger_now = _danger_tiles(grid, bombs, players)
        valid_actions = _valid_actions(grid, my_pos, blocked)

        # RULE 1: SURVIVAL
        if my_pos in danger_now or my_pos in danger_soon:
            escape_action = _move_to_safe(grid, my_pos, blocked, danger_soon)
            if escape_action is not None:
                return escape_action
            safe_now = [a for a in valid_actions if _next_pos(my_pos, a) not in danger_now]
            return random.choice(safe_now) if safe_now else 0

        # Phân tích kẻ địch gần nhất
        closest_enemy_pos = None
        min_dist = float('inf')
        for i, p in enumerate(players):
            if i != self.agent_id and p[2] == 1:
                e_pos = (int(p[0]), int(p[1]))
                dist = _shortest_path_length(grid, my_pos, e_pos, blocked)
                if dist < min_dist:
                    min_dist = dist
                    closest_enemy_pos = e_pos

        # RULE 2: AGGRESSIVE TRAPPING
        if bombs_left > 0 and closest_enemy_pos and min_dist <= 3:
            e_space = _enemy_trapped_score(grid, closest_enemy_pos, blocked)
            if e_space <= 6:
                if _can_escape_after_placing(grid, my_pos, blocked, danger_now, bomb_radius):
                    return 5

        # RULE 3: FARMING ITEMS & BOXES
        items = _item_tiles(grid, prefer_capacity=(bombs_left <= 1), prefer_radius=(bomb_bonus <= 1))
        if items:
            move = _move_to_targets(grid, my_pos, items, blocked, danger_soon)
            if move is not None:
                return move

        boxes_here = _count_boxes_in_blast(grid, my_pos[0], my_pos[1], bomb_radius)
        if bombs_left > 0 and boxes_here > 0:
            if _can_escape_after_placing(grid, my_pos, blocked, danger_now, bomb_radius):
                return 5

        box_spots = _box_bomb_spots(grid, blocked)
        if box_spots:
            safe_spots = {spot for spot in box_spots if not _is_dead_end(grid, spot, blocked)}
            if not safe_spots:
                safe_spots = box_spots
            move = _move_to_targets(grid, my_pos, safe_spots, blocked, danger_soon)
            if move is not None:
                return move
                
        # RULE 4: HUNT ENEMY OR WANDER
        if closest_enemy_pos:
            move = _move_to_targets(grid, my_pos, {closest_enemy_pos}, blocked, danger_soon)
            if move is not None:
                return move

        safe_moves = [a for a in valid_actions if _next_pos(my_pos, a) not in danger_soon]
        return random.choice(safe_moves) if safe_moves else 0
