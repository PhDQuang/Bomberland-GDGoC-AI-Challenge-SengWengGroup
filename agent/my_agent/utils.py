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
    """Sử dụng A* để tìm độ dài đường đi ngắn nhất đến 1 target cụ thể."""
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

def _move_to_targets(grid, start, targets, blocked_set, danger_tiles):
    """Tìm hành động đầu tiên để đi đến bất kỳ target nào gần nhất."""
    if start in targets:
        return 0
    if not targets:
        return None
        
    q = deque([(start, None)])
    seen = {start}
    
    while q:
        pos, first_action = q.popleft()
        
        if pos in targets and first_action is not None:
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
            if npos in danger_tiles:
                continue
                
            seen.add(npos)
            q.append((npos, a if first_action is None else first_action))
            
    return None

# ---------------------------------------------------------
# 3. Bomb / Explosion / Chain Reaction
# ---------------------------------------------------------

def _blast_tiles(grid, bx, by, radius):
    """Trả về các ô bị ảnh hưởng bởi 1 quả bom."""
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
    """Tính toán nổ dây chuyền."""
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
    """Tính 2 vùng nguy hiểm: sắp nổ và sẽ nổ ngay."""
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
    """Đếm số ô an toàn có thể đến được."""
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
    """Kiểm tra xem vị trí có phải ngõ cụt không."""
    return _escape_space_score(grid, start, blocked_set, max_depth=6) <= 4

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
    """Đánh giá xem kẻ địch có đang bị kẹt hay không (ít không gian tẩu thoát)."""
    return _escape_space_score(grid, enemy_pos, blocked_set, max_depth=5)
