import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    _next_pos, _valid_actions, _danger_tiles, _move_to_safe, 
    _can_escape_after_placing, _count_boxes_in_blast, _box_bomb_spots, 
    _item_tiles, _move_to_targets, _shortest_path_length, _is_dead_end,
    _enemy_trapped_score, _chain_bomb_prediction
)

class Agent:
    team_id = "AdvancedHeuristicBot"
    
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)

    def act(self, obs: dict) -> int:
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
        # RULE 2: AGGRESSIVE TRAPPING (Áp sát & Đặt bom ép góc)
        # ========================================================
        if bombs_left > 0 and closest_enemy_pos and min_dist <= 3:
            # Kiểm tra xem địch có bị kẹt không
            e_space = _enemy_trapped_score(grid, closest_enemy_pos, blocked)
            if e_space <= 6:
                # Địch đang kẹt hoặc trong ngõ cụt, thử ném bom
                if _can_escape_after_placing(grid, my_pos, blocked, danger_now, bomb_radius):
                    return 5

        # ========================================================
        # RULE 3: FARMING ITEMS & BOXES
        # ========================================================
        # Ưu tiên nhặt đồ
        items = _item_tiles(grid, prefer_capacity=(bombs_left <= 1), prefer_radius=(bomb_bonus <= 1))
        if items:
            move = _move_to_targets(grid, my_pos, blocked, items, danger_soon)
            if move is not None:
                return move

        # Đặt bom phá hộp nếu đang đứng cạnh hộp
        boxes_here = _count_boxes_in_blast(grid, my_pos[0], my_pos[1], bomb_radius)
        if bombs_left > 0 and boxes_here > 0:
            if _can_escape_after_placing(grid, my_pos, blocked, danger_now, bomb_radius):
                return 5

        # Đi tìm chỗ đặt bom farm hộp
        box_spots = _box_bomb_spots(grid, blocked)
        if box_spots:
            # Lọc bỏ ngõ cụt để tránh rủi ro nếu có kẻ thù đẩy vào
            safe_spots = {spot for spot in box_spots if not _is_dead_end(grid, spot, blocked)}
            if not safe_spots:
                safe_spots = box_spots # Nếu chỗ nào cũng là ngõ cụt thì đành chịu
            move = _move_to_targets(grid, my_pos, blocked, safe_spots, danger_soon)
            if move is not None:
                return move
                
        # ========================================================
        # RULE 4: HUNT ENEMY OR WANDER
        # ========================================================
        if closest_enemy_pos:
            move = _move_to_targets(grid, my_pos, blocked, {closest_enemy_pos}, danger_soon)
            if move is not None:
                return move

        # Đi bừa cho an toàn
        safe_moves = [a for a in valid_actions if _next_pos(my_pos, a) not in danger_soon]
        return random.choice(safe_moves) if safe_moves else 0
