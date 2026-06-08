import os
import sys
import torch
import numpy as np
from collections import deque

MY_AGENT_DIR = os.path.abspath(os.path.dirname(__file__))
if MY_AGENT_DIR not in sys.path:
    sys.path.insert(0, MY_AGENT_DIR)

# Inherit from the new V2 expert
from agent_ultimate_heuristic_v2 import Agent as UltimateHeuristicV2Agent
import torch.nn as nn

class DirectorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(19, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 13 * 13, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4) # NUM_CLASSES = 4 (economy, unstuck, combat, escape)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class Agent(UltimateHeuristicV2Agent):
    team_id = "HeuristicDirector"

    def __init__(self, agent_id: int):
        super().__init__(agent_id)
        
        self.director = DirectorNet()
        # The model that will be trained by train_director.py
        model_path = os.path.join(os.path.dirname(__file__), "models", "director_19ch.pt")
        self.director_loaded = os.path.exists(model_path)
        if self.director_loaded:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            # Remove '_orig_mod.' prefix added by torch.compile
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.director.load_state_dict(state_dict)
            self.director.eval()
            
        self.force_tempo = None

    def _extract_features(self, ctx):
        grid = ctx["grid"]
        h, w = grid.shape
        C = 19
        features = np.zeros((C, h, w), dtype=np.float32)
        
        # 0-9: Core Base Features
        features[0] = (grid == 1).astype(np.float32)
        features[1] = (grid == 2).astype(np.float32)
        features[2] = ((grid == 3) | (grid == 4)).astype(np.float32)
        
        my_pos = ctx["my_pos"]
        features[3, my_pos[0], my_pos[1]] = 1.0
        
        for e in ctx["enemies"]:
            features[4, e["pos"][0], e["pos"][1]] = 1.0
            
        for b in ctx["bombs"]:
            features[5, b["pos"][0], b["pos"][1]] = max(0, 1.0 - (b["timer"] / 40.0))

        danger_at = ctx.get("danger_at", {})
        for (x, y), t_set in danger_at.items():
            min_t = min(t_set) if isinstance(t_set, set) and t_set else (t_set if not isinstance(t_set, set) else 40)
            features[6, x, y] = max(0, 1.0 - (min_t / 40.0))

        earliest = ctx.get("earliest", {})
        for (x, y), t in earliest.items():
            features[7, x, y] = max(0, 1.0 - (t / 40.0))

        for b in ctx["bombs"]:
            features[8, b["pos"][0], b["pos"][1]] = b["radius"] / 5.0

        features[9, my_pos[0], my_pos[1]] = ctx.get("my_bombs_left", 1) / 5.0
        for e in ctx["enemies"]:
            features[9, e["pos"][0], e["pos"][1]] = e.get("bombs_left", 1) / 5.0

        # 10. Escape Distance Map
        def get_min_d(r, c):
            v = danger_at.get((r, c), 40)
            return min(v) if isinstance(v, set) and v else (v if not isinstance(v, set) else 40)

        safe_tiles = [(r, c) for r in range(h) for c in range(w) 
                      if grid[r, c] not in (1, 2) and get_min_d(r, c) > 2]
        dist_escape = np.full((h, w), -1.0)
        q = deque(safe_tiles)
        for r, c in safe_tiles: dist_escape[r, c] = 0
        while q:
            r, c = q.popleft()
            d = dist_escape[r, c]
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<h and 0<=nc<w and grid[nr, nc] not in (1, 2) and dist_escape[nr, nc] == -1:
                    dist_escape[nr, nc] = d + 1
                    q.append((nr, nc))
        features[10] = np.where(dist_escape != -1, np.maximum(0, 1.0 - dist_escape / 15.0), 0)

        # 11. Enemy Distance Map
        dist_enemy = np.full((h, w), -1.0)
        q = deque([e["pos"] for e in ctx["enemies"]])
        for r, c in q: dist_enemy[r, c] = 0
        while q:
            r, c = q.popleft()
            d = dist_enemy[r, c]
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<h and 0<=nc<w and grid[nr, nc] not in (1, 2) and dist_enemy[nr, nc] == -1:
                    dist_enemy[nr, nc] = d + 1
                    q.append((nr, nc))
        features[11] = np.where(dist_enemy != -1, np.maximum(0, 1.0 - dist_enemy / 15.0), 0)

        # 12. My Distance Map
        dist_my = np.full((h, w), -1.0)
        q = deque([my_pos])
        dist_my[my_pos] = 0
        while q:
            r, c = q.popleft()
            d = dist_my[r, c]
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<h and 0<=nc<w and grid[nr, nc] not in (1, 2) and dist_my[nr, nc] == -1:
                    dist_my[nr, nc] = d + 1
                    q.append((nr, nc))
        features[12] = np.where(dist_my != -1, np.maximum(0, 1.0 - dist_my / 15.0), 0)

        # 13 & 14. Fast Bomb Score & Trap Score Maps
        my_radius = ctx.get("my_radius", 1)
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 0:
                    boxes = 0
                    enemies = 0
                    for e in ctx["enemies"]:
                        if e["pos"] == (r, c): enemies += 1
                    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                        for dist in range(1, my_radius + 1):
                            nr, nc = r + dr*dist, c + dc*dist
                            if not (0 <= nr < h and 0 <= nc < w): break
                            if grid[nr, nc] == 1: break
                            if grid[nr, nc] == 2:
                                boxes += 1
                                break
                            for e in ctx["enemies"]:
                                if e["pos"] == (nr, nc): enemies += 1
                    features[13, r, c] = boxes / 5.0
                    features[14, r, c] = enemies / 3.0

        # 15. Safe Reachable Area Component Map
        visited = set()
        for r in range(h):
            for c in range(w):
                if grid[r, c] not in (1, 2) and (r, c) not in visited:
                    v_d = danger_at.get((r, c), 40)
                    d_val = min(v_d) if isinstance(v_d, set) and v_d else (v_d if not isinstance(v_d, set) else 40)
                    if d_val > 2:
                        comp = []
                        q_comp = deque([(r, c)])
                        visited.add((r, c))
                        while q_comp:
                            curr = q_comp.popleft()
                            comp.append(curr)
                            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                                nr, nc = curr[0]+dr, curr[1]+dc
                                if 0<=nr<h and 0<=nc<w and grid[nr, nc] not in (1, 2) and (nr, nc) not in visited:
                                    v_n = danger_at.get((nr, nc), 40)
                                    n_d_val = min(v_n) if isinstance(v_n, set) and v_n else (v_n if not isinstance(v_n, set) else 40)
                                    if n_d_val > 2:
                                        visited.add((nr, nc))
                                        q_comp.append((nr, nc))
                        size_norm = len(comp) / 20.0
                        for cr, cc in comp:
                            features[15, cr, cc] = min(1.0, size_norm)

        # 16. Enemy Risk Map
        for pos in ctx.get("enemy_risk", set()):
            if 0 <= pos[0] < h and 0 <= pos[1] < w:
                features[16, pos[0], pos[1]] = 1.0

        # 17. Game Phase
        features[17, :, :] = getattr(self, 'turn', 0) / 500.0

        # 18. Alive Count
        features[18, :, :] = ctx.get("alive_count", 4) / 4.0

        return features

    def _analyze_macro_state(self, ctx):
        base_mode, aggression, target_enemy_id = super()._analyze_macro_state(ctx)
        
        if hasattr(self, 'force_tempo') and self.force_tempo is not None:
            return self.force_tempo, aggression, target_enemy_id
            
        if hasattr(self, 'director_loaded') and self.director_loaded:
            f = self._extract_features(ctx)
            batch_t = torch.tensor(np.array([f]), dtype=torch.float32)
            with torch.no_grad():
                logits = self.director(batch_t)
                pred_idx = torch.argmax(logits, dim=1).item()
                
            id_to_tempo = {
                0: "economy", 
                1: "unstuck", 
                2: "combat", 
                3: "escape"
            }
            return id_to_tempo.get(pred_idx, "economy"), aggression, target_enemy_id
            
        return base_mode, aggression, target_enemy_id
