import os
import sys
import torch
import numpy as np

# Ensure root is in sys.path for dynamic importing
DQN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dqn_agent'))
if DQN_DIR not in sys.path:
    sys.path.insert(0, DQN_DIR)
    
MY_AGENT_DIR = os.path.abspath(os.path.dirname(__file__))
if MY_AGENT_DIR not in sys.path:
    sys.path.insert(0, MY_AGENT_DIR)

# Import the base Agent from AdaptiveProV2
from agent_adaptive_pro_v2 import Agent as AdaptiveProV2Agent
import torch.nn as nn

class DirectorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=3, padding=1),
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
            nn.Linear(128, 6) # NUM_CLASSES = 6
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class Agent(AdaptiveProV2Agent):
    team_id = "RLDirector_HybridProV2"

    def __init__(self, agent_id: int):
        super().__init__(agent_id)
        
        self.director = DirectorNet()
        model_path = os.path.join(os.path.dirname(__file__), "models", "director10chanelcu.pt")
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
        C = 10
        features = np.zeros((C, h, w), dtype=np.float32)
        
        # 1. Tường cứng (Wall)
        features[0] = (grid == 1).astype(np.float32)
        # 2. Hộp gỗ (Box)
        features[1] = (grid == 2).astype(np.float32)
        # 3. Vật phẩm (Items: 3 là Radius, 4 là Capacity)
        features[2] = ((grid == 3) | (grid == 4)).astype(np.float32)
        
        # 4. Vị trí bản thân
        my_pos = ctx["my_pos"]
        features[3, my_pos[0], my_pos[1]] = 1.0
        
        # 5. Vị trí kẻ thù
        for e in ctx["enemies"]:
            features[4, e["pos"][0], e["pos"][1]] = 1.0
            
        # 6. Bom và thời gian nổ
        for b in ctx["bombs"]:
            features[5, b["pos"][0], b["pos"][1]] = max(0, 1.0 - (b["timer"] / 40.0))

        # 7. Danger Map
        danger_at = ctx.get("danger_at", {})
        for (x, y), t_set in danger_at.items():
            min_t = min(t_set) if isinstance(t_set, set) and t_set else (t_set if not isinstance(t_set, set) else 40)
            features[6, x, y] = max(0, 1.0 - (min_t / 40.0))

        # 8. Earliest Explosion
        earliest = ctx.get("earliest", {})
        for (x, y), t in earliest.items():
            features[7, x, y] = max(0, 1.0 - (t / 40.0))

        # 9. Bomb Radius Map
        for b in ctx["bombs"]:
            features[8, b["pos"][0], b["pos"][1]] = b["radius"] / 5.0

        # 10. Bombs Left Map
        features[9, my_pos[0], my_pos[1]] = ctx["my_bombs_left"] / 5.0
        for e in ctx["enemies"]:
            features[9, e["pos"][0], e["pos"][1]] = e.get("bombs_left", 1) / 5.0

        return features

    def _tempo_mode(self, ctx):
        # Preserve strategic modes from AdaptiveProV3!
        strategic = ctx.get("strategic_mode", "adaptive")
        if strategic in ("escape", "hunt", "item_race", "box_farm"):
            return strategic
            
        # 1. Ép Tempo (Dùng cho quá trình Huấn luyện DQN/PPO)
        if hasattr(self, 'force_tempo') and self.force_tempo is not None:
            return self.force_tempo
            
        # 2. Dùng Neural Network để đoán Tempo liên tục mỗi turn
        if hasattr(self, 'director_loaded') and self.director_loaded:
            f = self._extract_features(ctx)
            batch_t = torch.tensor(np.array([f]), dtype=torch.float32)
            with torch.no_grad():
                logits = self.director(batch_t)
                pred_idx = torch.argmax(logits, dim=1).item()
                
            id_to_tempo = {
                0: "economy", 
                1: "unstuck", 
                2: "pressure", 
                3: "duel", 
                4: "tiebreak", 
                5: "survive_duel"
            }
            return id_to_tempo[pred_idx]
            
        # 3. Fallback: logic gốc của AdaptiveProV3
        return super()._tempo_mode(ctx)
