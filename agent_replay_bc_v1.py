import importlib.util
import os
from pathlib import Path

import torch


_MODEL_PATH = Path(__file__).parent / "agent" / "ppo_agent" / "ppo_model.py"
_SPEC = importlib.util.spec_from_file_location("ppo_model_runtime", _MODEL_PATH)
_PPO_MODEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PPO_MODEL)

MaskedActorCritic = _PPO_MODEL.MaskedActorCritic
encode_obs = _PPO_MODEL.encode_obs
legal_action_mask = _PPO_MODEL.legal_action_mask
masked_logits = _PPO_MODEL.masked_logits


class Agent:
    team_id = "ReplayBCV1"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.turn = 0
        self.model = MaskedActorCritic()
        self.model.eval()
        self.has_model = False
        self._load_checkpoint()

    def _load_checkpoint(self):
        env_path = os.environ.get("BOMBERLAND_PPO_CHECKPOINT", "").strip()
        candidates = []
        if env_path:
            candidates.append(Path(env_path))
        candidates.extend(
            [
                Path(__file__).parent / "agent" / "ppo_agent" / "replay_bc.pt",
                Path(__file__).parent / "agent" / "ppo_agent" / "ppo_replay_finetuned.pt",
                Path(__file__).parent / "agent" / "ppo_agent" / "ppo_bomber.pt",
                Path(__file__).parent / "agent" / "ppo_agent" / "ppo_smoke.pt",
            ]
        )

        for path in candidates:
            if not path.exists():
                continue
            try:
                checkpoint = torch.load(path, map_location="cpu")
                state = checkpoint.get("model", checkpoint)
                self.model.load_state_dict(state)
                self.has_model = True
                return
            except Exception:
                continue

    def act(self, obs: dict) -> int:
        self.turn += 1
        try:
            mask = legal_action_mask(obs, self.agent_id)
            if not self.has_model:
                return self._safe_fallback(obs, mask)
            map_feat, aux_feat = encode_obs(obs, self.agent_id, self.turn)
            with torch.no_grad():
                logits, _ = self.model(
                    torch.from_numpy(map_feat).unsqueeze(0),
                    torch.from_numpy(aux_feat).unsqueeze(0),
                )
                logits = masked_logits(logits, torch.from_numpy(mask).unsqueeze(0))
                return int(torch.argmax(logits, dim=1).item())
        except Exception:
            return 0

    def _safe_fallback(self, obs, mask):
        grid = obs["map"]
        players = obs["players"]
        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            return 0
        my = players[self.agent_id]
        x, y = int(my[0]), int(my[1])
        danger = set()
        for b in obs["bombs"]:
            bx, by, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            if timer > 2:
                continue
            radius = 1 + int(players[owner][4]) if 0 <= owner < len(players) else 1
            danger.add((bx, by))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                for r in range(1, min(5, radius) + 1):
                    nx, ny = bx + dx * r, by + dy * r
                    if not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]):
                        break
                    if int(grid[nx, ny]) == 1:
                        break
                    danger.add((nx, ny))
                    if int(grid[nx, ny]) == 2:
                        break
        for action in (1, 2, 3, 4, 0):
            if not mask[action]:
                continue
            dx, dy = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
            if (x + dx, y + dy) not in danger:
                return action
        return 0
