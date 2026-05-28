from collections import deque

import numpy as np
import torch
import torch.nn as nn


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
N_ACTIONS = 6
N_CHANNELS = 12
AUX_DIM = 10


def in_bounds(grid, x, y):
    return 0 <= int(x) < grid.shape[0] and 0 <= int(y) < grid.shape[1]


def passable(grid, x, y):
    return in_bounds(grid, x, y) and int(grid[x, y]) in (GRASS, ITEM_RADIUS, ITEM_CAPACITY)


def blast_tiles(grid, bx, by, radius):
    tiles = {(int(bx), int(by))}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, int(radius) + 1):
            x = int(bx) + dx * r
            y = int(by) + dy * r
            if not in_bounds(grid, x, y):
                break
            cell = int(grid[x, y])
            if cell == WALL:
                break
            tiles.add((x, y))
            if cell == BOX:
                break
    return tiles


def legal_action_mask(obs, agent_id):
    grid = obs["map"]
    players = obs["players"]
    bombs = obs["bombs"]
    mask = np.zeros(N_ACTIONS, dtype=np.bool_)
    if agent_id >= len(players) or int(players[agent_id][2]) != 1:
        mask[STOP] = True
        return mask

    x, y = int(players[agent_id][0]), int(players[agent_id][1])
    bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
    mask[STOP] = True
    for action in DIRS:
        dx, dy = MOVES[action]
        nx, ny = x + dx, y + dy
        if passable(grid, nx, ny) and (nx, ny) not in bomb_positions:
            mask[action] = True
    if int(players[agent_id][3]) > 0 and (x, y) not in bomb_positions:
        mask[BOMB] = True
    return mask


def reachable_area(obs, agent_id, limit=24):
    grid = obs["map"]
    players = obs["players"]
    bombs = {(int(b[0]), int(b[1])) for b in obs["bombs"]}
    if int(players[agent_id][2]) != 1:
        return 0
    start = (int(players[agent_id][0]), int(players[agent_id][1]))
    q = deque([start])
    seen = {start}
    while q and len(seen) < limit:
        pos = q.popleft()
        for action in DIRS:
            dx, dy = MOVES[action]
            npos = (pos[0] + dx, pos[1] + dy)
            if npos in seen or npos in bombs:
                continue
            if passable(grid, npos[0], npos[1]):
                seen.add(npos)
                q.append(npos)
    return len(seen)


def encode_obs(obs, agent_id, turn=0):
    grid = obs["map"]
    players = obs["players"]
    bombs = obs["bombs"]
    h, w = grid.shape
    channels = []
    for cell in (GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY):
        channels.append((grid == cell).astype(np.float32))

    self_pos = np.zeros((h, w), dtype=np.float32)
    enemy_pos = np.zeros((h, w), dtype=np.float32)
    if agent_id < len(players) and int(players[agent_id][2]) == 1:
        self_pos[int(players[agent_id][0]), int(players[agent_id][1])] = 1.0
    alive_enemies = []
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue
        ex, ey = int(p[0]), int(p[1])
        enemy_pos[ex, ey] = 1.0
        alive_enemies.append((ex, ey))

    bomb_timer = np.zeros((h, w), dtype=np.float32)
    own_bomb = np.zeros((h, w), dtype=np.float32)
    enemy_bomb = np.zeros((h, w), dtype=np.float32)
    danger_now = np.zeros((h, w), dtype=np.float32)
    danger_soon = np.zeros((h, w), dtype=np.float32)
    for b in bombs:
        bx, by, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        radius = 1
        if 0 <= owner < len(players):
            radius = min(5, 1 + int(players[owner][4]))
        bomb_timer[bx, by] = max(bomb_timer[bx, by], float(timer) / 7.0)
        if owner == agent_id:
            own_bomb[bx, by] = 1.0
        else:
            enemy_bomb[bx, by] = 1.0
        if timer <= 1:
            for tx, ty in blast_tiles(grid, bx, by, radius):
                danger_now[tx, ty] = 1.0
        if timer <= 3:
            for tx, ty in blast_tiles(grid, bx, by, radius):
                danger_soon[tx, ty] = 1.0

    map_feat = np.stack(
        [
            *channels,
            self_pos,
            enemy_pos,
            bomb_timer,
            own_bomb,
            enemy_bomb,
            danger_now,
            danger_soon,
        ],
        axis=0,
    ).astype(np.float32)

    my = players[agent_id]
    mx, my_y = int(my[0]), int(my[1])
    nearest_enemy = 0.0
    if alive_enemies:
        nearest_enemy = min(abs(mx - ex) + abs(my_y - ey) for ex, ey in alive_enemies) / float(h + w)
    nearest_bomb_timer = 1.0
    if len(bombs):
        nearest_bomb_timer = min(int(b[2]) for b in bombs) / 7.0
    aux = np.array(
        [
            float(mx) / h,
            float(my_y) / w,
            float(int(my[3])) / 5.0,
            float(int(my[4])) / 4.0,
            float(len(alive_enemies)) / 3.0,
            nearest_enemy,
            nearest_bomb_timer,
            float(reachable_area(obs, agent_id)) / 24.0,
            min(float(turn) / 500.0, 1.0),
            1.0 if int(my[2]) == 1 else 0.0,
        ],
        dtype=np.float32,
    )
    return map_feat, aux


class MaskedActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.map_encoder(torch.zeros(1, N_CHANNELS, 13, 13)).shape[1]
        self.aux_encoder = nn.Sequential(
            nn.Linear(AUX_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.body = nn.Sequential(
            nn.Linear(flat_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, N_ACTIONS)
        self.critic = nn.Linear(128, 1)

    def forward(self, map_x, aux_x):
        feat = torch.cat([self.map_encoder(map_x), self.aux_encoder(aux_x)], dim=1)
        body = self.body(feat)
        return self.actor(body), self.critic(body).squeeze(-1)


def masked_logits(logits, masks):
    return logits.masked_fill(~masks.bool(), -1.0e9)
