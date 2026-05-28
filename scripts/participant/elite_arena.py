import argparse
import contextlib
import csv
import itertools
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.game import BomberEnv
from competition.evaluation.runtime_guard import load_agent_instance


OFFICIAL_BASELINES = {
    "RandomAgent",
    "SimpleRuleAgent",
    "SmarterRuleAgent",
    "GeniusRuleAgent",
    "BoxFarmerAgent",
    "TacticalRuleAgent",
    "Random",
    "None",
}

BASELINE_CLASSES = {
    "RandomAgent": "RandomAgent",
    "SimpleRuleAgent": "SimpleRuleAgent",
    "SmarterRuleAgent": "SmarterRuleAgent",
    "GeniusRuleAgent": "GeniusRuleAgent",
    "BoxFarmerAgent": "BoxFarmerAgent",
    "TacticalRuleAgent": "TacticalRuleAgent",
}

DIRS = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


def _resolve_agent_spec(spec: str) -> str:
    spec = str(spec).strip()
    if spec in OFFICIAL_BASELINES:
        return spec
    path = Path(spec)
    if path.exists():
        return str(path)

    typo_fallbacks = {
        "agentEnemy1.py": "agentEnermy1.py",
        "agent_enemy1.py": "agentEnermy1.py",
    }
    alt = typo_fallbacks.get(spec)
    if alt and Path(alt).exists():
        return alt
    return spec


def _agent_label(spec: str) -> str:
    resolved = _resolve_agent_spec(spec)
    if resolved in OFFICIAL_BASELINES:
        return resolved
    path = Path(resolved)
    if path.name == "agent.py" and path.parent == Path("."):
        return "agent.py"
    return path.name if path.name else str(path)


def _load_agent(spec: str, agent_id: int):
    resolved = _resolve_agent_spec(spec)
    if resolved in {"Random", "None"}:
        resolved = "RandomAgent"
    if resolved in BASELINE_CLASSES:
        from agent import (
            BoxFarmerAgent,
            GeniusRuleAgent,
            RandomAgent,
            SimpleRuleAgent,
            SmarterRuleAgent,
            TacticalRuleAgent,
        )

        cls = {
            "RandomAgent": RandomAgent,
            "SimpleRuleAgent": SimpleRuleAgent,
            "SmarterRuleAgent": SmarterRuleAgent,
            "GeniusRuleAgent": GeniusRuleAgent,
            "BoxFarmerAgent": BoxFarmerAgent,
            "TacticalRuleAgent": TacticalRuleAgent,
        }[resolved]
        return cls(agent_id)

    path = Path(resolved)
    if path.is_dir():
        path = path / "agent.py"
    return load_agent_instance(str(path), agent_id)


def _agent_worker(spec, agent_id, recv_conn, send_conn):
    try:
        agent = _load_agent(spec, agent_id)
    except Exception as exc:
        send_conn.send({"ok": False, "error": f"load_failed:{exc}"})
        return

    send_conn.send({"ok": True})
    devnull = open(os.devnull, "w", encoding="utf-8")
    while True:
        try:
            payload = recv_conn.recv()
        except EOFError:
            return
        cmd = payload.get("cmd")
        if cmd == "close":
            return
        if cmd != "act":
            send_conn.send({"ok": False, "error": f"unknown_cmd:{cmd}"})
            continue
        try:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                action = int(agent.act(payload.get("obs")))
            send_conn.send({"ok": True, "action": action})
        except Exception as exc:
            send_conn.send({"ok": False, "error": f"act_failed:{exc}"})


class AgentExecutor:
    def __init__(self, spec, agent_id, startup_timeout_s=5.0):
        self.spec = spec
        self.agent_id = agent_id
        self.startup_timeout_s = startup_timeout_s
        method = "spawn" if sys.platform.startswith("win") else "fork"
        self.ctx = mp.get_context(method)
        self.proc = None
        self.parent_recv = None
        self.parent_send = None

    def start(self):
        if self.proc is not None and self.proc.is_alive():
            return True, None

        parent_recv, child_send = self.ctx.Pipe(duplex=False)
        child_recv, parent_send = self.ctx.Pipe(duplex=False)
        proc = self.ctx.Process(
            target=_agent_worker,
            args=(self.spec, self.agent_id, child_recv, child_send),
            daemon=True,
        )
        proc.start()
        self.proc = proc
        self.parent_recv = parent_recv
        self.parent_send = parent_send

        if not self.parent_recv.poll(self.startup_timeout_s):
            self.terminate()
            return False, "startup_timeout"
        boot = self.parent_recv.recv()
        if not boot.get("ok"):
            self.terminate()
            return False, boot.get("error", "startup_failed")
        return True, None

    def is_alive(self):
        return self.proc is not None and self.proc.is_alive()

    def act(self, obs, timeout_s):
        if not self.is_alive():
            return 0, False, False, "not_started"
        try:
            self.parent_send.send({"cmd": "act", "obs": obs})
        except Exception as exc:
            self.terminate()
            return 0, False, False, f"send_failed:{exc}"

        if not self.parent_recv.poll(timeout_s):
            self.terminate()
            return 0, True, False, "timeout"

        try:
            reply = self.parent_recv.recv()
        except Exception as exc:
            self.terminate()
            return 0, False, False, f"recv_failed:{exc}"

        if not reply.get("ok"):
            self.terminate()
            return 0, False, False, reply.get("error", "act_failed")
        try:
            action = int(reply.get("action"))
        except Exception:
            return 0, False, True, "invalid_action"
        if action < 0 or action > 5:
            return 0, False, True, "invalid_action"
        return action, False, False, None

    def terminate(self):
        if self.parent_send is not None:
            try:
                self.parent_send.send({"cmd": "close"})
            except Exception:
                pass
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=0.2)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(timeout=0.2)
        self.proc = None
        self.parent_recv = None
        self.parent_send = None


def _blast_tiles(grid, bomb):
    tiles = {(bomb["x"], bomb["y"])}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, int(bomb["radius"]) + 1):
            x = bomb["x"] + dx * r
            y = bomb["y"] + dy * r
            if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
                break
            cell = int(grid[x, y])
            if cell == 1:
                break
            tiles.add((x, y))
            if cell == 2:
                break
    return tiles


def _predict_affected_tiles(env, actions):
    grid = env.map.grid.copy()
    players = [
        {
            "x": p.x,
            "y": p.y,
            "alive": bool(p.alive),
            "bombs_left": p.bombs_left,
            "bonus": p.bomb_radius_bonus,
        }
        for p in env.players
    ]
    bombs = [
        {
            "x": b.x,
            "y": b.y,
            "timer": b.timer,
            "owner": b.owner_id,
            "radius": b.radius,
            "exploded": False,
        }
        for b in env.bombs
    ]
    old_bomb_positions = {(b["x"], b["y"]) for b in bombs}
    pending_bombs = {}

    for player_id, action in enumerate(actions):
        player = players[player_id]
        if not player["alive"]:
            continue

        if action == 5:
            if player["bombs_left"] <= 0:
                continue
            pos = (player["x"], player["y"])
            if pos in old_bomb_positions:
                continue
            radius = 1 + int(player["bonus"])
            if pos not in pending_bombs or radius > pending_bombs[pos][1]:
                pending_bombs[pos] = (player_id, radius)
            continue

        if action in DIRS:
            dx, dy = DIRS[action]
            nx = player["x"] + dx
            ny = player["y"] + dy
            if not (0 < nx < grid.shape[0] - 1 and 0 < ny < grid.shape[1] - 1):
                continue
            if int(grid[nx, ny]) in (1, 2):
                continue
            if (nx, ny) in old_bomb_positions:
                continue
            player["x"], player["y"] = nx, ny

    for (x, y), (owner, radius) in pending_bombs.items():
        bombs.append(
            {
                "x": x,
                "y": y,
                "timer": 7,
                "owner": owner,
                "radius": radius,
                "exploded": False,
            }
        )

    exploding = []
    for bomb in bombs:
        if bomb["exploded"]:
            continue
        bomb["timer"] -= 1
        if bomb["timer"] <= 0:
            bomb["exploded"] = True
            exploding.append(bomb)

    idx = 0
    while idx < len(exploding):
        bomb = exploding[idx]
        idx += 1
        blast = _blast_tiles(grid, bomb)
        for other in bombs:
            if other["exploded"]:
                continue
            if (other["x"], other["y"]) in blast:
                other["exploded"] = True
                exploding.append(other)

    affected = {}
    for bomb in exploding:
        for tile in _blast_tiles(grid, bomb):
            affected.setdefault(tile, set()).add(bomb["owner"])
    return affected


def _rank_match(env, death_order):
    alive = [i for i, player in enumerate(env.players) if player.alive]
    if alive:
        def stats(player_id):
            s = env.players[player_id].stats
            return (s["kills"], s["boxes"], s["items"], s["bombs"])

        alive.sort(key=stats, reverse=True)
        groups = []
        current = [alive[0]]
        current_stats = stats(alive[0])
        for player_id in alive[1:]:
            player_stats = stats(player_id)
            if player_stats == current_stats:
                current.append(player_id)
            else:
                groups.append(current)
                current = [player_id]
                current_stats = player_stats
        groups.append(current)
        death_order.extend(reversed(groups))

    ranks = [0] * len(env.players)
    for rank, group in enumerate(reversed(death_order)):
        for player_id in group:
            ranks[player_id] = rank
    return ranks


def _new_metrics():
    return {
        "matches": 0,
        "wins": 0,
        "draws": 0,
        "rank_sum": 0,
        "first_deaths": 0,
        "deaths": 0,
        "self_deaths": 0,
        "enemy_deaths": 0,
        "unknown_deaths": 0,
        "crashes": 0,
        "timeouts": 0,
        "survival_step_sum": 0,
        "runtime_ms": [],
    }


def _slot_orders(agent_specs, mode):
    indices = list(range(len(agent_specs)))
    if mode == "fixed":
        return [indices]
    if mode == "permutations":
        return [list(order) for order in itertools.permutations(indices)]
    return [indices[i:] + indices[:i] for i in range(len(indices))]


def run_lobby(lobby_id, agent_specs, matches, seed, max_steps, slot_mode, timeout_ms):
    resolved_specs = [_resolve_agent_spec(spec) for spec in agent_specs]
    labels = [_agent_label(spec) for spec in agent_specs]
    orders = _slot_orders(resolved_specs, slot_mode)
    metrics = defaultdict(_new_metrics)

    for match_index in range(matches):
        order = orders[match_index % len(orders)]
        slot_specs = [resolved_specs[i] for i in order]
        slot_labels = [labels[i] for i in order]
        match_seed = seed + match_index

        executors = [
            AgentExecutor(spec=slot_specs[slot], agent_id=slot)
            for slot in range(len(slot_specs))
        ]
        env = BomberEnv(max_steps=max_steps, seed=match_seed)
        obs = env.reset(seed=match_seed)

        previous_alive = [bool(player.alive) for player in env.players]
        death_order = []
        death_reasons = [""] * 4
        survival_steps = [max_steps] * 4
        terminated = False
        truncated = False

        try:
            while not (terminated or truncated):
                actions = []
                for slot, executor in enumerate(executors):
                    label = slot_labels[slot]
                    if not env.players[slot].alive:
                        actions.append(0)
                        continue

                    if not executor.is_alive():
                        ok, error = executor.start()
                        if not ok:
                            metrics[label]["crashes"] += 1
                            actions.append(0)
                            continue

                    started = time.perf_counter()
                    action, timed_out, invalid, error = executor.act(obs, timeout_ms / 1000.0)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    metrics[label]["runtime_ms"].append(elapsed_ms)
                    if timed_out:
                        metrics[label]["timeouts"] += 1
                    if invalid or (error and not timed_out):
                        metrics[label]["crashes"] += 1
                    actions.append(action)

                affected = _predict_affected_tiles(env, actions)
                obs, terminated, truncated = env.step(actions)

                alive_now = [bool(player.alive) for player in env.players]
                deaths = []
                for slot in range(4):
                    if previous_alive[slot] and not alive_now[slot]:
                        deaths.append(slot)
                        survival_steps[slot] = env.current_step
                        pos = (env.players[slot].x, env.players[slot].y)
                        owners = affected.get(pos, set())
                        if slot in owners:
                            death_reasons[slot] = "self"
                        elif owners:
                            death_reasons[slot] = "enemy"
                        else:
                            death_reasons[slot] = "unknown"
                if deaths:
                    death_order.append(deaths)
                previous_alive = alive_now
        finally:
            for executor in executors:
                executor.terminate()

        for slot, player in enumerate(env.players):
            if player.alive:
                survival_steps[slot] = env.current_step

        ranks = _rank_match(env, death_order)
        best_rank = min(ranks)
        winners = [slot for slot, rank in enumerate(ranks) if rank == best_rank]
        first_death_group = death_order[0] if death_order else []

        for slot, label in enumerate(slot_labels):
            row = metrics[label]
            row["matches"] += 1
            row["rank_sum"] += ranks[slot]
            row["survival_step_sum"] += survival_steps[slot]
            if slot in winners and len(winners) == 1:
                row["wins"] += 1
            elif slot in winners:
                row["draws"] += 1
            if slot in first_death_group:
                row["first_deaths"] += 1
            reason = death_reasons[slot]
            if reason:
                row["deaths"] += 1
                row[f"{reason}_deaths"] += 1

    return lobby_id, metrics


def _percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def _format_rows(lobby_id, metrics):
    rows = []
    for label, data in sorted(metrics.items()):
        matches = max(1, data["matches"])
        runtimes = data["runtime_ms"]
        rows.append(
            {
                "lobby": lobby_id,
                "agent": label,
                "matches": data["matches"],
                "wins": data["wins"],
                "draws": data["draws"],
                "win_rate": data["wins"] / matches,
                "draw_rate": data["draws"] / matches,
                "avg_rank": data["rank_sum"] / matches,
                "first_death_rate": data["first_deaths"] / matches,
                "death_rate": data["deaths"] / matches,
                "self_death_rate": data["self_deaths"] / matches,
                "enemy_death_rate": data["enemy_deaths"] / matches,
                "unknown_death_rate": data["unknown_deaths"] / matches,
                "timeouts": data["timeouts"],
                "crashes": data["crashes"],
                "avg_survival_step": data["survival_step_sum"] / matches,
                "runtime_p95_ms": _percentile(runtimes, 95),
                "runtime_max_ms": max(runtimes) if runtimes else 0.0,
            }
        )
    return rows


def _preset_lobbies(variant, enemy1):
    return [
        ("A", [variant, "agent_temporal_hybrid_v1.py", enemy1, "TacticalRuleAgent"]),
        ("B", [variant, "agent_temporal_hybrid_v1.py", enemy1, "GeniusRuleAgent"]),
        ("C", [variant, "agent_temporal_hybrid_v1.py", enemy1, "SmarterRuleAgent"]),
        ("D", [variant, "agent_temporal_hybrid_v1.py", "agent_temporal_hybrid_v1.py", "TacticalRuleAgent"]),
        ("E", [variant, enemy1, enemy1, "TacticalRuleAgent"]),
        ("F", [variant, "TacticalRuleAgent", "GeniusRuleAgent", "SmarterRuleAgent"]),
    ]


def _parse_lobby_args(values):
    lobbies = []
    for idx, value in enumerate(values or [], start=1):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 4:
            raise ValueError(f"--lobby must contain exactly 4 comma-separated agents: {value}")
        lobbies.append((f"custom_{idx}", parts))
    return lobbies


def main():
    parser = argparse.ArgumentParser(description="Run elite Bomberland lobbies with slot rotation.")
    parser.add_argument("--agents", nargs=4, help="Run one lobby with exactly four agents.")
    parser.add_argument("--lobby", action="append", help="Comma-separated 4-agent lobby. Can be repeated.")
    parser.add_argument("--preset", choices=["elite"], help="Run predefined elite protocol lobbies.")
    parser.add_argument("--variant", default="agent_phase_v1.py", help="Variant path used by --preset elite.")
    parser.add_argument("--enemy1", default="agentEnemy1.py", help="Strong external enemy path for preset lobbies.")
    parser.add_argument("--matches_per_lobby", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=86000)
    parser.add_argument("--slot_mode", choices=["fixed", "rotations", "permutations"], default="rotations")
    parser.add_argument("--timeout_ms", type=float, default=100.0)
    parser.add_argument("--csv", default="logs/elite_arena_summary.csv")
    args = parser.parse_args()

    lobbies = []
    if args.agents:
        lobbies.append(("agents", args.agents))
    lobbies.extend(_parse_lobby_args(args.lobby))
    if args.preset == "elite":
        lobbies.extend(_preset_lobbies(args.variant, args.enemy1))
    if not lobbies:
        raise SystemExit("Provide --agents, --lobby, or --preset elite.")

    all_rows = []
    for lobby_index, (lobby_id, specs) in enumerate(lobbies):
        seed = args.seed + lobby_index * 10000
        print(f"\n=== Lobby {lobby_id}: {specs} ===")
        lobby_id, metrics = run_lobby(
            lobby_id=lobby_id,
            agent_specs=specs,
            matches=args.matches_per_lobby,
            seed=seed,
            max_steps=args.max_steps,
            slot_mode=args.slot_mode,
            timeout_ms=args.timeout_ms,
        )
        rows = _format_rows(lobby_id, metrics)
        all_rows.extend(rows)
        for row in rows:
            print(
                f"{row['agent']:<32} "
                f"W {row['wins']:>3}/{row['matches']:<3} "
                f"D {row['draws']:>3} "
                f"avg_rank {row['avg_rank']:.2f} "
                f"first {row['first_death_rate']:.2f} "
                f"self {row['self_death_rate']:.2f} "
                f"surv {row['avg_survival_step']:.1f} "
                f"p95 {row['runtime_p95_ms']:.1f}ms "
                f"max {row['runtime_max_ms']:.1f}ms "
                f"to/crash {row['timeouts']}/{row['crashes']}"
            )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lobby",
        "agent",
        "matches",
        "wins",
        "draws",
        "win_rate",
        "draw_rate",
        "avg_rank",
        "first_death_rate",
        "death_rate",
        "self_death_rate",
        "enemy_death_rate",
        "unknown_death_rate",
        "timeouts",
        "crashes",
        "avg_survival_step",
        "runtime_p95_ms",
        "runtime_max_ms",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved CSV summary to {csv_path}")


if __name__ == "__main__":
    main()
