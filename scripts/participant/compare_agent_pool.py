import argparse
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine.game import BomberEnv
from scripts.participant.run_local_match import make_agents


def rank_match(env, alive, death_steps):
    stats = [p.stats for p in env.players]
    survivor_order = [i for i, is_alive in enumerate(alive) if is_alive]
    survivor_order.sort(
        key=lambda i: (
            stats[i]["kills"],
            stats[i]["boxes"],
            stats[i]["items"],
            stats[i]["bombs"],
        ),
        reverse=True,
    )

    groups = []
    if survivor_order:
        current_key = None
        current_group = []
        for i in survivor_order:
            key = (
                stats[i]["kills"],
                stats[i]["boxes"],
                stats[i]["items"],
                stats[i]["bombs"],
            )
            if current_key is None or key == current_key:
                current_group.append(i)
            else:
                groups.append(current_group)
                current_group = [i]
            current_key = key
        groups.append(current_group)

    dead_steps = sorted({s for s in death_steps if s is not None}, reverse=True)
    for step in dead_steps:
        groups.append([i for i, s in enumerate(death_steps) if s == step])

    ranks = [0] * 4
    rank = 0
    for group in groups:
        for i in group:
            ranks[i] = rank
        rank += len(group)
    return ranks, stats


def compare(agent_paths, num_matches, max_steps, seed):
    totals = {
        "wins": [0] * 4,
        "best_draws": [0] * 4,
        "rank_sum": [0] * 4,
        "kills": [0] * 4,
        "boxes": [0] * 4,
        "items": [0] * 4,
        "bombs": [0] * 4,
    }
    names = None

    for match in range(num_matches):
        match_seed = seed + match if seed is not None else None
        env = BomberEnv(max_steps=max_steps, seed=match_seed)
        agents, names = make_agents(agent_paths, seed=match_seed)
        obs = env.reset(seed=match_seed)
        prev_alive = [bool(p[2]) for p in obs["players"]]
        death_steps = [None] * 4
        done = False
        step = 0

        while not done and step < max_steps:
            actions = []
            for i, agent in enumerate(agents):
                try:
                    actions.append(int(agent.act(obs)))
                except Exception:
                    actions.append(0)
            obs, terminated, truncated = env.step(actions)
            step += 1
            alive = [bool(p[2]) for p in obs["players"]]
            for i in range(4):
                if prev_alive[i] and not alive[i]:
                    death_steps[i] = step
            prev_alive = alive
            done = terminated or truncated

        ranks, stats = rank_match(env, prev_alive, death_steps)
        best_rank = min(ranks)
        winners = [i for i, r in enumerate(ranks) if r == best_rank]
        for i in range(4):
            totals["rank_sum"][i] += ranks[i]
            totals["kills"][i] += stats[i]["kills"]
            totals["boxes"][i] += stats[i]["boxes"]
            totals["items"][i] += stats[i]["items"]
            totals["bombs"][i] += stats[i]["bombs"]
        if len(winners) == 1:
            totals["wins"][winners[0]] += 1
        else:
            for i in winners:
                totals["best_draws"][i] += 1

    print("slot,name,wins,best_draws,avg_rank,kills,boxes,items,bombs")
    for i in range(4):
        print(
            f"{i},{names[i]},{totals['wins'][i]},{totals['best_draws'][i]},"
            f"{totals['rank_sum'][i] / num_matches:.3f},"
            f"{totals['kills'][i]},{totals['boxes'][i]},"
            f"{totals['items'][i]},{totals['bombs'][i]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_paths", nargs=4, required=True)
    parser.add_argument("--num_matches", type=int, default=40)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    compare(args.agent_paths, args.num_matches, args.max_steps, args.seed)
