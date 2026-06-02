import argparse
import csv
import json
import random
import sys
from pathlib import Path

import trueskill

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine.game import BomberEnv
from scripts.participant.run_local_match import make_agents


DEFAULT_AGENTS = [
    "agentsangv4.py",
    "agentpromax_v1.py",
    "agent_adaptive_pro_v1.py",
    "agent_phase_v1.py",
    "agent_temporal_hybrid_v1.py",
    "agent_economy_v1.py",
    "TacticalRuleAgent",
]

WIN_REASON_KEYS = ("survival", "kills", "boxes", "items", "bombs")


def score(rating):
    return float(rating.mu) - 3.0 * float(rating.sigma)


def normalize_agent_spec(spec):
    # The baseline file defines TacticalRuleAgent, not class Agent, so use the
    # built-in loader name accepted by run_local_match.make_agents().
    normalized = spec.replace("\\", "/")
    if normalized in {"agent/tactical_rule_agent.py", "tactical_rule_agent.py"}:
        return "TacticalRuleAgent"
    return spec


def label_for_spec(spec):
    if spec == "TacticalRuleAgent":
        return "tactical_rule_agent"
    path = Path(spec)
    if path.name == "agent.py" and path.parent.name:
        return path.parent.name
    return path.stem if path.suffix else spec


def validate_agent_specs(agent_specs):
    missing = []
    for spec in agent_specs:
        if spec in {
            "RandomAgent",
            "SimpleRuleAgent",
            "SmarterRuleAgent",
            "GeniusRuleAgent",
            "BoxFarmerAgent",
            "TacticalRuleAgent",
            "Random",
            "None",
        }:
            continue
        path = Path(spec)
        if path.is_dir():
            path = path / "agent.py"
        if not path.exists():
            missing.append(spec)
    if missing:
        raise FileNotFoundError("Missing agent file(s): " + ", ".join(missing))


def stat_key(env, slot):
    stats = env.players[slot].stats
    return (
        int(stats["kills"]),
        int(stats["boxes"]),
        int(stats["items"]),
        int(stats["bombs"]),
    )


def grouped_by_equal_key(items, key_fn):
    groups = []
    current_group = []
    current_key = None
    for item in items:
        key = key_fn(item)
        if current_key is None or key == current_key:
            current_group.append(item)
        else:
            groups.append(current_group)
            current_group = [item]
        current_key = key
    if current_group:
        groups.append(current_group)
    return groups


def compute_dense_ranks(env, alive_final, death_steps):
    """Return TrueSkill ranks, lower is better. Tied agents share a rank."""
    groups_best_to_worst = []

    survivors = [i for i, alive in enumerate(alive_final) if alive]
    if survivors:
        survivors.sort(key=lambda i: stat_key(env, i), reverse=True)
        groups_best_to_worst.extend(grouped_by_equal_key(survivors, lambda i: stat_key(env, i)))

    dead_steps = sorted({step for step in death_steps if step is not None}, reverse=True)
    for step in dead_steps:
        groups_best_to_worst.append([i for i, s in enumerate(death_steps) if s == step])

    ranks = [0] * 4
    for rank, group in enumerate(groups_best_to_worst):
        for slot in group:
            ranks[slot] = rank
    return ranks


def determine_win_reason(env, alive_final, ranks):
    winners = [i for i, rank in enumerate(ranks) if rank == min(ranks)]
    if len(winners) != 1:
        return None

    winner = winners[0]
    survivors = [i for i, alive in enumerate(alive_final) if alive]
    if not alive_final[winner] or len(survivors) <= 1:
        return "survival"

    # At max_steps, multiple survivors are separated by the official tie-break
    # order: kills, boxes, items, bombs. Return the first stat that makes the
    # unique winner strictly better than every other survivor.
    winner_stats = env.players[winner].stats
    for key in ("kills", "boxes", "items", "bombs"):
        if all(int(winner_stats[key]) > int(env.players[i].stats[key]) for i in survivors if i != winner):
            return key
    return "survival"


def run_one_match(agent_specs, seed, max_steps):
    env = BomberEnv(max_steps=max_steps, seed=seed)
    agents, _ = make_agents(agent_specs, seed=seed)
    names = [label_for_spec(spec) for spec in agent_specs]
    obs = env.reset(seed=seed)

    prev_alive = [bool(p[2]) for p in obs["players"]]
    death_steps = [None] * 4
    action_errors = [0] * 4
    invalid_actions = [0] * 4
    done = False
    step = 0

    while not done and step < max_steps:
        actions = []
        for slot, agent in enumerate(agents):
            try:
                action = int(agent.act(obs))
            except Exception:
                action = 0
                action_errors[slot] += 1
            if action not in (0, 1, 2, 3, 4, 5):
                action = 0
                invalid_actions[slot] += 1
            actions.append(action)

        obs, terminated, truncated = env.step(actions)
        step += 1
        alive_now = [bool(p[2]) for p in obs["players"]]
        for slot in range(4):
            if prev_alive[slot] and not alive_now[slot]:
                death_steps[slot] = step
        prev_alive = alive_now
        done = terminated or truncated

    alive_final = [bool(p[2]) for p in obs["players"]]
    ranks = compute_dense_ranks(env, alive_final, death_steps)
    win_reason = determine_win_reason(env, alive_final, ranks)
    survival_steps = [step if alive_final[i] else int(death_steps[i] or step) for i in range(4)]
    stats = [dict(env.players[i].stats) for i in range(4)]
    return {
        "seed": seed,
        "agent_specs": list(agent_specs),
        "names": names,
        "ranks": ranks,
        "win_reason": win_reason,
        "steps": step,
        "alive_final": alive_final,
        "death_steps": death_steps,
        "survival_steps": survival_steps,
        "stats": stats,
        "action_errors": action_errors,
        "invalid_actions": invalid_actions,
    }


def update_table(table, ratings, ts_env, match_result, recency_by_name):
    names = match_result["names"]
    ranks = match_result["ranks"]
    rating_groups = [(ratings[name],) for name in names]
    new_groups = ts_env.rate(rating_groups, ranks=ranks)
    for slot, name in enumerate(names):
        ratings[name] = new_groups[slot][0]

    best_rank = min(ranks)
    winners = [i for i, rank in enumerate(ranks) if rank == best_rank]
    for slot, name in enumerate(names):
        row = table[name]
        row["games"] += 1
        row["wins"] += 1 if slot in winners and len(winners) == 1 else 0
        row["draws"] += 1 if slot in winners and len(winners) > 1 else 0
        row["losses"] += 1 if slot not in winners else 0
        row["deaths"] += 1 if match_result["death_steps"][slot] is not None else 0
        row["total_rank"] += int(ranks[slot])
        row["total_steps"] += int(match_result["survival_steps"][slot])
        row["action_errors"] += int(match_result["action_errors"][slot])
        row["invalid_actions"] += int(match_result["invalid_actions"][slot])
        for key in ("kills", "boxes", "items", "bombs"):
            row[key] += int(match_result["stats"][slot][key])
        if slot in winners and len(winners) == 1 and match_result["win_reason"] in WIN_REASON_KEYS:
            row[f"win_by_{match_result['win_reason']}"] += 1
        row["recency"] = recency_by_name.get(name, row["recency"])


def leaderboard_rows(table, ratings):
    rows = []
    for name, row in table.items():
        rating = ratings[name]
        games = max(1, row["games"])
        wins = max(1, row["wins"])
        rows.append(
            {
                "name": name,
                "score": score(rating),
                "mu": float(rating.mu),
                "sigma": float(rating.sigma),
                "games": row["games"],
                "wins": row["wins"],
                "draws": row["draws"],
                "losses": row["losses"],
                "deaths": row["deaths"],
                "win_rate": row["wins"] / games,
                "avg_rank": row["total_rank"] / games,
                "avg_steps": row["total_steps"] / games,
                "kills": row["kills"],
                "boxes": row["boxes"],
                "items": row["items"],
                "bombs": row["bombs"],
                "win_by_survival": row["win_by_survival"],
                "win_by_kills": row["win_by_kills"],
                "win_by_boxes": row["win_by_boxes"],
                "win_by_items": row["win_by_items"],
                "win_by_bombs": row["win_by_bombs"],
                "win_survival_pct": row["win_by_survival"] / wins,
                "win_kills_pct": row["win_by_kills"] / wins,
                "win_boxes_pct": row["win_by_boxes"] / wins,
                "win_items_pct": row["win_by_items"] / wins,
                "win_bombs_pct": row["win_by_bombs"] / wins,
                "action_errors": row["action_errors"],
                "invalid_actions": row["invalid_actions"],
                "recency": row["recency"],
            }
        )
    rows.sort(key=lambda r: (-r["score"], -r["mu"], r["sigma"], -r["recency"]))
    return rows


def print_leaderboard(rows):
    header = (
        "rank name                         score      mu   sigma games  W  D  L "
        "death avgR  win% kills boxes items bombs surv% kill% box% item% bomb%"
    )
    print(header)
    print("-" * len(header))
    for i, row in enumerate(rows, start=1):
        print(
            f"{i:>4} {row['name'][:28]:<28} "
            f"{row['score']:>7.2f} {row['mu']:>7.2f} {row['sigma']:>6.2f} "
            f"{row['games']:>5} {row['wins']:>2} {row['draws']:>2} {row['losses']:>2} "
            f"{row['deaths']:>5} {row['avg_rank']:>4.2f} {row['win_rate'] * 100:>5.1f} "
            f"{row['kills']:>5} {row['boxes']:>5} {row['items']:>5} {row['bombs']:>5} "
            f"{row['win_survival_pct'] * 100:>5.1f} {row['win_kills_pct'] * 100:>5.1f} "
            f"{row['win_boxes_pct'] * 100:>4.1f} {row['win_items_pct'] * 100:>5.1f} "
            f"{row['win_bombs_pct'] * 100:>5.1f}"
        )


def write_outputs(rows, matches, csv_path, json_path):
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"leaderboard": rows, "matches": matches}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS)
    parser.add_argument("--matches", type=int, default=300)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--csv", default="logs/arena/random_trueskill_leaderboard.csv")
    parser.add_argument("--json", default="logs/arena/random_trueskill_matches.json")
    args = parser.parse_args()

    agent_specs = [normalize_agent_spec(spec) for spec in args.agents]
    if len(agent_specs) < 4:
        raise ValueError("Need at least 4 agents")
    validate_agent_specs(agent_specs)

    rng = random.Random(args.seed)
    ts_env = trueskill.TrueSkill(mu=100.0, sigma=100.0 / 3.0, draw_probability=0.1)
    ratings = {}
    table = {}
    # Local tie-break: if everything is equal, later items in --agents are treated as newer.
    recency_by_name = {}
    for idx, spec in enumerate(agent_specs):
        name = label_for_spec(spec)
        ratings[name] = ts_env.Rating()
        recency_by_name[name] = idx
        table[name] = {
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "deaths": 0,
            "total_rank": 0,
            "total_steps": 0,
            "kills": 0,
            "boxes": 0,
            "items": 0,
            "bombs": 0,
            "win_by_survival": 0,
            "win_by_kills": 0,
            "win_by_boxes": 0,
            "win_by_items": 0,
            "win_by_bombs": 0,
            "action_errors": 0,
            "invalid_actions": 0,
            "recency": idx,
        }

    matches = []
    for match_idx in range(args.matches):
        chosen_specs = rng.sample(agent_specs, 4)
        rng.shuffle(chosen_specs)
        match_seed = rng.randrange(1, 2**31 - 1)
        result = run_one_match(chosen_specs, match_seed, args.max_steps)
        matches.append(result)
        update_table(table, ratings, ts_env, result, recency_by_name)

        if args.progress_every and (match_idx + 1) % args.progress_every == 0:
            print(f"\nAfter {match_idx + 1}/{args.matches} matches")
            print_leaderboard(leaderboard_rows(table, ratings))

    rows = leaderboard_rows(table, ratings)
    print("\nFinal leaderboard")
    print_leaderboard(rows)
    write_outputs(rows, matches, args.csv, args.json)
    print(f"\nSaved CSV:  {args.csv}")
    print(f"Saved JSON: {args.json}")


if __name__ == "__main__":
    main()
