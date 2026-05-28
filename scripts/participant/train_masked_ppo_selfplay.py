import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from agent import (  # noqa: E402
    BoxFarmerAgent,
    GeniusRuleAgent,
    RandomAgent,
    SimpleRuleAgent,
    SmarterRuleAgent,
    TacticalRuleAgent,
)
from agent.ppo_agent.ppo_model import (  # noqa: E402
    MaskedActorCritic,
    encode_obs,
    legal_action_mask,
    masked_logits,
)
from competition.evaluation.runtime_guard import load_agent_instance  # noqa: E402
from engine.game import BomberEnv  # noqa: E402


BASELINES = {
    "RandomAgent": RandomAgent,
    "SimpleRuleAgent": SimpleRuleAgent,
    "SmarterRuleAgent": SmarterRuleAgent,
    "GeniusRuleAgent": GeniusRuleAgent,
    "BoxFarmerAgent": BoxFarmerAgent,
    "TacticalRuleAgent": TacticalRuleAgent,
}


def make_agent(spec, agent_id):
    if spec in BASELINES:
        return BASELINES[spec](agent_id)
    path = Path(spec)
    if path.is_dir():
        path = path / "agent.py"
    return load_agent_instance(str(path), agent_id)


def pick_opponents(pool, controlled_slot):
    agents = [None] * 4
    specs = random.choices(pool, k=3)
    j = 0
    for slot in range(4):
        if slot == controlled_slot:
            continue
        agents[slot] = make_agent(specs[j], slot)
        j += 1
    return agents


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
        last_key = None
        group = []
        for i in survivor_order:
            key = (
                stats[i]["kills"],
                stats[i]["boxes"],
                stats[i]["items"],
                stats[i]["bombs"],
            )
            if last_key is None or key == last_key:
                group.append(i)
            else:
                groups.append(group)
                group = [i]
            last_key = key
        groups.append(group)
    for step in sorted({s for s in death_steps if s is not None}, reverse=True):
        groups.append([i for i, s in enumerate(death_steps) if s == step])

    ranks = [0] * 4
    rank = 0
    for group in groups:
        for i in group:
            ranks[i] = rank
        rank += len(group)
    return ranks


def terminal_reward(env, slot, alive, death_steps):
    ranks = rank_match(env, alive, death_steps)
    best_rank = min(ranks)
    winners = [i for i, rank in enumerate(ranks) if rank == best_rank]
    if ranks[slot] == best_rank and len(winners) == 1:
        return 7.0
    if ranks[slot] == best_rank:
        return 2.5
    return -2.0 - 0.5 * ranks[slot]


def shaped_reward(prev_stats, next_stats, died):
    reward = 0.01
    reward += (next_stats["kills"] - prev_stats["kills"]) * 2.5
    reward += (next_stats["boxes"] - prev_stats["boxes"]) * 0.08
    reward += (next_stats["items"] - prev_stats["items"]) * 0.22
    reward += (next_stats["bombs"] - prev_stats["bombs"]) * 0.015
    if died:
        reward -= 4.0
    return reward


def tensor_batch(items, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(items), dtype=dtype, device=device)


def collect_bc_dataset(expert_spec, opponent_pool, episodes, max_steps, seed):
    maps, auxes, masks, actions = [], [], [], []
    for ep in range(episodes):
        env = BomberEnv(max_steps=max_steps, seed=seed + ep)
        obs = env.reset(seed=seed + ep)
        controlled_slot = random.randrange(4)
        agents = pick_opponents(opponent_pool, controlled_slot)
        agents[controlled_slot] = make_agent(expert_spec, controlled_slot)
        done = False
        step = 0
        while not done and step < max_steps:
            action = int(agents[controlled_slot].act(obs))
            mask = legal_action_mask(obs, controlled_slot)
            if mask[action]:
                map_feat, aux_feat = encode_obs(obs, controlled_slot, step)
                maps.append(map_feat)
                auxes.append(aux_feat)
                masks.append(mask)
                actions.append(action)

            joint_actions = []
            for slot, agent in enumerate(agents):
                try:
                    joint_actions.append(int(agent.act(obs)))
                except Exception:
                    joint_actions.append(0)
            obs, terminated, truncated = env.step(joint_actions)
            done = terminated or truncated
            step += 1
    return maps, auxes, masks, actions


def behavior_clone(model, optimizer, dataset, device, epochs=3, batch_size=256):
    maps, auxes, masks, actions = dataset
    if not actions:
        return
    n = len(actions)
    indices = np.arange(n)
    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            map_t = tensor_batch([maps[i] for i in idx], device)
            aux_t = tensor_batch([auxes[i] for i in idx], device)
            mask_t = torch.as_tensor(np.asarray([masks[i] for i in idx]), dtype=torch.bool, device=device)
            action_t = torch.as_tensor(np.asarray([actions[i] for i in idx]), dtype=torch.long, device=device)
            logits, _ = model(map_t, aux_t)
            loss = F.cross_entropy(masked_logits(logits, mask_t), action_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
        print(f"BC epoch {epoch + 1}: loss={total_loss / n:.4f}, samples={n}")


def sample_action(model, obs, slot, turn, device):
    map_feat, aux_feat = encode_obs(obs, slot, turn)
    mask = legal_action_mask(obs, slot)
    with torch.no_grad():
        logits, value = model(
            torch.as_tensor(map_feat, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(aux_feat, dtype=torch.float32, device=device).unsqueeze(0),
        )
        dist = Categorical(logits=masked_logits(logits, torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)))
        action = dist.sample()
    return map_feat, aux_feat, mask, int(action.item()), float(dist.log_prob(action).item()), float(value.item())


def train_ppo(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = MaskedActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    if args.resume and Path(args.resume).exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        print(f"Resumed model from {args.resume}")

    opponent_pool = args.opponents
    if args.bc_episodes > 0:
        print(f"Collecting BC data from {args.expert} for {args.bc_episodes} episodes...")
        dataset = collect_bc_dataset(args.expert, opponent_pool, args.bc_episodes, args.max_steps, args.seed + 10000)
        behavior_clone(model, optimizer, dataset, device, args.bc_epochs, args.batch_size)
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "steps": 0, "stage": "bc"}, args.save_path)

    env = None
    obs = None
    agents = None
    controlled_slot = None
    prev_alive = None
    death_steps = None
    episode_step = 0
    episode = 0
    wins = 0
    returns = []
    current_return = 0.0

    def reset_episode(global_step):
        nonlocal env, obs, agents, controlled_slot, prev_alive, death_steps, episode_step, episode, current_return
        episode += 1
        controlled_slot = random.randrange(4)
        env = BomberEnv(max_steps=args.max_steps, seed=args.seed + global_step + episode * 17)
        obs = env.reset(seed=args.seed + global_step + episode * 17)
        agents = pick_opponents(opponent_pool, controlled_slot)
        prev_alive = [bool(p[2]) for p in obs["players"]]
        death_steps = [None] * 4
        episode_step = 0
        current_return = 0.0

    reset_episode(0)
    global_step = 0
    while global_step < args.total_steps:
        rollout = {k: [] for k in ("maps", "auxes", "masks", "actions", "logps", "values", "rewards", "dones")}
        for _ in range(args.rollout_steps):
            map_feat, aux_feat, mask, action, logp, value = sample_action(model, obs, controlled_slot, episode_step, device)
            prev_stats = dict(env.players[controlled_slot].stats)
            joint_actions = []
            for slot in range(4):
                if slot == controlled_slot:
                    joint_actions.append(action)
                else:
                    try:
                        joint_actions.append(int(agents[slot].act(obs)))
                    except Exception:
                        joint_actions.append(0)

            obs, terminated, truncated = env.step(joint_actions)
            alive = [bool(p[2]) for p in obs["players"]]
            for i in range(4):
                if prev_alive[i] and not alive[i]:
                    death_steps[i] = episode_step + 1
            died = prev_alive[controlled_slot] and not alive[controlled_slot]
            prev_alive = alive
            done = terminated or truncated

            next_stats = env.players[controlled_slot].stats
            reward = shaped_reward(prev_stats, next_stats, died)
            if done:
                reward += terminal_reward(env, controlled_slot, alive, death_steps)

            rollout["maps"].append(map_feat)
            rollout["auxes"].append(aux_feat)
            rollout["masks"].append(mask)
            rollout["actions"].append(action)
            rollout["logps"].append(logp)
            rollout["values"].append(value)
            rollout["rewards"].append(reward)
            rollout["dones"].append(done)

            current_return += reward
            global_step += 1
            episode_step += 1

            if done:
                ranks = rank_match(env, alive, death_steps)
                if ranks[controlled_slot] == min(ranks) and ranks.count(min(ranks)) == 1:
                    wins += 1
                returns.append(current_return)
                if len(returns) % 10 == 0:
                    avg_ret = sum(returns[-10:]) / 10.0
                    print(f"step={global_step} episodes={len(returns)} wins={wins} avg_return_10={avg_ret:.3f}")
                reset_episode(global_step)
            if global_step >= args.total_steps:
                break

        with torch.no_grad():
            if rollout["dones"] and rollout["dones"][-1]:
                next_value = 0.0
            else:
                map_feat, aux_feat = encode_obs(obs, controlled_slot, episode_step)
                _, value = model(
                    torch.as_tensor(map_feat, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.as_tensor(aux_feat, dtype=torch.float32, device=device).unsqueeze(0),
                )
                next_value = float(value.item())

        rewards = rollout["rewards"]
        dones = rollout["dones"]
        values = rollout["values"]
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 0.0 if dones[t] else 1.0
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + args.gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns_t = advantages + np.asarray(values, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(rewards)
        indices = np.arange(n)
        for _ in range(args.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, args.batch_size):
                idx = indices[start : start + args.batch_size]
                map_t = tensor_batch([rollout["maps"][i] for i in idx], device)
                aux_t = tensor_batch([rollout["auxes"][i] for i in idx], device)
                mask_t = torch.as_tensor(np.asarray([rollout["masks"][i] for i in idx]), dtype=torch.bool, device=device)
                action_t = torch.as_tensor(np.asarray([rollout["actions"][i] for i in idx]), dtype=torch.long, device=device)
                old_logp_t = tensor_batch([rollout["logps"][i] for i in idx], device)
                adv_t = tensor_batch(advantages[idx], device)
                ret_t = tensor_batch(returns_t[idx], device)

                logits, value_t = model(map_t, aux_t)
                dist = Categorical(logits=masked_logits(logits, mask_t))
                logp_t = dist.log_prob(action_t)
                ratio = torch.exp(logp_t - old_logp_t)
                pg_loss = -torch.min(
                    adv_t * ratio,
                    adv_t * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef),
                ).mean()
                value_loss = F.mse_loss(value_t, ret_t)
                entropy = dist.entropy().mean()
                loss = pg_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "steps": global_step}, args.save_path)

    print(f"Saved PPO model to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=20000)
    parser.add_argument("--rollout_steps", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--bc_episodes", type=int, default=20)
    parser.add_argument("--bc_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_path", default="agent/ppo_agent/ppo_bomber.pt")
    parser.add_argument("--resume", default="")
    parser.add_argument("--expert", default="agent_adaptive_pro_v1.py")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=[
            "agent_adaptive_pro_v1.py",
            "agentpromax_v1.py",
            "agent_phase_v1.py",
            "agent_temporal_hybrid_v1.py",
            "TacticalRuleAgent",
            "GeniusRuleAgent",
            "SmarterRuleAgent",
        ],
    )
    train_ppo(parser.parse_args())
