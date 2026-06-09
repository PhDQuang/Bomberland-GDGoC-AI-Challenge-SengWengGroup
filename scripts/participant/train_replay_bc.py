import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.ppo_agent.ppo_model import (  # noqa: E402
    BOMB,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    MaskedActorCritic,
    encode_obs,
    legal_action_mask,
    masked_logits,
)
from scripts.participant.submission_json_report import (  # noqa: E402
    download_drive_json,
    iter_sheet_rows,
    load_tracked_ids,
    parse_submission_ids,
    replay_match,
    win_reason,
)


ACTION_FLIP_X = {
    LEFT: RIGHT,
    RIGHT: LEFT,
    UP: UP,
    DOWN: DOWN,
    BOMB: BOMB,
    0: 0,
}
ACTION_FLIP_Y = {
    LEFT: LEFT,
    RIGHT: RIGHT,
    UP: DOWN,
    DOWN: UP,
    BOMB: BOMB,
    0: 0,
}


def action_transform(action, flip_x=False, flip_y=False):
    action = int(action)
    if flip_x:
        action = ACTION_FLIP_X.get(action, action)
    if flip_y:
        action = ACTION_FLIP_Y.get(action, action)
    return action


def obs_from_entry(entry):
    return {
        "map": np.asarray(entry["map"], dtype=np.int8),
        "players": np.asarray(entry["players"], dtype=np.int8),
        "bombs": np.asarray(entry["bombs"], dtype=np.int8),
    }


def transform_obs(obs, flip_x=False, flip_y=False):
    if not flip_x and not flip_y:
        return obs

    grid = obs["map"]
    players = np.array(obs["players"], copy=True)
    bombs = np.array(obs["bombs"], copy=True)
    h, w = grid.shape

    if flip_x:
        grid = np.flip(grid, axis=0).copy()
        players[:, 0] = h - 1 - players[:, 0]
        if len(bombs):
            bombs[:, 0] = h - 1 - bombs[:, 0]
    else:
        grid = np.array(grid, copy=True)

    if flip_y:
        grid = np.flip(grid, axis=1).copy()
        players[:, 1] = w - 1 - players[:, 1]
        if len(bombs):
            bombs[:, 1] = w - 1 - bombs[:, 1]

    return {"map": grid, "players": players, "bombs": bombs}


def load_ids_file(path):
    if not path:
        return {}
    class Args:
        ids = None
        ids_file = path
    return load_tracked_ids(Args())


def match_sources(args, target_ids):
    if args.json_files:
        for json_path in args.json_files:
            yield str(json_path), None
        return

    for row, ids_col, json_col in iter_sheet_rows(args.sheet):
        row_ids = parse_submission_ids(row.get(ids_col))
        if target_ids:
            if len(set(row_ids) & target_ids) < args.min_overlap:
                continue
        yield row.get(json_col), row


def slot_weight(match, slot, reason, args):
    ranks = match["ranks"]
    best_rank = min(ranks)
    rank = int(ranks[slot])
    weight = max(args.min_rank_weight, args.rank_decay ** rank)
    if rank == best_rank and ranks.count(best_rank) == 1:
        weight *= args.winner_weight
    if reason == "kill":
        weight *= args.kill_win_weight
    elif reason == "box":
        weight *= args.box_win_weight
    return float(weight)


def should_take_slot(match, slot, target_ids, args):
    submission_id = match["team_ids"][slot]
    if target_ids and submission_id not in target_ids:
        return False
    rank = int(match["ranks"][slot])
    if args.winner_only:
        return rank == min(match["ranks"])
    return rank <= args.max_rank


def collect_samples(args):
    rng = random.Random(args.seed)
    target_ids = set(load_ids_file(args.ids_file)) if args.ids_file else set()
    maps, auxes, masks, actions, weights = [], [], [], [], []
    seen_match_ids = set()
    stats = Counter()
    transforms = [(False, False)]
    if args.augment:
        transforms.extend([(True, False), (False, True), (True, True)])

    sources = list(match_sources(args, target_ids))
    rng.shuffle(sources)
    if args.max_json:
        sources = sources[: args.max_json]

    for source, sheet_row in sources:
        if len(actions) >= args.max_samples:
            break
        try:
            payload = download_drive_json(source, cache_dir=args.cache_dir)
            match_id = str(payload.get("seed", "")) + ":" + ",".join(payload.get("team_ids", []))
            if match_id in seen_match_ids:
                continue
            seen_match_ids.add(match_id)

            match = replay_match(payload)
            reason = win_reason(match)
            history = payload.get("history") or []
            if len(history) < 2:
                continue

            slot_base_weights = [
                slot_weight(match, slot, reason, args) if should_take_slot(match, slot, target_ids, args) else 0.0
                for slot in range(min(4, len(match["team_ids"])))
            ]
            if not any(slot_base_weights):
                continue

            kill_step_sets = [set(steps) for steps in match.get("kill_steps", [[] for _ in range(4)])]
            for idx in range(1, len(history)):
                action_row = history[idx].get("actions")
                if action_row is None:
                    continue
                if args.sample_stride > 1 and (idx % args.sample_stride) != 0:
                    continue
                obs = obs_from_entry(history[idx - 1])
                turn = int(history[idx - 1].get("step", idx - 1))
                action_step = int(history[idx].get("step", idx))

                for slot, base_weight in enumerate(slot_base_weights):
                    if base_weight <= 0:
                        continue
                    if slot >= len(action_row):
                        continue
                    action = int(action_row[slot])
                    sample_weight = base_weight
                    if any(action_step <= kill_step <= action_step + args.kill_window for kill_step in kill_step_sets[slot]):
                        sample_weight *= args.prekill_weight
                    if action == BOMB:
                        sample_weight *= args.bomb_action_weight

                    for flip_x, flip_y in transforms:
                        aug_obs = transform_obs(obs, flip_x=flip_x, flip_y=flip_y)
                        aug_action = action_transform(action, flip_x=flip_x, flip_y=flip_y)
                        mask = legal_action_mask(aug_obs, slot)
                        if aug_action < 0 or aug_action >= len(mask) or not mask[aug_action]:
                            stats["illegal_labels"] += 1
                            continue
                        map_feat, aux_feat = encode_obs(aug_obs, slot, turn)
                        maps.append(map_feat.astype(np.float16))
                        auxes.append(aux_feat.astype(np.float16))
                        masks.append(mask)
                        actions.append(aug_action)
                        weights.append(sample_weight)
                        stats[f"action_{aug_action}"] += 1
                        if len(actions) >= args.max_samples:
                            break
                    if len(actions) >= args.max_samples:
                        break
                if len(actions) >= args.max_samples:
                    break

            stats["matches_used"] += 1
        except Exception as exc:
            stats["errors"] += 1
            label = source if sheet_row is None else sheet_row.get("Match ID", source)
            print(f"[warn] skipped {label}: {exc}")

    return maps, auxes, masks, actions, weights, stats


def train_bc(args, dataset):
    maps, auxes, masks, actions, weights, stats = dataset
    if not actions:
        raise ValueError("No BC samples collected")

    device = torch.device(args.device)
    model = MaskedActorCritic().to(device)
    if args.resume and Path(args.resume).exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        print(f"Resumed model from {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(actions)
    indices = np.arange(n)
    weights_np = np.asarray(weights, dtype=np.float32)
    weights_np = weights_np / max(float(weights_np.mean()), 1.0e-6)

    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        total_acc = 0
        for start in range(0, n, args.batch_size):
            idx = indices[start : start + args.batch_size]
            map_t = torch.as_tensor(np.asarray([maps[i] for i in idx]), dtype=torch.float32, device=device)
            aux_t = torch.as_tensor(np.asarray([auxes[i] for i in idx]), dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(np.asarray([masks[i] for i in idx]), dtype=torch.bool, device=device)
            action_t = torch.as_tensor(np.asarray([actions[i] for i in idx]), dtype=torch.long, device=device)
            weight_t = torch.as_tensor(weights_np[idx], dtype=torch.float32, device=device)

            logits, _ = model(map_t, aux_t)
            logits = masked_logits(logits, mask_t)
            loss_each = F.cross_entropy(logits, action_t, reduction="none")
            loss = (loss_each * weight_t).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item()) * len(idx)
            total_acc += int((torch.argmax(logits, dim=1) == action_t).sum().item())

        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"loss={total_loss / n:.4f} acc={total_acc / n:.3f} samples={n}"
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "stage": "replay_bc",
            "samples": n,
            "stats": dict(stats),
            "args": vars(args),
        },
        args.output,
    )
    print(f"Saved replay BC checkpoint: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Behavior clone MaskedActorCritic from Bomberland replay JSONs.")
    parser.add_argument("--sheet", help="Google Sheet URL/export CSV URL with JSON Drive URL column.")
    parser.add_argument("--json-files", nargs="*", help="Local replay JSON files.")
    parser.add_argument("--ids-file", default="", help="CSV with Submission ID column. If set, train only these IDs.")
    parser.add_argument("--min-overlap", type=int, default=1)
    parser.add_argument("--cache-dir", default="logs/json_cache")
    parser.add_argument("--output", default="agent/ppo_agent/replay_bc.pt")
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--max-json", type=int, default=0, help="Limit number of replay JSONs. 0 means no limit.")
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--max-rank", type=int, default=1, help="Train slots with rank <= this value.")
    parser.add_argument("--winner-only", action="store_true")
    parser.add_argument("--augment", action="store_true", help="Use board flips to multiply data by 4.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--winner-weight", type=float, default=2.0)
    parser.add_argument("--rank-decay", type=float, default=0.65)
    parser.add_argument("--min-rank-weight", type=float, default=0.25)
    parser.add_argument("--kill-win-weight", type=float, default=1.25)
    parser.add_argument("--box-win-weight", type=float, default=1.1)
    parser.add_argument("--prekill-weight", type=float, default=1.8)
    parser.add_argument("--kill-window", type=int, default=7)
    parser.add_argument("--bomb-action-weight", type=float, default=1.05)
    args = parser.parse_args()

    if not args.sheet and not args.json_files:
        raise ValueError("Provide --sheet or --json-files")

    dataset = collect_samples(args)
    _, _, _, actions, _, stats = dataset
    print(f"Collected samples: {len(actions)}")
    print("Collection stats:", json.dumps(dict(stats), sort_keys=True))
    train_bc(args, dataset)


if __name__ == "__main__":
    main()
