import argparse
import csv
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.game import BomberEnv
from scripts.participant.elite_arena import _predict_affected_tiles, _rank_match


STAT_KEYS = ("kills", "boxes", "items", "bombs")
REASON_KEYS = ("surv", "kill", "box", "item", "bomb")
UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def norm_header(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def parse_submission_ids(value):
    text = str(value or "")
    ids = UUID_RE.findall(text)
    if ids:
        return [item.lower() for item in ids]
    return [part.strip().lower() for part in text.split(",") if part.strip()]


def load_tracked_ids(args):
    tracked = {}
    for item in args.ids or []:
        tracked[item.strip().lower()] = ""

    if args.ids_file:
        with open(args.ids_file, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            headers = {norm_header(name): name for name in (reader.fieldnames or [])}
            id_col = headers.get("submissionid") or headers.get("submissionids")
            team_col = headers.get("team") or headers.get("name")
            if not id_col:
                raise ValueError("ids-file must contain a 'Submission ID' column")
            for row in reader:
                sid = str(row.get(id_col, "")).strip().lower()
                if sid:
                    tracked[sid] = str(row.get(team_col, "")).strip() if team_col else ""

    if not tracked:
        raise ValueError("Provide at least one id with --ids or --ids-file")
    return tracked


def sheet_csv_url(source):
    if not source:
        return source
    if "docs.google.com/spreadsheets" not in source:
        return source
    match = re.search(r"/spreadsheets/d/([^/]+)", source)
    if not match:
        return source
    sheet_id = match.group(1)
    parsed = urllib.parse.urlparse(source)
    query = urllib.parse.parse_qs(parsed.query)
    gid = (query.get("gid") or ["0"])[0]
    if parsed.fragment:
        frag_qs = urllib.parse.parse_qs(parsed.fragment)
        gid = (frag_qs.get("gid") or [gid])[0]
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def read_text_source(source):
    path = Path(source)
    if path.exists():
        return path.read_text(encoding="utf-8-sig")
    url = sheet_csv_url(source)
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8-sig")


def drive_file_id(url):
    text = str(url or "").strip()
    patterns = (
        r"/file/d/([^/]+)",
        r"[?&]id=([^&]+)",
        r"/d/([^/]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return urllib.parse.unquote(match.group(1))
    return None


def download_drive_json(url, cache_dir=None):
    path = Path(str(url))
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    file_id = drive_file_id(url)
    if not file_id:
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    if cache_dir:
        cache_path = Path(cache_dir) / f"{file_id}.json"
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as handle:
                return json.load(handle)

    download_url = f"https://drive.google.com/uc?export=download&id={urllib.parse.quote(file_id)}"
    data = urllib.request.urlopen(download_url, timeout=60).read()
    text = data.decode("utf-8")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
    return json.loads(text)


def replay_match(payload):
    history = payload.get("history") or []
    if not history:
        raise ValueError("match JSON has no history")

    seed = int(payload.get("seed", 0))
    max_steps = int(history[-1].get("step", len(history) - 1))
    env = BomberEnv(seed=seed, max_steps=max_steps)
    death_order = []
    death_steps = [[] for _ in range(4)]
    kill_steps = [[] for _ in range(4)]
    obs = env._get_obs()
    prev_alive = [bool(row[2]) for row in obs["players"]]

    for entry in history[1:]:
        actions = entry.get("actions")
        if actions is None:
            continue
        actions = [int(action) for action in actions]
        step = int(entry.get("step", env.current_step + 1))
        affected_tiles = _predict_affected_tiles(env, actions)
        obs, _, _ = env.step(actions)
        alive_now = [bool(row[2]) for row in obs["players"]]
        died = [slot for slot in range(len(alive_now)) if prev_alive[slot] and not alive_now[slot]]
        if died:
            death_order.append(died)
            for victim in died:
                death_steps[victim].append(step)
                pos = (int(obs["players"][victim][0]), int(obs["players"][victim][1]))
                for owner in affected_tiles.get(pos, set()):
                    owner = int(owner)
                    if 0 <= owner < len(kill_steps) and owner != victim:
                        kill_steps[owner].append(step)
        prev_alive = alive_now

    computed_ranks = _rank_match(env, list(death_order))
    ranks = [int(x) for x in payload.get("ranks", computed_ranks)]
    stats = [dict(env.players[slot].stats) for slot in range(len(env.players))]
    alive_final = [bool(player.alive) for player in env.players]
    return {
        "seed": seed,
        "team_ids": [str(item).lower() for item in (payload.get("team_ids") or [])],
        "ranks": ranks,
        "computed_ranks": computed_ranks,
        "stats": stats,
        "alive_final": alive_final,
        "death_order": death_order,
        "death_steps": death_steps,
        "kill_steps": kill_steps,
        "survival_steps": payload.get("survival_steps") or [],
    }


def win_reason(match):
    ranks = match["ranks"]
    best_rank = min(ranks)
    winners = [slot for slot, rank in enumerate(ranks) if rank == best_rank]
    if len(winners) != 1:
        return None

    winner = winners[0]
    alive = match["alive_final"]
    survivors = [slot for slot, is_alive in enumerate(alive) if is_alive]
    if not alive[winner] or len(survivors) <= 1:
        return "surv"

    stats = match["stats"]
    for stat_key, reason_key in zip(STAT_KEYS, ("kill", "box", "item", "bomb")):
        winner_value = int(stats[winner].get(stat_key, 0))
        other_values = [int(stats[slot].get(stat_key, 0)) for slot in survivors if slot != winner]
        if other_values and winner_value > max(other_values):
            return reason_key
    return None


def new_summary(team_name=""):
    row = {
        "team": team_name,
        "matches": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "deaths": 0,
        "rank_sum": 0,
        "survival_step_sum": 0,
        "death_steps": [],
        "kill_steps": [],
    }
    for key in STAT_KEYS:
        row[key] = 0
    for key in REASON_KEYS:
        row[f"win_by_{key}"] = 0
    return row


def update_summary(summary, slot, match, reason):
    row = summary
    ranks = match["ranks"]
    best_rank = min(ranks)
    winners = [idx for idx, rank in enumerate(ranks) if rank == best_rank]
    rank = int(ranks[slot])

    row["matches"] += 1
    row["rank_sum"] += rank
    if match["survival_steps"]:
        row["survival_step_sum"] += int(match["survival_steps"][slot])
    row["wins"] += 1 if slot in winners and len(winners) == 1 else 0
    row["draws"] += 1 if slot in winners and len(winners) > 1 else 0
    row["losses"] += 1 if slot not in winners else 0
    row["deaths"] += 1 if not match["alive_final"][slot] else 0
    row["death_steps"].extend(match["death_steps"][slot])
    row["kill_steps"].extend(match["kill_steps"][slot])
    for key in STAT_KEYS:
        row[key] += int(match["stats"][slot].get(key, 0))
    if slot in winners and len(winners) == 1 and reason:
        row[f"win_by_{reason}"] += 1


def step_stats(steps):
    if not steps:
        return {
            "avg": 0.0,
            "p50": 0.0,
            "mode": "",
            "top3": "",
        }

    ordered = sorted(int(step) for step in steps)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        p50 = float(ordered[mid])
    else:
        p50 = (ordered[mid - 1] + ordered[mid]) / 2.0
    counts = Counter(ordered)
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    return {
        "avg": sum(ordered) / n,
        "p50": p50,
        "mode": top[0][0],
        "top3": ";".join(f"{step}:{count}" for step, count in top),
    }


def final_rows(summaries):
    rows = []
    for submission_id, row in summaries.items():
        matches = max(1, int(row["matches"]))
        wins = max(1, int(row["wins"]))
        death_step_stats = step_stats(row["death_steps"])
        kill_step_stats = step_stats(row["kill_steps"])
        out = {
            "team": row["team"],
            "submission_id": submission_id,
            "matches": row["matches"],
            "wins": row["wins"],
            "draws": row["draws"],
            "losses": row["losses"],
            "deaths": row["deaths"],
            "avg_rank": row["rank_sum"] / matches,
            "avg_survival": row["survival_step_sum"] / matches,
            "death_step_avg": death_step_stats["avg"],
            "death_step_p50": death_step_stats["p50"],
            "death_step_mode": death_step_stats["mode"],
            "death_step_top3": death_step_stats["top3"],
            "kill_step_avg": kill_step_stats["avg"],
            "kill_step_p50": kill_step_stats["p50"],
            "kill_step_mode": kill_step_stats["mode"],
            "kill_step_top3": kill_step_stats["top3"],
        }
        for key in STAT_KEYS:
            out[key] = row[key]
        for key in REASON_KEYS:
            out[f"{key}%"] = 100.0 * row[f"win_by_{key}"] / wins
        rows.append(out)
    rows.sort(key=lambda item: (-item["wins"], item["avg_rank"], -item["matches"], item["submission_id"]))
    return rows


def print_rows(rows):
    header = (
        "team,submission_id,matches,wins,draws,losses,deaths,avg_rank,avg_survival,"
        "death_step_avg,death_step_p50,death_step_mode,death_step_top3,"
        "kill_step_avg,kill_step_p50,kill_step_mode,kill_step_top3,"
        "kills,boxes,items,bombs,surv%,kill%,box%,item%,bomb%"
    )
    print(header)
    for row in rows:
        print(
            f"{row['team']},{row['submission_id']},{row['matches']},{row['wins']},"
            f"{row['draws']},{row['losses']},{row['deaths']},"
            f"{row['avg_rank']:.3f},{row['avg_survival']:.1f},"
            f"{row['death_step_avg']:.1f},{row['death_step_p50']:.1f},"
            f"{row['death_step_mode']},{row['death_step_top3']},"
            f"{row['kill_step_avg']:.1f},{row['kill_step_p50']:.1f},"
            f"{row['kill_step_mode']},{row['kill_step_top3']},"
            f"{row['kills']},{row['boxes']},{row['items']},{row['bombs']},"
            f"{row['surv%']:.1f},{row['kill%']:.1f},{row['box%']:.1f},"
            f"{row['item%']:.1f},{row['bomb%']:.1f}"
        )


def write_csv(rows, output):
    if not output:
        return
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def iter_sheet_rows(sheet_source):
    text = read_text_source(sheet_source)
    reader = csv.DictReader(text.splitlines())
    headers = {norm_header(name): name for name in (reader.fieldnames or [])}
    ids_col = headers.get("submissionids")
    json_col = headers.get("jsondriveurl") or headers.get("jsonurl")
    if not ids_col or not json_col:
        raise ValueError("sheet CSV must contain 'Submission IDs' and 'JSON Drive URL' columns")
    for row in reader:
        yield row, ids_col, json_col


def analyze_sources(args, tracked):
    summaries = defaultdict(lambda: None)
    for submission_id, team in tracked.items():
        summaries[submission_id] = new_summary(team)

    seen_match_ids = set()
    analyzed = 0
    skipped = 0
    errors = []

    if args.json_files:
        sources = [(str(path), None, None) for path in args.json_files]
    else:
        sources = []
        for row, ids_col, json_col in iter_sheet_rows(args.sheet):
            row_ids = parse_submission_ids(row.get(ids_col))
            overlap = set(row_ids) & set(tracked)
            if len(overlap) < args.min_overlap:
                skipped += 1
                continue
            sources.append((row.get(json_col), row, row_ids))

    for source, sheet_row, row_ids in sources:
        try:
            payload = download_drive_json(source, cache_dir=args.cache_dir)
            match_id = str(payload.get("seed", "")) + ":" + ",".join(payload.get("team_ids", []))
            if match_id in seen_match_ids:
                continue
            seen_match_ids.add(match_id)
            match = replay_match(payload)
            reason = win_reason(match)
            for slot, submission_id in enumerate(match["team_ids"]):
                if submission_id in tracked:
                    update_summary(summaries[submission_id], slot, match, reason)
            analyzed += 1
        except Exception as exc:
            label = source if sheet_row is None else sheet_row.get("Match ID", source)
            errors.append(f"{label}: {exc}")

    rows = final_rows(summaries)
    return rows, analyzed, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Bomberland match JSONs for selected Submission IDs."
    )
    parser.add_argument("--sheet", help="Local CSV path or Google Sheets URL/export CSV URL.")
    parser.add_argument("--json-files", nargs="*", help="Analyze local JSON files directly.")
    parser.add_argument("--ids", nargs="*", help="Submission IDs to track.")
    parser.add_argument("--ids-file", help="CSV with Team and Submission ID columns.")
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=1,
        help="For sheet mode, only analyze rows whose Submission IDs contain at least this many tracked IDs.",
    )
    parser.add_argument("--cache-dir", default="logs/json_cache", help="Cache downloaded Drive JSON files.")
    parser.add_argument("--output", default="logs/submission_json_report.csv")
    args = parser.parse_args()

    if not args.sheet and not args.json_files:
        raise ValueError("Provide --sheet or --json-files")

    tracked = load_tracked_ids(args)
    rows, analyzed, skipped, errors = analyze_sources(args, tracked)
    print_rows(rows)
    write_csv(rows, args.output)
    print(f"\nAnalyzed matches: {analyzed}")
    if skipped:
        print(f"Skipped sheet rows by min-overlap: {skipped}")
    if args.output:
        print(f"CSV written: {args.output}")
    if errors:
        print("\nErrors:")
        for item in errors[:20]:
            print(f"- {item}")
        if len(errors) > 20:
            print(f"- ... {len(errors) - 20} more")


if __name__ == "__main__":
    main()
