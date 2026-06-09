import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


DEFAULT_OPPONENTS = [
    "agent_adaptive_pro_v2.py",
    "agentpromax_v1.py",
    "agent_phase_v1.py",
    "agent_economy_v1.py",
    "agent_aggressive_endgame_v1.py",
    "TacticalRuleAgent",
    "GeniusRuleAgent",
    "SmarterRuleAgent",
]


def run_stage(name, cmd, log_dir, env=None):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    print(f"log: {log_path}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        code = proc.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)


def main():
    parser = argparse.ArgumentParser(description="Run replay BC -> PPO fine-tune -> arena evaluate.")
    parser.add_argument("--sheet", help="Google Sheet URL/export CSV URL with JSON Drive URL column.")
    parser.add_argument("--json-files", nargs="*", help="Local replay JSON files for BC.")
    parser.add_argument("--ids-file", default="tracked_submission_ids.csv")
    parser.add_argument("--cache-dir", default="logs/json_cache")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260603)

    parser.add_argument("--bc-output", default="agent/ppo_agent/replay_bc.pt")
    parser.add_argument("--bc-max-json", type=int, default=0)
    parser.add_argument("--bc-max-samples", type=int, default=120000)
    parser.add_argument("--bc-sample-stride", type=int, default=2)
    parser.add_argument("--bc-max-rank", type=int, default=1)
    parser.add_argument("--bc-epochs", type=int, default=5)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-augment", action="store_true")
    parser.add_argument("--winner-only", action="store_true")
    parser.add_argument("--skip-bc", action="store_true")

    parser.add_argument("--ppo-output", default="agent/ppo_agent/ppo_replay_finetuned.pt")
    parser.add_argument("--ppo-total-steps", type=int, default=100000)
    parser.add_argument("--ppo-rollout-steps", type=int, default=512)
    parser.add_argument("--ppo-batch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-lr", type=float, default=2.0e-4)
    parser.add_argument("--skip-ppo", action="store_true")

    parser.add_argument("--eval-matches", type=int, default=120)
    parser.add_argument("--eval-csv", default="logs/arena/replay_pipeline_eval.csv")
    parser.add_argument("--eval-json", default="logs/arena/replay_pipeline_eval.json")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--opponents", nargs="+", default=DEFAULT_OPPONENTS)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else ROOT_DIR / "logs" / "pipeline" / timestamp

    if not args.skip_bc:
        if not args.sheet and not args.json_files:
            raise ValueError("BC stage needs --sheet or --json-files")
        bc_cmd = [
            sys.executable,
            "-m",
            "scripts.participant.train_replay_bc",
            "--ids-file",
            args.ids_file,
            "--cache-dir",
            args.cache_dir,
            "--output",
            args.bc_output,
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--max-samples",
            str(args.bc_max_samples),
            "--sample-stride",
            str(args.bc_sample_stride),
            "--max-rank",
            str(args.bc_max_rank),
            "--epochs",
            str(args.bc_epochs),
            "--batch-size",
            str(args.bc_batch_size),
        ]
        if args.sheet:
            bc_cmd.extend(["--sheet", args.sheet])
        if args.json_files:
            bc_cmd.append("--json-files")
            bc_cmd.extend(args.json_files)
        if args.bc_max_json:
            bc_cmd.extend(["--max-json", str(args.bc_max_json)])
        if args.bc_augment:
            bc_cmd.append("--augment")
        if args.winner_only:
            bc_cmd.append("--winner-only")
        run_stage("01_replay_bc", bc_cmd, log_dir)

    if not args.skip_ppo:
        ppo_cmd = [
            sys.executable,
            "-m",
            "scripts.participant.train_masked_ppo_selfplay",
            "--resume",
            args.bc_output,
            "--save_path",
            args.ppo_output,
            "--total_steps",
            str(args.ppo_total_steps),
            "--rollout_steps",
            str(args.ppo_rollout_steps),
            "--batch_size",
            str(args.ppo_batch_size),
            "--ppo_epochs",
            str(args.ppo_epochs),
            "--lr",
            str(args.ppo_lr),
            "--bc_episodes",
            "0",
            "--seed",
            str(args.seed + 1000),
            "--device",
            args.device,
            "--opponents",
        ]
        ppo_cmd.extend(args.opponents)
        run_stage("02_ppo_selfplay", ppo_cmd, log_dir)

    if not args.skip_eval:
        eval_cmd = [
            sys.executable,
            "-m",
            "scripts.participant.random_trueskill_arena",
            "--agents",
            "agent_replay_bc_v1.py",
        ]
        eval_cmd.extend(args.opponents)
        eval_cmd.extend(
            [
                "--matches",
                str(args.eval_matches),
                "--seed",
                str(args.seed + 2000),
                "--progress_every",
                "25",
                "--csv",
                args.eval_csv,
                "--json",
                args.eval_json,
            ]
        )
        run_stage(
            "03_arena_eval",
            eval_cmd,
            log_dir,
            env={"BOMBERLAND_PPO_CHECKPOINT": str(ROOT_DIR / args.ppo_output)},
        )

    print(f"\nPipeline complete. Logs: {log_dir}")


if __name__ == "__main__":
    main()
