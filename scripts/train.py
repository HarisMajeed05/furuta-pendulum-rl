"""
scripts/train.py
==================
Trains a SAC agent to balance the Furuta pendulum. SAC over PPO here
specifically because this is a continuous-control balance task where
sample efficiency matters more than in the highway-driving project, SAC's
off-policy replay buffer learns balance policies in far fewer environment
steps than PPO typically needs for this kind of task.

USAGE
-----
    python scripts/train.py --timesteps 300000
    python scripts/train.py --timesteps 300000 --device cpu
    python scripts/train.py --resume
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from envs.furuta_pendulum_env import FurutaPendulumEnv
from envs.disturbance_wrapper import DisturbanceWrapper


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("--device cuda requested but no GPU found, falling back to CPU.")
            return "cpu"
        return "cuda"
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("No GPU detected, training on CPU.")
    return "cpu"


def prune_old_checkpoints(checkpoint_dir: str, keep: int):
    files = glob.glob(os.path.join(checkpoint_dir, "*_steps.zip"))

    def step_number(path):
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    files.sort(key=step_number)
    while len(files) > keep:
        try:
            os.remove(files.pop(0))
        except OSError:
            pass


class PruningCheckpointCallback(CheckpointCallback):
    def __init__(self, *args, keep_checkpoints=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_checkpoints = keep_checkpoints

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            prune_old_checkpoints(self.save_path, self.keep_checkpoints)
        return result


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    files = glob.glob(os.path.join(checkpoint_dir, "*_steps.zip"))
    if not files:
        raise SystemExit(f"--resume given but no checkpoint found in {checkpoint_dir}")

    def step_number(path):
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(files, key=step_number)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--checkpoint-interval", type=int, default=20_000)
    ap.add_argument("--keep-checkpoints", type=int, default=3)
    ap.add_argument("--run-name", default="furuta")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--domain-randomization", action="store_true",
                     help="Train with random disturbance 'bumps' injected during episodes, "
                          "so the agent actually learns to recover from them, not just gets "
                          "tested against them afterward. Real physical ceiling applies, see "
                          "envs/disturbance_wrapper.py docstring before pushing --disturb-max-strength high.")
    ap.add_argument("--disturb-max-strength", type=float, default=3.0,
                     help="Max random velocity kick magnitude during training, rad/s")
    ap.add_argument("--disturb-interval-min", type=float, default=1.0)
    ap.add_argument("--disturb-interval-max", type=float, default=3.0)
    args = ap.parse_args()

    device = resolve_device(args.device)
    checkpoint_dir = os.path.abspath(os.path.join("checkpoints", args.run_name))
    tensorboard_dir = os.path.abspath("runs")
    os.makedirs(checkpoint_dir, exist_ok=True)

    def make_env_fn():
        env = FurutaPendulumEnv()
        if args.domain_randomization:
            env = DisturbanceWrapper(
                env,
                interval_min_sec=args.disturb_interval_min,
                interval_max_sec=args.disturb_interval_max,
                max_strength=args.disturb_max_strength,
            )
        return env

    env = make_vec_env(make_env_fn, n_envs=1)

    if args.resume:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        print(f"Resuming from: {checkpoint_path}")
        model = SAC.load(checkpoint_path, env=env, device=device, tensorboard_log=tensorboard_dir)
        reset_num_timesteps = False
    else:
        print("Starting fresh training")
        model = SAC("MlpPolicy", env, verbose=1, device=device, tensorboard_log=tensorboard_dir,
                     policy_kwargs=dict(net_arch=[64, 64]))
        reset_num_timesteps = True

    callback = PruningCheckpointCallback(
        save_freq=args.checkpoint_interval,
        save_path=checkpoint_dir,
        name_prefix=args.run_name,
        keep_checkpoints=args.keep_checkpoints,
    )

    print(f"Device: {device} | Checkpoint every {args.checkpoint_interval} steps, "
          f"keeping the last {args.keep_checkpoints}")
    if args.domain_randomization:
        print(f"Domain randomization ON: random bumps up to {args.disturb_max_strength} rad/s, "
              f"every {args.disturb_interval_min}-{args.disturb_interval_max}s")
    else:
        print("Domain randomization OFF, training on an undisturbed pendulum only")
    print(f"Watch live: tensorboard --logdir {tensorboard_dir}")

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback,
                    reset_num_timesteps=reset_num_timesteps, tb_log_name=args.run_name)
    except KeyboardInterrupt:
        interrupt_path = os.path.join(checkpoint_dir, f"{args.run_name}_interrupted.zip")
        model.save(interrupt_path)
        print(f"\nStopped early. Saved to: {interrupt_path}")
        print(f"Resume with: python scripts/train.py --run-name {args.run_name} --resume")
        sys.exit(0)

    final_path = os.path.join(checkpoint_dir, f"{args.run_name}_final.zip")
    model.save(final_path)
    print(f"\nDone. Final model: {final_path}")
    print(f"Watch it: python scripts/watch.py --model {final_path}")
    print(f"Export to C for hardware: python scripts/export_to_c.py --model {final_path}")


if __name__ == "__main__":
    main()