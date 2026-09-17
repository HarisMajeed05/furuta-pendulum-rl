"""
scripts/watch.py
==================
Loads a trained model and renders it balancing the pendulum, reports how
long it stayed upright per episode.

USAGE
-----
    python scripts/watch.py --model checkpoints/furuta/furuta_final.zip
    python scripts/watch.py --model path/to/model.zip --episodes 20 --no-render
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import SAC
from envs.furuta_pendulum_env import FurutaPendulumEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    env = FurutaPendulumEnv(render_mode=None if args.no_render else "human")
    model = SAC.load(args.model)

    survival_times = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if not args.no_render:
                env.render()
            steps += 1
            if steps > 2000:  # 20 seconds at dt=0.01, treat as a full success
                break
        seconds = steps * env.dt
        survival_times.append(seconds)
        print(f"Episode {ep + 1}/{args.episodes}: balanced for {seconds:.2f}s ({steps} steps)")

    env.close()
    print(f"\nAverage balance time: {sum(survival_times) / len(survival_times):.2f}s")


if __name__ == "__main__":
    main()
