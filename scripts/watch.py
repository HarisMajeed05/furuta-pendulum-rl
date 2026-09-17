"""
scripts/watch.py
==================
Loads a trained model and renders it balancing the pendulum, reports how
long it stayed upright per episode.

USAGE
-----
    python scripts/watch.py --model checkpoints/furuta/furuta_final.zip
    python scripts/watch.py --model path/to/model.zip --episodes 20 --no-render

Real-world disturbance simulation, --disturb:
    python scripts/watch.py --model path/to/model.zip --disturb

At random intervals, a random "bump" is applied directly to the
pendulum's velocity, simulating someone physically knocking it, then you
watch whether the trained policy recovers balance on its own. This is a
genuinely meaningful robustness check: a policy that only ever balanced a
perfectly undisturbed pendulum in training may or may not generalize to
recovering from a real push, this is how you'd actually find out, in
simulation, before ever touching hardware.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import SAC
from envs.furuta_pendulum_env import FurutaPendulumEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--disturb", action="store_true",
                     help="Randomly bump the pendulum during the episode, to test recovery")
    ap.add_argument("--disturb-interval-min", type=float, default=1.0,
                     help="Minimum seconds between disturbances")
    ap.add_argument("--disturb-interval-max", type=float, default=3.0,
                     help="Maximum seconds between disturbances")
    ap.add_argument("--disturb-strength", type=float, default=2.5,
                     help="Max magnitude of the random velocity kick, rad/s. "
                          "Bigger = harder to recover from. Tune this to find where your agent's recovery limit actually is.")
    args = ap.parse_args()

    env = FurutaPendulumEnv(render_mode=None if args.no_render else "human")
    model = SAC.load(args.model)

    survival_times = []
    disturbance_count_total = 0

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        disturbances_this_ep = 0
        next_disturbance_at = (
            random.uniform(args.disturb_interval_min, args.disturb_interval_max) / env.dt
            if args.disturb else None
        )

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            if args.disturb and steps >= next_disturbance_at:
                kick = random.uniform(-args.disturb_strength, args.disturb_strength)
                env.perturb(alpha_dot_kick=kick)
                disturbances_this_ep += 1
                print(f"  [disturbance] kicked alpha_dot by {kick:+.2f} rad/s at step {steps}")
                next_disturbance_at = steps + random.uniform(args.disturb_interval_min, args.disturb_interval_max) / env.dt

            if not args.no_render:
                env.render()
            steps += 1
            if steps > 2000:
                break

        seconds = steps * env.dt
        survival_times.append(seconds)
        disturbance_count_total += disturbances_this_ep
        suffix = f", survived {disturbances_this_ep} disturbance(s)" if args.disturb else ""
        print(f"Episode {ep + 1}/{args.episodes}: balanced for {seconds:.2f}s ({steps} steps){suffix}")

    env.close()
    print(f"\nAverage balance time: {sum(survival_times) / len(survival_times):.2f}s")
    if args.disturb:
        print(f"Total disturbances applied across all episodes: {disturbance_count_total}")
        print("A policy that keeps recovering after real bumps here is a genuinely stronger result")
        print("than one that only balances an undisturbed pendulum, closer to what real hardware needs.")


if __name__ == "__main__":
    main()