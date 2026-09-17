"""
envs/disturbance_wrapper.py
==============================
Wraps FurutaPendulumEnv so random "bumps" (see env.perturb()) happen
DURING training, not just during evaluation in watch.py. This is domain
randomization: the agent only gets good at recovering from disturbances
if it actually experiences them while learning, testing an undisturbed-
trained policy against bumps in watch.py tells you how robust it happens
to be, it doesn't make it more robust.

REAL PHYSICAL CEILING, not a training limitation: a disturbance strong
enough to push the pendulum past the failure angle within a single control
step (10ms here) ends the episode before the agent can react at all. No
amount of training helps against that, it's not a skill gap, it's a
physics/timing fact. Roughly (using default fail_angle=25 deg, dt=0.01s):
a kick above ~30-40 rad/s starting from near-upright will frequently cross
that line in one step on its own. Practically-recoverable disturbances,
ones where the agent has a real chance to fight back, top out well below
that, and exactly where depends on your motor's max_torque too. Find the
real ceiling for your setup empirically:
    python scripts/watch.py --model <trained>.zip --disturb --disturb-strength 5
    (increase --disturb-strength until recovery rate visibly collapses)

USAGE
-----
This is used automatically by train.py when you pass --domain-randomization,
you don't need to import this directly.
"""

import random

import gymnasium as gym


class DisturbanceWrapper(gym.Wrapper):
    def __init__(self, env, interval_min_sec=1.0, interval_max_sec=3.0, max_strength=3.0):
        super().__init__(env)
        self.interval_min_sec = interval_min_sec
        self.interval_max_sec = interval_max_sec
        self.max_strength = max_strength
        self._steps_since_reset = 0
        self._next_disturbance_step = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._steps_since_reset = 0
        self._schedule_next()
        return obs, info

    def _schedule_next(self):
        interval_sec = random.uniform(self.interval_min_sec, self.interval_max_sec)
        self._next_disturbance_step = self._steps_since_reset + int(interval_sec / self.env.unwrapped.dt)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps_since_reset += 1

        if self._steps_since_reset >= self._next_disturbance_step and not (terminated or truncated):
            kick = random.uniform(-self.max_strength, self.max_strength)
            self.env.unwrapped.perturb(alpha_dot_kick=kick)
            info["disturbance_applied"] = kick
            self._schedule_next()
            # Refresh the observation, the kick changed the state the
            # agent should actually see and react to next step.
            obs = self.env.unwrapped._get_obs()

        return obs, reward, terminated, truncated, info