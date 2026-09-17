"""
envs/furuta_pendulum_env.py
=============================
Rotary inverted pendulum (Furuta pendulum): a motor spins an arm in the
horizontal plane; a free rod pivots on the arm's tip and must be balanced
upright. Task here is BALANCE, not swing-up, matching "balance the rod once
placed on top" -- episodes start near-upright, not hanging down.

State (observation), 4 values:
    theta       arm angle (rad)
    theta_dot   arm angular velocity (rad/s)
    alpha       pendulum angle from upright, 0 = perfectly upright (rad)
    alpha_dot   pendulum angular velocity (rad/s)

Action: 1 continuous value, normalized motor torque/voltage in [-1, 1].

PHYSICS, worth being precise about: this uses the standard nonlinear
Furuta pendulum equations of motion (same structural form as Quanser's
published QUBE-Servo model, widely used in the RL-for-control literature).
The mass/length/inertia constants below are reasonable placeholder values
for a small desktop rig, NOT measured from any specific hardware. Before
trusting sim-to-real transfer, replace PARAMS below with values measured
from your actual build (arm length, pendulum mass/length, motor torque
constant, joint friction), via basic system identification. Training on
unvalidated physics and expecting it to balance real hardware unmodified
is the single most common way this kind of project fails.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

PARAMS = {
    "Mp": 0.024,   # pendulum mass, kg
    "Lp": 0.129,   # pendulum length, m
    "Lr": 0.085,   # arm length, m
    "Jr": 4.0e-6,  # arm moment of inertia about pivot (motor + arm), kg*m^2
    "Dr": 0.0015,  # arm viscous damping
    "Dp": 0.0005,  # pendulum viscous damping
    "g": 9.81,
    "max_torque": 0.05,   # N*m, at action = 1.0
}


class FurutaPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, dt=0.01, fail_angle_deg=25):
        super().__init__()
        self.dt = dt
        self.fail_angle = np.deg2rad(fail_angle_deg)
        self.render_mode = render_mode
        self._screen = None
        self._clock = None

        p = PARAMS
        self.Mp, self.Lp, self.Lr = p["Mp"], p["Lp"], p["Lr"]
        self.Jr, self.Dr, self.Dp = p["Jr"], p["Dr"], p["Dp"]
        self.g = p["g"]
        self.max_torque = p["max_torque"]

        self.lp = self.Lp / 2.0
        self.Jp = (1.0 / 3.0) * self.Mp * self.Lp ** 2  # rod pivoted at one end
        self.J0 = self.Jr + self.Mp * self.Lr ** 2

        high = np.array([np.pi, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self.state = np.zeros(4, dtype=np.float64)

    def _dynamics(self, state, torque):
        theta, theta_dot, alpha, alpha_dot = state
        Mp, lp, Lr, Jp, J0, Dr, Dp, g = (
            self.Mp, self.lp, self.Lr, self.Jp, self.J0, self.Dr, self.Dp, self.g
        )

        sin_a, cos_a = np.sin(alpha), np.cos(alpha)

        M11 = J0 + Mp * lp ** 2 * sin_a ** 2
        M12 = Mp * Lr * lp * cos_a
        M21 = M12
        M22 = Jp
        M = np.array([[M11, M12], [M21, M22]])

        c1 = (Mp * lp ** 2 * sin_a * cos_a * theta_dot * alpha_dot * 2.0
              + Mp * Lr * lp * sin_a * alpha_dot ** 2)
        c2 = -Mp * lp ** 2 * sin_a * cos_a * theta_dot ** 2 - Mp * g * lp * sin_a

        rhs = np.array([
            torque - Dr * theta_dot - c1,
            -Dp * alpha_dot - c2,
        ])

        theta_ddot, alpha_ddot = np.linalg.solve(M, rhs)
        return np.array([theta_dot, theta_ddot, alpha_dot, alpha_ddot])

    def perturb(self, alpha_dot_kick: float = None, theta_dot_kick: float = None):
        """Simulates an external physical disturbance, someone bumping the
        arm or pendulum, by directly kicking the velocity terms. This is
        NOT part of the action space, the agent doesn't cause this and
        can't predict it, exactly like a real-world bump. Called from
        outside the normal step() loop (see watch.py --disturb) rather
        than from the RL training loop itself, since training with random
        disturbances baked in would be a deliberate robustness-training
        choice (domain randomization), a different, bigger decision than
        just testing an already-trained agent's recovery behavior."""
        if alpha_dot_kick is not None:
            self.state[3] += alpha_dot_kick
        if theta_dot_kick is not None:
            self.state[1] += theta_dot_kick

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([
            self.np_random.uniform(-0.05, 0.05),
            0.0,
            self.np_random.uniform(-0.08, 0.08),
            0.0,
        ], dtype=np.float64)
        return self._get_obs(), {}

    def step(self, action):
        torque = float(np.clip(action[0], -1.0, 1.0)) * self.max_torque

        substeps = 5
        h = self.dt / substeps
        for _ in range(substeps):
            deriv = self._dynamics(self.state, torque)
            self.state = self.state + h * deriv
            self.state[0] = np.arctan2(np.sin(self.state[0]), np.cos(self.state[0]))
            self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))

        theta, theta_dot, alpha, alpha_dot = self.state

        terminated = abs(alpha) > self.fail_angle
        reward = 1.0 - 2.0 * (alpha ** 2) - 0.1 * (theta ** 2) - 0.001 * (theta_dot ** 2) - 0.01 * float(action[0]) ** 2
        if terminated:
            reward -= 10.0

        return self._get_obs(), float(reward), bool(terminated), False, {}

    def _get_obs(self):
        return self.state.astype(np.float32)

    def render(self):
        if self.render_mode != "human":
            return
        import pygame

        w, h = 500, 500
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((w, h))
            self._clock = pygame.time.Clock()
            pygame.display.set_caption("Furuta Pendulum")

        theta, _, alpha, _ = self.state
        cx, cy = w // 2, h // 2
        arm_len_px = 120
        pend_len_px = 160

        ax = cx + arm_len_px * np.cos(theta)
        ay = cy + arm_len_px * np.sin(theta)
        tip_x = ax + pend_len_px * np.sin(alpha)
        tip_y = ay - pend_len_px * np.cos(alpha)

        self._screen.fill((15, 15, 20))
        pygame.draw.circle(self._screen, (200, 200, 210), (cx, cy), 6)
        pygame.draw.line(self._screen, (62, 232, 220), (cx, cy), (ax, ay), 4)
        color = (255, 107, 107) if abs(alpha) > self.fail_angle * 0.7 else (139, 123, 247)
        pygame.draw.line(self._screen, color, (ax, ay), (tip_x, tip_y), 4)
        pygame.draw.circle(self._screen, color, (int(tip_x), int(tip_y)), 6)

        pygame.event.pump()
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None


def register():
    gym.register(id="FurutaPendulum-v0", entry_point="envs.furuta_pendulum_env:FurutaPendulumEnv")