# Furuta Pendulum, Reinforcement Learning to Hardware

A SAC agent learns to balance a rod on a spinning arm (a rotary inverted pendulum). Training happens in a custom physics simulator. The trained policy is exported as a plain C header, so it can run on a microcontroller.

## Setup

```bash
conda create -n furuta-rl python=3.11 -y
conda activate furuta-rl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124   # GPU build
pip install -r requirements.txt
```

## Train and watch

| Command | What it does |
|---|---|
| `python scripts/train.py --timesteps 300000` | New run. The GPU is used if available. |
| `python scripts/train.py --timesteps 300000 --device cpu` | Force CPU |
| `python scripts/train.py --resume` | Continue after Ctrl+C or a crash |
| `python scripts/train.py --domain-randomization` | Add random disturbance bumps during training |
| `tensorboard --logdir runs` | Watch training live |
| `python scripts/watch.py --model checkpoints/furuta/furuta_final.zip` | Render a trained agent |

A checkpoint is saved every 20,000 steps (`--checkpoint-interval`) and the last 3 are kept (`--keep-checkpoints`). Ctrl+C saves the state and prints the resume command.

The disturbance options are `--disturb-max-strength` (default 3.0), `--disturb-interval-min` (default 1.0) and `--disturb-interval-max` (default 3.0). See `envs/disturbance_wrapper.py` for details.

## Export to hardware

```bash
python scripts/export_to_c.py --model checkpoints/furuta/furuta_final.zip --out furuta_policy.h
```

This writes one `.h` file in plain C with no dependencies. The policy has 2 hidden layers of 64 units. Use it like this:

```c
float obs[4] = {theta, theta_dot, alpha, alpha_dot};  // radians and rad/s
float action = furuta_policy_forward(obs);             // in [-1, 1]
motor_command = action * MAX_TORQUE;
```

The control loop should read the encoders, compute `[theta, theta_dot, alpha, alpha_dot]`, call `furuta_policy_forward()`, scale the result to a motor command and repeat at about 100 Hz, which matches the training step of `dt = 0.01 s`.

## Limitations

- **Sim to real gap.** The simulator uses the standard Furuta pendulum equations, but the mass, length and friction values in `PARAMS` (`envs/furuta_pendulum_env.py`) are placeholders, not measured from real hardware. Replace them with values from your own build before using the policy on a real rig.
- **Unmodeled effects.** Encoder noise, control loop delay and motor deadband are not simulated.

## Project structure

```
envs/
  furuta_pendulum_env.py   simulator with the equations of motion
  disturbance_wrapper.py   adds random disturbances during training
scripts/
  train.py                 SAC training
  watch.py                 render and evaluate a trained agent
  export_to_c.py           export the policy to a C header
environment.yml
requirements.txt
```
