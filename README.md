# Furuta Pendulum Balance, RL to Hardware Export

Rotary inverted pendulum. SAC agent learns to balance a rod on a spinning
arm. Trained in a custom physics sim, exported to a dependency-free C
header for MCU deployment.

## Setup

```bash
conda create -n furuta-rl python=3.11 -y
conda activate furuta-rl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124   # GPU
pip install -r requirements.txt
```

## Train / watch

| Command | Does |
|---|---|
| `python scripts/train.py --timesteps 300000` | fresh run, GPU auto-detected |
| `python scripts/train.py --timesteps 300000 --device cpu` | force CPU |
| `python scripts/train.py --resume` | continue after Ctrl+C or crash |
| `tensorboard --logdir runs` | watch training live |
| `python scripts/watch.py --model checkpoints/furuta/furuta_final.zip` | render trained agent |

Checkpointing: every 20k steps, last 3 kept, same pattern as the earlier
YOLO/self-driving projects. Ctrl+C saves state and prints the resume
command.

## Export to hardware

```bash
python scripts/export_to_c.py --model checkpoints/furuta/furuta_final.zip --out furuta_policy.h
```

Produces one `.h` file, plain C, no runtime dependency. Policy is ~2
hidden layers x 64 units, a few thousand floats. Include it and call:

```c
float obs[4] = {theta, theta_dot, alpha, alpha_dot};  // radians, rad/s
float action = furuta_policy_forward(obs);             // in [-1, 1]
motor_command = action * MAX_TORQUE;
```

## MCU recommendation

| Board | Why |
|---|---|
| **ESP32 (recommended)** | dual-core, fast enough, native PWM/ADC/encoder-interrupt support, cheap, huge community |
| STM32 (F4/G4) | true hard real-time if FreeRTOS jitter on ESP32 becomes a problem later |
| Arduino Uno/Mega | avoid, CPU is marginal for encoder ISR + inference at useful control rates |

Control loop, roughly: read encoders -> compute `[theta, theta_dot, alpha,
alpha_dot]` -> `furuta_policy_forward()` -> scale to motor command -> repeat
at ~100Hz (matches training `dt=0.01s`).

## What this does NOT solve

**Sim-to-real gap.** The physics model uses standard Furuta pendulum
equations (same structural form as Quanser's published QUBE-Servo model)
with placeholder mass/length/friction constants, not measured from any
real hardware. A policy trained on unvalidated physics will very likely
need retuning, or fail outright, on a real rig. Before trusting this on
hardware: replace `PARAMS` in `envs/furuta_pendulum_env.py` with values
measured from your actual build, and expect to add domain randomization
(randomizing friction/mass slightly during training) if you want real
robustness margin, not included here.

**Encoder noise, control loop delay, motor deadband** are all unmodeled.
Real hardware has all three.

## Project structure

```
furuta-rl/
├── envs/furuta_pendulum_env.py   physics sim, Furuta pendulum equations of motion
├── scripts/
│   ├── train.py                    SAC training
│   ├── watch.py                    render + evaluate a trained agent
│   └── export_to_c.py              extract policy weights to a C header
├── environment.yml / requirements.txt
```
