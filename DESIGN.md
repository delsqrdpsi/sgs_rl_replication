# SGS rod-in-hole: design

Goal: reproduce the SGS teacher result on UR5e rod-in-hole in MuJoCo Warp, comparing the SGS start-state
sampler against uniform sampling at the same budget.

Sources, in priority order:

1. **SGS** — *A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control*,
   [arXiv 2610.12465v1](https://arxiv.org/html/2610.12465v1). Code not released (project page checked 2026-10-26).
2. **OmniReset / UWLab**, which SGS builds on, for values SGS leaves unstated:
   [UWLab @ 2b40afb](https://github.com/UW-Lab/UWLab/tree/2b40afb9693d46bc1b1a91c7ba95e03c04daa60d/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset)
   and the asset metadata in [UW-Lab/uwlab-assets](https://huggingface.co/datasets/UW-Lab/uwlab-assets).
3. **Ours** — choices neither source specifies. Each is marked *assumption* below.

## Fixed by the user

| Item | Value |
|---|---|
| Task | UR5e + Robotiq 2F-85, rod-in-hole |
| Simulator | MuJoCo Warp, 5 ms physics step |
| Scale | ~10,000 parallel environments on one RTX 5090 (RunPod) |
| Policy input | Point cloud representation |
| Critic input | Same as the actor |
| Exploration | Plain Gaussian, state-independent std |
| Bank | SGS's: 32,768 configurations, three families |
| Rod / hole | 16 mm rod, ~0.5 mm clearance |
| Out of scope | Student distillation, sim-to-real |

## Task and timing

| Item | Value | Source |
|---|---|---|
| Physics step | 5 ms (200 Hz) | SGS |
| Policy rate | 10 Hz (20 physics steps per action) | SGS |
| Episode length | 16 s = 160 policy steps | OmniReset `episode_length_s` |
| Rod | Cylinder, 16 mm diameter, 50 mm long | SGS (diameter); length *assumption* |
| Hole | 16.5 mm bore (0.5 mm diametral clearance), 25 mm deep, in a block fixed to the table | Clearance per user; depth *assumption*; board fixed, as in SGS's real setup |
| Gripper pad friction | μ = 2, max combine (MuJoCo's default combine rule) | SGS |
| Robot gravity | Compensated (robot bodies gravity-free; objects keep gravity) | OmniReset assets (`disable_gravity=True`) |

## Action and control

Action: 7-D = 3 position increments, 3 axis-angle increments, 1 binary gripper command (close if > 0). SGS.

Relative OSC on the `wrist_3_link` pose, following OmniReset's `RelCartesianOSCAction`:

- Each policy step: target = current pose + action × scale, scale = (0.02, 0.02, 0.02 m, 0.02, 0.02, 0.2 rad).
- Each physics step: τ = Jᵀ(Kp·e − Kd·J q̇), Kp = (200, 200, 200, 3, 3, 3), Kd = 2·√Kp·ζ with
  ζ = (3, 3, 3, 1, 1, 1); torques clamped to (150, 150, 150, 28, 28, 28) N·m.

## Reward, success and failure (SGS)

r = 100·S − F − 1e-5·C(‖a‖²) − 1e-4·C(‖a − a_prev‖²) − 1e-4·C(‖q̇_arm‖²), C(x) = min(x, 1e4).

- **Success S:** the rod's seated reference point is within **2.5 mm** of the hole's seated point and
  |roll| + |pitch| of the rod relative to the hole is below **0.025 rad** (yaw ignored).
  Thresholds: OmniReset `PegHole/metadata.yaml`.
- **Failure F (−1):** any arm joint speed above 2× its limit, or the rod outside the workspace bounds. SGS.
- **Episode end:** success, failure, or timeout. Ending on success is an *assumption*; the outcome recorded for
  the sampler is whether the episode succeeded.

## Observations (actor and critic identical)

| Term | Size | Source |
|---|---|---|
| Arm joint positions (6) + gripper opening (1) | 7 | SGS |
| End-effector pose (position + quaternion) | 7 | SGS §A.3.2 (Appendix C.3 omits it; we include it) |
| Previous action | 7 | SGS |
| Point cloud in the gripper frame: gripper, rod, board | N × 3 | SGS |

Point cloud: fixed points sampled once on each body's surface, moved with the body's pose each step, expressed
in the gripper frame. N = 32 per body (96 points) is an *assumption*. 1 mm Gaussian noise per step on the
points (SGS, rod-in-hole).

## Reset bank (SGS)

32,768 collision-checked configurations, split evenly across:

1. **Near-Goal:** rod grasped at its grasp point and placed along the insertion path, depth fraction in
   [0, 1.1] (0 = fully seated); gripper placed by IK; random gripper aperture.
2. **Stable Grasp:** rod grasped and held above the hole, higher than Near-Goal; random aperture.
3. **Reaching:** rod anywhere in the workspace; gripper aligned to the rod's grasp frame by IK; random aperture.
   The grasp is not guaranteed.

Every candidate is collision-checked and settled before it enters the bank. Ranges SGS does not give
(workspace box, heights, aperture range) are *assumptions*, listed in the generator config.

Evaluation: a separate set of Reaching configurations generated with a different seed, never seen by the
sampler (SGS reports success from Reaching starts).

## SGS sampler

- Keep the last H = 100 Boolean outcomes per configuration; p_i = their mean.
- w_i = (p_i + ε)^(κt) · (1 − p_i + ε)^(κ(1−t)), ℓ_i = log(w_i + ε), sample from softmax(ℓ / max(T, 1)).
- Manipulation values: t = 0.5, κ = 1, ε = 1e-4, T = 2.
- Untried configurations count as p = 0 (*assumption*; all start equal, so sampling starts uniform).

Baseline: `--sampler uniform` draws every reset uniformly from the same bank.

## PPO (OmniReset's runner config; SGS gives none)

| Setting | Value |
|---|---|
| Actor / critic | Separate MLPs [512, 256, 128, 64], ELU, empirical observation normalization |
| Rollout | 32 steps per environment |
| Epochs / mini-batches | 5 / 4 (mini-batch count fixed as envs grow, per SGS) |
| Learning rate | 1e-4, adaptive schedule, desired KL 0.01 |
| γ / λ | 0.99 / 0.95 |
| Clip / entropy / value coef | 0.2 / 0.006 / 1.0, clipped value loss |
| Max grad norm | 1.0 |
| Advantage normalization | Over the whole rollout |
| Initial action std | 1.0 |

## Not reproduced

Domain randomization beyond point-cloud noise, symmetry-aware resets (the rod is symmetric about its axis,
and success ignores yaw), the gravity curriculum (SGS found it unnecessary), PLR, students, sim-to-real.
