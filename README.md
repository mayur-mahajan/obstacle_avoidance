# Teaching a robot car to follow instructions with a VLA

A hands-on course in **vision-language-action (VLA) model training**, from an empty simulator to a car that drives
to *"the red ball"* using only its camera. Everything runs on one machine, in six Jupyter notebooks that explain every
step for readers without a maths background.

* **Robot:** a simulated mecanum-wheel car (it can drive, turn and slide sideways) in the [Genesis](https://genesis-world.readthedocs.io) simulator.
* **Task:** random rooms with obstacles and coloured objects; an instruction names one object; the car must drive to
  it and stop, seeing only its front camera.
* **Model:** [smolVLA](https://huggingface.co/lerobot/smolvla_base) (450 M parameters, Hugging Face LeRobot), trained by
  **imitation** on 296 scripted drives, then fine-tuned by **reinforcement learning** (PPO).

## The notebooks

Run them in order; each one saves what the next one needs into `runs/`.

| # | Notebook | What you do and learn | Time* |
|---|---|---|---|
| 00 | `00_setup_and_first_drive` | hardware check (NVIDIA / Apple Silicon / CPU), Genesis basics, how a mecanum car is modelled, drive it with the keyboard | 15 min |
| 01 | `01_environment_design` | random rooms, goal objects and prompts, path planning, success and collisions, a scripted expert driver | 15 min |
| 02 | `02_recording_demonstrations` | the LeRobot dataset format, the image↔action pairing check, record a few drives yourself, record **300 scripted drives** (the training data) | 30 min |
| 03 | `03_preparing_training_data` | split by layout, action chunks, scaling, mirroring (and when not to), prompt paraphrases, the data loader | 10 min |
| 04 | `04_imitation_training` | how a VLA works (reader + flow-matching action head), feeding the reader correctly, an auxiliary quiz, training **two action heads** (smolVLA's and a car-specific one), driving tests with charts and luck ranges | 2 h |
| 05 | `05_reinforcement_learning_ppo` | reward design, parallel CPU simulators, making a flow-matching head trainable by PPO, reading RL graphs, comparing RL with imitation | 4 h |
| 06 | `06_overhead_camera_guidance` | a ceiling camera read by an off-car server (SmolVLM2): extra input tokens for the car, testing whether help is used, why direction help fails and route help works, a learned obstacle map, marker tracking, a route planner | ~1 day* |
| 07 | `07_route_guided_rl_ppo` | notebook 05's practice (PPO) for the route-guided car, with the real server in the practice loop (planning once per drive, tracking every half second), training-health charts (KL, train vs validation demo loss) | 4–8 h |

\*On an RTX 4080 SUPER (16 GB) with a 20-core CPU. Notebook 06 from scratch takes about a day (mostly the server's
practice pictures and training); every long step is a script that is skipped when its results exist. On an Apple Silicon Mac, notebooks 00–03 are fine; training in
04–05 runs but is several times slower.

## Results (imitation, notebook 04)

Fixed tests, the same drives for every model (`scripts/rl_tests.py`). "Pair" = two objects in view and each layout
driven twice, asking for each object: a model that ignores the words scores 50%.

| | smolVLA's head (seed 0 / 1) | car head (seed 0 / 1) | same recipe, smolVLA's published input format |
|---|---|---|---|
| Words test, pair: drove to the asked-for object | **73% / 82%** | **79% / 87%** | 48–60% |
| Goal behind the car, decoy in front | 87% / 87% | 80% / 80% | 47–73% |
| One object alone | 93% / 97% | 97% / 97% | 87–97% |
| Standard test: 128 ordinary rooms | **78% / 70%** | **74% / 74%** | 48–62% |
| ... goal hidden at the start | 61% / 59% | 59% / 65% | |

Notebook 04 reproduces the seed-0 columns exactly. The two heads are equally good within luck; the input format of the
reader makes the difference (see below). Still weak: same-colour lookalikes (cube vs cylinder), and in ordinary rooms
searching for a hidden goal and getting stuck on obstacles, which notebook 05's reinforcement learning practises.

## Results (reinforcement learning, notebook 05)

100 rounds of PPO starting from the car head (seed 0), tested on the same drives:

| | imitation | after RL | luck test |
|---|---|---|---|
| Standard test: drives that touched something | 62% | **43%** | p = 0.0004, real |
| ... seconds touching per drive | 5.0 | **1.9** | |
| ... failed drives stuck on an obstacle | 16 | **1** | |
| ... failed drives that saw the goal, then lost it | 7 | 16 | |
| Standard test: success | 74% | 76% | p = 0.84, luck |
| Words test, pair: drove to the asked-for object | 79% | 83% | p = 0.43, luck |

RL taught what the demonstrations couldn't (the scripted driver never touched anything, so never showed how to back
off), without hurting the reading. Success didn't rise because the failures moved: freed from obstacles, the car now
more often loses sight of the goal and doesn't find it again. With one camera picture at a time it has no memory of
where the goal was; that is the next limit (notebook 05, section 9.5).

## Results (ceiling camera, notebook 06)

An off-car server reads a ceiling camera and sends the car 9 numbers every half second; same 128 test rooms (seed 0):

| | front camera only | perfect direction to the goal | **real route from the server** | perfect route |
|---|---|---|---|---|
| success | 75% | 77% | **91%** | 99% |
| ... goal hidden at the start | 69% | 63% | **88%** | 100% |
| ... a box in the way (20 rooms) | 50% | 50% | **80%** | 95% |
| drives that touched something | 52% | 46% | **30%** | 4% |

Real route vs front camera, drive by drive: 29 rooms won, 9 lost (p = 0.002). Training the server's obstacle map
longer (notebook 06, section 8.7) raised the real-route car to 94% (90% with a box in the way, 22% touching), 28 rooms
won and 4 lost against the front camera; its gain over 91% is not yet beyond luck. Direction help (even perfect) points the
car straight into whatever is in the way; a route goes around it. The working server uses the VLM for *what* (which
object, what is an obstacle) and simple geometry for *where* (marker tracking of the car, colour for the goal, a
shortest-path planner).

## Results (practice with route help, notebook 07)

140 rounds of notebook 05's PPO for the route car of notebook 06 (better map), same 128 test rooms:

| | before practice | **after practice** | perfect route |
|---|---|---|---|
| success | 94% | **98%** | 99% |
| ... goal hidden at the start | 92% | **96%** | 100% |
| failed drives (stuck) | 8 (7) | **3 (3)** | 1 (1) |
| seconds touching per drive | 3.0 | **1.6** | |
| with help switched off | 67% | **76%** | 58% |

Room by room the gain over "before" is not yet beyond luck (7 won, 2 lost, p = 0.18), but the car is now within luck of
the perfect route (1 won, 3 lost) and far ahead of the front-camera car (29 won, 0 lost). Practice teaches *how to get
unstuck*, the route says *where to go*: notebook 05 had only the first, notebook 06 only the second.

## Install

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), about 15 GB of free disk space (frame caches ~7 GB,
checkpoints ~1 GB, each RL run ~1.5 GB, plus the downloaded models), and for training an NVIDIA GPU with ≥ 16 GB or an Apple Silicon Mac with ≥ 32 GB of memory.

```bash
git clone <this repo> && cd obstacle_avoidance
uv sync                     # creates .venv with the exact versions in uv.lock
uv run jupyter lab          # then open notebooks/00_setup_and_first_drive.ipynb
```

* **Linux / Windows with an NVIDIA GPU:** `uv sync` installs PyTorch built for CUDA 13.4 (see `pyproject.toml`). You need
  a recent NVIDIA driver.
* **macOS (Apple Silicon M1–M4):** `uv sync` installs the standard PyTorch with Apple's GPU support (`mps`). Genesis
  uses Metal. The first time you drive the car, macOS may ask to allow keyboard input for the terminal. Intel Macs are
  not supported (Genesis has no build for them).
* **No GPU:** everything selects the CPU automatically; notebooks 00–03 work, training would take days.

The device is picked in one place, `carsim/device.py` (`cuda` > `mps` > `cpu`); force one with `CARSIM_DEVICE=cpu`.
The first run downloads smolVLA (~0.9 GB) and SmolVLM2 from Hugging Face.

## Repository layout

```
carsim/            the project code, imported by the notebooks (each file starts with a comment explaining it)
  config.py car.py mecanum.py arena.py   the car and its physics
  world.py env.py viz.py words.py curriculum.py   rooms, episodes, the Gymnasium-style environment, maps
  expert.py         scripted drivers (route follower; search-then-go with a human-like finish)
  recording.py data.py                  LeRobot recording format; training samples, splits, scaling, augmentation
  vla.py bc_train.py                     smolVLA loading, the reader input format, the car head; imitation training
  rl_env.py rl_policy.py ppo.py          reward, parallel simulators, PPO with a flow-matching head
  evaluate.py device.py teleop.py        test charts and statistics; device selection; keyboard driving
  overhead.py server_native.py server_map.py   notebook 06: the ceiling camera, the SmolVLM2 server, the learned obstacle map
  guidance.py guided_train.py route.py tracker.py nb06.py   help tokens for the car, route planning, marker tracking, charts
  route_ppo.py      notebook 07: PPO with the real server's route help in the practice loop
scripts/           programs the notebooks start in their own process (driving windows, recorder, tests)
  drive_empty_arena.py teleop.py record.py prompt_dialog.py   driving and recording
  rl_tests.py rl_reward_check.py train_bc.py                   driving tests, reward check, training from the command line
  overhead_data*.py train_server*.py overhead_frames.py route_frames.py track_frames.py server_route_frames.py
  train_guided.py car06_tests.py route_follow_prep.py           notebook 06's pipeline (each step skipped if done)
  train_guided_ppo.py                                           practice runs from the command line (notebook 07: --route)
assets/            the car model (URDF) and the 300 scripted room layouts (seeds) behind the results
notebooks/         the course
runs/              everything the notebooks produce (datasets, caches, checkpoints, test results); not in git
```

## What the project found (the short version)

1. **Feed a pretrained model the way it was trained.** smolVLA's published code multiplies the reader's inputs by
   √960 ≈ 31, uses two-way attention and leaves out image markers, settings carried over from π0's different language
   model. Fed like that, SmolVLM2's reader cannot tell which of two objects the instruction names (50%, a coin flip);
   in its own format it can (93%). Training in SmolVLM2's format raised every driving score (notebook 04, section 3).
2. **Test what you care about, in a way the model can't pass by accident.** In most scenes "drive to the obvious
   object" works without reading. Only tests with two objects and both instructions reveal whether the words are used.
3. **Scripted demonstrations work if the camera can explain them.** A driver that turns towards a hidden goal it
   knows from the map teaches "magic"; a driver that searches like a person, and finishes like one, teaches skills.
4. **Measure luck.** Retraining with another random seed moved scores by up to ~10 points; every comparison in the
   notebooks shows 95% ranges and drive-by-drive sign tests.
5. **RL fixes what demonstrations can't show, and reveals what's hidden behind it.** Practice with a crash cost taught
   the car to get off obstacles; that uncovered the next weakness (losing the goal), which a small reward alone did not fix.

6. **Ask what kind of help the driver needs.** The perfect direction to a hidden goal didn't raise success; the same
   9 numbers as a route around the obstacles did (75% → 91% with the real server, 99% with a perfect map).
7. **VLMs for *what*, geometry for *where*.** SmolVLM2 locates things only to ~0.5 m (one picture token covers more
   than half a metre of floor) and reads the car's heading backwards ~4% of the time; marker tracking gets 1 cm.

8. **Practice and guidance do different jobs.** Practice (RL) taught getting unstuck; the route taught where to go.
   Each alone left the car at 75–94%; together, 98%.

## References

* smolVLA: Shukor et al. 2025, *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*, arXiv 2506.01844
* π0: Black et al. 2024, arXiv 2410.24164 · PaliGemma: Beyer et al. 2024, arXiv 2407.07726
* ReinFlow (PPO for flow policies): Zhang et al. 2025 · PPO: Schulman et al. 2017 · GAE: Schulman et al. 2016
* Data scaling in imitation learning: Lin et al. 2024, arXiv 2410.18647 · DAgger / compounding errors: Ross et al. 2011
* Language use in VLAs: LIBERO-Plus (arXiv 2510.13626), LangGap (arXiv 2603.00592)
* Notebook 06: SmolVLM2 / SmolVLM (arXiv 2504.05299) · Mobility VLA (arXiv 2407.07775) · HAMSTER (arXiv 2502.05485) ·
  RT-Trajectory (arXiv 2311.01977) · ODG-NoMaD (arXiv 2608.21395) · CoFL (arXiv 2603.02854) · A Good Time to Ask
  (arXiv 2206.10606) · Ask4Help (arXiv 2211.09960) · VLMs are blind (arXiv 2407.06581) · grounding-aware token pruning
  (arXiv 2506.21873) · the Robotarium (arXiv 1609.04730)
