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
| 05 | `05_reinforcement_learning_ppo` | reward design, parallel CPU simulators, making a flow-matching head trainable by PPO, reading RL graphs, comparing RL with imitation | 1 h + |

\*On an RTX 4080 SUPER (16 GB) with a 20-core CPU. On an Apple Silicon Mac, notebooks 00–03 are fine; training in
04–05 runs but is several times slower.

## Results (imitation, notebook 04)

Fixed tests, the same drives for every model (`scripts/rl_tests.py`). "Pair" = two objects in view and each layout
driven twice, asking for each object: a model that ignores the words scores 50%.

*(filled in after the final runs)*

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
scripts/           programs the notebooks start in their own process (driving windows, recorder, tests)
  drive_empty_arena.py teleop.py record.py prompt_dialog.py   driving and recording
  rl_tests.py rl_reward_check.py train_bc.py                   driving tests, reward check, training from the command line
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

## References

* smolVLA: Shukor et al. 2025, *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*, arXiv 2506.01844
* π0: Black et al. 2024, arXiv 2410.24164 · PaliGemma: Beyer et al. 2024, arXiv 2407.07726
* ReinFlow (PPO for flow policies): Zhang et al. 2025 · PPO: Schulman et al. 2017 · GAE: Schulman et al. 2016
* Data scaling in imitation learning: Lin et al. 2024, arXiv 2410.18647 · DAgger / compounding errors: Ross et al. 2011
* Language use in VLAs: LIBERO-Plus (arXiv 2510.13626), LangGap (arXiv 2603.00592)
