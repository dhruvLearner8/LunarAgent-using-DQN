# Findings

Notes from debugging and experimenting with the Lunar Rover DQN. Ordered roughly as discovered.

## Bugs found and fixed

1. **Reward shaping made dying fast cheaper than surviving.** The original reward paid `-0.01 * distance_to_goal` every step just for being far from the goal, on top of a `-1` step penalty. Over a ~1000-step episode that could total -2500 to -3500 - far worse than the -100 crater penalty. A real 1000-episode training run confirmed this empirically: crater-death rate rose from 34% to 94% and average survival time fell from ~730 to ~94 steps as epsilon decreased, i.e. the *learned* policy got worse at surviving, not better, because dying immediately was mathematically the best available option once reaching the goal looked unreachable.

   **Fix:** switched to potential-based shaping - reward the *change* in distance each step (`prev_distance - new_distance`) instead of the absolute distance. This is a telescoping sum bounded by net displacement regardless of episode length, so merely being far from the goal is never inherently a lost cause. (`environment.py`)

2. **MSE loss + no gradient clipping caused the network to collapse.** TD targets here legitimately range roughly -1000 to +500. Squaring large early-training errors (MSE) produced destabilizing gradients; loss spiked into the thousands over a training run instead of decreasing, and the trained network ended up outputting nearly-identical Q-values regardless of input (confirmed directly: fed it clearly different real terrain patches from an actual rollout, got the same output every time).

   **Fix:** Huber loss (`SmoothL1Loss`) instead of MSE, plus gradient-norm clipping (`max_grad_norm=10.0`). Matches the original DeepMind Atari DQN paper's stabilization approach, which this architecture is otherwise modeled on. (`agent.py`)

3. **Replay buffer (10k) was too small relative to episode length and reward sparsity.** Episodes run several hundred steps and successes were rare (single digits per hundred episodes). Every one of a 500-episode run's ~20 successful transitions had already been evicted from the buffer more than 69,000 steps before training ended - the network had no positive examples left to learn from for the entire back half of training, which produced a visible late-training performance regression.

   **Fix:** raised `buffer_size` to 100,000. (`agent.py`)

4. **Device selection never actually checked MPS.** Hardcoded `cuda` or `cpu`; Apple Silicon GPU was never used even when available. Benchmarked ~2.3x speedup on MPS vs CPU for this network size once fixed. (`agent.py`)

5. **`agent.save()` didn't create its parent directory.** `torch.save()` errors if the directory doesn't exist rather than creating it - a real training run crashed mid-run after its `models/` directory was deleted externally. Fixed defensively regardless of cause. (`agent.py`)

## Architecture redesign

Original: 2 conv layers (kernel 4→3, stride 2→1) → flatten → `Linear(9216, 256)` → `Linear(256, 4)`. 2.38M params, of which 2.36M lived in the single flatten→FC layer. Effective receptive field on the 30x30 input: only 8x8 - the conv layers could never relate "danger on the left" to "safety on the right," leaving the giant FC layer to reconstruct spatial relationships from a positionally-scrambled flatten.

Redesigned: 4 conv layers (all kernel 3, stride 2/2/1/1) → global average pooling → dueling head (separate value/advantage streams). **110,405 params (21.6x smaller)**, receptive field grown to 23x23.

Tried BatchNorm first for training stability - made things worse. BatchNorm normalizes using per-batch statistics, but in DQN both the sampled batch composition *and* the bootstrapped targets keep shifting as the policy improves - two compounding sources of non-stationarity. A validation run showed loss climbing 770x (0.135 → 104) with BatchNorm despite Huber loss + gradient clipping already in place. Switched to `GroupNorm(1, C)` (LayerNorm-equivalent for conv nets, normalizes per-sample) - no batch-statistics dependency, same stabilization benefit.

## Tiny single/multi-crater experiments (`main.py`, `notebooks/tiny_crater_test_jupyter.ipynb`)

Built a minimal, deterministic, small-grid version of the environment (one fixed crater, fixed start/goal) to isolate "does the DQN pipeline learn anything at all" from "is the full procedural task just hard." Confirmed: yes, it learns a real detour around the one crater (visually verified path).

**Memorization vs. generalization, tested directly:** with a *fixed* crater position across all training episodes, the trained agent succeeded on its training map but failed the moment the crater was moved to an unseen position - the path it walked was the *same fixed motion sequence* regardless of where the crater actually was, confirming it had memorized a route rather than learned to read danger from the image. Randomizing the crater's center/radius every training episode fixed this: the same test (crater moved to an unseen position) then succeeded, with a visibly different, adapted path.

**2-crater generalization gap:** an agent trained on 1 randomized crater, tested on maps with 2 craters (never trained on), showed a real but partial transfer - roughly 25-33% success vs. 50-65% on 1-crater maps it was actually trained for. Mostly failed by hitting a crater, not by timing out.

**Fine-tuning / continual learning:** continuing training of the *same* (already 1-crater-trained) agent on 2-crater maps - with epsilon boosted back up first, since it had already decayed near its floor - clearly outperformed the cold 1-crater-trained agent on 2-crater maps: 33% → 58% success in one comparison. Evidence that the crater-detection features learned on the easier task transfer usefully to the harder one.

**Map-solvability check added:** `TinyFixedEnvironment.reset()` originally only checked that individual craters didn't cover the start/goal cells. With 2+ craters, that doesn't guarantee a *path* exists between them. Added a BFS reachability retry loop (reusing `LunarEnvironment._path_exists()`, already used by the full project) so every generated map is confirmed solvable.

## Open items / not yet resolved

- The full project's `environment.py` always places the start in the top-left corner and the goal in the bottom-right - every episode is the same broad directional task. Combined with the agent never observing the goal's direction/distance (only the local terrain patch), this is a likely generalization gap in the *full* procedural environment, analogous to the fixed-crater memorization issue found and fixed in the tiny test. Not yet addressed there.
- A full-scale (200x200, curriculum, 1000+ episode) training run with all current fixes applied together has not yet been run to completion/convergence.
- Real NASA lunar imagery/DEM data as a test surface (converting to the format this environment expects) discussed as a future direction, not yet attempted.
