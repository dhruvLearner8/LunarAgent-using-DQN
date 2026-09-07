"""environment.py - Gym-style RL environment for the lunar rover.

Wraps terrain.py's procedural generation into a stateful environment with
the standard reset()/step() interface used throughout RL (matches the
OpenAI Gym / Gymnasium convention, which is what most DQN tutorials and
libraries expect).

Design decisions worth understanding:

- Local 30x30 view instead of the full 200x200 map: a real rover only has
  local sensors, not a satellite view, so training on local views forces
  the agent to learn a *reactive navigation policy* ("if a crater is here,
  go around it") rather than memorizing where the one goal is on one
  specific map. It also keeps the CNN's input size fixed regardless of
  grid size or agent position.

- Reachability check on reset(): with dense terrains (e.g. "hard" at ~70%
  crater coverage), there's a real chance the sampled start and goal have
  no safe path between them at all. Without checking, some episodes would
  be unsolvable through no fault of the agent, which pollutes the reward
  curve with noise that looks like learning failure. We verify with a
  flood-fill (BFS) over non-crater pixels and regenerate if unreachable -
  cheap (at most 40,000 cells) relative to how rarely it's needed.

- Reward shaping uses potential-based distance *progress*
  (prev_distance - new_distance), not raw distance-to-goal. An earlier
  version paid `-0.01 * distance` every single step just for being far
  from the goal, so a long episode's total shaping penalty scaled with
  episode LENGTH rather than with how well the agent was actually doing -
  over the ~1000-step horizon that alone (~-0.01*250*1000 = -2500) dwarfed
  both terminal rewards (+500/-100). A real 1000-episode training run
  measurably learned the wrong lesson from this: crater-death rate rose
  from 34% to 94% and average survival time fell from ~730 to ~94 steps as
  epsilon decreased, i.e. the *learned* policy got worse at survival, not
  better - because once reaching the goal looked unreachable, dying in a
  crater immediately (-100) was cheaper than surviving a long time without
  progress (which could cost -1000 or more in accumulated step+distance
  penalty alone). Rewarding progress instead (positive for closing the
  distance, negative for opening it) is a telescoping sum bounded by net
  displacement regardless of episode length, so merely being far from the
  goal is never inherently a lost cause the way raw-distance shaping made
  it - only actually failing to make progress is penalized.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from terrain import generate_terrain

# Action space: discrete up/down/left/right, encoded as (row_delta, col_delta).
ACTIONS: dict[int, tuple[int, int]] = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}
ACTION_NAMES: list[str] = ["up", "down", "left", "right"]

# Reward constants (see module docstring for the reasoning behind each).
CRATER_PENALTY = -100.0
GOAL_REWARD = 500.0
TIMEOUT_PENALTY = -50.0
STEP_PENALTY = -1.0
SAFE_STEP_BONUS = 1.0  # cancels STEP_PENALTY on safe steps - see step()
DISTANCE_PROGRESS_WEIGHT = 0.5
GOAL_RADIUS = 5.0


class LunarEnvironment:
    """A procedurally-generated lunar surface the rover must cross.

    Attributes:
        size: Width/height of the square terrain grid, in pixels.
        view_size: Width/height of the local window returned as state.
        max_steps: Episode horizon; exceeding this ends the episode.
        difficulty: Current terrain difficulty ("easy" | "medium" | "hard").
    """

    def __init__(
        self,
        size: int = 200,
        view_size: int = 30,
        max_steps: int = 1000,
        difficulty: str = "easy",
        seed: Optional[int] = None,
    ) -> None:
        self.size = size
        self.view_size = view_size
        self.max_steps = max_steps
        self.difficulty = difficulty
        self.action_space_size = len(ACTIONS)

        self._rng = np.random.default_rng(seed)

        # Populated by reset(); typed here for clarity.
        self.terrain: np.ndarray = np.zeros((size, size), dtype=np.float32)
        self.crater_mask: np.ndarray = np.zeros((size, size), dtype=bool)
        self._padded_terrain: np.ndarray = np.zeros(
            (size + view_size, size + view_size), dtype=np.float32
        )
        self.agent_pos: tuple[int, int] = (0, 0)
        self.goal_pos: tuple[int, int] = (0, 0)
        self.steps_taken: int = 0
        self.path: list[tuple[int, int]] = []

    # ------------------------------------------------------------------ #
    # Core Gym-style API
    # ------------------------------------------------------------------ #

    def reset(self, difficulty: Optional[str] = None) -> np.ndarray:
        """Generate a fresh terrain and place the agent/goal, then return the
        initial state.

        Args:
            difficulty: Optionally override the environment's current
                difficulty for this episode (used by the curriculum in
                train.py to escalate difficulty across episodes).

        Returns:
            The initial 30x30 normalized local view around the agent.
        """
        if difficulty is not None:
            self.difficulty = difficulty

        max_attempts = 50
        terrain = mask = start = goal = None
        for _ in range(max_attempts):
            terrain, mask = generate_terrain(self.difficulty, size=self.size)
            try:
                start = self._sample_safe_position(mask, corner="top_left")
                goal = self._sample_safe_position(mask, corner="bottom_right")
            except RuntimeError:
                # This particular terrain's corner region is entirely
                # crater-covered (can happen on dense "hard" maps) - it's
                # not salvageable, so discard it and generate a fresh one
                # rather than crashing the whole training run.
                continue
            if self._path_exists(mask, start, goal):
                break
        else:
            raise RuntimeError(
                f"Failed to generate a valid '{self.difficulty}' episode "
                f"after {max_attempts} attempts (start/goal unplaceable or "
                f"unreachable every time). The crater density for this "
                f"difficulty is likely too high - consider lowering "
                f"num_craters or radius_range in terrain.py's "
                f"DIFFICULTY_PRESETS."
            )

        self.terrain = terrain
        self.crater_mask = mask
        self.agent_pos = start
        self.goal_pos = goal
        self.steps_taken = 0
        self.path = [self.agent_pos]

        # Pad once per episode (terrain doesn't change during an episode),
        # rather than re-padding on every single step() call.
        half = self.view_size // 2
        self._padded_terrain = np.pad(self.terrain, half, mode="edge")

        return self.get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Move the agent one cell and compute the resulting reward.

        Args:
            action: Integer in [0, 3] - see ACTIONS / ACTION_NAMES.

        Returns:
            (state, reward, done, info) - the standard Gym step tuple.
        """
        if action not in ACTIONS:
            raise ValueError(f"Invalid action {action}; expected one of {list(ACTIONS)}")

        d_row, d_col = ACTIONS[action]
        row, col = self.agent_pos
        prev_distance = self._distance_to_goal()  # BEFORE moving, for progress shaping

        new_row = int(np.clip(row + d_row, 0, self.size - 1))
        new_col = int(np.clip(col + d_col, 0, self.size - 1))
        self.agent_pos = (new_row, new_col)
        self.path.append(self.agent_pos)
        self.steps_taken += 1

        done = False
        distance = self._distance_to_goal()
        progress = prev_distance - distance  # positive = moved closer this step

        if self.crater_mask[new_row, new_col]:
            reward = CRATER_PENALTY
            done = True
        elif distance <= GOAL_RADIUS:
            reward = GOAL_REWARD
            done = True
        elif self.steps_taken >= self.max_steps:
            reward = TIMEOUT_PENALTY
            done = True
        else:
            reward = STEP_PENALTY + SAFE_STEP_BONUS + DISTANCE_PROGRESS_WEIGHT * progress

        info = {
            "steps": self.steps_taken,
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "distance_to_goal": distance,
        }
        return self.get_state(), reward, done, info

    def get_state(self) -> np.ndarray:
        """Extract the local view_size x view_size window around the agent,
        normalized to [0, 1] float32.

        Uses the pre-padded terrain (padded with mode='edge', i.e. edge
        pixels are repeated outward) so the agent can be near the border of
        the map without the window going out of bounds.
        """
        half = self.view_size // 2
        row, col = self.agent_pos
        # Padded array index = original index + half (since we padded by
        # `half` on every side in reset()).
        p_row, p_col = row + half, col + half
        window = self._padded_terrain[
            p_row - half : p_row + half, p_col - half : p_col + half
        ]
        return (window / 255.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _sample_safe_position(
        self, mask: np.ndarray, corner: str, margin_fraction: float = 0.2
    ) -> tuple[int, int]:
        """Sample a random non-crater position within a corner region.

        Args:
            mask: Crater danger mask for the current terrain.
            corner: "top_left" (used for the start) or "bottom_right"
                (used for the goal).
            margin_fraction: Size of the corner region, as a fraction of
                the grid size (0.2 -> a 40x40 region on a 200x200 grid).
        """
        margin = max(1, int(self.size * margin_fraction))
        for _ in range(1000):
            if corner == "top_left":
                row = int(self._rng.integers(0, margin))
                col = int(self._rng.integers(0, margin))
            elif corner == "bottom_right":
                row = int(self._rng.integers(self.size - margin, self.size))
                col = int(self._rng.integers(self.size - margin, self.size))
            else:
                raise ValueError(f"Unknown corner '{corner}'")

            if not mask[row, col]:
                return (row, col)

        raise RuntimeError(
            f"Could not find a safe '{corner}' position after 1000 attempts "
            f"- terrain may be too dense."
        )

    def _path_exists(
        self, mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
    ) -> bool:
        """Breadth-first flood-fill over non-crater pixels to verify a safe
        path exists between start and goal (4-directional movement, matching
        the agent's actual action space).
        """
        if mask[start] or mask[goal]:
            return False

        visited = np.zeros_like(mask, dtype=bool)
        visited[start] = True
        queue: deque[tuple[int, int]] = deque([start])

        while queue:
            row, col = queue.popleft()
            if (row, col) == goal:
                return True
            for d_row, d_col in ACTIONS.values():
                n_row, n_col = row + d_row, col + d_col
                if (
                    0 <= n_row < self.size
                    and 0 <= n_col < self.size
                    and not visited[n_row, n_col]
                    and not mask[n_row, n_col]
                ):
                    visited[n_row, n_col] = True
                    queue.append((n_row, n_col))
        return False

    def _distance_to_goal(self) -> float:
        """Euclidean distance from the agent's current position to the goal."""
        row, col = self.agent_pos
        goal_row, goal_col = self.goal_pos
        return float(np.hypot(row - goal_row, col - goal_col))


if __name__ == "__main__":
    # Manual smoke test: reset the env, print start/goal, then run a
    # random-action episode to completion and confirm step() behaves.
    import time

    for difficulty in ["easy", "medium", "hard"]:
        env = LunarEnvironment(difficulty=difficulty, seed=None)
        t0 = time.time()
        state = env.reset()
        reset_time = time.time() - t0

        print(f"\n=== difficulty={difficulty} ===")
        print(f"reset() took {reset_time*1000:.1f} ms")
        print(f"start={env.agent_pos} goal={env.goal_pos} "
              f"initial_distance={env._distance_to_goal():.1f}")
        print(f"state shape={state.shape} dtype={state.dtype} "
              f"min={state.min():.3f} max={state.max():.3f}")

        total_reward = 0.0
        done = False
        steps = 0
        outcome = "max_steps_debug_cap"
        while not done and steps < 2000:
            action = np.random.randint(0, 4)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                if reward == GOAL_REWARD:
                    outcome = "reached_goal"
                elif reward == CRATER_PENALTY:
                    outcome = "hit_crater"
                elif reward == TIMEOUT_PENALTY:
                    outcome = "timed_out"

        print(f"random episode: steps={steps} total_reward={total_reward:.1f} "
              f"outcome={outcome}")