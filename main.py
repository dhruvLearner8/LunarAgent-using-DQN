# def main():
#     print("Hello from lunaragent!")


# if __name__ == "__main__":
#     main()

"""test_single_crater.py - Minimal sanity check for the DQN pipeline.

One tiny fixed grid, ONE crater sitting directly on the straight line
between a fixed start and fixed goal, so the agent is FORCED to learn a
real detour - it can't just get lucky by going straight. No procedural
generation, no curriculum, no randomness in the map. This isolates one
question: does the DQN pipeline (environment + CNN + replay buffer +
target network) actually learn anything at all, separate from whether the
full procedural/curriculum problem is just hard and needs more episodes.

Run from your project root: uv run python test_single_crater.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from environment import LunarEnvironment, GOAL_REWARD, CRATER_PENALTY
from agent import DQNAgent


class TinyFixedEnvironment(LunarEnvironment):
    """Same LunarEnvironment class (same step()/get_state()/reward logic)
    but reset() builds a tiny map directly instead of calling
    generate_terrain(). Crater math below is copied verbatim from
    terrain.py's add_craters() - same bowl-depth + raised-rim formula,
    just applied to one crater instead of N.

    If randomize_crater=True, reset() samples NEW crater center(s) and
    radius/radii every episode instead of using one fixed crater. This
    matters: with a single always-identical crater, the network can get
    +500 just as reliably by memorizing one fixed sequence of moves as it
    could by actually reading the crater's position out of the image - and
    a real test confirmed exactly that (it failed the moment the crater
    moved to an unseen position). Varying the crater's position/size every
    episode removes that shortcut - the only way to keep reliably reaching
    the goal across many different crater placements is to actually learn
    "sense a crater nearby in the image, steer around it."

    num_craters controls how many separate craters get stamped when
    randomize_crater=True (still 1 by default - only raise this for
    testing generalization to a harder situation than what was trained
    on, not for training itself, unless that's specifically what you want
    to try next).
    """

    def __init__(self, size=20, crater_center=(9, 9), crater_radius=4,
                 start=(1, 1), goal=(18, 18), view_size=30, max_steps=150,
                 randomize_crater=False, radius_range=(2, 5), num_craters=1,
                 seed=None):
        super().__init__(size=size, view_size=view_size, max_steps=max_steps)
        self._crater_center = crater_center
        self._crater_radius = crater_radius
        self._start = start
        self._goal = goal
        self._randomize_crater = randomize_crater
        self._radius_range = radius_range
        self._num_craters = num_craters
        self._rng = np.random.default_rng(seed)

    def reset(self, difficulty=None):
        size = self.size
        rows, cols = np.mgrid[0:size, 0:size]

        # Individual craters already avoid covering start/goal directly
        # (checked below), but with 2+ craters they could still jointly
        # wall off every route between start and goal even without
        # touching either cell - so the outer loop here regenerates the
        # WHOLE map (not just one crater) until _path_exists() (inherited
        # from LunarEnvironment - the same BFS reachability check the full
        # project uses) confirms a real route exists, same pattern as
        # LunarEnvironment.reset()'s own retry loop.
        for _attempt in range(50):
            terrain = np.full((size, size), 150.0, dtype=np.float32)  # flat base, no Perlin noise
            mask = np.zeros((size, size), dtype=bool)

            if self._randomize_crater:
                # For each crater: retry until it doesn't cover start or
                # goal - otherwise the episode would be unsolvable from
                # step 0 regardless of what the path-reachability check
                # below finds. Craters are allowed to overlap each other
                # (not checked) - terrain.py's real multi-crater stamping
                # doesn't guard against that either.
                margin = 3
                min_r, max_r = self._radius_range
                for _ in range(self._num_craters):
                    for _ in range(100):
                        cr = int(self._rng.integers(margin, size - margin))
                        cc = int(self._rng.integers(margin, size - margin))
                        radius = int(self._rng.integers(min_r, max_r + 1))
                        dist = np.sqrt((rows - cr) ** 2 + (cols - cc) ** 2)
                        inside = dist <= radius
                        if not inside[self._start] and not inside[self._goal]:
                            break
                    self._stamp_crater(terrain, dist, radius, inside)
                    mask |= inside
            else:
                cr, cc = self._crater_center
                radius = self._crater_radius
                dist = np.sqrt((rows - cr) ** 2 + (cols - cc) ** 2)
                inside = dist <= radius
                self._stamp_crater(terrain, dist, radius, inside)
                mask = inside.copy()

            if self._path_exists(mask, self._start, self._goal):
                break
        else:
            raise RuntimeError(
                f"Could not generate a solvable tiny map with {self._num_craters} "
                f"crater(s) after 50 attempts - craters are probably too big/many "
                f"for a {size}x{size} grid. Try a smaller radius_range or fewer craters."
            )

        terrain = np.clip(terrain, 0.0, 255.0).astype(np.float32)

        self.terrain = terrain
        self.crater_mask = mask
        self.agent_pos = self._start
        self.goal_pos = self._goal
        self.steps_taken = 0
        self.path = [self.agent_pos]

        half = self.view_size // 2
        self._padded_terrain = np.pad(self.terrain, half, mode="edge")
        return self.get_state()

    @staticmethod
    def _stamp_crater(terrain: np.ndarray, dist: np.ndarray, radius: float, inside: np.ndarray) -> None:
        """Stamp one crater's bowl-depth + raised-rim into terrain, in
        place. Factored out so reset() can call it once per crater when
        placing more than one (see num_craters)."""
        depth_factor = np.clip(1.0 - dist / radius, 0.0, 1.0)
        terrain[inside] -= depth_factor[inside] * 120.0

        rim_width = max(1.0, radius * 0.3)
        rim_dist = np.abs(dist - radius)
        rim_region = rim_dist <= rim_width
        rim_factor = np.clip(1.0 - rim_dist / rim_width, 0.0, 1.0)
        terrain[rim_region] += rim_factor[rim_region] * 60.0


def main():
    env = TinyFixedEnvironment(randomize_crater=True)
    agent = DQNAgent(input_size=30, num_actions=4, target_update_freq=50, batch_size=32)

    episodes = 400
    reward_history = []
    outcome_history = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        outcome = "timed_out"
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
            if done:
                if reward == GOAL_REWARD:
                    outcome = "reached_goal"
                elif reward == CRATER_PENALTY:
                    outcome = "hit_crater"
        agent.decay_epsilon()  # once per episode, not per step
        reward_history.append(total_reward)
        outcome_history.append(outcome)

        if ep % 40 == 0:
            recent = outcome_history[-40:]
            success_rate = recent.count("reached_goal") / len(recent) * 100
            print(f"Episode {ep}/{episodes} | Reward: {total_reward:.1f} | "
                  f"Steps: {env.steps_taken} | Epsilon: {agent.epsilon:.3f} | "
                  f"last40_success_rate: {success_rate:.0f}%")

    # Final check: run the trained agent GREEDILY (epsilon=0, no randomness)
    # and see what it actually does - this is the real test of what it learned.
    agent.epsilon = 0.0
    state = env.reset()
    done = False
    reward = 0.0
    while not done:
        action = agent.choose_action(state)
        state, reward, done, info = env.step(action)
    final_outcome = ("reached_goal" if reward == GOAL_REWARD
                      else "hit_crater" if reward == CRATER_PENALTY
                      else "timed_out")
    print(f"\nFinal greedy run (epsilon=0): steps={env.steps_taken} outcome={final_outcome}")

    # Generalization test: SAME trained agent (still epsilon=0, no further
    # training happens here), but on a NEW environment where the crater's
    # center is moved upward (row 9 -> row 4, same column, same radius).
    # If the agent only memorized "avoid pixels around (9,9)" rather than
    # learning a general "sense a crater nearby, steer around it" policy,
    # it should fail here even though this is an easier map in every other
    # respect (same start, same goal, same single-crater structure).
    shifted_env = TinyFixedEnvironment(crater_center=(4, 9))
    state = shifted_env.reset()
    done = False
    reward = 0.0
    while not done:
        action = agent.choose_action(state)
        state, reward, done, info = shifted_env.step(action)
    shifted_outcome = ("reached_goal" if reward == GOAL_REWARD
                        else "hit_crater" if reward == CRATER_PENALTY
                        else "timed_out")
    print(f"Generalization test (crater moved upward to row 4): "
          f"steps={shifted_env.steps_taken} outcome={shifted_outcome}")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    axes[0].plot(reward_history, color="lightgray", linewidth=0.8)
    window = 20
    rolling = np.convolve(reward_history, np.ones(window) / window, mode="valid")
    axes[0].plot(range(window - 1, episodes), rolling, color="tab:blue", linewidth=2)
    axes[0].set_title("Tiny-environment reward curve")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].grid(alpha=0.3)

    axes[1].imshow(env.terrain, cmap="gray", vmin=0, vmax=255)
    overlay = np.zeros((*env.crater_mask.shape, 4))
    overlay[env.crater_mask] = [1, 0, 0, 0.3]
    axes[1].imshow(overlay)
    path = np.array(env.path)
    axes[1].plot(path[:, 1], path[:, 0], color="cyan", linewidth=2)
    axes[1].scatter(*env._start[::-1], c="blue", s=100, label="start", zorder=5)
    axes[1].scatter(*env._goal[::-1], c="lime", s=120, marker="*", label="goal", zorder=5)
    axes[1].set_title(f"Trained-on map (epsilon=0): {final_outcome}")
    axes[1].legend()
    axes[1].axis("off")

    axes[2].imshow(shifted_env.terrain, cmap="gray", vmin=0, vmax=255)
    overlay2 = np.zeros((*shifted_env.crater_mask.shape, 4))
    overlay2[shifted_env.crater_mask] = [1, 0, 0, 0.3]
    axes[2].imshow(overlay2)
    shifted_path = np.array(shifted_env.path)
    axes[2].plot(shifted_path[:, 1], shifted_path[:, 0], color="cyan", linewidth=2)
    axes[2].scatter(*shifted_env._start[::-1], c="blue", s=100, label="start", zorder=5)
    axes[2].scatter(*shifted_env._goal[::-1], c="lime", s=120, marker="*", label="goal", zorder=5)
    axes[2].set_title(f"Crater moved upward (epsilon=0): {shifted_outcome}")
    axes[2].legend()
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("tiny_test_result.png", dpi=120)
    print("saved tiny_test_result.png")


if __name__ == "__main__":
    main()