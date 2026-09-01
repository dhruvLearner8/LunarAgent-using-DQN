"""train.py - Curriculum training loop for the Lunar Rover DQN agent.

Lives at the project root (not in src/) since it's the entry point a human
runs, not a module other src/ files import from. Every src/ module uses
flat imports internally (e.g. agent.py does `from model import LunarDQN`,
not `from .model import`) so that each file can also be run standalone as
`python src/<file>.py` for its own smoke test. Because train.py imports
those flat-importing modules from outside src/, it inserts src/ onto
sys.path *before* importing them - importing via `from src.agent import
DQNAgent` instead would work here, but would break agent.py's own internal
`from model import LunarDQN`, since that import is resolved relative to
sys.path, not relative to agent.py's package.

Curriculum learning (why progressively raise difficulty instead of training
on "hard" the whole time): "hard" terrain is ~53% crater coverage, so a
freshly-initialized agent choosing largely random actions dies in a crater
almost immediately, almost every episode. The replay buffer fills with
"random action -> death" transitions and barely any "successfully closed
the distance to the goal" transitions, so there's very little signal to
learn from. Training on "easy" terrain first (10 craters, ~well under 53%
coverage) means far more episodes end in either reaching the goal or a slow
timeout instead of instant death, which gives the agent a chance to learn
the basic "move toward the goal, don't step somewhere dark" policy under
much better odds. Once that policy exists, raising the difficulty transfers
and refines it, rather than asking the agent to discover it from scratch
under the hardest odds available.

Per-step vs per-episode training and epsilon decay: train_step() is called
after every single environment step (not once per episode) because that's
standard DQN - every step's transition should immediately contribute to
training, since with a 10k-capacity replay buffer and random sampling,
waiting until episode end to train wouldn't decorrelate anything further,
just delay learning. Epsilon decay is the opposite: decay_epsilon() is
called exactly once per episode (see agent.py's decay_epsilon() docstring)
because decaying once per step would collapse epsilon to its floor within a
single ~1000-step episode, long before the curriculum even reaches medium
difficulty.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from agent import DQNAgent  # noqa: E402
from environment import (  # noqa: E402
    CRATER_PENALTY,
    GOAL_REWARD,
    TIMEOUT_PENALTY,
    LunarEnvironment,
)

LOG_DIR = "logs"
MODEL_DIR = "models"
LOG_CSV_PATH = os.path.join(LOG_DIR, "training_log.csv")
REWARD_CURVE_PATH = os.path.join(LOG_DIR, "reward_curve.png")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pt")

CSV_HEADER = ["episode", "total_reward", "steps", "epsilon", "avg_loss", "difficulty", "outcome"]


def _try_import_visualizer():
    """Import LunarVisualizer lazily so training can still run headless
    (no pygame installed, or no display available - e.g. over SSH or in
    CI) - rendering is simply disabled rather than the whole run crashing.
    """
    try:
        from visualizer import LunarVisualizer

        return LunarVisualizer
    except Exception as exc:  # pygame missing, no display driver, etc.
        print(f"[train.py] Visualization disabled (couldn't import visualizer: {exc})")
        return None


def get_difficulty(episode: int, easy_until: int = 300, medium_until: int = 600) -> str:
    """Curriculum schedule: easy -> medium -> hard as training progresses.
    See the module docstring for why curriculum learning matters here.

    Args:
        episode: 1-indexed current episode number.
        easy_until: Last episode (inclusive) trained on "easy" terrain.
        medium_until: Last episode (inclusive) trained on "medium" terrain
            (must be >= easy_until for the schedule to make sense).

    Returns:
        One of "easy", "medium", "hard".
    """
    if episode <= easy_until:
        return "easy"
    if episode <= medium_until:
        return "medium"
    return "hard"


def _outcome_from_reward(reward: float) -> str:
    """Classify how an episode ended from its terminal reward. Safe to
    compare by exact equality (rather than isclose) because environment.py
    assigns these three constants directly on the terminal branches of
    step() - they're never combined with the per-step distance shaping."""
    if reward == GOAL_REWARD:
        return "reached_goal"
    if reward == CRATER_PENALTY:
        return "hit_crater"
    if reward == TIMEOUT_PENALTY:
        return "timed_out"
    return "unknown"


def _plot_reward_curve(
    episodes: list[int], rewards: list[float], path: str, rolling_window: int = 20
) -> None:
    """(Re)save a reward-curve PNG: raw per-episode reward (faint) plus a
    rolling average (bold) - the raw signal alone is far too noisy episode
    to episode (a single unlucky crater near the start dominates one data
    point) to see a training trend by eye, so the rolling average is what
    actually shows whether the agent is improving.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless backend - never tries to open a GUI window
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.3, color="tab:blue", label="reward")

    if len(rewards) >= rolling_window:
        rolling = np.convolve(rewards, np.ones(rolling_window) / rolling_window, mode="valid")
        rolling_episodes = episodes[rolling_window - 1 :]
        ax.plot(
            rolling_episodes,
            rolling,
            color="tab:orange",
            linewidth=2,
            label=f"rolling avg ({rolling_window})",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Lunar Rover DQN Training Reward")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(
    num_episodes: int = 1000,
    easy_until: int = 300,
    medium_until: int = 600,
    max_steps: int = 1000,
    render: bool = False,
    render_every: int = 25,
    log_every: int = 10,
    plot_every: int = 25,
    seed: Optional[int] = None,
) -> DQNAgent:
    """Run the full curriculum training loop end to end.

    Args:
        num_episodes: Total episodes to train for.
        easy_until: Last episode (inclusive) on "easy" terrain.
        medium_until: Last episode (inclusive) on "medium" terrain; episodes
            after this run on "hard".
        max_steps: Per-episode step horizon passed to LunarEnvironment.
        render: If True, attempt to open a live pygame window and render a
            subset of episodes (every `render_every`th). Silently falls
            back to no rendering if pygame/visualizer can't be imported.
        render_every: Render every Nth episode's steps live, when
            `render=True`. Rendering every single episode over a
            1000-episode run would make training far too slow to actually
            finish; rendering only a per-episode snapshot instead of live
            steps wouldn't be watchable enough to see *how* the agent
            moves. Rendering all steps of one in every `render_every`
            episodes is the compromise.
        log_every: Print a progress line every N episodes.
        plot_every: Re-save the reward curve PNG every N episodes.
        seed: Optional seed for the environment's internal RNG (used for
            start/goal placement; terrain itself is unseeded per episode
            unless environment.py is changed).

    Returns:
        The trained DQNAgent (also saved to disk at FINAL_MODEL_PATH).
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(LOG_CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)

    env = LunarEnvironment(size=200, view_size=30, max_steps=max_steps, difficulty="easy", seed=seed)
    agent = DQNAgent(input_size=env.view_size, num_actions=env.action_space_size)
    print(f"Training for {num_episodes} episodes on device: {agent.device}")

    visualizer = None
    rendering_enabled = render
    if rendering_enabled:
        LunarVisualizer = _try_import_visualizer()
        if LunarVisualizer is not None:
            visualizer = LunarVisualizer()
        else:
            rendering_enabled = False

    episode_numbers: list[int] = []
    episode_rewards: list[float] = []
    best_reward = -float("inf")
    start_time = time.time()

    try:
        for episode in range(1, num_episodes + 1):
            difficulty = get_difficulty(episode, easy_until, medium_until)
            state = env.reset(difficulty=difficulty)
            total_reward = 0.0
            episode_losses: list[float] = []
            done = False
            reward = 0.0

            should_render = rendering_enabled and visualizer is not None and episode % render_every == 0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _info = env.step(action)
                agent.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)

                if should_render:
                    keep_going = visualizer.render(
                        env, episode, total_reward, agent.epsilon, path=env.path
                    )
                    if not keep_going:
                        # User closed the window - stop rendering for the
                        # rest of training rather than calling into a
                        # shut-down pygame again.
                        rendering_enabled = False
                        should_render = False
                        visualizer = None

            agent.decay_epsilon()

            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            outcome = _outcome_from_reward(reward)

            episode_numbers.append(episode)
            episode_rewards.append(total_reward)

            with open(LOG_CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow(
                    [episode, total_reward, env.steps_taken, agent.epsilon, avg_loss, difficulty, outcome]
                )

            if total_reward > best_reward:
                best_reward = total_reward
                agent.save(BEST_MODEL_PATH)

            if episode % log_every == 0:
                print(
                    f"Episode {episode}/{num_episodes} | Reward: {total_reward:.1f} | "
                    f"Steps: {env.steps_taken} | Epsilon: {agent.epsilon:.3f} | "
                    f"Difficulty: {difficulty}"
                )

            if episode % plot_every == 0:
                _plot_reward_curve(episode_numbers, episode_rewards, REWARD_CURVE_PATH)

    except KeyboardInterrupt:
        print("\n[train.py] Interrupted - saving final model and reward curve before exiting.")

    agent.save(FINAL_MODEL_PATH)
    if episode_numbers:
        _plot_reward_curve(episode_numbers, episode_rewards, REWARD_CURVE_PATH)
    if visualizer is not None:
        visualizer.close()

    elapsed = time.time() - start_time
    print(
        f"Done. {len(episode_numbers)} episodes in {elapsed / 60:.1f} min. "
        f"Best reward: {best_reward:.1f}. Models saved to {MODEL_DIR}/."
    )
    return agent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Lunar Rover DQN agent.")
    parser.add_argument("--episodes", type=int, default=1000, dest="num_episodes")
    parser.add_argument("--easy-until", type=int, default=300)
    parser.add_argument("--medium-until", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--render", action="store_true", help="Open a live pygame window.")
    parser.add_argument("--render-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--plot-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        num_episodes=args.num_episodes,
        easy_until=args.easy_until,
        medium_until=args.medium_until,
        max_steps=args.max_steps,
        render=args.render,
        render_every=args.render_every,
        log_every=args.log_every,
        plot_every=args.plot_every,
        seed=args.seed,
    )
