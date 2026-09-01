"""visualizer.py - Pygame-based live rendering of a training episode.

train.py calls LunarVisualizer.render() once per environment step (for a
subset of episodes - see train.py's `render_every`) so a human can watch the
agent actually move across the terrain in real time, rather than only ever
seeing aggregate reward numbers scroll by in a log. Watching failure modes
directly (e.g. "it keeps walking straight into the same crater rim") is often
far faster for debugging a policy than staring at a loss curve.

Why cache the background instead of redrawing terrain+craters every frame:
rasterizing a 200x200 float array into a colored, upscaled 800x800 surface
is the expensive part of a frame (a full-array numpy pass plus a scale
blit); the terrain itself is fixed for the whole episode (see environment.py
- reset() is the only place a new terrain is generated), so redoing that
work on every single step (potentially thousands of times per episode) would
make rendering the bottleneck. We instead rebuild it only when we detect a
genuinely new terrain array via `id()` - an identity check is O(1) and safe
here because environment.py never mutates self.terrain in place after
reset(), it only ever reassigns it wholesale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pygame

if TYPE_CHECKING:
    from environment import LunarEnvironment

WINDOW_SIZE = 800  # 200-cell grid * 4px/cell
TRAIL_LENGTH = 60  # how many recent positions the fading trail keeps
AGENT_RADIUS = 4
GOAL_RADIUS = 4
CRATER_OVERLAY_COLOR = np.array([255, 60, 60], dtype=np.float32)
CRATER_OVERLAY_ALPHA = 0.35
AGENT_COLOR = (30, 90, 255)
GOAL_COLOR = (40, 200, 80)
TRAIL_COLOR = (60, 130, 255)
PANEL_BG = (0, 0, 0, 160)
PANEL_TEXT_COLOR = (255, 255, 255)


class LunarVisualizer:
    """Live pygame window showing terrain, craters, agent, goal, and a
    fading path trail for one training episode at a time.

    Not safe to share across concurrent episodes - it caches one
    background surface at a time, keyed by the identity of the terrain
    array it was built from.
    """

    def __init__(self, grid_size: int = 200, window_size: int = WINDOW_SIZE, fps: int = 30) -> None:
        pygame.init()
        pygame.display.set_caption("Lunar Rover DQN")

        self.grid_size = grid_size
        self.window_size = window_size
        self.scale = window_size / grid_size
        self.fps = fps

        self.screen = pygame.display.set_mode((window_size, window_size))
        self.font = pygame.font.SysFont("monospace", 18)
        self.clock = pygame.time.Clock()

        # Cache: rebuilt only when a new episode's terrain array shows up.
        self._background: Optional[pygame.Surface] = None
        self._background_terrain_id: Optional[int] = None

    def render(
        self,
        env: "LunarEnvironment",
        episode: int,
        reward: float,
        epsilon: float,
        path: Optional[list[tuple[int, int]]] = None,
    ) -> bool:
        """Draw one frame of the current environment state.

        Args:
            env: The LunarEnvironment to render (reads terrain, crater_mask,
                agent_pos, goal_pos, difficulty).
            episode: Current episode number, shown in the stats panel.
            reward: Running (or final) episode reward, shown in the stats
                panel - typically the caller's running total so far.
            epsilon: Current exploration rate, shown in the stats panel.
            path: History of (row, col) agent positions this episode, used
                to draw the fading trail. Defaults to `[]` inside the
                function body rather than in the signature - a mutable
                default argument would be shared and silently accumulate
                across every call that doesn't pass one explicitly.

        Returns:
            True if rendering should continue. False if the user closed
            the window - the caller should stop calling render() for the
            rest of training (pygame has already been shut down at that
            point, so calling render() again would error).
        """
        if path is None:
            path = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        if self._background is None or self._background_terrain_id != id(env.terrain):
            self._background = self._build_background(env.terrain, env.crater_mask)
            self._background_terrain_id = id(env.terrain)

        self.screen.blit(self._background, (0, 0))
        self._draw_trail(path)

        ax, ay = self._to_screen(*env.agent_pos)
        pygame.draw.circle(self.screen, AGENT_COLOR, (ax, ay), AGENT_RADIUS)

        gx, gy = self._to_screen(*env.goal_pos)
        pygame.draw.circle(self.screen, GOAL_COLOR, (gx, gy), GOAL_RADIUS)

        self._draw_stats_panel(episode, reward, epsilon, env.difficulty)

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self) -> None:
        """Shut down pygame explicitly (e.g. when training ends normally
        without the user closing the window)."""
        pygame.quit()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_background(self, terrain: np.ndarray, crater_mask: np.ndarray) -> pygame.Surface:
        """Rasterize the grayscale terrain plus a semi-transparent red
        crater overlay into a window-sized surface (see module docstring
        for why this is cached rather than rebuilt every frame).
        """
        gray = np.clip(terrain, 0, 255).astype(np.float32)
        rgb = np.stack([gray, gray, gray], axis=-1)

        # Blend crater pixels toward the overlay color rather than
        # replacing them outright, so texture (e.g. the crater floor's
        # own bowl shading) still reads through the red tint.
        rgb[crater_mask] = (
            (1 - CRATER_OVERLAY_ALPHA) * rgb[crater_mask]
            + CRATER_OVERLAY_ALPHA * CRATER_OVERLAY_COLOR
        )
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # numpy arrays are indexed (row, col) = (y, x); pygame surfarrays
        # are indexed (x, y), so swap the first two axes before building
        # the surface.
        surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        return pygame.transform.scale(surf, (self.window_size, self.window_size))

    def _draw_trail(self, path: list[tuple[int, int]]) -> None:
        """Draw the last TRAIL_LENGTH path points as a blue trail that
        fades from transparent (oldest) to opaque (most recent), via a
        dedicated SRCALPHA overlay surface (the main screen surface has no
        per-pixel alpha channel of its own).
        """
        if len(path) < 2:
            return

        trail = path[-TRAIL_LENGTH:]
        overlay = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        n = len(trail)
        for i, (row, col) in enumerate(trail):
            alpha = int(255 * (i + 1) / n)
            x, y = self._to_screen(row, col)
            pygame.draw.circle(overlay, (*TRAIL_COLOR, alpha), (x, y), 3)
        self.screen.blit(overlay, (0, 0))

    def _draw_stats_panel(self, episode: int, reward: float, epsilon: float, difficulty: str) -> None:
        """Draw a semi-transparent stats box in the top-left corner."""
        lines = [
            f"Episode: {episode}",
            f"Reward: {reward:.1f}",
            f"Epsilon: {epsilon:.3f}",
            f"Difficulty: {difficulty}",
        ]
        line_height = 20
        # Size the panel to the widest rendered line rather than a fixed
        # guess - a hardcoded width silently clips longer text (e.g.
        # "Difficulty: medium"/"hard") depending on font/content.
        panel_width = max(self.font.size(line)[0] for line in lines) + 12
        panel = pygame.Surface((panel_width, line_height * len(lines) + 10), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        for i, line in enumerate(lines):
            text = self.font.render(line, True, PANEL_TEXT_COLOR)
            panel.blit(text, (6, 6 + i * line_height))
        self.screen.blit(panel, (0, 0))

    def _to_screen(self, row: int, col: int) -> tuple[int, int]:
        """Grid (row, col) -> screen (x, y) pixel coordinates. `col` maps to
        the horizontal axis and `row` to the vertical axis, matching the
        numpy/imshow convention used everywhere else in this project."""
        return int(col * self.scale), int(row * self.scale)


if __name__ == "__main__":
    # Manual/visual smoke test: run a short random-action episode against a
    # real LunarEnvironment and render every step, so you can eyeball
    # terrain, craters, agent, goal, trail, and the stats panel all at
    # once. Close the window (or let it hit max_steps) to end the demo.
    import time

    from environment import LunarEnvironment

    env = LunarEnvironment(difficulty="medium", size=200, view_size=30, max_steps=300)
    env.reset()

    viz = LunarVisualizer(grid_size=200, fps=30)
    print("Rendering a random-action episode - close the window to stop early.")

    total_reward = 0.0
    done = False
    keep_rendering = True
    steps = 0
    t0 = time.time()

    while not done and keep_rendering:
        action = np.random.randint(0, 4)
        _, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        keep_rendering = viz.render(
            env, episode=1, reward=total_reward, epsilon=1.0, path=env.path
        )

    elapsed = time.time() - t0
    print(f"Ran {steps} steps in {elapsed:.2f}s ({steps / max(elapsed, 1e-9):.1f} steps/s).")
    print(f"Final: total_reward={total_reward:.1f} done={done} "
          f"agent_pos={env.agent_pos} goal_pos={env.goal_pos}")

    if keep_rendering:
        # Save a final-frame screenshot for headless verification (e.g. CI
        # or reviewing over SSH) in addition to the live window.
        pygame.image.save(viz.screen, "visualizer_smoke_test.png")
        print("Saved visualizer_smoke_test.png")
        viz.close()
