"""agent.py - Experience replay buffer and the DQN agent itself.

See model.py for the CNN architecture. This file wires that network into
a full DQN learner: epsilon-greedy action selection, a replay buffer to
decorrelate training samples, and a target network to stabilize the
bootstrapped Q-learning update. See the module-level explanation given
alongside this file for the reasoning behind experience replay and the
target network - the short version is repeated in each method's docstring
below.
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import LunarDQN


class ReplayBuffer:
    """Fixed-size ring buffer of past transitions, sampled randomly during
    training to decorrelate consecutive updates (see agent.py's module
    docstring for why that matters).
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self.buffer: deque = deque(maxlen=max_size)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition. `deque(maxlen=...)` automatically discards
        the oldest transition once full, so old (possibly stale, from an
        undertrained early policy) experience gets naturally phased out.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random minibatch, returned as stacked numpy arrays
        (one array per field, not a list of tuples) so agent.py can convert
        each straight to a tensor in one call.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def is_ready(self, batch_size: int) -> bool:
        """Whether there's enough experience to draw a full batch from."""
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent: epsilon-greedy policy over a LunarDQN, trained
    with experience replay and a periodically-synced target network.

    Hyperparameters (why these specific values):
        gamma=0.99: discount factor. Close to 1 because the real payoff
            (+500 for reaching the goal) can be hundreds of steps away -
            a low gamma would make the agent short-sighted and effectively
            blind to the goal reward, chasing only the small per-step
            distance shaping instead.
        epsilon=1.0 -> epsilon_min=0.01, decay=0.995: start fully random
            (the network's Q-values are meaningless noise at init, so
            there's nothing worth exploiting yet) and gradually shift
            toward trusting the learned Q-values as they become
            informative. epsilon_min keeps a small permanent floor of
            exploration so the agent never fully stops probing.
        target_update_freq=100 steps: how often the target network is
            resynced from the online network - see the "why target
            network" explanation in this project's design discussion.
        lr=0.0001 (Adam): DQN training is already somewhat unstable
            (bootstrapping + function approximation + off-policy data) -
            a conservative learning rate trades slower convergence for a
            lower risk of the loss diverging.
        buffer_size=100_000 (raised from an initial 10_000): episodes here
            run several hundred steps and the goal reward is rare (single
            digits of episodes out of every hundred reach it), so a small
            buffer acts as a short sliding window - a real training run
            showed every successful transition getting evicted within
            10-25 episodes of occurring, well before the network had a
            chance to repeatedly resample and consolidate learning from
            it, and performance measurably regressed once the buffer had
            gone success-free for a long stretch. 100k gives rare positive
            transitions a much longer runway to actually get learned from.
    """

    def __init__(
        self,
        input_size: int = 30,
        num_actions: int = 4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        target_update_freq: int = 100,
        batch_size: int = 32,
        buffer_size: int = 100_000,
        lr: float = 0.0001,
        device: Optional[str] = None,
    ) -> None:
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.online_network = LunarDQN(input_size, num_actions).to(self.device)
        self.target_network = LunarDQN(input_size, num_actions).to(self.device)
        # Target starts as an exact copy of online - both networks should
        # agree at step 0, they only diverge as online gets trained.
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()  # target network is never trained directly

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)
        # Huber loss (SmoothL1), not MSE: our TD targets (reward + gamma *
        # next_q) legitimately range roughly -1000 to +500 given this env's
        # reward scale and up-to-1000-step horizon. Squaring errors of that
        # magnitude (as MSE does) produces huge, destabilizing gradients
        # early in training when the bootstrapped estimate is still very
        # wrong - this is exactly the instability the original DeepMind
        # Atari DQN paper (which this architecture follows, see model.py)
        # addresses with Huber loss instead of MSE: it's quadratic for
        # small errors (same behavior as MSE near convergence) but linear
        # for large ones, so one badly-wrong bootstrapped target can't
        # blow up the gradient the way it can under MSE.
        self.loss_fn = nn.SmoothL1Loss()
        self.max_grad_norm = 10.0  # paired with Huber loss - see train_step()

        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.train_steps = 0  # counts train_step() calls, drives target sync + epsilon decay

    def choose_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection.

        Args:
            state: (input_size, input_size) normalized local view.

        Returns:
            Integer action in [0, num_actions).
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        # eval() mode: BatchNorm should normalize using its accumulated
        # running statistics here, not statistics computed from this one
        # single observed state - the latter is what train() mode would do
        # and would let acting-time samples silently perturb the running
        # stats that train_step()'s batched updates are supposed to own.
        # Restored to train() immediately after so train_step() is unaffected.
        self.online_network.eval()
        with torch.no_grad():
            state_tensor = (
                torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
            )  # (H, W) -> (1, 1, H, W): add channel dim, then batch dim
            q_values = self.online_network(state_tensor)
        self.online_network.train()
        return int(torch.argmax(q_values, dim=1).item())

    def train_step(self) -> Optional[float]:
        """Sample a batch from the replay buffer and perform one gradient
        update on the online network.

        Returns:
            The scalar loss value, or None if the buffer doesn't have
            enough transitions yet to sample a full batch.
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        self.online_network.train()  # defensive: BatchNorm must use batch stats here, not eval()'s running stats

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # (batch, H, W) -> (batch, 1, H, W): add the channel dimension the
        # CNN expects.
        states_t = torch.from_numpy(states).float().unsqueeze(1).to(self.device)
        actions_t = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().unsqueeze(1).to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        # Current Q: the value the online network currently assigns to the
        # action that was *actually taken* in each stored transition.
        # gather(1, actions_t) picks out one Q-value per row according to
        # the action index.
        current_q = self.online_network(states_t).gather(1, actions_t).squeeze(1)

        # Target Q: reward + gamma * best possible next-state value,
        # computed from the frozen target network (not online) - this is
        # the piece that keeps the bootstrapped target stable, see the
        # module docstring. (1 - dones_t) zeroes out the future-value term
        # for terminal transitions - there is no "next state" to bootstrap
        # from once an episode has ended.
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip the gradient norm (paired with Huber loss above) as a second,
        # complementary guard against destabilizing updates: Huber loss
        # bounds how much a single large TD error can contribute to the
        # loss, but doesn't bound the resulting gradient norm across all
        # parameters combined - clipping that directly is standard practice
        # in DQN implementations for exactly this reason.
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.train_steps += 1

        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Decay epsilon by one step. Call this once per completed EPISODE,
        not once per environment/train step: a ~1000-step episode calling
        this every step would collapse epsilon to epsilon_min well before
        the episode even ends, killing exploration long before the
        curriculum reaches medium/hard difficulty. Once per episode gives
        a decay schedule measured in hundreds of episodes instead.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Save the online network's weights plus training state needed to
        resume (epsilon, step count). The target network is NOT saved
        separately - on load() it's resynced from the online weights,
        since it's always meant to be a (lagged) copy of online anyway.

        Recreates path's parent directory if it's missing - torch.save()
        does not do this itself and errors instead. A real training run
        crashed exactly this way after its output directory was deleted
        out from under it mid-run; a multi-hour run shouldn't be lost to
        something this recoverable.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.online_network.state_dict(),
                "epsilon": self.epsilon,
                "train_steps": self.train_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load weights + training state, and sync the target network to
        match so both networks agree immediately after loading.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.online_network.load_state_dict(checkpoint["model_state_dict"])
        self.target_network.load_state_dict(checkpoint["model_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.train_steps = checkpoint["train_steps"]


if __name__ == "__main__":
    # Smoke tests on random dummy data - no real environment yet, just
    # verifying the mechanics: does train_step actually reduce loss on a
    # fixed batch (sanity check that gradients flow correctly), does the
    # target network sync on schedule, does epsilon decay, does save/load
    # round-trip correctly.
    print("=== ReplayBuffer ===")
    buffer = ReplayBuffer(max_size=100)
    for i in range(50):
        state = np.random.rand(30, 30).astype(np.float32)
        next_state = np.random.rand(30, 30).astype(np.float32)
        buffer.add(state, random.randint(0, 3), random.uniform(-1, 1), next_state, False)
    print(f"buffer size: {len(buffer)}")
    print(f"is_ready(32): {buffer.is_ready(32)}")
    s, a, r, ns, d = buffer.sample(8)
    print(f"sample shapes: states={s.shape} actions={a.shape} rewards={r.shape} "
          f"next_states={ns.shape} dones={d.shape}")

    print("\n=== DQNAgent: toy overfit test ===")
    # Fill a buffer with random transitions, then hammer train_step() on it
    # repeatedly. This isn't realistic RL (same buffer, no new experience)
    # but it's a standard sanity check: if the loss doesn't trend down at
    # all here, something is wrong with the gradient computation itself,
    # independent of anything about the environment.
    agent = DQNAgent(input_size=30, num_actions=4, target_update_freq=20, batch_size=32)
    for i in range(200):
        state = np.random.rand(30, 30).astype(np.float32)
        next_state = np.random.rand(30, 30).astype(np.float32)
        action = random.randint(0, 3)
        reward = random.uniform(-1, 1)
        done = random.random() < 0.05
        agent.replay_buffer.add(state, action, reward, next_state, done)

    losses = []
    online_before = agent.online_network.conv[0].weight.clone()
    target_before = agent.target_network.conv[0].weight.clone()
    # Simulate 300 train_step() calls across ~15 "episodes" of 20 steps
    # each, decaying epsilon once per simulated episode (as train.py does)
    # rather than once per step, to verify decay_epsilon() works without
    # reproducing the per-step collapse bug this was split out to fix.
    steps_per_episode = 20
    for step in range(300):
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)
        if (step + 1) % steps_per_episode == 0:
            agent.decay_epsilon()

    print(f"train_step() calls that returned a loss: {len(losses)}")
    print(f"first 5 losses:  {[round(l, 4) for l in losses[:5]]}")
    print(f"last 5 losses:   {[round(l, 4) for l in losses[-5:]]}")
    early_avg = float(np.mean(losses[:20]))
    late_avg = float(np.mean(losses[-20:]))
    print(f"avg loss (first 20): {early_avg:.4f}   avg loss (last 20): {late_avg:.4f}")

    online_after = agent.online_network.conv[0].weight
    target_after = agent.target_network.conv[0].weight
    print(f"\nonline network weights changed: {not torch.equal(online_before, online_after)}")
    print(f"target network weights changed: {not torch.equal(target_before, target_after)} "
          f"(should be True - {300 // agent.target_update_freq} syncs expected "
          f"at target_update_freq={agent.target_update_freq})")
    print(f"train_steps counter: {agent.train_steps}")
    print(f"epsilon after {300 // steps_per_episode} simulated episodes: {agent.epsilon:.4f} "
          f"(started at 1.0, decay=0.995 per episode)")

    print("\n=== save/load round trip ===")
    q_before = (
        agent.online_network(torch.rand(1, 1, 30, 30, device=agent.device))
        .detach()
        .cpu()
        .numpy()
    )
    agent.save("/tmp/agent_test_checkpoint.pt")

    fresh_agent = DQNAgent(input_size=30, num_actions=4)
    print(f"fresh agent epsilon before load: {fresh_agent.epsilon}")
    fresh_agent.load("/tmp/agent_test_checkpoint.pt")
    print(f"fresh agent epsilon after load: {fresh_agent.epsilon:.4f} "
          f"(should match saved agent: {agent.epsilon:.4f})")
    print(f"fresh agent train_steps after load: {fresh_agent.train_steps} "
          f"(should match: {agent.train_steps})")

    online_match = all(
        torch.equal(p1, p2)
        for p1, p2 in zip(agent.online_network.parameters(), fresh_agent.online_network.parameters())
    )
    target_match = all(
        torch.equal(p1, p2)
        for p1, p2 in zip(agent.target_network.parameters(), fresh_agent.target_network.parameters())
    )
    print(f"loaded online weights match saved: {online_match}")
    print(f"loaded target weights match saved online (resynced): {target_match}")