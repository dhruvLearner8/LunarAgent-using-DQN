"""model.py - The CNN that estimates Q-values for the lunar rover.

Why a CNN instead of a flat Q-table or a plain MLP on raw pixels:

A tabular Q-learning approach needs one entry per (state, action) pair. Our
state is a 30x30 grayscale patch - even at just 2 brightness levels per
pixel that's 2^900 possible states, hopelessly larger than any table could
hold, let alone visit during training. A plain fully-connected network on
the flattened 900 pixels would technically fit, but it has no notion of
*spatial structure* - it would have to independently (re)learn that "a dark
blob near the top of the patch" and "a dark blob near the bottom" are the
same kind of hazard, from scratch, for every possible position. A CNN's
convolutional filters are applied identically across the whole image, so a
"crater edge" detector learned in one region automatically applies
everywhere else too - the same trick that makes CNNs effective for any
image task, here reused to make a spatial danger-detector.

Architecture history: the original version here was 2 conv layers
(k=4,s=2 then k=3,s=1) followed by a flatten into a single
Linear(9216, 256) layer - 2.36M of the network's 2.38M total parameters
lived in that one FC layer. The problem: those 2 conv layers only reach an
8x8 effective receptive field on the 30x30 input, meaning information from
one side of the local view could never be related to information from the
other side by convolution at all - "crater on my left, clear path on my
right" had to be reconstructed, if at all, by the giant unstructured FC
layer trying to learn arbitrary pixel-position-specific combinations from
a flattened, spatially-scrambled feature grid. That's a bad trade: almost
all the model's capacity sitting in the part of the network with the least
spatial reasoning ability.

This version fixes that with three changes, in order of importance:
  1. A deeper conv stack (4 layers, stride-2 then stride-1) reaching a
     23x23 effective receptive field - each unit in the final feature map
     now sees most of the 30x30 patch, so relating "danger here" to
     "safety there" is something convolution can actually do.
  2. GroupNorm(1, C) after every conv layer - equivalent to LayerNorm for
     a conv net (normalizes each sample's own channel+spatial activations,
     with no dependency on the rest of the batch). BatchNorm was tried
     first here and made things worse: it normalizes using statistics
     computed from whatever batch happens to be sampled, and in DQN that
     batch is drawn from a replay buffer whose contents keep shifting as
     the policy improves, while the bootstrapped targets are *also*
     shifting as the online network trains - two compounding sources of
     non-stationarity that BatchNorm's batch-statistics assumption doesn't
     handle well. A real run showed loss climbing 770x over training
     (0.135 -> 104) with BatchNorm despite Huber loss and gradient
     clipping already in place - the same instability signature as the
     original MSE-loss bug, just reintroduced through a different
     mechanism. This is also why the original DeepMind Atari DQN paper
     does not use BatchNorm at all. GroupNorm(1, C) keeps the benefit of
     normalized activations (still counters unstable early TD targets)
     without depending on batch composition.
  3. Global average pooling instead of flatten-into-FC: collapsing each of
     the 64 final-layer channels to a single number (its spatial average)
     forces every channel to encode something meaningful about the whole
     patch, rather than a positional flatten needing a separate weight per
     pixel-per-channel-per-output. This alone cuts total parameters by
     roughly 20x (see the __main__ smoke test for the exact count) while
     the deeper conv stack *increases* effective spatial reasoning - a
     much better trade than the original's "huge FC layer, tiny receptive
     field." It also means the flatten size is always exactly the last
     conv layer's channel count, independent of input_size - no more
     dummy-forward-pass trick needed to discover it.

Dueling head: splits the final Q-value into a state-value stream V(s) (how
good is this situation, independent of action) and an advantage stream
A(s,a) (how much better is each action than the others), recombined as
Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)). This is a good fit here because
in most patches all 4 actions are probably roughly equally fine - only when
a crater is immediately adjacent does the choice of action actually matter
much - and dueling architectures learn V(s) from every single transition
(regardless of which action was taken) instead of only from transitions
where that specific action happened to be sampled, which is more
data-efficient exactly in this "most states don't discriminate much
between actions" regime.

This mirrors DeepMind's original Atari DQN architecture (small conv layers
-> flatten -> fully connected layers -> one output per action) but updated
with the standard modern refinements above.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LunarDQN(nn.Module):
    """CNN that maps a local terrain view to a Q-value for each action.

    Architecture:
        Conv2d(1, 32, k=3, s=2, pad=1) -> BatchNorm -> ReLU   # 30x30 -> 15x15
        Conv2d(32, 64, k=3, s=2, pad=1) -> BatchNorm -> ReLU  # 15x15 -> 8x8
        Conv2d(64, 64, k=3, s=1, pad=1) -> BatchNorm -> ReLU  # 8x8   -> 8x8
        Conv2d(64, 64, k=3, s=1, pad=1) -> BatchNorm -> ReLU  # 8x8   -> 8x8
        GlobalAveragePool -> 64-dim vector
        Dueling head:
            value stream:     Linear(64,128) -> ReLU -> Linear(128,1)
            advantage stream: Linear(64,128) -> ReLU -> Linear(128,num_actions)
            Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))

    Args:
        input_size: Height/width of the (square) input state (30 for the
            30x30 local view defined in environment.py). Unlike the
            previous flatten-based version, this no longer affects the
            head's parameter count - GAP always yields a fixed 64-dim
            vector regardless of input_size.
        num_actions: Size of the discrete action space (4: up/down/left/right).
    """

    def __init__(self, input_size: int = 30, num_actions: int = 4) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, 32),  # LayerNorm-for-conv: see module docstring
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (batch, 64, H, W) -> (batch, 64, 1, 1)

        feature_size = 64  # = last conv layer's out_channels, fixed by GAP

        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of states.

        Args:
            x: (batch, 1, input_size, input_size) float tensor.

        Returns:
            (batch, num_actions) tensor of Q-values.
        """
        features = self.conv(x)
        features = self.global_pool(features).flatten(1)  # (batch, 64, 1, 1) -> (batch, 64)

        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, num_actions)

        # Subtracting the mean advantage (rather than e.g. the max) is the
        # standard dueling-DQN identifiability fix: without some
        # normalization, V and A could each drift by an arbitrary constant
        # that cancels out in Q = V + A, making them individually
        # meaningless even though Q itself is still correct. Forcing the
        # advantages to average to zero pins both streams down uniquely.
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


if __name__ == "__main__":
    # Smoke test: verify shapes end-to-end, report the parameter-count
    # reduction from the previous architecture, and - directly regression-
    # testing the exact failure mode found in this project - confirm the
    # network actually responds differently to different inputs rather
    # than collapsing to an input-invariant output.
    model = LunarDQN(input_size=30, num_actions=4)
    print(model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {num_params:,} (previous architecture: 2,383,236)")

    for batch_size in [1, 8, 32]:
        dummy_states = torch.rand(batch_size, 1, 30, 30)
        q_values = model(dummy_states)
        expected_shape = (batch_size, 4)
        assert q_values.shape == expected_shape, (
            f"Shape mismatch: got {tuple(q_values.shape)}, expected {expected_shape}"
        )
        print(f"batch_size={batch_size}: input {tuple(dummy_states.shape)} "
              f"-> output {tuple(q_values.shape)}  OK")

    print("\n=== distinctness check (regression test for the input-collapse bug) ===")
    model.eval()  # BatchNorm needs eval mode to use running stats, not batch stats, for single-sample calls
    with torch.no_grad():
        blank = model(torch.zeros(1, 1, 30, 30))
        bright = model(torch.ones(1, 1, 30, 30))
        rand_a = model(torch.rand(1, 1, 30, 30))
        rand_b = model(torch.rand(1, 1, 30, 30))
    print(f"blank input:  Q = {blank.numpy().round(3)}")
    print(f"bright input: Q = {bright.numpy().round(3)}")
    print(f"random input A: Q = {rand_a.numpy().round(3)}")
    print(f"random input B: Q = {rand_b.numpy().round(3)}")
    all_different = len({tuple(q.numpy().round(4).tolist()[0]) for q in [blank, bright, rand_a, rand_b]}) == 4
    print(f"all four outputs distinct: {all_different} (should be True at random init)")

    sample_q = model(torch.rand(1, 1, 30, 30))
    best_action = torch.argmax(sample_q, dim=1).item()
    print(f"\nSample Q-values: {sample_q.detach().numpy().round(4)}")
    print(f"Best action (argmax): {best_action} ({['up','down','left','right'][best_action]})")
