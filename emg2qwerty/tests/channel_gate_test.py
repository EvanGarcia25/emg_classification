# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from emg2qwerty.modules import ChannelGate


def test_channel_gate_applies_per_channel_scaling() -> None:
    gate = ChannelGate(num_channels=3, threshold=0.01)
    with torch.no_grad():
        gate.gate.copy_(torch.tensor([1.0, 0.0, 2.0]))

    inputs = torch.ones(2, 1, 2, 3, 4)
    outputs = gate(inputs)

    assert outputs.shape == inputs.shape
    torch.testing.assert_close(outputs[..., 0, :], torch.ones(2, 1, 2, 4))
    torch.testing.assert_close(outputs[..., 1, :], torch.zeros(2, 1, 2, 4))
    torch.testing.assert_close(outputs[..., 2, :], torch.full((2, 1, 2, 4), 2.0))


def test_channel_gate_selected_channels_and_prune() -> None:
    gate = ChannelGate(num_channels=4, threshold=0.01)
    with torch.no_grad():
        gate.gate.copy_(torch.tensor([0.7, 0.009, -0.03, 0.0]))

    assert gate.selected_channels() == [0, 2]

    gate.prune_()
    torch.testing.assert_close(
        gate.gate.detach(),
        torch.tensor([0.7, 0.0, -0.03, 0.0]),
    )
