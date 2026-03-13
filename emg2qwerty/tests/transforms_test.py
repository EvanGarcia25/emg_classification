# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from emg2qwerty.transforms import SelectElectrodeChannels


def test_select_electrode_channels_prefix() -> None:
    inputs = torch.arange(4 * 2 * 16).reshape(4, 2, 16)

    transform = SelectElectrodeChannels(num_channels=6)
    actual = transform(inputs)
    expected = inputs[..., :6]

    assert actual.shape == (4, 2, 6)
    torch.testing.assert_close(actual, expected)


def test_select_electrode_channels_explicit_indices() -> None:
    inputs = torch.arange(3 * 2 * 8).reshape(3, 2, 8)

    transform = SelectElectrodeChannels(channels=[1, 3, 5])
    actual = transform(inputs)
    expected = inputs[..., [1, 3, 5]]

    assert actual.shape == (3, 2, 3)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "num_channels, channels",
    [
        (0, None),
        (4, []),
    ],
)
def test_select_electrode_channels_invalid_init(
    num_channels: int,
    channels: list[int] | None,
) -> None:
    with pytest.raises(ValueError):
        SelectElectrodeChannels(num_channels=num_channels, channels=channels)


def test_select_electrode_channels_invalid_runtime_indices() -> None:
    transform = SelectElectrodeChannels(channels=[0, 9])
    inputs = torch.zeros(2, 2, 8)

    with pytest.raises(ValueError):
        _ = transform(inputs)
