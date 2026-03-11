# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class RNNBlock(nn.Module):
    """An RNN block using LSTM/GRU with layer normalization and skip connections.
    Maintains similar structure to TDSConv2dBlock with skip connections and
    layer normalization for stable training.

    Args:
        num_features (int): Input/output feature dimension
        hidden_size (int): Hidden size of the RNN
        num_layers (int): Number of RNN layers (default: 1)
        rnn_type (str): Type of RNN - 'lstm' or 'gru' (default: 'lstm')
        bidirectional (bool): Whether to use bidirectional RNN (default: True)
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # Create RNN
        rnn_class = nn.LSTM if rnn_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_class(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=False,  # We use (T, N, C) format
        )

        # Project RNN output back to num_features for skip connection
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        self.output_proj = nn.Linear(rnn_output_size, num_features)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        x = inputs

        # RNN forward
        rnn_out, _ = self.rnn(x)  # (T, N, hidden_size * num_directions)

        # Project back to num_features
        x = self.output_proj(rnn_out)  # (T, N, num_features)

        # Skip connection
        x = x + inputs

        # Layer norm
        return self.layer_norm(x)  # (T, N, num_features)


class RNNEncoder(nn.Module):
    """An RNN-based encoder composing a sequence of stacked RNN blocks.
    Similar in structure to TDSConvEncoder but uses RNN layers instead of
    temporal convolutions. Each block consists of a bidirectional RNN with
    skip connections and layer normalization.

    Args:
        num_features (int): Input/output feature dimension for all blocks
        num_blocks (int): Number of RNN blocks to stack (default: 4)
        hidden_size (int): Hidden size for each RNN block (default: 256)
        rnn_type (str): Type of RNN - 'lstm' or 'gru' (default: 'lstm')
        bidirectional (bool): Whether to use bidirectional RNN (default: True)
    """

    def __init__(
        self,
        num_features: int,
        num_blocks: int = 4,
        hidden_size: int = 256,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        rnn_blocks: list[nn.Module] = []
        for _ in range(num_blocks):
            rnn_blocks.append(
                RNNBlock(
                    num_features=num_features,
                    hidden_size=hidden_size,
                    num_layers=1,
                    rnn_type=rnn_type,
                    bidirectional=bidirectional,
                )
            )

        self.rnn_blocks = nn.Sequential(*rnn_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.rnn_blocks(inputs)  # (T, N, num_features)


class CNNRNNEncoder(nn.Module):
    """A CNN + RNN encoder. Stacked 1-D conv blocks extract local temporal
    features followed by a bidirectional RNN for long-range dependencies.

    Inputs: (T, N, num_features)
    Returns: (T, N, num_features)

    Args:
        num_features (int): Input/output feature dimension.
        cnn_channels (list): Output channels for each 1-D conv block.
        cnn_kernel_size (int): Kernel size for each 1-D conv.
        cnn_dropout (float): Dropout after each conv block.
        rnn_type (str): 'lstm' or 'gru'.
        rnn_hidden_size (int): Hidden size of the RNN.
        rnn_num_layers (int): Number of RNN layers.
        rnn_dropout (float): Dropout between RNN layers.
        rnn_bidirectional (bool): Whether to use bidirectional RNN.
    """

    def __init__(
        self,
        num_features: int,
        cnn_channels: Sequence[int] = (128, 128),
        cnn_kernel_size: int = 5,
        cnn_dropout: float = 0.1,
        rnn_type: str = "lstm",
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0.1,
        rnn_bidirectional: bool = True,
    ) -> None:
        super().__init__()

        # Build CNN blocks: Conv1d + ReLU + Dropout
        cnn_layers: list[nn.Module] = []
        in_ch = num_features
        for out_ch in cnn_channels:
            cnn_layers.extend(
                [
                    nn.Conv1d(
                        in_ch, out_ch, cnn_kernel_size,
                        padding=cnn_kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.Dropout(cnn_dropout),
                ]
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Project CNN output back to num_features if needed
        self.cnn_proj = (
            nn.Linear(in_ch, num_features) if in_ch != num_features else nn.Identity()
        )

        # RNN
        rnn_class = nn.LSTM if rnn_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_class(
            input_size=num_features,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
            bidirectional=rnn_bidirectional,
            batch_first=False,
        )

        rnn_output_size = rnn_hidden_size * (2 if rnn_bidirectional else 1)
        self.rnn_proj = nn.Linear(rnn_output_size, num_features)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        T, N, C = inputs.shape

        # CNN expects (N, C, T)
        x = inputs.permute(1, 2, 0)  # (N, C, T)
        x = self.cnn(x)  # (N, cnn_out, T)
        x = x.permute(2, 0, 1)  # (T, N, cnn_out)
        x = self.cnn_proj(x)  # (T, N, num_features)

        # RNN
        rnn_out, _ = self.rnn(x)  # (T, N, rnn_output_size)
        x = self.rnn_proj(rnn_out)  # (T, N, num_features)

        # Skip connection + layer norm
        x = x + inputs
        return self.layer_norm(x)  # (T, N, num_features)


### Evan's transformer / ViT implementation

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in 'Attention Is All You Need'.

    Adds position-dependent sinusoidal signals to encode temporal ordering
    for inputs of shape (T, N, d_model).

    Args:
        d_model (int): Embedding/feature dimension.
        max_len (int): Maximum sequence length. (default: 10000)
        dropout (float): Dropout probability. (default: 0.1)
    """

    def __init__(
        self, d_model: int, max_len: int = 10000, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        self.register_buffer("pe", self._build_pe(max_len, d_model))

    @staticmethod
    def _build_pe(length: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(length, 1, d_model)
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, d_model)
        T = inputs.shape[0]
        if T > self.pe.shape[0]:
            self.pe = self._build_pe(T, self.d_model).to(self.pe.device)
        return self.dropout(inputs + self.pe[:T])


class EMGPatchEmbedding(nn.Module):
    """ViT-style 2D patch embedding for EMG spectrograms.

    For each band, treats per-timestep (electrode_channels, freq_bins) data
    as a 2D spatial input, partitions into non-overlapping patches via Conv2d,
    and projects each patch to the embedding dimension. Learned band embeddings
    distinguish left/right bands and learned positional embeddings encode
    spatial patch positions.

    Inputs: (T, N, num_bands, electrode_channels, freq_bins)
    Returns: (T, N, total_patches, d_model)

    Args:
        electrode_channels (int): Electrode channels per band. (default: 16)
        freq_bins (int): Frequency bins from spectrogram. (default: 33)
        patch_size (tuple): (patch_h, patch_w) for electrode and freq dims.
        d_model (int): Embedding dimension per patch. (default: 256)
        num_bands (int): Number of EMG bands. (default: 2)
        dropout (float): Dropout probability. (default: 0.1)
    """

    def __init__(
        self,
        electrode_channels: int = 16,
        freq_bins: int = 33,
        patch_size: Sequence[int] = (4, 11),
        d_model: int = 256,
        num_bands: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.electrode_channels = electrode_channels
        self.freq_bins = freq_bins
        self.patch_size = tuple(patch_size)
        self.num_bands = num_bands
        self.d_model = d_model

        # Pad to ensure even patch tiling
        self.pad_h = (
            (self.patch_size[0] - electrode_channels % self.patch_size[0])
            % self.patch_size[0]
        )
        self.pad_w = (
            (self.patch_size[1] - freq_bins % self.patch_size[1])
            % self.patch_size[1]
        )
        eff_h = electrode_channels + self.pad_h
        eff_w = freq_bins + self.pad_w

        self.num_patches_per_band = (eff_h // self.patch_size[0]) * (
            eff_w // self.patch_size[1]
        )
        self.total_patches = self.num_patches_per_band * num_bands

        # Per-band Conv2d patch projection
        self.patch_projs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=d_model,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                )
                for _ in range(num_bands)
            ]
        )

        # Learned spatial positional embedding
        self.patch_pos_embed = nn.Parameter(
            torch.randn(1, self.total_patches, d_model) * 0.02
        )
        # Band embedding to distinguish left vs right
        self.band_embed = nn.Parameter(
            torch.randn(1, num_bands, 1, d_model) * 0.02
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape

        patches_per_band = []
        for b in range(self.num_bands):
            x = inputs[:, :, b].reshape(T * N, 1, C, freq)  # (T*N, 1, C, freq)
            if self.pad_h > 0 or self.pad_w > 0:
                x = nn.functional.pad(x, (0, self.pad_w, 0, self.pad_h))
            x = self.patch_projs[b](x)  # (T*N, d_model, H', W')
            x = x.flatten(2).transpose(1, 2)  # (T*N, patches_per_band, d_model)
            x = x + self.band_embed[0, b]
            patches_per_band.append(x)

        # (T*N, total_patches, d_model)
        x = torch.cat(patches_per_band, dim=1)
        x = x + self.patch_pos_embed
        x = self.norm(x)
        x = self.dropout_layer(x)
        return x.reshape(T, N, self.total_patches, self.d_model)


class SpatialTransformerEncoder(nn.Module):
    """Transformer self-attention across spatial patches per timestep.

    Applies a Transformer encoder to attend over electrode-frequency
    patches within each timestep, aggregating via a CLS token or mean
    pooling into a single feature vector per timestep.

    Inputs: (T, N, num_patches, d_model)
    Returns: (T, N, d_model)

    Args:
        d_model (int): Patch feature dimension. (default: 256)
        nhead (int): Number of attention heads. (default: 8)
        num_layers (int): Transformer encoder layers. (default: 2)
        dim_feedforward (int): FFN hidden dimension. (default: 512)
        dropout (float): Dropout probability. (default: 0.1)
        use_cls_token (bool): Use CLS token (True) or mean pool (False).
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_cls_token = use_cls_token

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, P, D = inputs.shape
        x = inputs.reshape(T * N, P, D)  # (T*N, patches, d_model)

        if self.use_cls_token:
            cls = self.cls_token.expand(T * N, -1, -1)
            x = torch.cat([cls, x], dim=1)  # (T*N, 1+P, D)

        x = self.transformer(x)

        if self.use_cls_token:
            x = x[:, 0]  # CLS output: (T*N, D)
        else:
            x = x.mean(dim=1)  # Mean pool: (T*N, D)

        x = self.norm(x)
        return x.reshape(T, N, D)  # (T, N, d_model)


class TemporalTransformerEncoder(nn.Module):
    """Transformer encoder for the temporal dimension.

    Drop-in replacement for ``TDSConvEncoder`` and ``RNNEncoder``.
    Applies multi-head self-attention and feed-forward layers across the
    time axis with sinusoidal positional encoding.

    Inputs: (T, N, num_features)
    Returns: (T, N, num_features)

    Args:
        num_features (int): Feature dimension.
        nhead (int): Attention heads (must divide num_features). (default: 8)
        num_layers (int): Transformer encoder layers. (default: 4)
        dim_feedforward (int): FFN hidden dimension. (default: 1024)
        dropout (float): Dropout probability. (default: 0.1)
        max_len (int): Max sequence length for positional encoding.
            (default: 10000)
    """

    def __init__(
        self,
        num_features: int,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 10000,
    ) -> None:
        super().__init__()
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=num_features, max_len=max_len, dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # (T, N, D) format
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(inputs)  # (T, N, num_features)
        x = self.transformer(x)
        return self.norm(x)  # (T, N, num_features)
