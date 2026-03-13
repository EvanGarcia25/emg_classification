# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    ChannelGate,
    CNNRNNEncoder,
    MultiBandRotationInvariantMLP,
    RNNEncoder,
    SpectrogramNorm,
    TDSConvEncoder,
    TemporalTransformerEncoder,
)
from emg2qwerty.transforms import Transform

log = logging.getLogger(__name__)


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        electrode_channels: int = 16,
        l1_lambda: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, C=electrode_channels, freq)
        self.spec_norm = SpectrogramNorm(channels=self.NUM_BANDS * electrode_channels)
        self.channel_gate = ChannelGate(num_channels=electrode_channels)
        self.model = nn.Sequential(
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.spec_norm(inputs)
        x = self.channel_gate(x)
        return self.model(x)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )
        if phase == "train":
            loss = loss + self.hparams.l1_lambda * self.channel_gate.l1_loss()

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def on_train_end(self) -> None:
        self.channel_gate.prune_()
        selected = self.channel_gate.selected_channels()
        log.info(
            f"Selected channels after training: {selected} "
            f"({len(selected)} / {self.hparams.electrode_channels})"
        )

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class RNNCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        num_blocks: int,
        hidden_size: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        electrode_channels: int = 16,
        l1_lambda: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, C=electrode_channels, freq)
        self.spec_norm = SpectrogramNorm(channels=self.NUM_BANDS * electrode_channels)
        self.channel_gate = ChannelGate(num_channels=electrode_channels)
        self.model = nn.Sequential(
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            RNNEncoder(
                num_features=num_features,
                num_blocks=num_blocks,
                hidden_size=hidden_size,
                bidirectional=True,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.spec_norm(inputs)
        x = self.channel_gate(x)
        return self.model(x)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # RNN doesn't change temporal dimension, so no need to adjust T_diff
        # like in TDSConvEncoder. Emissions have same time length as inputs.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )
        if phase == "train":
            loss = loss + self.hparams.l1_lambda * self.channel_gate.l1_loss()

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def on_train_end(self) -> None:
        self.channel_gate.prune_()
        selected = self.channel_gate.selected_channels()
        log.info(
            f"Selected channels after training: {selected} "
            f"({len(selected)} / {self.hparams.electrode_channels})"
        )

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )



class CNNRNNCTCModule(pl.LightningModule):
    """CTC model with a CNN + RNN encoder. Stacked 1D conv blocks extract
    local temporal features; a bidirectional LSTM/GRU captures long-range
    sequence dependencies."""

    NUM_BANDS: ClassVar[int] = 2

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        cnn_channels: Sequence[int],
        cnn_kernel_size: int,
        cnn_dropout: float,
        rnn_type: str,
        rnn_hidden_size: int,
        rnn_num_layers: int,
        rnn_dropout: float,
        rnn_bidirectional: bool,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        electrode_channels: int = 16,
        l1_lambda: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.spec_norm = SpectrogramNorm(
            channels=self.NUM_BANDS * electrode_channels
        )
        self.channel_gate = ChannelGate(num_channels=electrode_channels)
        self.multiband_mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.encoder = CNNRNNEncoder(
            num_features=num_features,
            cnn_channels=cnn_channels,
            cnn_kernel_size=cnn_kernel_size,
            cnn_dropout=cnn_dropout,
            rnn_type=rnn_type,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        self.output_proj = nn.Linear(num_features, charset().num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.spec_norm(inputs)
        x = self.channel_gate(x)
        x = self.multiband_mlp(x)
        x = self.flatten(x)
        x = self.encoder(x)
        x = self.output_proj(x)
        return self.log_softmax(x)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )
        if phase == "train":
            loss = loss + self.hparams.l1_lambda * self.channel_gate.l1_loss()

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def on_train_end(self) -> None:
        self.channel_gate.prune_()
        selected = self.channel_gate.selected_channels()
        log.info(
            f"Selected channels after training: {selected} "
            f"({len(selected)} / {self.hparams.electrode_channels})"
        )

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class TransformerCTCModule(pl.LightningModule):
    """CTC module using the standard front-end (SpectrogramNorm +
    MultiBandRotationInvariantMLP) with a Transformer temporal encoder
    replacing the TDS convolutional or RNN encoder."""

    NUM_BANDS: ClassVar[int] = 2
    # maximum chunk size for test-time inference. Full sessions can be
    # 140k+ timesteps, causing O(t^2) attention collapse. Chunking keeps
    # each window within the trained context length.
    TEST_CHUNK_SIZE: ClassVar[int] = 8000

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        nhead: int,
        num_transformer_layers: int,
        dim_feedforward: int,
        dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        electrode_channels: int = 16,
        l1_lambda: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # model components
        # inputs: (T, N, bands=2, C=electrode_channels, freq)
        self.spec_norm = SpectrogramNorm(
            channels=self.NUM_BANDS * electrode_channels,
        )
        self.channel_gate = ChannelGate(num_channels=electrode_channels)
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.encoder = TemporalTransformerEncoder(
            num_features=num_features,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.spec_norm(inputs)
        x = self.channel_gate(x)
        x = self.mlp(x)
        x = self.flatten(x)
        x = self.encoder(x)
        return self.output_proj(x)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # transformer preserves temporal dimension (no downsampling)
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions, # (T, N, num_classes)
            targets=targets.transpose(0, 1), # (T, N) -> (N, T)
            input_lengths=emission_lengths, # (N,)
            target_lengths=target_lengths, # (N,)
        )
        if phase == "train":
            loss = loss + self.hparams.l1_lambda * self.channel_gate.l1_loss()

        # decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def _chunked_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run forward in overlapping chunks with padding context to avoid O(T^2)
        attention collapse and preserve local context from training."""
        T = inputs.shape[0]
        if T <= self.TEST_CHUNK_SIZE:
            return self.forward(inputs)
        
        # Use same padding as training for context around each chunk
        pre_pad, post_pad = 1800, 200
        pad_size = pre_pad + post_pad
        
        outputs = []
        for start in range(0, T, self.TEST_CHUNK_SIZE):
            # Extract chunk with padding context
            pad_start = max(0, start - pre_pad)
            pad_end = min(T, start + self.TEST_CHUNK_SIZE + post_pad)
            chunk_padded = inputs[pad_start:pad_end]
            
            # Run forward
            chunk_output = self.forward(chunk_padded)
            
            # Extract only the middle part (without padding regions) from output
            output_start = start - pad_start  # frames from pre-padding
            output_end = output_start + self.TEST_CHUNK_SIZE
            
            # Clamp to valid output range
            output_start = max(0, output_start)
            output_end = min(chunk_output.shape[0], output_end)
            
            outputs.append(chunk_output[output_start:output_end])
        
        return torch.cat(outputs, dim=0)

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(
        self, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self._chunked_forward(inputs)

        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics["test_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log("test/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def on_train_end(self) -> None:
        self.channel_gate.prune_()
        selected = self.channel_gate.selected_channels()
        log.info(
            f"Selected channels after training: {selected} "
            f"({len(selected)} / {self.hparams.electrode_channels})"
        )

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )




