import copy
import functools
import math
from typing import Any, Optional, Union, Sequence, Tuple

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_conv1d_layer(
    in_channels: int, out_channels: int, ln_shape: tuple[int], kernel_size: int, pool_size: int, use_pooling: bool, dropout_rate: float
) -> nn.Module:
    layers = [
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size),
        nn.LayerNorm(ln_shape),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
    ]
    if use_pooling:
        layers.append(nn.MaxPool1d(kernel_size=pool_size))
    return nn.Sequential(*layers)

def _get_time_shape(
    time_dim: int, ksize: int, dilation: int = 1, stride: int = 1, padding: int = 0, pool_size: int = 1,
) -> int:
    return int(((time_dim // pool_size + 2 * padding - dilation * (ksize-1) - 1) / stride + 1))

def _get_conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int] = (3, 3),
    padding: tuple[int, int] = (1, 1),
    activation_fn: Any = nn.ReLU,
    max_pool_size: Optional[Union[tuple[int, int], int]] = 3,
    dropout: Optional[float] = 0.3,
    bn: bool = False,
) -> nn.Sequential:
    """Returns a CBAD layer: Convolution, Batch normalization, Activation, and Dropout."""
    layers = [nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding
    )]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation_fn())
    if max_pool_size is not None:
        layers.append(nn.MaxPool2d(max_pool_size))
    if dropout is not None:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


@gin.configurable
class FPM(nn.Module):
    def __init__(self,
                 ssl_shape: tuple[int],
                 conv_channels: Sequence[int] = (32, 32),
                 use_poolings: Sequence[bool] = (True, True),
                 kernel_size: int = 5,
                 pool_size: int = 5,
                 dropout_rate: float = 0.3
                 ):
        super(FPM, self).__init__()
        assert len(conv_channels) == len(use_poolings)

        conv_layer = functools.partial(_get_conv1d_layer, kernel_size=kernel_size, pool_size=pool_size, dropout_rate=dropout_rate)
        projection_channels = 128
        ln_shape = (projection_channels, _get_time_shape(ssl_shape[1], ksize=5))
        layers = [_get_conv1d_layer(ssl_shape[0], projection_channels, ln_shape=ln_shape, kernel_size=5, pool_size=5, use_pooling=False, dropout_rate=0.3)]
        conv_channels = [projection_channels] + list(conv_channels)
        use_poolings_shape = (False,) + use_poolings[:-1]

        for in_channels, out_channels, use_pooling, use_pooling_shape in zip(conv_channels[:-1], conv_channels[1:], use_poolings, use_poolings_shape):
            ln_shape = (out_channels, _get_time_shape(ln_shape[-1], ksize=kernel_size, pool_size=1 if not use_pooling_shape else pool_size))
            layers.append(conv_layer(in_channels, out_channels, ln_shape=ln_shape, use_pooling=use_pooling))

        self._encoder = nn.Sequential(*layers)
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self._encoder(spec)


@gin.configurable
class SPM(nn.Module):

    def __init__(self, bn: bool = True, max_pool_size: int = 3, activation_fn: Any = nn.ReLU):
        super(SPM, self).__init__()
        self.encoder = nn.Sequential(
            _get_conv_layer(1, 32, bn=bn, max_pool_size=None, activation_fn=activation_fn),
            _get_conv_layer(32, 32, bn=bn, max_pool_size=max_pool_size, activation_fn=activation_fn),
            _get_conv_layer(32, 64, bn=bn, max_pool_size=None, activation_fn=activation_fn),
            _get_conv_layer(64, 64, bn=bn, max_pool_size=None, dropout=None, activation_fn=activation_fn),
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # input speech_spectrum shape (batch, 1, max_seq_len, n_features)
        embeddings = self.encoder(spec)  # shape (batch, 64, max_seq_len, n_features)
        embeddings = F.max_pool2d(embeddings, kernel_size=embeddings.size()[2:])
        return embeddings


class MOS_mapping_module(nn.Module):
    def __init__(self, in_dense: int, dense_neurons: Sequence[int] = (64, 32, 1)):
        super(MOS_mapping_module, self).__init__()
        assert len(dense_neurons) == 3
        
        hidden_dense1, hidden_dense2, out_dense = dense_neurons
        self._mu_head = nn.Sequential(
            nn.Linear(in_dense, hidden_dense1),
            nn.ReLU(),
            nn.Linear(hidden_dense1, hidden_dense2),
            nn.ReLU(),
            nn.Linear(hidden_dense2, out_dense),
        )
        self._var_head = nn.Sequential(
            nn.Linear(in_dense, hidden_dense1),
            nn.ReLU(),
            nn.Linear(hidden_dense1, hidden_dense2),
            nn.ReLU(),
            nn.Linear(hidden_dense2, out_dense),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return 2 * self._mu_head(x), 4 * self._var_head(x)


@gin.configurable
class SA_SSL_MOS(nn.Module):

    def __init__(
            self,
            ssl_shape: tuple[int],
            stft_shape: tuple[int],
            dense_neurons: Sequence[int] = (64, 32, 1)
            ):
        super(SA_SSL_MOS, self).__init__()
        self._fpm = FPM(ssl_shape)
        self._spm = SPM()

        self._flatten = nn.Flatten()

        in_dense = self._get_in_dense(ssl_shape, stft_shape)
        self._mos_mapping_module = MOS_mapping_module(in_dense, dense_neurons=dense_neurons)

    def _get_in_dense(self, ssl_shape: tuple[int], stft_shape: tuple[int]) -> int:
        x1 = torch.zeros((1,)+ssl_shape)
        x1 = self._fpm(x1)

        x2 = torch.zeros((1,)+stft_shape).unsqueeze(1)
        x2 = self._spm(x2)
        
        return torch.cat((self._flatten(x1), self._flatten(x2)), dim=1).shape[-1]

    def forward(self, ssl_data: torch.Tensor, stft_data: torch.Tensor) -> torch.Tensor:
        # FPM
        ssl_vector = self._fpm(ssl_data)
        ssl_vector = self._flatten(ssl_vector)

        # SPM
        stft_data = stft_data.unsqueeze(1)
        stft_vector = self._spm(stft_data)
        stft_vector = self._flatten(stft_vector)

        # Concatenate
        vector = torch.cat((ssl_vector, stft_vector), dim=1)

        # MOS mapping
        mean, var = self._mos_mapping_module(vector)

        predictions = torch.cat((mean, var), dim=1)
        return predictions

@gin.configurable
class SSL_Layer_MOS(nn.Module):

    def __init__(
            self,
            ssl_shape: tuple[int],
            dense_neurons: Sequence[int] = (64, 32, 1)
            ):
        super(SSL_Layer_MOS, self).__init__()
        self._fpm = FPM(ssl_shape)

        self._flatten = nn.Flatten()

        in_dense = self._get_in_dense(ssl_shape)
        self._mos_mapping_module = MOS_mapping_module(in_dense, dense_neurons=dense_neurons)

    def _get_in_dense(self, ssl_shape: tuple[int]) -> int:
        x1 = torch.zeros((1,)+ssl_shape)
        x1 = self._fpm(x1)
        return self._flatten(x1).shape[-1]

    def forward(self, ssl_data: torch.Tensor, stft_data: torch.Tensor = torch.empty(1)) -> torch.Tensor:
        # FPM
        ssl_vector = self._fpm(ssl_data)
        ssl_vector = self._flatten(ssl_vector)

        # MOS mapping
        mean, var = self._mos_mapping_module(ssl_vector)

        predictions = torch.cat((mean, var), dim=1)
        return predictions

