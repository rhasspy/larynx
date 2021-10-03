import logging
import math
import typing

import torch
from torch import nn
from torch.nn import functional as F

from glow_tts.attentions import CouplingBlock, Encoder
from glow_tts.config import TrainingConfig
from glow_tts.layers import ActNorm, ConvReluNorm, InvConvNear, LayerNorm
from glow_tts.utils import generate_path, sequence_mask, squeeze, unsqueeze

_LOGGER = logging.getLogger("glow_tts.models")

# -----------------------------------------------------------------------------


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        filter_channels_dp,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        window_size=None,
        block_length=None,
        mean_only=False,
        prenet=False,
        gin_channels=0,
    ):

        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.mean_only = mean_only
        self.prenet = prenet
        self.gin_channels = gin_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        if prenet:
            self.pre = ConvReluNorm(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.5,
            )
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
        )

        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj_w = DurationPredictor(
            hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout
        )

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        if self.prenet:
            x = self.pre(x, x_mask)
        x = self.encoder(x, x_mask)

        if g is not None:
            g_exp = g.expand(-1, -1, x.size(-1))
            x_dp = torch.cat([torch.detach(x), g_exp], 1)
        else:
            x_dp = torch.detach(x)

        x_m = self.proj_m(x) * x_mask
        if not self.mean_only:
            x_logs = self.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)

        logw = self.proj_w(x_dp, x_mask)
        return x_m, x_logs, logw, x_mask


class FlowSpecDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_blocks,
        n_layers,
        p_dropout=0.0,
        n_split=4,
        n_sqz=2,
        sigmoid_scale=False,
        gin_channels=0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for _ in range(n_blocks):
            self.flows.append(ActNorm(channels=in_channels * n_sqz))
            self.flows.append(
                InvConvNear(channels=in_channels * n_sqz, n_split=n_split)
            )
            self.flows.append(
                CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        if self.n_sqz > 1:
            x, x_mask = squeeze(x, x_mask, self.n_sqz)
        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
        if self.n_sqz > 1:
            x, x_mask = unsqueeze(x, x_mask, self.n_sqz)
        return x, logdet_tot

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()


class FlowGenerator(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        hidden_channels: int,
        filter_channels: int,
        filter_channels_dp: int,
        out_channels: int,
        kernel_size: int = 3,
        n_heads: int = 2,
        n_layers_enc: int = 6,
        p_dropout: float = 0.0,
        n_blocks_dec: int = 12,
        kernel_size_dec: int = 5,
        dilation_rate: int = 5,
        n_block_layers: int = 4,
        p_dropout_dec: float = 0.0,
        n_speakers: int = 0,
        gin_channels: int = 0,
        n_split: int = 4,
        n_sqz: int = 1,
        sigmoid_scale: bool = False,
        window_size: typing.Optional[int] = None,
        block_length: typing.Optional[int] = None,
        mean_only: bool = False,
        hidden_channels_enc: typing.Optional[int] = None,
        hidden_channels_dec: typing.Optional[int] = None,
        prenet: bool = False,
    ):

        super().__init__()
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.n_layers_enc = n_layers_enc
        self.p_dropout = p_dropout
        self.n_blocks_dec = n_blocks_dec
        self.kernel_size_dec = kernel_size_dec
        self.dilation_rate = dilation_rate
        self.n_block_layers = n_block_layers
        self.p_dropout_dec = p_dropout_dec
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.window_size = window_size
        self.block_length = block_length
        self.mean_only = mean_only
        self.hidden_channels_enc = hidden_channels_enc
        self.hidden_channels_dec = hidden_channels_dec
        self.prenet = prenet

        self.encoder = TextEncoder(
            n_vocab,
            out_channels,
            hidden_channels_enc or hidden_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_layers_enc,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
            mean_only=mean_only,
            prenet=prenet,
            gin_channels=gin_channels,
        )

        self.decoder = FlowSpecDecoder(
            out_channels,
            hidden_channels_dec or hidden_channels,
            kernel_size_dec,
            dilation_rate,
            n_blocks_dec,
            n_block_layers,
            p_dropout=p_dropout_dec,
            n_split=n_split,
            n_sqz=n_sqz,
            sigmoid_scale=sigmoid_scale,
            gin_channels=gin_channels,
        )

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

    def forward(
        self,
        x,
        x_lengths,
        y=None,
        y_lengths=None,
        g=None,
        noise_scale=1.0,
        length_scale=1.0,
    ):
        if g is not None:
            g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h]

        x_m, x_logs, logw, x_mask = self.encoder(x, x_lengths, g=g)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = None

        y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
        z_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

        # gen = True
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        z_m = torch.matmul(
            attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)
        ).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        z_logs = torch.matmul(
            attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)
        ).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

        z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask
        y, logdet = self.decoder(z, z_mask, g=g, reverse=True)
        return (
            (y, z_m, z_logs, logdet, z_mask),
            (x_m, x_logs, x_mask),
            (attn, logw, logw_),
        )

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = (
            torch.div(y_lengths, self.n_sqz, rounding_mode="trunc") * self.n_sqz
        )
        return y, y_lengths, y_max_length

    def store_inverse(self):
        self.decoder.store_inverse()


# -----------------------------------------------------------------------------

ModelType = FlowGenerator


def setup_model(
    config: TrainingConfig,
    model: typing.Optional[ModelType] = None,
    model_factory=ModelType,
    use_cuda: bool = True,
) -> ModelType:
    if model is None:
        # Create new generator
        model = model_factory(
            n_vocab=config.model.num_symbols,
            hidden_channels=config.model.hidden_channels,
            filter_channels=config.model.filter_channels,
            filter_channels_dp=config.model.filter_channels_dp,
            out_channels=config.audio.mel_channels,
            kernel_size=config.model.kernel_size,
            n_heads=config.model.n_heads,
            n_layers_enc=config.model.n_layers_enc,
            p_dropout=config.model.p_dropout,
            n_blocks_dec=config.model.n_blocks_dec,
            kernel_size_dec=config.model.kernel_size_dec,
            dilation_rate=config.model.dilation_rate,
            n_block_layers=config.model.n_block_layers,
            p_dropout_dec=config.model.p_dropout_dec,
            n_speakers=config.model.n_speakers,
            gin_channels=config.model.gin_channels,
            n_split=config.model.n_split,
            n_sqz=config.model.n_sqz,
            sigmoid_scale=config.model.sigmoid_scale,
            window_size=config.model.window_size,
            block_length=config.model.block_length,
            mean_only=config.model.mean_only,
            hidden_channels_enc=config.model.hidden_channels_enc,
            hidden_channels_dec=config.model.hidden_channels_dec,
            prenet=config.model.prenet,
        )

    if use_cuda:
        model.cuda()

    return model
