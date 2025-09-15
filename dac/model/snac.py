import json
import math
import os
from typing import List, Tuple, Dict, Union

import numpy as np
import torch
from torch import nn

from .layers import Encoder, Decoder
from .vq import ResidualVectorQuantize

from audiotools.ml import BaseModel
from .base import CodecMixin

class SNAC(BaseModel, CodecMixin):
    def __init__(
        self,
        sampling_rate=24000,
        encoder_dim=64,
        encoder_rates=[3, 3, 7, 7],
        latent_dim=None,
        decoder_dim=1536,
        decoder_rates=[7, 7, 3, 3],
        codebook_size=4096,
        codebook_dim=8,
        vq_strides=[8, 4, 2, 1],
        noise=True,
        depthwise=True,
        commitment_weight=0.25,
        codebook_weight=1.0,
        ema_decay=0.99,
        use_ema=True,
        attn_window_size=None, # just for hubert compatibility
        apply_rotation_trick=True,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        
        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)

        self.n_codebooks = len(vq_strides)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.vq_strides = vq_strides

        self.encoder = Encoder(
            d_model=encoder_dim,
            strides=encoder_rates,
            depthwise=depthwise,
        )
        
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=3,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=0.5,
            vq_strides=vq_strides,
            apply_rotation_trick=apply_rotation_trick,
        )

        self.decoder = Decoder(
            input_channel=latent_dim,
            channels=decoder_dim,
            rates=decoder_rates,
            noise=noise,
            depthwise=depthwise,
        )

    def preprocess(self, audio_data):
        length = audio_data.shape[-1]
        lcm = math.lcm(self.vq_strides[0], 1)
        pad_to = self.hop_length * lcm
        right_pad = math.ceil(length / pad_to) * pad_to - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data

    # The forward signature now mirrors `dac.model.DAC.forward` so that existing
    # training/validation loops can use SNAC as a drop-in replacement.
    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int | None = None,
        n_quantizers: int | None = None,
    ) -> dict:
        """Forward pass compatible with scripts/train.py.

        Parameters
        ----------
        audio_data : torch.Tensor [B x 1 x T]
            Input waveform.
        sample_rate : int | None
            Ignored but kept for API compatibility (trainer always passes it).
        n_quantizers : int | None
            How many quantisers to use; forwarded to RVQ.

        Returns
        -------
        dict
            Same keys as DAC.forward so that the trainer does not need to
            change. "latents" is the pre-quantisation encoder output.
        """
        length = audio_data.shape[-1]

        # Preprocess/pad so that hop_length divides length.
        audio_data = self.preprocess(audio_data)

        # Encode ↦ Quantise ↦ Decode
        z_q, codes, latents, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers=n_quantizers
        )

        audio_hat = self.decoder(z_q)

        return {
            "audio": audio_hat[..., :length],
            "z": z_q,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode audio and return representations matching DAC.encode."""
        audio_data = self.preprocess(audio_data)
        latents = self.encoder(audio_data)

        z_q, codes, commitment_loss, codebook_loss = self.quantizer(
            latents, n_quantizers=n_quantizers
        )
        
        return z_q, codes, latents, commitment_loss, codebook_loss

    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        z_q = self.quantizer.from_codes(codes)
        audio_hat = self.decoder(z_q)
        return audio_hat

    @property
    def device(self):
        return next(self.parameters()).device

    def get_codebook_usage_stats(self) -> Dict[str, float]:
        """Get codebook usage statistics from all quantizers"""
        return self.quantizer.get_usage_stats()

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(**config)
        return model

    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        from huggingface_hub import hf_hub_download

        if not os.path.isdir(repo_id):
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json", **kwargs)
            model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", **kwargs)
            model = cls.from_config(config_path)
            state_dict = torch.load(model_path, map_location="cpu")
        else:
            model = cls.from_config(os.path.join(repo_id, "config.json"))
            state_dict = torch.load(os.path.join(repo_id, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model


if __name__ == "__main__":
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    model.eval()
    model.to("cuda")
    model.forward(torch.randn(1, 1, 24000).to("cuda"))