from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layers import WNConv1d


class VectorQuantize(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, stride: int = 1, apply_rotation_trick: bool = True):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.stride = stride
        self.apply_rotation_trick = apply_rotation_trick

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        if self.stride > 1:
            z = F.avg_pool1d(z, self.stride, stride=self.stride)

        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        # Apply rotation trick if enabled
        if self.apply_rotation_trick:
            # Rearrange to (B, T, D) for easier computation
            z_e_rot = rearrange(z_e, "b d t -> b t d")
            z_q_rot = rearrange(z_q, "b d t -> b t d")
            
            with torch.no_grad():
                # Normalize vectors for computing rotation
                e_norm = F.normalize(z_e_rot.detach(), dim=-1)
                q_norm = F.normalize(z_q_rot.detach(), dim=-1)
                
                # Compute r = (e + q)/||e + q|| for Householder reflection
                r = (e_norm + q_norm)
                r = F.normalize(r, dim=-1)
                
                # Compute rotation matrix R = I - 2rr^T + 2qe^T
                B, T, D = z_e_rot.shape
                I = torch.eye(D, device=z_e_rot.device).expand(B, T, D, D)
                rrt = torch.einsum('bti,btj->btij', r, r)
                qet = torch.einsum('bti,btj->btij', q_norm, e_norm)
                R = I - 2 * rrt + 2 * qet

                # Scale factor to preserve norms
                scaling = (z_q_rot.norm(dim=-1) / z_e_rot.norm(dim=-1).clamp(min=1e-8)).unsqueeze(-1)

            # Apply rotation and scaling as constants during backprop
            z_q_rotated = scaling * torch.einsum('btij,btj->bti', R, z_e_rot)
            
            # Rearrange back to (B, D, T)
            z_q = rearrange(z_q_rotated, "b t d -> b d t")
        else:
            # Standard straight-through estimator
            z_q = z_e + (z_q - z_e).detach()

        z_q = self.out_proj(z_q)
        if self.stride > 1:
            z_q = F.interpolate(z_q, scale_factor=self.stride, mode="linear")

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.0,
        vq_strides: list[int] = [1, 1, 1, 1],
        apply_rotation_trick: bool = True,
    ):
        super().__init__()

        assert len(vq_strides) == n_codebooks, "Number of codebooks and strides must match"

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [VectorQuantize(input_dim, codebook_size, codebook_dim, vq_strides[i], apply_rotation_trick) for i in range(self.n_codebooks)]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        z_q = 0
        residual = z

        commitment_loss = 0
        codebook_loss = 0

        codes = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks

        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i in range(self.n_codebooks):
            quantizer = self.quantizers[i]
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)

            mask = torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codes.append(indices_i)

        return z_q, codes, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0

        for i in range(self.n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[i])
            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q_i = z_q_i.repeat_interleave(self.quantizers[i].stride, dim=-1)
            z_q += z_q_i

        return z_q

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[0]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


if __name__ == "__main__":
    rvq = ResidualVectorQuantize(quantizer_dropout=True)
    x = torch.randn(16, 512, 80)
    y = rvq(x)
    print(y["latents"].shape)