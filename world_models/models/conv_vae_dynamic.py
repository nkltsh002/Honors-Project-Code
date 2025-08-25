from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    """
    Dynamic Conv VAE that works with square images of size img_size (e.g., 32 or 64).
    It computes conv flatten shape at runtime via a dummy forward, so no hardcoded dims.
    """
    def __init__(self, img_channels: int = 3, img_size: int = 32, latent_dim: int = 32,
                 enc_channels=(32, 64, 128, 256)):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Encoder: downsample by 2 at each block (stride=2), "same-ish" padding
        c = [img_channels] + list(enc_channels)
        enc = []
        for i in range(len(enc_channels)):
            enc += [
                nn.Conv2d(c[i], c[i+1], kernel_size=3, stride=2, padding=1),  # /2
                nn.BatchNorm2d(c[i+1]),
                nn.ReLU(inplace=True),
            ]
        self.encoder = nn.Sequential(*enc)

        # compute conv flatten dim dynamically with a dummy pass
        with torch.no_grad():
            x = torch.zeros(1, img_channels, img_size, img_size)
            h = self.encoder(x)
            self._conv_shape = h.shape[1:]         # (C, H, W)
            self._conv_flat = h.view(1, -1).size(1)  # Flatten per sample, not total batch

        self.fc_mu     = nn.Linear(self._conv_flat, latent_dim)
        self.fc_logvar = nn.Linear(self._conv_flat, latent_dim)

        # Decoder: mirror of encoder (ConvTranspose to upsample by 2 each block)
        dec_channels = list(enc_channels)[::-1]
        D = []
        in_c = dec_channels[0]
        for j, out_c in enumerate(dec_channels[1:] + [img_channels]):
            # last layer goes to img_channels
            is_last = (j == len(dec_channels) - 1)
            if not is_last:
                D += [
                    nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),  # *2
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ]
                in_c = out_c
            else:
                D += [
                    nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                ]
        self.decoder = nn.Sequential(*D)

        self.fc_up = nn.Linear(latent_dim, self._conv_flat)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1)  # Use reshape instead of view for memory compatibility
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        h = self.fc_up(z)
        h = h.reshape(z.size(0), *self._conv_shape)   # Use reshape instead of view
        x = self.decoder(h)
        return torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
