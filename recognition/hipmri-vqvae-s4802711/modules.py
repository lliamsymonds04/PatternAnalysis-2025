import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e):
        # flatten input
        device = z_e.device
        embeddings = self.embeddings.weight.to(device)
        z_e_flattened = (
            z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        )

        # compute the distances between the outputs and the embedding vectors
        distances = (
            torch.sum(z_e_flattened**2, dim=1, keepdim=True)
            + torch.sum(embeddings, dim=1)
            - 2 * torch.matmul(z_e_flattened, embeddings.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embeddings(encoding_indices).view(z_e.shape)

        # loss terms
        e_latent_loss = F.mse_loss(quantized.detach(), z_e)
        q_latent_loss = F.mse_loss(quantized, z_e.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = z_e + (quantized - z_e).detach()  # Straight-through estimator

        return quantized, loss


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, hidden_channels, kernel_size=4, stride=2, padding=1
            ),  # 128
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1
            ),  # 64
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels, latent_dim, kernel_size=4, stride=2, padding=1
            ),  # 32
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim, hidden_channels, kernel_size=4, stride=2, padding=1
            ),  # 32
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1
            ),  # 64
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_channels, in_channels, kernel_size=4, stride=2, padding=1
            ),  # 128
        )

    def forward(self, z):
        return self.net(z)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        num_embeddings: int,
        commitment_cost: float,
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_channels, in_channels)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

    def forward(self, x):
        z_e = self.encoder.forward(x)
        z_q, vq_loss = self.quantizer.forward(z_e)
        x_recon = self.decoder.forward(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss

        return x_recon, total_loss, recon_loss, vq_loss
