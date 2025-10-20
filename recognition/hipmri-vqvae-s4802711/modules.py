import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer:
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e):
        # flatten input
        z_e_flattened = (
            z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        )

        # compute the distances between the outputs and the embedding vectors
        distances = (
            torch.sum(z_e_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(z_e_flattened, self.embeddings.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embeddings(encoding_indices).view(z_e.shape)

        # loss terms
        e_latent_loss = F.mse_loss(quantized.detach(), z_e)
        q_latent_loss = F.mse_loss(quantized, z_e.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = z_e + (quantized - z_e).detach()  # Straight-through estimator

        return quantized, loss


class Encoder:
    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int):
        super().__init__()
        pass

    def forward(self, x):
        pass


class Decoder:
    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int):
        pass

    def forward(self, z):
        pass


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        num_embeddings: int,
        commitment_cost: float,
    ):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_channels, in_channels)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

    def forward(self, x):
        pass
