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
            + torch.sum(embeddings**2, dim=1)
            - 2 * torch.matmul(z_e_flattened, embeddings.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # quantized = self.embeddings(encoding_indices).view(z_e.shape)
        quantized = F.embedding(encoding_indices, embeddings).view(
            z_e.permute(0, 2, 3, 1).shape
        )
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # loss terms
        e_latent_loss = F.mse_loss(quantized.detach(), z_e)
        q_latent_loss = F.mse_loss(quantized, z_e.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = z_e + (quantized - z_e).detach()  # Straight-through estimator

        return quantized, loss


class Encoder(nn.Module):
    def __init__(
        self, in_channels: int = 1, hidden_channels: int = 128, latent_dim: int = 64
    ):
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
                hidden_channels, latent_dim, kernel_size=3, stride=1, padding=1
            ),  # 32
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(
        self, out_channels: int = 1, hidden_channels: int = 128, latent_dim: int = 64
    ):
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
            nn.Conv2d(
                hidden_channels, out_channels, kernel_size=3, stride=1, padding=1
            ),  # 128
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim)
        self.decoder = Decoder(in_channels, hidden_channels, embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, x):
        z_e = self.encoder.forward(x)
        z_q, vq_loss = self.quantizer.forward(z_e)
        x_recon = self.decoder.forward(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss

        return x_recon, total_loss, recon_loss, vq_loss


# modules for vqvae2
class EncoderTop(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        top_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, top_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class DecoderTop(nn.Module):
    def __init__(
        self,
        top_dim: int = 64,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(top_dim, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.net(z)


class VQVAE2(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        seg_channels: int = 4,
        hidden_channels: int = 128,
        bottom_dim: int = 64,
        top_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.bottom_dim = bottom_dim
        self.top_dim = top_dim

        # encoders
        self.encoder_bottom = Encoder(
            in_channels + seg_channels, hidden_channels, bottom_dim
        )
        self.encoder_top = EncoderTop(bottom_dim, top_dim)

        # decoders
        self.decoder_top = DecoderTop(top_dim, bottom_dim)
        # self.decoder_bottom = Decoder(in_channels, hidden_channels, bottom_dim * 2)
        self.decoder_bottom = nn.Sequential(
            nn.Conv2d(
                bottom_dim * 2 + seg_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            Decoder(
                hidden_channels=hidden_channels,
                latent_dim=hidden_channels,
                out_channels=in_channels,
            ),
        )

        # quantizers
        self.vp_top = VectorQuantizer(num_embeddings, top_dim, commitment_cost)
        self.vp_bottom = VectorQuantizer(num_embeddings, bottom_dim, commitment_cost)

    def forward(self, x, seg):
        # Concatenate image and segmentation for encoding
        x_cond = torch.cat([x, seg], dim=1)

        # Encode
        z_bottom = self.encoder_bottom(x_cond)
        z_top = self.encoder_top(z_bottom)

        # Quantize
        z_top_q, vq_top_loss = self.vp_top(z_top)
        z_bottom_q, vq_bottom_loss = self.vp_bottom(z_bottom)

        # Decode
        z_top_dec = self.decoder_top(z_top_q)

        # Upsample segmentation to match z_bottom_q spatial dimensions
        seg_upsampled = F.interpolate(seg, size=z_bottom_q.shape[2:], mode="nearest")

        # Combine for decoding
        z_combined = torch.cat([z_top_dec, z_bottom_q, seg_upsampled], dim=1)
        x_recon = self.decoder_bottom(z_combined)

        # Losses
        recon_loss = F.mse_loss(x_recon, x)
        vq_loss = vq_top_loss + vq_bottom_loss
        total_loss = recon_loss + vq_loss

        return x_recon, total_loss, recon_loss, vq_loss

    def generate(self, seg):
        """Generate image from segmentation only."""
        # Use random latent codes or zeros for unconditional generation
        batch_size = seg.shape[0]
        device = seg.device

        # Sample random codes from embeddings
        top_codes = torch.randint(
            0, self.vp_top.num_embeddings, (batch_size, 32, 32), device=device
        )
        bottom_codes = torch.randint(
            0, self.vp_bottom.num_embeddings, (batch_size, 64, 64), device=device
        )

        # Get quantized vectors
        z_top_q = F.embedding(top_codes, self.vp_top.embeddings.weight)
        z_top_q = z_top_q.permute(0, 3, 1, 2)

        z_bottom_q = F.embedding(bottom_codes, self.vp_bottom.embeddings.weight)
        z_bottom_q = z_bottom_q.permute(0, 3, 1, 2)

        # Decode
        z_top_dec = self.decoder_top(z_top_q)
        seg_upsampled = F.interpolate(seg, size=z_bottom_q.shape[2:], mode="nearest")
        z_combined = torch.cat([z_top_dec, z_bottom_q, seg_upsampled], dim=1)
        x_gen = self.decoder_bottom(z_combined)

        return x_gen


class TransformerPrior(nn.Module):
    def __init__(
        self,
        num_embeddings,
        hidden_dim=512,
        n_layers=6,
        n_heads=8,
        seq_len=32 * 32,
        dropout=0.1,
        seg_channels=0,
    ):
        """
        Transformer prior for VQ-VAE-2.
        Args:
            num_embeddings: number of discrete codes (same as VectorQuantizer)
            hidden_dim: embedding dim inside transformer
            n_layers: number of transformer layers
            n_heads: attention heads
            seq_len: length of flattened latent map (H*W)
            seg_channels: number of segmentation channels for conditioning (0 = unconditional)
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.seq_len = seq_len
        self.seg_channels = seg_channels

        # code embedding
        self.token_emb = nn.Embedding(num_embeddings, hidden_dim)
        # positional embedding
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden_dim))

        # segmentation conditioning
        if seg_channels > 0:
            self.seg_encoder = nn.Sequential(
                nn.Conv2d(
                    seg_channels, hidden_dim // 4, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    hidden_dim // 4, hidden_dim // 2, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1
                ),
            )
        else:
            self.seg_encoder = None

        # transformer with optimizations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,  # Use batch_first for better performance
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # output projection to logits over discrete codes
        self.to_logits = nn.Linear(hidden_dim, num_embeddings)

    def forward(self, x_seq, seg=None):
        """
        Args:
            x_seq: LongTensor of shape (B, H*W), discrete code indices from VQ-VAE
            seg: Optional segmentation mask (B, C, H, W) for conditioning
        Returns:
            logits: (B, seq_len, num_embeddings)
        """
        # embed tokens + positions
        x = self.token_emb(x_seq) + self.pos_emb[:, : x_seq.size(1), :]

        # add segmentation conditioning if available
        if seg is not None and self.seg_encoder is not None:
            seg_feat = self.seg_encoder(seg)  # (B, hidden_dim, 32, 32)
            seg_feat = seg_feat.flatten(2).transpose(1, 2)  # (B, 32*32, hidden_dim)
            x = x + seg_feat[:, : x_seq.size(1), :]

        out = self.transformer(x)
        logits = self.to_logits(out)
        return logits

    @torch.inference_mode()
    def sample(
        self,
        device,
        batch_size=1,
        seq_len=None,
        temperature=1.0,
        seg=None,
        top_k=None,
        top_p=None,
    ):
        """
        Sample discrete codes autoregressively with optional conditioning
        Args:
            device: torch device
            batch_size: number of samples to generate
            seq_len: sequence length to generate
            temperature: sampling temperature (higher = more random)
            seg: optional segmentation for conditioning (B, C, H, W)
            top_k: if set, only sample from top k logits
            top_p: if set, use nucleus sampling (sample from smallest set of tokens with cumulative prob >= top_p)
        """
        if seq_len is None:
            seq_len = self.seq_len

        # Start with a special token (0) or random token
        codes = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        for t in range(seq_len):
            logits = self.forward(codes, seg)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[:, 0] = False

                # Scatter to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            codes = torch.cat([codes, next_token], dim=1)

        # Remove the initial token and return
        return codes[:, 1:]
