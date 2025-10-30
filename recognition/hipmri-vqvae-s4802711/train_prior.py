import argparse
import pathlib
import torch
import torch.nn.functional as F
from dataset import load_data_helper
from modules import VQVAE2, TransformerPrior
from torch.cuda.amp import autocast, GradScaler


@torch.inference_mode()
def extract_top_indices(vqvae, x, seg):
    """Get discrete top-level indices from VQ-VAE-2"""
    x_cond = torch.cat([x, seg], dim=1)
    z_bottom = vqvae.encoder_bottom(x_cond)
    z_top = vqvae.encoder_top(z_bottom)
    # compute nearest embedding indices
    flat = z_top.permute(0, 2, 3, 1).contiguous().view(-1, vqvae.top_dim)
    emb = vqvae.vp_top.embeddings.weight
    dists = (
        torch.sum(flat**2, dim=1, keepdim=True)
        + torch.sum(emb**2, dim=1)
        - 2 * torch.matmul(flat, emb.t())
    )
    indices = torch.argmin(dists, dim=1)
    indices = indices.view(z_top.size(0), z_top.size(2) * z_top.size(3))  # (B, H*W)
    return indices


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    root_dir = pathlib.Path(__file__).parent.resolve()
    _, train_loader = load_data_helper(
        root_dir / args.data_dir,
        "train",
        batch_size=args.batch_size,
        num_classes=6,
    )

    # Load pretrained VQ-VAE-2
    vqvae = VQVAE2(
        in_channels=1,
        seg_channels=6,
    ).to(device)
    vqvae.load_state_dict(torch.load(root_dir / args.vqvae_path, map_location=device))
    vqvae.eval()

    # Transformer prior
    seq_len = 32 * 32  # top latent spatial size
    prior = TransformerPrior(num_embeddings=512, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=args.lr)

    save_path = root_dir / args.save_path

    # Training loop
    try:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            total_loss = 0

            for img, seg in train_loader:
                img, seg = img.to(device), seg.to(device)
                top_indices = extract_top_indices(vqvae, img, seg).clone()

                logits = prior(top_indices[:, :-1])
                target = top_indices[:, 1:]
                loss = F.cross_entropy(
                    logits.reshape(-1, prior.num_embeddings), target.reshape(-1)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"    Loss: {total_loss / len(train_loader):.4f}")

        # save the final model:
        print("Finished training. Saving model...")
        torch.save(prior.state_dict(), save_path)
        print("Done")

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        torch.save(prior.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="transformer_prior.pth")
    parser.add_argument("--data_dir", type=str, default="keras_slices_data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision for faster training",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster training (PyTorch >= 2.0)",
    )
    args = parser.parse_args()
    train(args)
