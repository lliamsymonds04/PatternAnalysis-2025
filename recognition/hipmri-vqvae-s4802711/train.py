import torch
import pathlib
from dataset import load_data_helper
from modules import VQVAE


if __name__ == "__main__":
    root_dir = pathlib.Path(__file__).parent.resolve()
    train_dataset, train_loader = load_data_helper(
        root_dir / "keras_slices_data",
        "train",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    model = VQVAE(
        in_channels=1,
        hidden_channels=128,
        latent_dim=64,
        num_embeddings=512,
        commitment_cost=0.25,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(10):
        total_loss = 0.0
        for imgs in train_loader:
            imgs = imgs.to(device).float()
            optimizer.zero_grad()
            recon_imgs, vq_loss, recon_loss, vq_loss = model.forward(imgs)

            vq_loss.backward()
            optimizer.step()

            total_loss += vq_loss.item()
