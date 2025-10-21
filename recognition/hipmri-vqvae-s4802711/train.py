import torch
from torch.utils.data import DataLoader
import pathlib
from dataset import load_data_helper
from modules import VQVAE


def create_model():
    return VQVAE(
        in_channels=1,
        hidden_channels=128,
        embedding_dim=64,
        num_embeddings=512,
        commitment_cost=0.25,
    )


def train_model(
    model: VQVAE, train_loader: DataLoader, device: torch.device, epochs: int = 10
):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        for imgs in train_loader:
            imgs = imgs.to(device).float()
            optimizer.zero_grad()
            _, vq_loss, _, _ = model.forward(imgs)

            vq_loss.backward()
            optimizer.step()

            total_loss += vq_loss.item()

        print(f"  Loss: {total_loss / len(train_loader)}")


if __name__ == "__main__":
    root_dir = pathlib.Path(__file__).parent.resolve()
    train_dataset, train_loader = load_data_helper(
        root_dir / "keras_slices_data",
        "train",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    model = create_model().to(device)

    train_model(model, train_loader, device, epochs=10)
