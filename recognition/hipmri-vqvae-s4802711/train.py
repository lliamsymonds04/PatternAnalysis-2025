import torch
from torch.utils.data import DataLoader
from torch import nn
import pathlib
import time
from dataset import load_data_helper
from modules import VQVAE, VQVAE2


def create_vqvae():
    return VQVAE(
        in_channels=1,
        hidden_channels=128,
        embedding_dim=64,
        num_embeddings=512,
        commitment_cost=0.25,
    )


def create_vqvae2():
    return VQVAE2(
        in_channels=1,
        hidden_channels=128,
        bottom_dim=64,
        top_dim=64,
        num_embeddings=512,
        commitment_cost=0.25,
    )


def train_model(
    model: nn.Module, train_loader: DataLoader, device: torch.device, epochs: int = 10
):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        for imgs in train_loader:
            imgs = imgs.to(device).float()
            optimizer.zero_grad()
            _, total_loss_batch, _, _ = model.forward(imgs)

            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()

        print(f"  Loss: {total_loss / len(train_loader)}")


def save_model(model: nn.Module, path: pathlib.Path):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    root_dir = pathlib.Path(__file__).parent.resolve()
    train_dataset, train_loader = load_data_helper(
        root_dir / "keras_slices_data",
        "train",
        batch_size=32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    file_name = (
        input("Enter model filename (default: vqvae_model.pth): ") or "vqvae_model.pth"
    )
    model = create_vqvae2().to(device)

    start = time.time()
    print(f"training started at {time.ctime(start)}")

    try:
        train_model(model, train_loader, device, epochs=50)
        save_model(model, root_dir / file_name)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, saving model...")
        save_model(model, root_dir / file_name)
    finally:
        # output training duration
        end = time.time()
        print(f"training ended at {time.ctime(end)}")
        training_minutes = (end - start) / 60
        training_seconds = (end - start) % 60
        print(
            f"Total training time: {int(training_minutes)} minutes and {int(training_seconds)} seconds."
        )

    print("Model saved. Exiting...")
