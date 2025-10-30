import torch
from torch.utils.data import DataLoader
from torch import nn
import pathlib
import argparse
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
        seg_channels=6,
        hidden_channels=128,
        bottom_dim=64,
        top_dim=64,
        num_embeddings=512,
        commitment_cost=0.25,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    learning_rate: float = 2e-4,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        for batch in train_loader:
            imgs, segs = batch
            imgs = imgs.to(device).float()
            segs = segs.to(device).float()

            optimizer.zero_grad()
            _, total_loss_batch, _, _ = model.forward(imgs, segs)

            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()

        print(f"  Loss: {total_loss / len(train_loader)}")


def save_model(model: nn.Module, path: pathlib.Path):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQ-VAE model.")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--filename", type=str, default="vqvae_model.pth", help="Output model filename"
    )
    parser.add_argument(
        "--data_dir", type=str, default="keras_slices_data", help="Dataset directory"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate for optimizer"
    )
    args = parser.parse_args()

    root_dir = pathlib.Path(__file__).parent.resolve()
    train_dataset, train_loader = load_data_helper(
        root_dir / args.data_dir,
        "train",
        batch_size=args.batch_size,
        num_classes=6,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    model = create_vqvae2().to(device)

    start = time.time()
    print(f"training started at {time.ctime(start)}")

    try:
        train_model(model, train_loader, device, epochs=args.epochs)
        save_model(model, root_dir / args.filename)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, saving model...")
        save_model(model, root_dir / args.filename)
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
