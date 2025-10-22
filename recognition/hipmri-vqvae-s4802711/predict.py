import torch
import torch.nn as nn
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure as ssim
from modules import VQVAE, VQVAE2
from dataset import load_data_helper
import pathlib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vqvae():
    model = VQVAE(in_channels=1).to(device)
    root_dir = pathlib.Path(__file__).parent.resolve()
    model_path = root_dir / "vqvae_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_vqvae2():
    model = VQVAE2(in_channels=1).to(device)
    root_dir = pathlib.Path(__file__).parent.resolve()
    model_path = root_dir / "vqvae_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_model(model: nn.Module, path: pathlib.Path):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()


def get_test_loader():
    root_dir = pathlib.Path(__file__).parent.resolve()
    _, test_loader = load_data_helper(
        root_dir / "keras_slices_data",
        "test",
    )
    return test_loader


def calculate_ssim(model: nn.Module, test_loader):
    ssim_values = []

    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(device).float()
            recons, _, _, _ = model.forward(imgs)

            # clip to valid range
            recons = torch.clamp(recons, -1, 1)
            imgs = torch.clamp(imgs, -1, 1)

            batch_ssim = ssim(recons, imgs, data_range=2.0)
            if isinstance(batch_ssim, torch.Tensor):
                batch_ssim = batch_ssim.detach().cpu()
                if batch_ssim.ndim == 0:
                    ssim_values.append(batch_ssim.item())

    average_ssim = np.mean(ssim_values)
    return average_ssim


def plot_reconstructions(model: nn.Module, test_loader):
    imgs = next(iter(test_loader))
    imgs = imgs.to(device).float()
    with torch.no_grad():
        recons, _, _, _ = model.forward(imgs)

    recons = torch.clamp(recons, -1, 1)

    imgs = imgs.cpu().numpy()
    recons = recons.cpu().numpy()

    num_images = min(5, imgs.shape[0])
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(imgs[i, 0], cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(recons[i, 0], cmap="gray")
        plt.title("Reconstruction")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_loader = get_test_loader()
    model = VQVAE2(in_channels=1).to(device)
    file_name = (
        input("Enter model filename (default: vqvae_model.pth): ") or "vqvae_model.pth"
    )
    load_model(model, pathlib.Path(__file__).parent.resolve() / file_name)

    average_ssim = calculate_ssim(model, test_loader)
    print(f"average SSIM on test set: {average_ssim:.4f}")
    plot_reconstructions(model, test_loader)
