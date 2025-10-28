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


def generate_new_images(model: nn.Module, test_loader, num_images: int = 5):
    """
    Generate images by sampling and recombining latent codes from real images.
    Since there's no trained prior, we sample latent codes from actual images
    and recombine them to create variations.
    """
    with torch.no_grad():
        # Get a batch of real images to extract latent codes from
        imgs = next(iter(test_loader))
        imgs = imgs.to(device).float()

        # Encode images to get latent codes
        z_bottom = model.encoder_bottom(imgs)
        z_top = model.encoder_top(z_bottom)

        # Quantize to get discrete codes
        z_top_q, _ = model.vp_top(z_top)
        z_bottom_q, _ = model.vp_bottom(z_bottom)

        # For generation, randomly recombine top and bottom codes from different images
        num_available = min(imgs.shape[0], num_images * 2)
        samples_list = []

        for i in range(num_images):
            # Randomly select different images for top and bottom codes
            top_idx = torch.randint(0, num_available, (1,)).item()
            bottom_idx = torch.randint(0, num_available, (1,)).item()

            # Use the quantized codes from different images
            z_top_sample = z_top_q[top_idx : top_idx + 1]
            z_bottom_sample = z_bottom_q[bottom_idx : bottom_idx + 1]

            # Decode
            z_top_dec = model.decoder_top(z_top_sample)
            z_combined = torch.cat([z_top_dec, z_bottom_sample], dim=1)
            sample = model.decoder_bottom(z_combined)
            samples_list.append(sample)

        samples = torch.cat(samples_list, dim=0)
        samples = torch.clamp(samples, -1, 1)

    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(samples[i, 0].cpu().numpy(), cmap="gray")
        plt.title(f"image {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


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
    # plot_reconstructions(model, test_loader)
    generate_new_images(model, test_loader, num_images=5)
