import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure as ssim
from modules import VQVAE, VQVAE2, TransformerPrior
from dataset import load_data_helper
import pathlib
import matplotlib.pyplot as plt
import argparse

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


def load_prior(prior_path: pathlib.Path):
    prior = TransformerPrior(num_embeddings=512, seq_len=32 * 32).to(device)
    prior.load_state_dict(torch.load(prior_path, map_location=device))
    prior.eval()
    return prior


def get_test_loader(num_classes: int = 6, batch_size: int = 8):
    root_dir = pathlib.Path(__file__).parent.resolve()
    _, test_loader = load_data_helper(
        root_dir / "keras_slices_data",
        "test",
        num_classes=num_classes,
        batch_size=batch_size,
    )
    return test_loader


def calculate_ssim(model: nn.Module, test_loader):
    ssim_values = []

    with torch.no_grad():
        for imgs, segs in test_loader:
            imgs = imgs.to(device).float()
            segs = segs.to(device).float()
            recons, _, _, _ = model.forward(imgs, segs)

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
    batch = next(iter(test_loader))
    imgs, segs = batch
    imgs = imgs.to(device).float()
    segs = segs.to(device).float()
    with torch.no_grad():
        recons, _, _, _ = model.forward(imgs, segs)

    recons = torch.clamp(recons, -1, 1)

    imgs = imgs.cpu().numpy()
    recons = recons.cpu().numpy()

    num_images = min(5, imgs.shape[0])
    plt.figure(figsize=(10, 3))
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
    plt.show(block=False)


def generate_from_prior(vqvae: VQVAE2, prior: TransformerPrior, seg, temperature=1.0):
    """Generate images using the transformer prior and VQ-VAE-2 decoder."""
    with torch.no_grad():
        batch_size = seg.shape[0]

        # Sample top-level codes from prior
        top_codes = prior.sample(device, seq_len=32 * 32, temperature=temperature)
        top_codes = top_codes.repeat(batch_size, 1)

        # Reshape to spatial dimensions
        top_codes = top_codes.view(batch_size, 32, 32)

        # Convert codes to quantized latents
        z_top_q = F.embedding(top_codes, vqvae.vp_top.embeddings.weight)
        z_top_q = z_top_q.permute(0, 3, 1, 2).contiguous()

        # Sample random bottom codes
        bottom_codes = torch.randint(
            0, vqvae.vp_bottom.num_embeddings, (batch_size, 64, 64), device=device
        )
        z_bottom_q = F.embedding(bottom_codes, vqvae.vp_bottom.embeddings.weight)
        z_bottom_q = z_bottom_q.permute(0, 3, 1, 2).contiguous()

        # Decode
        z_top_dec = vqvae.decoder_top(z_top_q)
        seg_upsampled = F.interpolate(seg, size=z_bottom_q.shape[2:], mode="nearest")
        z_combined = torch.cat([z_top_dec, z_bottom_q, seg_upsampled], dim=1)
        x_gen = vqvae.decoder_bottom(z_combined)

        return x_gen


def plot_prior_generations(
    vqvae: VQVAE2, prior: TransformerPrior, test_loader, num_images=5, temperature=1.0
):
    """Generate and plot images using the transformer prior."""
    batch = next(iter(test_loader))
    _, segs = batch
    segs = segs.to(device).float()

    with torch.no_grad():
        generated = generate_from_prior(
            vqvae, prior, segs[:num_images], temperature=temperature
        )

    generated = torch.clamp(generated, -1, 1)
    generated = generated.cpu().numpy()
    segs_np = segs.cpu().numpy()

    plt.figure(figsize=(10, 4))
    for i in range(min(num_images, generated.shape[0])):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(segs_np[i].argmax(0), cmap="tab10")
        plt.title("Segmentation")
        plt.axis("off")

        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(generated[i, 0], cmap="gray")
        plt.title("Generated")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VQ-VAE-2 model on test set and generate images."
    )

    # arguments
    parser.add_argument(
        "--vqvae_path",
        type=str,
        default="vqvae_model.pth",
        help="Model filename to load (default: vqvae_model.pth)",
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=5,
        help="Number of new images to generate (default: 5)",
    )

    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Whether to save generated images (default: False)",
    )

    parser.add_argument(
        "--prior_path",
        type=str,
        default=None,
        help="Path to transformer prior model (default: None, skips prior generation)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for prior generation (default: 1.0)",
    )

    args = parser.parse_args()

    test_loader = get_test_loader()
    model = VQVAE2(in_channels=1, seg_channels=6).to(device)
    load_model(model, pathlib.Path(__file__).parent.resolve() / args.vqvae_path)

    average_ssim = calculate_ssim(model, test_loader)
    print(f"average SSIM on test set: {average_ssim:.4f}")
    plot_reconstructions(model, test_loader)

    # Generate from prior if path provided
    if args.prior_path:
        root_dir = pathlib.Path(__file__).parent.resolve()
        prior = load_prior(root_dir / args.prior_path)
        print(
            f"Generating {args.num_images} images from prior with temperature={args.temperature}"
        )
        plot_prior_generations(
            model, prior, test_loader, args.num_images, args.temperature
        )
