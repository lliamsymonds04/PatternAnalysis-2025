import torch
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure as ssim
from modules import VQVAE
from dataset import load_data_helper
import pathlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = VQVAE(in_channels=1).to(device)
    root_dir = pathlib.Path(__file__).parent.resolve()
    model_path = root_dir / "vqvae_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_test_loader():
    root_dir = pathlib.Path(__file__).parent.resolve()
    _, test_loader = load_data_helper(
        root_dir / "keras_slices_data",
        "test",
    )
    return test_loader


def calculate_ssim(model: VQVAE, test_loader):
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


if __name__ == "__main__":
    test_loader = get_test_loader()
    model = load_model()

    average_ssim = calculate_ssim(model, test_loader)
    print(f"average SSIM on test set: {average_ssim:.4f}")
