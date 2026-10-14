
"""Generate and save example faces from a trained GAN."""

import os
import torch
from torchvision.utils import save_image
from generator import Generator


def main():
    """Load trained weights and save 16 generated faces."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    os.makedirs("./generated", exist_ok=True)

    # Load generator
    generator = Generator().to(device)
    generator.load_state_dict(
        torch.load(
            "./models/generator.pth",
            map_location=device,
            weights_only=True,
        )
    )

    generator.eval()

    # Generate 16 faces
    with torch.no_grad():
        z = torch.randn(16, 100, device=device)
        faces = generator(z)

    # Save individual images
    for i, face in enumerate(faces):
        save_image(
            face,
            f"./generated/face_{i+1:03d}.png",
            normalize=True,
            value_range=(-1, 1)
        )

    # Save combined image grid
    save_image(
        faces,
        "./generated/grid.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )

    print(f"Saved {len(faces)} faces to ./generated/")

    return faces


if __name__ == "__main__":
    main()

