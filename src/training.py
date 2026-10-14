"""Train the Celebrity-GAN MLP generator and discriminator on CelebA.

Images are centre-cropped, resized to 32x32 and normalised to [-1, 1].
Each minibatch first updates the discriminator and then the generator.
Model weights are saved every five epochs and again at the end.
Run from the repository root because data and checkpoint paths are relative.
"""

import os

import descriminator as d
import generator as g
import torch
import torch.nn as nn
import constants as c

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Training settings: number of sampled images, full passes, and batch size.
MAX_IMAGES = c.max_images
EPOCHS = c.epochs
BATCH_SIZE = c.batch_size
# Number of random input values accepted by the generator.
LATENT_DIM = c.latent_dim
LEARNING_RATE = c.lr


def train():
    """Train both networks and save their weights to ./models/.

    Randomly selects up to MAX_IMAGES images once per run, then reuses
    that subset for every epoch. Prints mean losses after each epoch.
    """

    # GPU: prefer CUDA for NVIDIA acceleration.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training on: {device}")

    # Networks: generator makes faces; discriminator scores them.
    G = g.Generator().to(device)
    D = d.Descriminator().to(device)

    # The discriminator already ends in Sigmoid, so use BCELoss.
    lossFn = nn.BCELoss()

    # Separate optimisers allow independent updates to the two networks.
    optG = torch.optim.Adam(
        G.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999)
    )

    optD = torch.optim.Adam(
        D.parameters(),
        lr=0.0001,
        betas=(0.5, 0.999)
    )

    # Image preprocessing: square crop, downsample, tensor conversion,
    # and normalisation from [0, 1] to [-1, 1].
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((c.res, c.res)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    # Load dataset. ImageFolder expects images inside a subfolder of ./data.
    # Its class labels are not used by the GAN.
    dataset = datasets.ImageFolder(
        root="./data",
        transform=transform
    )

    # Select random images once; cap the requested count at dataset size.
    maxImages = min(MAX_IMAGES, len(dataset))

    indices = torch.randperm(
        len(dataset)
    )[:maxImages].tolist()

    dataset = Subset(dataset, indices)

    # DataLoader: shuffled minibatches, two worker processes for loading,
    # and pinned host memory when CUDA is used.
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        prefetch_factor=2
    )



    """
    real, _ = next(iter(dataloader))

    save_image(
        real[:16],
        "real_samples.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )
    """

    print(f"Loaded {len(dataset)} images")

    # Create the checkpoint destination if it does not exist.
    os.makedirs("./models", exist_ok=True)

    # Training: one epoch is one pass through the selected image subset.
    for epoch in range(EPOCHS):

        totalDLoss = 0.0
        totalGLoss = 0.0

        for real, _ in dataloader:

            real = real.to(device, non_blocking=True)
            batchSize = real.size(0)

            # -------------------------
            # Train discriminator
            # -------------------------

            # Sample latent noise and generate a fake minibatch.
            z = torch.randn(
                batchSize, LATENT_DIM, device=device
            )

            fake = G(z)

            # Real faces should be classified as 1 (real).
            realLoss = lossFn(
                D(real),
                torch.ones(batchSize, 1, device=device)
            )

            # Fake faces should be classified as 0 (fake).
            # detach() prevents this discriminator step from updating G.
            fakeLoss = lossFn(
                D(fake.detach()),
                torch.zeros(batchSize, 1, device=device)
            )

            # Average the real and fake classification losses.
            dLoss = (realLoss + fakeLoss) / 2

            # Reset discriminator gradients, backpropagate, update weights.
            optD.zero_grad()
            dLoss.backward()
            optD.step()

            # -------------------------
            # Train generator
            # -------------------------

            # Generate fresh fake faces for the generator update.
            z = torch.randn(
                batchSize, LATENT_DIM, device=device
            )

            fake = G(z)

            # Generator aims for D(fake) = 1, i.e. to fool D.
            gLoss = lossFn(
                D(fake),
                torch.ones(batchSize, 1, device=device)
            )

            # Update only generator parameters with its optimiser.
            optG.zero_grad()
            gLoss.backward()
            optG.step()

            # Record batch losses for the epoch summary.
            totalDLoss += dLoss.item()
            totalGLoss += gLoss.item()

        # Epoch results: arithmetic mean of batch losses.
        avgDLoss = totalDLoss / len(dataloader)
        avgGLoss = totalGLoss / len(dataloader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"D Loss: {avgDLoss:.4f} "
            f"G Loss: {avgGLoss:.4f}"
        )

        # Save every 5 epochs, overwriting the previous checkpoint files.
        # These state_dict files contain weights, not optimiser states.
        if (epoch + 1) % 5 == 0:

            torch.save(
                G.state_dict(),
                "./models/generator.pth"
            )

            torch.save(
                D.state_dict(),
                "./models/discriminator.pth"
            )

    # Save final models, including when EPOCHS is not divisible by five.
    torch.save(
        G.state_dict(),
        "./models/generator.pth"
    )

    torch.save(
        D.state_dict(),
        "./models/discriminator.pth"
    )

    print("Training complete!")


# Required on Windows when DataLoader uses multiprocessing workers.
if __name__ == "__main__":
    train()
