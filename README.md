# Celebrity-GAN

An educational, from-scratch **MLP-based generative adversarial network (GAN)** in PyTorch. It learns to synthesise **32 × 32 RGB celebrity-style faces** from the CelebA dataset. The generator and discriminator use fully connected layers, not convolutions.

## Project layout

```text
Celebrity-GAN/
├── data/                       # Dataset images (not committed)
│   └── img_align_celeba/       # Example image subfolder; any name works
├── generated/                  # Present in project; current script does not save here
├── model/                      # Present in project; not used by current scripts
├── models/                     # Saved model weights
│   ├── generator.pth
│   └── discriminator.pth
├── src/
│   ├── descriminator.py       # Real/fake classifier (original spelling)
│   ├── generate.py            # Returns generated image tensors
│   ├── generator.py           # Noise-to-image MLP
│   └── training.py            # GAN training loop
├── download_dataset.py        # Dataset download/extraction utility
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10+ with compatible PyTorch and torchvision builds
- PyTorch, torchvision, Pillow, gdown
- CUDA-compatible NVIDIA GPU recommended (CPU fallback is supported)

```bash
pip install torch torchvision pillow gdown
```

For a CUDA-enabled PyTorch build, follow the [official installation instructions](https://pytorch.org/get-started/locally/).

## Dataset setup

Download the **CelebA aligned and cropped JPG** archive (`img_align_celeba.zip`) using:

```bash
python download_dataset.py
```

The download utility is intended to extract the dataset under `./data/img_align_celeba/`. The full dataset contains **202,599 images**. If the Drive download fails, use the [official CelebA project page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and its download links. The downloader supports a file-sharing URL with `--url`.

`torchvision.datasets.ImageFolder(root="./data")` requires at least one subfolder containing images. That subfolder can have **any name**, for example `faces/` or `img_align_celeba/`; images directly in `./data/` are not sufficient for `ImageFolder`.

## Architecture

| Network | Layers | Input | Output |
| --- | --- | --- | --- |
| Generator | Linear 100→256→512→1024→3072, ReLU, Tanh | `(N, 100)` random noise | `(N, 3, 32, 32)` RGB values in `[-1, 1]` |
| Discriminator | Linear 3072→1024→512→256→1, LeakyReLU, Sigmoid | `(N, 3, 32, 32)` images | `(N, 1)` real-image probabilities |

The discriminator's `Sigmoid` is compatible with the current `BCELoss`. The GAN is unconditional: CelebA attribute and identity annotations are not used.

## Training

Run from the **project root** so relative paths resolve:

```bash
python src/training.py
```

Current training constants in `src/training.py`:

| Setting | Value |
| --- | --- |
| `MAX_IMAGES` | `1000` |
| `EPOCHS` | `50` |
| `BATCH_SIZE` | `64` |
| `LATENT_DIM` | `100` |
| `LEARNING_RATE` | `0.0002` |
| DataLoader workers | `2` |

Images are centre-cropped to 178 × 178, resized to 32 × 32, converted to tensors and normalised to `[-1, 1]`. The script randomly selects up to `MAX_IMAGES` images once per run, then trains over that subset for all epochs. For each minibatch, it updates the discriminator using real and detached fake images, then updates the generator to increase the discriminator's real prediction on new fake images.

Average discriminator and generator losses are printed after every epoch. Loss values alone are not reliable measures of generated image quality.

The script writes `./models/generator.pth` and `./models/discriminator.pth` every five epochs and after training. **Each save overwrites the previous files.** Only network weights are saved; optimiser states and epoch numbers are not included, so these files do not provide exact training-resume checkpoints.

On Windows, `train()` is called under `if __name__ == "__main__":` so the DataLoader can use multiprocessing workers.

## Generate faces

After training:

```bash
python src/generate.py
```

The script loads `./models/generator.pth`, draws 16 random 100-dimensional latent vectors and generates a tensor of shape `(16, 3, 32, 32)` in `[-1, 1]`. **It currently returns the tensor from `main()` but does not display or save the images.** The existing `generated/` directory is not written to by this script.

## CelebA dataset agreement and licensing

**CelebA is not a permissively licensed image dataset.** The [official CelebA agreement](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) makes the dataset available for **non-commercial research purposes only**. The accompanying dataset README also mentions educational purposes; users should follow the official agreement.

The maintainers state that:

- CelebA images were obtained from the internet and are **not owned by MMLAB**, The Chinese University of Hong Kong.
- Users must not reproduce, duplicate, copy, sell, trade, resell, or commercially exploit the images or derived data.
- Users must not further copy, publish, or distribute portions of the dataset, except permitted internal copies at a single site within the same organisation.
- MMLAB reserves the right to terminate dataset access.
- Identity annotations are available on request for research purposes.

**Do not commit or redistribute CelebA images.** Exercise care with generated outputs and trained weights because the agreement also restricts derived data. The dataset's terms are separate from the licensing of this repository's original source code.

### Dataset attribution

**CelebFaces Attributes Dataset (CelebA)** — Multimedia Laboratory, The Chinese University of Hong Kong. [Official dataset website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

### Academic citation

```bibtex
@inproceedings{liu2015faceattributes,
  title = {Deep Learning Face Attributes in the Wild},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
  month = {December},
  year = {2015}
}
```

## Code licensing

No licence is specified for this project's original code. A separate software `LICENSE` file would be required to grant explicit reuse permissions. Any future code licence does **not** override the CelebA dataset agreement.
