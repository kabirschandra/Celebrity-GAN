"""Download and extract the CelebA aligned JPG images into ./data.

Run from anywhere with:
    python download_dataset.py

Requires the gdown package:
    pip install gdown
"""

from pathlib import Path
import argparse
import shutil
import sys
import zipfile

# Google Drive file ID used by torchvision for the aligned JPG CelebA archive.
FILE_ID = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
EXPECTED_COUNT = 202599
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DEST = DATA / 'img_align_celeba'
ARCHIVE = DATA / 'img_align_celeba.zip'


def count_jpg(folder):
    """Count JPG files in a directory and return the total.

    Args:
        folder: Directory to scan for JPG images.

    Returns:
        The number of JPG files found in the directory.
    """
    return sum(1 for p in folder.glob('*.jpg')) if folder.is_dir() else 0


def main():
    """Download the CelebA archive if missing, validate it, and extract images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', help='Direct Google Drive FILE sharing URL for img_align_celeba.zip (not a folder URL)')
    parser.add_argument('--keep-zip', action='store_true', help='Keep downloaded archive after extraction')
    args = parser.parse_args()

    count = count_jpg(DEST)
    if count == EXPECTED_COUNT:
        print(f'Already extracted: {count:,} images in {DEST}')
        return

    DATA.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        try:
            import gdown
        except ImportError:
            sys.exit('Install downloader first: pip install gdown')
        url = args.url or f'https://drive.google.com/uc?id={FILE_ID}'
        print('Downloading CelebA aligned JPG archive...')
        result = gdown.download(url=url, output=str(ARCHIVE), quiet=False, fuzzy=True, resume=True)
        if not result:
            sys.exit('Download failed. Open the official CelebA Google Drive folder, copy the sharing link for img_align_celeba.zip, and rerun with --url "FILE_LINK".')

    if not zipfile.is_zipfile(ARCHIVE):
        sys.exit(f'{ARCHIVE} is not a valid ZIP. Remove it and retry using --url with the archive file link.')

    print('Extracting images...')
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        entries = [info for info in zf.infolist() if info.filename.lower().endswith('.jpg')]
        for i, info in enumerate(entries, 1):
            # Flatten any archive-level folder, with strict filename checks.
            name = Path(info.filename).name
            if not (len(name) == 10 and name[:6].isdigit() and name[6:].lower() == '.jpg'):
                raise ValueError(f'Unexpected image filename in archive: {info.filename}')
            target = DEST / name
            if not target.exists():
                with zf.open(info) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            if i % 10000 == 0 or i == len(entries):
                print(f'Extracted {i:,}/{len(entries):,}', flush=True)

    count = count_jpg(DEST)
    print(f'Found {count:,} JPG images at {DEST}')
    if count != EXPECTED_COUNT:
        sys.exit(f'Expected {EXPECTED_COUNT:,} images. Check the archive and extraction.')
    if not args.keep_zip:
        ARCHIVE.unlink()
        print('Deleted ZIP after successful extraction.')
    print('Ready for PyTorch ImageFolder(root="./data").')


if __name__ == '__main__':
    main()
