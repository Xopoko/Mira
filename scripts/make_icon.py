"""Build the application icons from a square source image.

    python scripts/make_icon.py path/to/new-icon.webp   # replace the icon
    python scripts/make_icon.py                         # rebuild icon.ico from icon.png
"""
from pathlib import Path
import sys

from PIL import Image

ASSETS = Path(__file__).resolve().parents[1] / 'context_operator' / 'assets'
ICON = ASSETS / 'icon.png'
ICO_SIZES = [(16, 16), (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64), (128, 128), (256, 256)]


def main():
    if len(sys.argv) > 1:
        source = Image.open(sys.argv[1]).convert('RGBA')
        if source.width != source.height:
            raise SystemExit('The icon source must be square.')
        source.resize((512, 512), Image.Resampling.LANCZOS).save(ICON, optimize=True)
    Image.open(ICON).convert('RGBA').save(ASSETS / 'icon.ico', sizes=ICO_SIZES)
    print('Application icon created.')


if __name__ == '__main__':
    main()
