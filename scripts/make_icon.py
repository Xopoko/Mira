"""Build the application icons from a square source image.

    python scripts/make_icon.py path/to/new-icon.webp   # replace the icon everywhere
    python scripts/make_icon.py                         # rebuild icon.ico from icon.png

A source image also writes the iPhone app icon, which needs the full 1024 px.
"""
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'context_operator' / 'assets'
ICON = ASSETS / 'icon.png'
PHONE_ICON = ROOT / 'phone' / 'Sources' / 'Assets.xcassets' / 'AppIcon.appiconset' / 'AppIcon.png'
ICO_SIZES = [(16, 16), (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64), (128, 128), (256, 256)]


def phone_icon(source):
    # iOS applies its own corner mask and shows transparency as black, so the
    # rounded corners are filled with the color along the icon's edge.
    image = source.resize((1024, 1024), Image.Resampling.LANCZOS)
    band = [image.getpixel((x, y)) for x in range(256, 768, 8) for y in (12, 1011)]
    band += [image.getpixel((x, y)) for y in range(256, 768, 8) for x in (12, 1011)]
    opaque = [pixel[:3] for pixel in band if pixel[3] >= 240]
    if not opaque:
        raise SystemExit('The icon source needs an opaque edge.')
    edge = tuple(round(sum(channel) / len(opaque)) for channel in zip(*opaque))
    canvas = Image.new('RGBA', image.size, edge + (255,))
    canvas.alpha_composite(image)
    return canvas.convert('RGB')


def main():
    if len(sys.argv) > 1:
        source = Image.open(sys.argv[1]).convert('RGBA')
        if source.width != source.height or source.width < 1024:
            raise SystemExit('The icon source must be square and at least 1024 px.')
        source.resize((512, 512), Image.Resampling.LANCZOS).save(ICON, optimize=True)
        phone_icon(source).save(PHONE_ICON, optimize=True)
    Image.open(ICON).convert('RGBA').save(ASSETS / 'icon.ico', sizes=ICO_SIZES)
    print('Application icon created.')


if __name__ == '__main__':
    main()
