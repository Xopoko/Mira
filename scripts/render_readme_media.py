"""Render Mira's README screenshots and simulated gaze demo offscreen."""

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QFontDatabase, QImage, QPainter
from PySide6.QtWidgets import QApplication, QFrame

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from context_operator import settings
from context_operator.core.contracts import Element, Gaze, Rect, Scene
from context_operator.core.fusion import Fusion
from context_operator.ui.overlay import paint_scene_light


FONTS = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
BACKGROUND = '#141414'
PRACTICE_SIZE = (1120, 582)
GAZE_STOPS = (('Top left', 640), ('Plus', 640), ('Top right', 400), ('Clock', 400), ('Reset', 720))


def prepare_practice(window):
    # Enter the real Practice mode without starting the live screen provider.
    window.c.running = True
    window.open_practice()
    page = window.stack.currentWidget().widget()
    stage = window.practice_buttons['Plus'].parentWidget()
    layout = page.layout()

    def hide_widgets(item):
        if widget := item.widget():
            widget.hide()
        elif children := item.layout():
            for index in range(children.count()):
                hide_widgets(children.itemAt(index))

    # The demo uses the title, controls and targets above the practice exercises.
    while layout.count() > layout.indexOf(stage) + 1:
        hide_widgets(layout.takeAt(layout.count() - 1))
    margins = layout.contentsMargins()
    margins.setBottom(14)
    layout.setContentsMargins(margins)
    window.content_status.hide()
    sidebar = window.findChild(QFrame, 'sidebar').layout()
    while sidebar.count() > len(window.nav) + 3:  # brand, spacer, navigation, stretch
        hide_widgets(sidebar.takeAt(sidebar.count() - 1))
    window.setMinimumHeight(0)
    window.resize(*PRACTICE_SIZE)
    window.setFocus()
    return page


def button_elements(window):
    elements = []
    for name, button in window.practice_buttons.items():
        pos = button.mapTo(window, QPoint())
        elements.append(Element(name, name, 'Button', Rect(pos.x(), pos.y(), button.width(), button.height())))
    return tuple(elements)


def rgb_image(image):
    image = image.convertToFormat(QImage.Format.Format_RGBX8888)
    return Image.frombytes('RGBX', (image.width(), image.height()), image.constBits()).convert('RGB')


def paint_gaze(base, elements, point, now, fusion):
    image = base.copy()
    gaze = Gaze(*point, now, settings.DEFAULTS['radius'])
    scene = Scene(0, now, Rect(0, 0, image.width(), image.height()), elements)
    target = fusion.choose(scene, gaze, now).target
    center = QPointF(gaze.x, gaze.y)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    paint_scene_light(painter, scene, center, gaze.radius, now, target.id if target else None)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(134, 221, 255, 190))
    painter.drawEllipse(center, 2.5, 2.5)
    painter.end()
    return rgb_image(image)


def target_point(element):
    x, y = element.rect.center
    return x + 18, y - 15


def render_gif(output, base, elements, fusion):
    targets = {element.id: element for element in elements}
    caption = Image.new('RGB', (960, 45), BACKGROUND)
    draw = ImageDraw.Draw(caption)
    draw.ellipse((21, 17, 31, 27), fill='#ffbd58')
    font = ImageFont.truetype(FONTS / 'segoeui.ttf', 19)
    draw.text((43, 12), 'Simulated gaze input', font=font, fill='#d2d2ce')
    frames, durations = [], []

    def add_frame(image, duration):
        frame = Image.new('RGB', (960, 543), BACKGROUND)
        frame.paste(image.resize((960, 498), Image.Resampling.LANCZOS))
        frame.paste(caption, (0, 498))
        frames.append(frame)
        durations.append(duration)

    add_frame(rgb_image(base), 320)
    previous = target_point(targets[GAZE_STOPS[0][0]])
    for index, (name, hold) in enumerate(GAZE_STOPS):
        point = target_point(targets[name])
        if index:
            for step in range(1, 10):
                t = step / 10
                t = t * t * (3 - 2 * t)
                between = tuple(a + (b - a) * t for a, b in zip(previous, point))
                frame = paint_gaze(base, elements, between, sum(durations) / 1000, fusion)
                add_frame(frame, 70 if step <= 5 else 80)
        add_frame(paint_gaze(base, elements, point, sum(durations) / 1000, fusion), hold)
        previous = point

    # One palette preserves the UI colors between frames and allows GIF delta encoding.
    palette = frames[1].quantize()
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    indexed[0].save(output, save_all=True, append_images=indexed[1:], duration=durations, loop=0)


def rounded_mask(size, radius):
    mask = Image.new('L', (size[0] * 4, size[1] * 4))
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, mask.width - 1, mask.height - 1), radius=radius * 4, fill=255)
    return mask.resize(size, Image.Resampling.LANCZOS)


def render_social(output, practice):
    card = Image.new('RGB', (1280, 640), BACKGROUND)
    with Image.open(settings.ASSETS / 'icon.png') as source:
        icon = source.convert('RGBA').resize((48, 48), Image.Resampling.LANCZOS)
    card.paste(icon, (56, 92), icon)
    draw = ImageDraw.Draw(card)
    bold = FONTS / 'segoeuib.ttf'
    draw.text((112, 87), 'Mira', font=ImageFont.truetype(bold, 36), fill='#f2f2ee')
    headline = ImageFont.truetype(bold, 58)
    for index, line in enumerate(('TrueDepth', 'gaze tracking', 'for Windows')):
        draw.text((58, 161 + index * 69), line, font=headline, fill='#f2f2ee')
    tagline = ImageFont.truetype(FONTS / 'segoeui.ttf', 23)
    draw.text((60, 423), 'See where you’re looking.', font=tagline, fill='#bdbdb7')

    size = (705, 454)
    panel = practice.resize((size[0] - 2, size[1] - 2), Image.Resampling.LANCZOS)
    bordered = Image.new('RGB', size, '#8c8c85')
    bordered.paste(panel, (1, 1), rounded_mask(panel.size, 13))
    card.paste(bordered, (510, 93), rounded_mask(size, 14))
    card.save(output)


def main():
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['QT_SCALE_FACTOR'] = '1'
    os.environ['QT_SCREEN_SCALE_FACTORS'] = '1'
    app = QApplication([])
    for font in ('segoeui.ttf', 'seguisb.ttf', 'segoeuib.ttf'):
        QFontDatabase.addApplicationFont(str(FONTS / font))

    output = ROOT / 'docs' / 'images'
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='mira-readme-') as folder, patch.object(settings, 'LOCAL', Path(folder)):
        # These modules bind their data paths at import time.
        from context_operator.controller import Controller
        from context_operator.ui.window import Window

        defaults = dict(settings.DEFAULTS, vision=False, ranker=False, anticipation=False, click_learning=False)
        controller = Controller(defaults)
        controller.timer.stop()
        with patch.object(controller.win, 'bounds', return_value=Rect(0, 0, 1920, 1080)):
            window = Window(controller)
            window.refresh_timer.stop()
            try:
                window.resize(1120, 790)
                window.show()
                window.setFocus()
                app.processEvents()
                rgb_image(window.grab().toImage()).save(output / 'mira-overview.png')

                page = prepare_practice(window)
                app.processEvents()
                base = window.grab().toImage()
                elements = button_elements(window)
                render_gif(output / 'mira-gaze.gif', base, elements, controller.fusion)
                plus = next(element for element in elements if element.id == 'Plus')
                preview = paint_gaze(base, elements, target_point(plus), 1., Fusion())
                pos = page.mapTo(window, QPoint())
                preview = preview.crop((pos.x(), pos.y(), pos.x() + page.width(), pos.y() + page.height()))
                render_social(output / 'mira-social.png', preview)
            finally:
                window.close()


if __name__ == '__main__':
    main()
