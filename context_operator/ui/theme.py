"""Mira's look: charcoal layers, one blue accent, and the small shared widgets."""
import ctypes
from functools import lru_cache
import math
import sys

from PySide6.QtCore import Qt, QByteArray, QPointF, QRectF, QSize
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetrics, QGuiApplication, QIcon, QPainter,
    QPainterPath, QPen, QPixmap)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QAbstractButton, QCheckBox, QComboBox, QFrame, QLabel, QPushButton, QStyle,
    QStyleOptionComboBox, QVBoxLayout, QWidget)

GROUND = '#16181C'
RAIL = '#111317'
CARD = '#1F2228'
FIELD = '#272B32'
LINE = '#2E333B'
LINE_STRONG = '#3A404A'
OUTLINE = '#5A626E'
RING = '#4A515C'
TEXT = '#E9EBEE'
MUTED = '#A4ABB6'
FAINT = '#8A919C'
ACCENT = '#3D8BFF'
ACCENT_FILL = '#2F6FE4'
ACCENT_TEXT = '#7DB1FF'
FOCUS = '#5EA0FF'
WAITING = '#F5B544'
SUCCESS = '#3DD68C'
DANGER = '#FF6B5E'
# The overlay's own light and focus colours, for pictures of it inside the app.
GAZE_LIGHT = '#78CAFF'
GAZE_DOT = '#BFEFFF'
GAZE_FOCUS = '#F5CD69'

COLORS = {name: value for name, value in globals().items() if name.isupper() and isinstance(value, str)}

STYLE = """
QMainWindow, QDialog, QWidget#root, QWidget#page, QWidget#canvas, QStatusBar {background:@GROUND;}
QWidget {@BODY;font-size:14px;color:@TEXT;}
QFrame#rail {background:@RAIL;border:none;border-right:1px solid @LINE;}
QFrame#card, QWidget#guide {background:@CARD;border:1px solid @LINE;border-radius:12px;}
QFrame#inset {background:@RAIL;border:1px solid @LINE;border-radius:10px;}
QFrame#rule {background:@LINE;border:none;}
QLabel {background:transparent;border:none;}
QLabel#muted {color:@MUTED;}
QLabel#faint {@SMALL;font-size:12px;color:@FAINT;}
QLabel#title {@TITLE;font-size:26px;}
QLabel#heading {@STRONG;font-size:15px;}
QLabel#call {@STRONG;font-size:16px;}
QLabel#metric, QLabel#practiceCount {@STRONG;}
QLabel#step {@SMALL_STRONG;font-size:12px;background:@FIELD;border-radius:11px;}
QStatusBar {border-top:1px solid @LINE;color:@MUTED;@SMALL;font-size:12px;}
QStatusBar::item {border:none;}
QStatusBar QLabel {@SMALL;font-size:12px;color:@MUTED;}
QPushButton {@STRONG;background:@FIELD;border:1px solid @LINE_STRONG;border-radius:8px;padding:8px 16px;}
QPushButton:hover {background:#2F343C;}
QPushButton:pressed {background:#23272D;}
QPushButton:focus {border:2px solid @FOCUS;padding:7px 15px;}
QPushButton#primary {background:@ACCENT_FILL;border-color:@ACCENT_FILL;color:#FFFFFF;}
QPushButton#primary:hover {background:#2A66D6;border-color:#2A66D6;}
QPushButton#primary:pressed {background:#255BC0;border-color:#255BC0;}
QPushButton#primary:focus {border-color:#D6E6FF;}
QPushButton:disabled, QPushButton#primary:disabled {background:#1B1E23;border-color:@LINE;color:#6B727D;}
QPushButton#advanced {text-align:left;}
QPushButton#advanced:checked {background:#2F343C;}
QPushButton#link {
    background:transparent;border:none;border-bottom:2px solid transparent;color:@ACCENT_TEXT;
    padding:3px 0 1px 0;text-align:left;
}
QPushButton#link:hover {text-decoration:underline;}
QPushButton#link:focus {border-bottom-color:@FOCUS;}
QPushButton#practiceTarget {background:@CARD;border:1px solid @LINE_STRONG;border-radius:12px;padding:10px 12px;}
QPushButton#practiceTarget:hover {background:@FIELD;}
QPushButton#practiceTarget:pressed {background:#2F343C;}
QPushButton#practiceTarget:focus {border:2px solid @FOCUS;padding:9px 11px;}
QPushButton#practiceTarget:disabled {background:#191C20;border-color:@LINE;color:#6B727D;}
QLineEdit, QPlainTextEdit, QComboBox {
    background:@GROUND;border:1px solid @LINE_STRONG;border-radius:8px;padding:8px 10px;
    selection-background-color:@ACCENT_FILL;selection-color:#FFFFFF;
}
QLineEdit, QPlainTextEdit {placeholder-text-color:@FAINT;}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {border:2px solid @FOCUS;padding:7px 9px;}
QLineEdit:disabled, QComboBox:disabled {color:#6B727D;border-color:@LINE;}
QComboBox {padding-right:30px;}
QComboBox:focus {padding-right:29px;}
QComboBox::drop-down {subcontrol-origin:padding;subcontrol-position:top right;width:26px;border:none;}
QComboBox QAbstractItemView {
    background:@CARD;border:1px solid @LINE_STRONG;outline:0;
    selection-background-color:@ACCENT_FILL;selection-color:#FFFFFF;
}
QSlider::groove:horizontal {background:@LINE_STRONG;height:4px;border-radius:2px;}
QSlider::sub-page:horizontal {background:@ACCENT;border-radius:2px;}
QSlider::handle:horizontal {background:#FFFFFF;border:2px solid @CARD;width:12px;margin:-6px 0;border-radius:8px;}
QSlider:focus::handle:horizontal {border-color:@FOCUS;}
QScrollArea {background:transparent;border:none;}
QScrollBar:vertical {background:transparent;width:10px;}
QScrollBar:horizontal {background:transparent;height:10px;}
QScrollBar::handle {background:@LINE_STRONG;border-radius:4px;min-height:30px;min-width:30px;margin:1px;}
QScrollBar::handle:hover {background:@OUTLINE;}
QScrollBar::add-line, QScrollBar::sub-line {height:0px;width:0px;}
QScrollBar::add-page, QScrollBar::sub-page {background:transparent;}
QToolTip {background:@FIELD;color:@TEXT;border:1px solid @LINE_STRONG;padding:6px 8px;}
"""

CHEVRON = QByteArray(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 8">'
                    b'<path d="M1 1l5 5 5-5" stroke="#A4ABB6" stroke-width="1.5" fill="none" '
                    b'stroke-linecap="round" stroke-linejoin="round"/></svg>')


@lru_cache(maxsize=None)
def faces():
    """Font family and weight per role. Segoe UI Variable where Windows has it."""
    def first(*candidates):
        return next((face for face in candidates if QFontDatabase.hasFamily(face[0])), candidates[-1])
    return {
        'BODY': first(('Segoe UI Variable Text', 400), ('Segoe UI', 400)),
        'STRONG': first(('Segoe UI Variable Text Semibold', 400), ('Segoe UI', 600)),
        'TITLE': first(('Segoe UI Variable Display Semibold', 400), ('Segoe UI', 600)),
        'SMALL': first(('Segoe UI Variable Small', 400), ('Segoe UI', 400)),
        'SMALL_STRONG': first(('Segoe UI Variable Small Semibold', 400), ('Segoe UI', 600)),
    }


def ui_font(role, pixels):
    family, weight = faces()[role]
    font = QFont(family)
    font.setPixelSize(pixels)
    font.setWeight(QFont.Weight(weight))
    return font


def theme():
    tokens = {'@' + role: f'font-family:"{family}";font-weight:{weight}' for role, (family, weight) in faces().items()}
    tokens.update({'@' + name: value for name, value in COLORS.items()})
    style = STYLE
    for token in sorted(tokens, key=len, reverse=True):
        style = style.replace(token, tokens[token])
    return style


def dark_title_bar(window):
    """Give the native title bar the app's ground. Windows 10 gets the dark frame only."""
    if sys.platform != 'win32' or QGuiApplication.platformName() != 'windows':
        return
    def colorref(color):
        value = QColor(color)
        return value.red() | value.green() << 8 | value.blue() << 16
    handle = int(window.winId())
    for attribute, value in ((20, 1), (35, colorref(GROUND)), (36, colorref(TEXT))):
        data = ctypes.c_int(value)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(handle, attribute, ctypes.byref(data), ctypes.sizeof(data))


def keyboard_focus(root):
    """Buttons take focus from the keyboard only, so a click never leaves a focus ring behind."""
    for button in root.findChildren(QAbstractButton):
        button.setFocusPolicy(Qt.TabFocus)


def label(text, style=None, wrap=False):
    widget = QLabel(text)
    widget.setWordWrap(wrap)
    widget.setTextFormat(Qt.PlainText)
    if style:
        widget.setObjectName(style)
    return widget


def card():
    frame = QFrame()
    frame.setObjectName('card')
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(10)
    return frame, layout


def rule():
    line = QFrame()
    line.setObjectName('rule')
    line.setFixedHeight(1)
    return line


def set_primary(button, primary):
    """Make a button the filled one, or an ordinary one, as the step it serves comes and goes."""
    name = 'primary' if primary else ''
    if button.objectName() != name:
        button.setObjectName(name)
        button.style().unpolish(button)
        button.style().polish(button)


def swatch(color, *, dot=False):
    mark = QFrame()
    mark.setFixedSize(*((8, 8) if dot else (12, 9)))
    mark.setStyleSheet(f'background:{color};border-radius:4px;' if dot else f'border:2px solid {color};border-radius:2px;')
    return mark


STATE_COLORS = {'off': FAINT, 'standby': WAITING, 'go': ACCENT, 'hold': DANGER}


class Lamp(QWidget):
    """A state dot. The word beside it always says the state; the dot repeats it."""
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.setFixedSize(8, 8)
        self.set_state('off', 'Off')

    def set_state(self, state, word):
        self.setAccessibleName(f'{self.name}, {word.lower()}')
        if getattr(self, 'state', None) != state:
            self.state = state
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(STATE_COLORS[self.state])
        if self.state == 'off':
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(color, 1.5))
            painter.drawEllipse(QRectF(self.rect()).adjusted(1, 1, -1, -1))
        else:
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QRectF(self.rect()))


class Pill(QWidget):
    """The page's state in one word, with its dot."""
    TEXT_COLORS = {'off': MUTED, 'standby': '#F8CE85', 'go': '#9CC4FF', 'hold': '#FF9A90'}

    def __init__(self):
        super().__init__()
        self.state, self.word = 'off', ''

    def set_state(self, state, word):
        if (state, word) != (self.state, self.word):
            self.state, self.word = state, word
            self.setAccessibleName(word)
            self.updateGeometry()
            self.update()

    def sizeHint(self):
        return QSize(QFontMetrics(ui_font('SMALL_STRONG', 12)).horizontalAdvance(self.word) + 36, 26)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        fill = QColor(FIELD if self.state == 'off' else STATE_COLORS[self.state])
        if self.state != 'off':
            fill.setAlpha(40)
        painter.setPen(Qt.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(QRectF(self.rect()), 13, 13)
        dot = QRectF(11, self.height()/2 - 3.5, 7, 7)
        if self.state == 'off':
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(FAINT), 1.5))
            painter.drawEllipse(dot.adjusted(.75, .75, -.75, -.75))
        else:
            painter.setBrush(QColor(STATE_COLORS[self.state]))
            painter.drawEllipse(dot)
        painter.setPen(QColor(self.TEXT_COLORS[self.state]))
        painter.setFont(ui_font('SMALL_STRONG', 12))
        painter.drawText(QRectF(25, 0, self.width() - 34, self.height()), Qt.AlignVCenter | Qt.AlignLeft, self.word)


class Switch(QCheckBox):
    """A setting row: its name on the left, the switch on the right."""
    def __init__(self, text=''):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)

    def sizeHint(self):
        metrics = self.fontMetrics()
        return QSize(metrics.horizontalAdvance(self.text()) + 54, max(28, metrics.height() + 8))

    def minimumSizeHint(self):
        return self.sizeHint()

    def hitButton(self, position):
        return self.rect().contains(position)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        enabled, on = self.isEnabled(), self.isChecked()
        track = QRectF(self.width() - 39, (self.height() - 20)/2, 36, 20)
        painter.setFont(self.font())
        painter.setPen(QColor(TEXT if enabled else '#6B727D'))
        painter.drawText(QRectF(0, 0, track.left() - 12, self.height()), Qt.AlignVCenter | Qt.AlignLeft, self.text())
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor((ACCENT if on else OUTLINE) if enabled else LINE_STRONG))
        painter.drawRoundedRect(track, 10, 10)
        painter.setBrush(QColor('#FFFFFF' if enabled else FAINT))
        painter.drawEllipse(QRectF(track.right() - 18 if on else track.left() + 2, track.top() + 2, 16, 16))
        if self.hasFocus():
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(FOCUS), 2))
            painter.drawRoundedRect(track.adjusted(-2, -2, 2, 2), 12, 12)


def nav_icon(name):
    """Line icons on a 24 px grid, one stroke weight."""
    path = QPainterPath()
    if name == 'Overview':
        for x, y, w, h in ((4, 4, 7, 9), (13, 4, 7, 5), (13, 11, 7, 9), (4, 15, 7, 5)):
            path.addRoundedRect(QRectF(x, y, w, h), 1.6, 1.6)
    elif name == 'Calibration':
        path.addEllipse(QPointF(12, 12), 7.5, 7.5)
        path.addEllipse(QPointF(12, 12), 2.4, 2.4)
        for start, end in (((12, 1.5), (12, 4.5)), ((12, 19.5), (12, 22.5)), ((1.5, 12), (4.5, 12)), ((19.5, 12), (22.5, 12))):
            path.moveTo(*start)
            path.lineTo(*end)
    elif name == 'Practice':
        path.moveTo(6.5, 4)
        path.lineTo(18.8, 10.4)
        path.lineTo(13.3, 12.5)
        path.lineTo(11, 18.6)
        path.closeSubpath()
    else:
        for tooth in range(8):
            for radius, offset in ((7.2, -14), (9.4, -8), (9.4, 8), (7.2, 14)):
                angle = math.radians(tooth*45 + offset)
                point = QPointF(12 + radius*math.cos(angle), 12 + radius*math.sin(angle))
                if path.elementCount():
                    path.lineTo(point)
                else:
                    path.moveTo(point)
        path.closeSubpath()
        path.addEllipse(QPointF(12, 12), 3, 3)
    return path


class NavButton(QPushButton):
    """A destination in the rail: its icon over its name."""
    def __init__(self, text):
        super().__init__(text)
        self.icon_path = nav_icon(text)
        self.setCheckable(True)
        self.setFixedSize(60, 58)
        self.setAttribute(Qt.WA_Hover)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        on = self.isChecked()
        if on or self.underMouse() or self.isDown():
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(CARD if on else '#1A1D22'))
            painter.drawRoundedRect(QRectF(self.rect()), 10, 10)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(ACCENT_TEXT if on else MUTED), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.save()
        painter.translate((self.width() - 24)/2, 8)
        painter.drawPath(self.icon_path)
        painter.restore()
        painter.setPen(QColor(TEXT if on else MUTED))
        painter.setFont(ui_font('SMALL_STRONG' if on else 'SMALL', 11))
        painter.drawText(QRectF(0, 35, self.width(), 16), Qt.AlignCenter, self.text())
        if self.hasFocus():
            painter.setPen(QPen(QColor(FOCUS), 2))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 9, 9)


class SettingsComboBox(QComboBox):
    """Keep a visible chevron with the flat input style."""
    def __init__(self):
        super().__init__()
        self.chevron = QSvgRenderer(CHEVRON, self)

    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        arrow = self.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxArrow, self)
        painter = QPainter(self)
        self.chevron.render(painter, QRectF(arrow.center().x()-5, arrow.center().y()-3, 10, 7))


def disclosure_icon():
    icon = QIcon()
    svg = QSvgRenderer(CHEVRON)
    for state, angle in ((QIcon.Off, -90), (QIcon.On, 0)):
        pixmap = QPixmap(32, 32)
        pixmap.setDevicePixelRatio(2)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.translate(8, 8)
        painter.rotate(angle)
        svg.render(painter, QRectF(-6, -4, 12, 8))
        painter.end()
        icon.addPixmap(pixmap, QIcon.Normal, state)
    return icon


def connect_disclosure(button, region):
    button.setCheckable(True)
    button.setIcon(disclosure_icon())
    button.setAccessibleDescription('Collapsed')
    button.toggled.connect(region.setVisible)
    button.toggled.connect(lambda expanded: button.setAccessibleDescription('Expanded' if expanded else 'Collapsed'))
    region.hide()


def paint_progress(painter, area, total, completed):
    """Show phase progress in the chrome, away from the target positions."""
    painter.setRenderHint(QPainter.Antialiasing)
    for index in range(total):
        painter.setBrush(QColor(SUCCESS) if index < completed else Qt.NoBrush)
        painter.setPen(QPen(QColor(SUCCESS if index < completed else WAITING if index == completed else OUTLINE), 1.5))
        painter.drawEllipse(QRectF(area.x() + index*15, area.y(), 8, 8))
