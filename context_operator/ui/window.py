"""Daily operator UI with one personal setup and no experimental model menus."""
from collections import Counter
import subprocess
import threading
import time
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, QSignalBlocker, QByteArray, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QFontDatabase, QIcon, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QStackedWidget, QLineEdit,
    QPlainTextEdit, QSlider, QCheckBox, QComboBox, QScrollArea, QSizePolicy,
    QFormLayout, QStatusBar, QStyle, QStyleOptionComboBox)
from ..core.webcam import fresh as fresh_gaze
from ..foundations.speech.runtime import microphone_matches
from ..providers.models import Speech
from ..providers.phone_connection import PhoneConnection, WIFI_PHONE_NEXT_STEP, WifiPhoneNotFound, find_phones
from ..settings import ASSETS, LOCAL, save
from .overlay import paint_scene_light,paint_magnetic_focus,current_focus_target
from .light_cache import LightCache
from .overview_cache import OverviewCache

STYLE = """
QMainWindow, QDialog, QWidget#root, QWidget#page, QStatusBar {background:#F3F3F0;}
QWidget {font-family:"BODY_FONT";font-size:14px;color:#141414;}
QFrame#sidebar {background:#141414;}
QFrame#panel, QWidget#guide {background:#FAFAF8;border:1px solid #D9D9D3;border-radius:10px;}
QFrame#cue {background:#141414;border-radius:14px;}
QFrame#mapPanel {background:#141414;border-radius:10px;}
QFrame#practiceStage {background:#141414;border-radius:14px;}
QLabel#practiceCount {font-family:"MONO_FONT";font-size:16px;font-weight:600;}
QPushButton#practiceTarget {
    background:#282826;color:#F2F2EE;border:2px solid #656560;border-radius:12px;
    padding:10px 12px;
}
QPushButton#practiceTarget:hover {background:#343431;}
QPushButton#practiceTarget:pressed {background:#40403B;}
QPushButton#practiceTarget:focus {border-color:#83A1FF;}
QPushButton#practiceTarget:disabled {background:#222220;color:#A6A6A0;border-color:#44443F;}
QFrame#rule {background:#DCDCD6;border:none;}
QFrame#railRule {background:#4A4A4A;border:none;}
QLabel {background:transparent;border:none;}
QLabel#muted, QStatusBar {color:#5C5C57;}
QLabel#title, QLabel#heading, QLabel#brand, QLabel#call, QLabel#guideHeading {
    font-weight:600;
}
QLabel#title {font-size:28px;}
QLabel#heading, QLabel#guideHeading {font-size:16px;}
QLabel#brand {color:#F2F2EE;font-size:26px;font-weight:700;}
QLabel#railText {color:#F2F2EE;}
QLabel#railMuted {color:#A6A6A0;font-size:12px;}
QLabel#call {color:#FFFFFF;font-size:30px;}
QLabel#cueDetail {color:#BDBDB7;}
QLabel#metric, QLabel#step {font-family:"MONO_FONT";}
QLabel#step {color:#FFFFFF;background:#141414;border-radius:11px;font-size:12px;}
QPushButton {background:#FAFAF8;border:1px solid #8C8C85;border-radius:8px;padding:10px 18px;font-weight:600;}
QPushButton:hover {background:#EDEDE8;}
QPushButton:pressed {background:#DCDCD6;}
QPushButton:focus {border:2px solid #3D6BFF;padding:9px 17px;}
QPushButton#primary {background:#141414;color:#FFFFFF;border-color:#141414;}
QPushButton#primary:hover {background:#2A2A2A;}
QPushButton#primary:pressed {background:#000000;}
QPushButton#inverse {background:#FFFFFF;color:#141414;border-color:#FFFFFF;}
QPushButton#inverse:hover {background:#EDEDE8;}
QPushButton#inverse:pressed {background:#DCDCD6;}
QPushButton#primary:focus, QPushButton#inverse:focus {border-color:#3D6BFF;}
QPushButton:disabled, QPushButton#primary:disabled, QPushButton#inverse:disabled {
    background:#D9D9D3;color:#8A8A85;border-color:#D9D9D3;
}
QPushButton#nav {text-align:left;background:transparent;border:2px solid transparent;color:#F2F2EE;padding:8px 12px;}
QPushButton#nav:hover, QPushButton#nav:checked {background:#2C2C2C;}
QPushButton#nav:focus {border-color:#3D6BFF;}
QPushButton#advanced {text-align:left;}
QPushButton#advanced:checked {background:#EDEDE8;}
QPushButton#link {background:transparent;border:2px solid transparent;color:#5C5C57;padding:4px 6px;}
QPushButton#link:hover {color:#141414;text-decoration:underline;}
QPushButton#link:focus {border-color:#3D6BFF;}
QLineEdit,QPlainTextEdit,QComboBox {
    background:#FFFFFF;border:1px solid #8C8C85;border-radius:8px;padding:8px 10px;
    selection-background-color:#2C4FD1;selection-color:#FFFFFF;
}
QLineEdit,QPlainTextEdit {placeholder-text-color:#5C5C57;}
QLineEdit:focus,QPlainTextEdit:focus,QComboBox:focus {border:2px solid #3D6BFF;padding:7px 9px;}
QComboBox {padding-right:30px;}
QComboBox:focus {padding-right:29px;}
QComboBox::drop-down {subcontrol-origin:padding;subcontrol-position:top right;width:26px;border:none;}
QComboBox QAbstractItemView {background:#FFFFFF;color:#141414;selection-background-color:#2C4FD1;selection-color:#FFFFFF;}
QSlider::groove:horizontal {background:#D9D9D3;height:4px;border-radius:2px;}
QSlider::sub-page:horizontal {background:#141414;border-radius:2px;}
QSlider::handle:horizontal {background:#141414;border:2px solid #FFFFFF;width:12px;margin:-6px 0;border-radius:8px;}
QSlider:focus::handle:horizontal {border-color:#3D6BFF;}
QCheckBox {spacing:9px;padding:3px 0;}
QCheckBox::indicator {width:14px;height:14px;border:1.5px solid #141414;border-radius:4px;background:#FFFFFF;}
QCheckBox::indicator:checked {background:#141414;image:url("CHECKMARK");}
QCheckBox::indicator:focus {border-color:#3D6BFF;}
QCheckBox:disabled {color:#8A8A85;}
QCheckBox::indicator:disabled {border-color:#8A8A85;}
QCheckBox::indicator:checked:disabled {background:#8A8A85;}
QScrollArea {background:transparent;border:none;}
QScrollBar:vertical {background:transparent;width:8px;}
QScrollBar:horizontal {background:transparent;height:8px;}
QScrollBar::handle {background:#C9C9C2;border-radius:4px;min-height:30px;}
QScrollBar::add-line,QScrollBar::sub-line {height:0px;width:0px;}
QScrollBar::add-page,QScrollBar::sub-page {background:transparent;}
QToolTip {background:#FAFAF8;color:#141414;border:1px solid #D9D9D3;padding:8px;}
"""

DARK_SETUP_STYLE = """
QMainWindow, QWidget#canvas {background:#141414;}
QWidget#canvas QLabel#title {color:#F2F2EE;}
QWidget#canvas QLabel#stageMuted {color:#A6A6A0;}
"""

GO_CONFIRMATION_SECONDS = 2.0

CHEVRON = QByteArray(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 8">'
                    b'<path d="M1 1l5 5 5-5" stroke="#5C5C57" stroke-width="1.5" fill="none" '
                    b'stroke-linecap="round" stroke-linejoin="round"/></svg>')

def theme(extra=""):
    fonts = {
        'BODY_FONT': ('Segoe UI Variable Text', 'Segoe UI Variable', 'Segoe UI'),
        'MONO_FONT': ('Cascadia Mono', 'Consolas'),
    }
    style = STYLE + extra
    for token, families in fonts.items():
        family = next((name for name in families if QFontDatabase.hasFamily(name)), families[-1])
        style = style.replace(token, family)
    return style.replace('CHECKMARK', (ASSETS / 'check.svg').as_posix())


class Lamp(QWidget):
    """An accessible cue light with an explicit state and no animation."""
    COLORS = {'off': '#3A3A3A', 'standby': '#FFB020', 'go': '#2DBE60', 'hold': '#E5483B'}

    def __init__(self, name, diameter=10, *, on_stage=False):
        super().__init__()
        self.name = name
        self.off_ring = '#4A4A4A' if on_stage else '#BDBDB6'
        self.setFixedSize(diameter, diameter)
        self.set_state('off', 'Off')

    def set_state(self, state, word):
        self.setAccessibleName(f'{self.name} lamp, {word.lower()}')
        if getattr(self, 'state', None) != state:
            self.state = state
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(self.COLORS[self.state]))
        if self.state == 'off':
            painter.setPen(QPen(QColor(self.off_ring), 2))
            circle = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        else:
            painter.setPen(Qt.NoPen)
            circle = QRectF(self.rect())
        painter.drawEllipse(circle)


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
    button.toggled.connect(lambda expanded:button.setAccessibleDescription('Expanded' if expanded else 'Collapsed'))
    region.hide()


def paint_progress(painter, area, total, completed):
    """Show phase progress in the chrome, away from the target positions."""
    painter.setRenderHint(QPainter.Antialiasing)
    for index in range(total):
        current = index == completed
        painter.setBrush(QColor('#2DBE60') if index < completed else Qt.NoBrush)
        painter.setPen(QPen(QColor('#FFB020' if current else '#2DBE60' if index < completed else '#4A4A4A'), 1.5))
        painter.drawEllipse(QRectF(area.x() + index*15, area.y(), 8, 8))


def rule(*, on_stage=False):
    line = QFrame()
    line.setObjectName('railRule' if on_stage else 'rule')
    line.setFixedHeight(1)
    return line


def swatch(color, *, dot=False):
    mark = QFrame()
    mark.setFixedSize(*( (8, 8) if dot else (12, 9)))
    decoration = f'background:{color};border-radius:4px;' if dot else f'border:2px solid {color};border-radius:2px;'
    mark.setStyleSheet(decoration)
    return mark


def label(text, style=None, wrap=False):
    widget=QLabel(text);widget.setWordWrap(wrap);widget.setTextFormat(Qt.PlainText)
    if style:widget.setObjectName(style)
    if style in ('brand', 'title', 'heading', 'call', 'guideHeading'):
        family = 'Segoe UI Variable'
        if QFontDatabase.hasFamily(family):
            face = 'Bold Display' if style == 'brand' else 'Semibold Display'
            widget.setFont(QFontDatabase.font(family, face, 14))
    return widget


def panel(name='panel'):
    frame=QFrame();frame.setObjectName(name);layout=QVBoxLayout(frame)
    layout.setContentsMargins(20,16,20,16);layout.setSpacing(10)
    return frame,layout


class SceneCanvas(QWidget):
    """Numeric element geometry, never a saved screen image."""
    def __init__(self,controller):
        super().__init__();self.controller=controller;self.setMinimumHeight(180)
        self.setAccessibleName('Live map of screen elements and gaze focus')
        self.overview=OverviewCache();self.light_cache=LightCache(max_pixels=1_000_000)

    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing)
        area=QRectF(1,1,self.width()-2,self.height()-2)
        p.setPen(Qt.NoPen);p.setBrush(QColor('#141414'));p.drawRoundedRect(area,10,10)
        if self.controller.settings.get('quiet_mode',False):return
        c=self.controller;scene=c.scene;now=time.monotonic()
        elements=scene.current_elements(now) if scene and 0<=now-scene.captured<3 else ()
        if elements:
            bounds=scene.bounds;scale=min((self.width()-48)/max(1,bounds.w),(self.height()-40)/max(1,bounds.h))
            ox=(self.width()-bounds.w*scale)/2;oy=(self.height()-bounds.h*scale)/2
            origin=(bounds.x-ox/scale,bounds.y-oy/scale)
            rects=[QRectF(ox+(e.rect.x-bounds.x)*scale,oy+(e.rect.y-bounds.y)*scale,
                         max(3,e.rect.w*scale),max(3,e.rect.h*scale)) for e in elements]
            self.overview.paint(p,rects,self.width(),self.height())
            gaze=c.attention if c.attention and c.attention.valid and 0<=now-c.attention.captured<.25 else None
            prediction=c.prediction if c.prediction and c.prediction.is_fresh(now) else None
            magnetic=c.settings.get('highlight_mode')=='magnetic'
            selected_until=c.fusion.selection_until(c.focus.target,now,gaze)
            target=current_focus_target(scene,c.focus.target,gaze,now,selected_until)
            if magnetic:
                paint_magnetic_focus(p,scene,target,now,origin,scale)
            else:
                paint_scene_light(p,scene,QPointF(gaze.x,gaze.y) if gaze else None,
                                  gaze.radius if gaze else 230.,now,target.id if target and not selected_until else None,
                                  origin=origin,scale=scale,prediction=prediction,cache=self.light_cache)
                if selected_until:
                    paint_magnetic_focus(p,scene,target,now,origin,scale)
            if gaze and not magnetic and c.settings.get('show_gaze',True):
                p.setPen(Qt.NoPen);p.setBrush(QColor('#bfefff'))
                p.drawEllipse(QPointF(ox+(gaze.x-bounds.x)*scale,oy+(gaze.y-bounds.y)*scale),2.5,2.5)
        else:
            self.overview.paint(p,[],self.width(),self.height())
            p.setFont(self.font());p.setPen(QColor('#A6A6A0'))
            message = ('Mira is waiting for the screen map.' if c.running else
                       'Select “Start” to show the screen map.')
            p.drawText(area.adjusted(16,0,-16,0), Qt.AlignCenter | Qt.TextWordWrap, message)



class Window(QMainWindow):
    phone_search_finished = Signal(object, object, object)
    microphone_list_finished = Signal(object)

    def __init__(self,controller):
        super().__init__();self.c=controller;self.settings=controller.settings;self.overlays=[];self.setup_window=None
        self.setWindowTitle('Mira');self.resize(1120,760);self.setMinimumSize(900,640)
        self.setWindowIcon(QIcon(str(ASSETS/'icon.png')));self.setStyleSheet(theme())
        self.last_paint=0.;self.counter=0;self.practice_active=False;self._profile_stamp=None;self._profile_status={}
        self._practice_started_control = False
        self._phone_search = None
        self._microphone_listing = False
        self._saved_phone_host = self.settings.get('phone_wifi_host', '')
        self.phone_search_finished.connect(self.finish_phone_search)
        self.microphone_list_finished.connect(self.show_microphones)
        root=QWidget();root.setObjectName('root');self.setCentralWidget(root)
        layout=QHBoxLayout(root);layout.setContentsMargins(0,0,0,0);layout.setSpacing(0)
        side=QFrame();side.setObjectName('sidebar');side.setFixedWidth(216)
        sl=QVBoxLayout(side);sl.setContentsMargins(18,23,18,18);sl.setSpacing(6)
        brand=QHBoxLayout();brand.setSpacing(9)
        logo=label('');logo.setPixmap(self.windowIcon().pixmap(34,34))
        brand.addWidget(logo);brand.addWidget(label('Mira','brand'),1)
        sl.addLayout(brand);sl.addSpacing(21)
        self.stack=QStackedWidget();self.nav=[]
        for index,text in enumerate(('Overview','Profile','Practice','Settings')):
            button=QPushButton(text);button.setObjectName('nav');button.setCheckable(True)
            button.clicked.connect(lambda _,i=index:self.navigate(i));self.nav.append(button)
            sl.addWidget(button)
        sl.addStretch()
        sl.addWidget(rule(on_stage=True));sl.addSpacing(7)
        sl.addWidget(label('Sources','railText'))
        self.rail_sources={}
        for name in ('iPhone','Microphone'):
            row=QHBoxLayout();row.setSpacing(8)
            lamp=Lamp(name,on_stage=True);state=label('Off','railMuted')
            row.addWidget(lamp);row.addWidget(label(name,'railText'),1);row.addWidget(state)
            sl.addLayout(row);self.rail_sources[name]=(lamp,state)
        sl.addSpacing(9)
        sl.addWidget(label('Ctrl + Alt + F12 stops control','railMuted',True))
        sl.addWidget(label('Version 1.3.1','railMuted'))

        content=QWidget();content_layout=QVBoxLayout(content)
        content_layout.setContentsMargins(0,0,0,0);content_layout.setSpacing(0)
        content_layout.addWidget(self.stack,1)
        self.content_status=QStatusBar(content)
        self.content_status.setSizeGripEnabled(False)
        self.content_status.setContentsMargins(26,0,26,0)
        content_layout.addWidget(self.content_status)
        self.status=label('Ready to connect','muted')
        self.content_status.addWidget(self.status,1)
        layout.addWidget(side);layout.addWidget(content,1)
        self.build_control();self.build_profile();self.build_practice();self.build_settings();self.navigate(0)
        self.c.changed.connect(self.refresh);self.c.message.connect(self.add_message)
        self.refresh_timer=QTimer(self);self.refresh_timer.timeout.connect(self.refresh);self.refresh_timer.start(150)
        self.refresh()

    def page(self,title,description):
        page=QWidget();page.setObjectName('page');layout=QVBoxLayout(page)
        layout.setContentsMargins(28,20,28,18);layout.setSpacing(10)
        header=QVBoxLayout();header.setSpacing(3)
        header.addWidget(label(title,'title'));header.addWidget(label(description,'muted',True))
        layout.addLayout(header)
        area=QScrollArea();area.setWidgetResizable(True);area.setWidget(page);self.stack.addWidget(area)
        return layout

    def build_control(self):
        layout=self.page('Overview','See the area of the screen you’re looking at.')
        self.build_phone_card(layout)

        self.control_panel,box=panel('cue');box.setContentsMargins(22,17,22,17)
        row=QHBoxLayout();row.setSpacing(20)
        call = QWidget()
        call.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        column = QVBoxLayout(call)
        column.setContentsMargins(0, 0, 0, 0);column.setSpacing(2)
        self.target = label('Paused', 'call', True)
        self.reason = label('', 'cueDetail', True)
        column.addWidget(self.target)
        column.addWidget(self.reason)
        row.addWidget(call, 1)
        self.run = QPushButton('Start')
        self.run.setObjectName('inverse')
        self.run.setMinimumWidth(84)
        self.run.clicked.connect(lambda: self.c.pause() if self.c.running else self.c.start())
        row.addWidget(self.run, 0, Qt.AlignVCenter)
        box.addLayout(row);layout.addWidget(self.control_panel)

        self.more_controls = QPushButton('More controls')
        self.more_controls.setObjectName('advanced')
        layout.addWidget(self.more_controls, 0, Qt.AlignLeft)
        controls = QWidget()
        details = QVBoxLayout(controls)
        details.setContentsMargins(0, 0, 0, 0)
        connect_disclosure(self.more_controls, controls)
        self.more_controls.toggled.connect(self.refresh_control_panel)
        layout.addWidget(controls)
        frame, box = panel()
        row = QHBoxLayout()
        self.voice_lamp = Lamp('Microphone')
        row.addWidget(self.voice_lamp)
        column = QVBoxLayout()
        column.addWidget(label('Microphone', 'heading'))
        self.voice_value = label('Off', 'muted', True)
        column.addWidget(self.voice_value)
        row.addLayout(column, 1)
        self.voice_btn = QPushButton('Turn on microphone')
        self.voice_btn.clicked.connect(self.c.toggle_microphone)
        row.addWidget(self.voice_btn)
        box.addLayout(row)
        details.addWidget(frame)

        self.scene_map_button=QPushButton('Show screen map')
        self.scene_map_button.clicked.connect(lambda:self.update_setting(
            'show_scene_map', not self.settings.get('show_scene_map',False)))
        details.addWidget(self.scene_map_button,0,Qt.AlignLeft)
        self.focus_panel=QFrame();self.focus_panel.setObjectName('mapPanel')
        self.focus_panel.setMaximumHeight(350)
        box=QVBoxLayout(self.focus_panel)
        box.setContentsMargins(0,0,0,0);box.setSpacing(0)
        self.canvas=SceneCanvas(self.c);box.addWidget(self.canvas,1)
        legend_strip=QWidget();legend_strip.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed)
        legend=QHBoxLayout(legend_strip);legend.setContentsMargins(12,0,12,10);legend.setSpacing(16)
        self.legend_gaze=QWidget();gaze_legend=QHBoxLayout(self.legend_gaze)
        gaze_legend.setContentsMargins(0,0,0,0);gaze_legend.setSpacing(6)
        gaze_legend.addWidget(swatch('#86DDFF',dot=True));gaze_legend.addWidget(label('Gaze dot','railMuted'))
        legend.addWidget(self.legend_gaze)
        action_legend=QHBoxLayout();action_legend.setSpacing(6)
        action_legend.addWidget(swatch('#F5CD69'));action_legend.addWidget(label('Yellow frame · current focus','railMuted'))
        legend.addLayout(action_legend);legend.addStretch()
        self.latency=label('','railMuted');legend.addWidget(self.latency)
        box.addWidget(legend_strip);details.addWidget(self.focus_panel)
        layout.addStretch()

    def build_phone_card(self, layout):
        frame, box = panel()
        box.setContentsMargins(22, 18, 22, 18)
        self.gaze_heading = label('iPhone gaze tracking', 'heading', True)
        self.gaze_value = label('', 'muted', True)
        box.addWidget(self.gaze_heading)
        box.addWidget(self.gaze_value)

        row = QHBoxLayout()
        self.gaze_btn = QPushButton('Set up iPhone')
        self.gaze_btn.setObjectName('primary')
        self.gaze_btn.clicked.connect(self.activate_phone)
        row.addWidget(self.gaze_btn)
        self.mouse_preview = QPushButton('Try with mouse')
        self.mouse_preview.clicked.connect(self.open_mouse_preview)
        row.addWidget(self.mouse_preview)
        self.disconnect_phone = QPushButton('Disconnect iPhone')
        self.disconnect_phone.clicked.connect(self.c.toggle_phone)
        row.addWidget(self.disconnect_phone)
        row.addStretch()
        box.addLayout(row)

        links = QHBoxLayout()
        self.phone_settings_button = QPushButton('Connection settings')
        self.phone_settings_button.setObjectName('link')
        self.phone_settings_button.clicked.connect(self.open_phone_settings)
        links.addWidget(self.phone_settings_button)
        self.use_wifi_button = QPushButton('Use Wi‑Fi instead')
        self.use_wifi_button.setObjectName('link')
        self.use_wifi_button.clicked.connect(self.use_wifi)
        links.addWidget(self.use_wifi_button)
        links.addStretch()
        box.addLayout(links)
        layout.addWidget(frame)

    def build_profile(self):
        layout=self.page('Profile','Setup creates your gaze profile. Click learning can refine it.')
        frame,box=panel();self.profile_title=label('Set up your gaze profile','heading',True)
        self.profile_detail=label('','muted',True);box.addWidget(self.profile_title);box.addWidget(self.profile_detail)
        steps=QHBoxLayout();steps.setSpacing(12)
        for number,title,hint in (
            ('1','Connect','Sit comfortably'),('2','Gaze setup','9 points, about a minute'),('3','Check','5 new points'),
        ):
            step=QHBoxLayout();step.setSpacing(7);step.setAlignment(Qt.AlignTop)
            number_label=label(number,'step');number_label.setFixedSize(22,22);number_label.setAlignment(Qt.AlignCenter)
            step.addWidget(number_label,0,Qt.AlignTop)
            caption=QVBoxLayout();caption.setSpacing(2)
            caption.addWidget(label(title,'heading'))
            caption.addWidget(label(hint,'muted',True))
            step.addLayout(caption);steps.addLayout(step,1)
        box.addLayout(steps);row=QHBoxLayout();self.profile_setup=QPushButton('Set up · about a minute')
        self.profile_setup.setObjectName('primary');self.profile_setup.clicked.connect(self.open_setup)
        row.addWidget(self.profile_setup);row.addStretch();box.addLayout(row);layout.addWidget(frame)
        self.no_setup_results=label('No setup results yet','muted')
        layout.addWidget(self.no_setup_results)
        self.validation_panel,box=panel()
        row=QHBoxLayout()
        self.validation_verdict=label('','heading',True);row.addWidget(self.validation_verdict,1)
        self.validation_toggle=QPushButton('Details');self.validation_toggle.setObjectName('advanced')
        row.addWidget(self.validation_toggle);box.addLayout(row)
        self.validation_details=QWidget();details=QVBoxLayout(self.validation_details)
        details.setContentsMargins(0,0,0,0);details.setSpacing(8)
        box.addWidget(self.validation_details)
        connect_disclosure(self.validation_toggle,self.validation_details)
        self.validation_hint=label('','muted',True)
        details.addWidget(self.validation_hint);self.metrics={}
        for key,title in (('median_px','Typical error'),('p90_px','Hard points'),('coverage','Coverage')):
            details.addWidget(rule())
            row=QHBoxLayout();value=label('—','metric');self.metrics[key]=value
            row.addWidget(label(title),1);row.addWidget(value)
            details.addLayout(row)
        layout.addWidget(self.validation_panel)
        frame,box=panel();row=QHBoxLayout();row.addWidget(label('Click learning','heading'),1)
        self.click_details_toggle=QPushButton('Details');self.click_details_toggle.setAccessibleName('Click learning details')
        row.addWidget(self.click_details_toggle);box.addLayout(row)
        self.click_plain_status=label('','muted',True);box.addWidget(self.click_plain_status)
        row=QHBoxLayout()
        self.click_learning=QCheckBox('Refine gaze using clicks');self.click_learning.toggled.connect(self.c.set_click_learning)
        row.addWidget(self.click_learning,1)
        self.undo_learning=QPushButton('Undo last correction');self.undo_learning.clicked.connect(self.c.undo_learning)
        row.addWidget(self.undo_learning);box.addLayout(row)
        self.click_details=QWidget();details=QVBoxLayout(self.click_details)
        details.setContentsMargins(0,0,0,0);details.setSpacing(6);details.addWidget(rule())
        self.click_hint=label('','muted',True);details.addWidget(self.click_hint)
        self.learning_value=label('','muted',True);details.addWidget(self.learning_value)
        self.click_summary=label('','muted',True);details.addWidget(self.click_summary)
        self.learning_model=label('','muted',True);details.addWidget(self.learning_model)
        box.addWidget(self.click_details);connect_disclosure(self.click_details_toggle,self.click_details)
        layout.addWidget(frame)
        frame,box=panel();row=QHBoxLayout();row.addWidget(label('Next target prediction','heading'),1)
        self.prediction_details_toggle=QPushButton('Details');self.prediction_details_toggle.setAccessibleName('Next target prediction details')
        row.addWidget(self.prediction_details_toggle);box.addLayout(row)
        self.prediction_plain_status=label('','muted',True);box.addWidget(self.prediction_plain_status)
        self.prediction_status=label('','muted',True);box.addWidget(self.prediction_status)
        connect_disclosure(self.prediction_details_toggle,self.prediction_status)
        layout.addWidget(frame)
        layout.addWidget(label('Run setup again if you move your iPhone or monitor. Mira doesn’t automatically save video or audio.','muted',True));layout.addStretch()

    def build_practice(self):
        layout = self.page('Practice', 'See how Mira highlights a button as you look at it or move your mouse.')
        row = QHBoxLayout()
        row.setSpacing(12)
        self.practice_button = QPushButton('Connect iPhone')
        self.practice_button.setObjectName('primary')
        self.practice_button.clicked.connect(self.toggle_practice)
        row.addWidget(self.practice_button)
        self.pointer = QCheckBox('Try with mouse')
        self.pointer.toggled.connect(self.toggle_pointer)
        row.addWidget(self.pointer)
        row.addStretch()
        layout.addLayout(row)
        layout.addSpacing(4)

        stage = QFrame()
        stage.setObjectName('practiceStage')
        stage.setMinimumHeight(288)
        grid = QGridLayout(stage)
        grid.setContentsMargins(24, 24, 24, 24)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(24)
        for column in range(3):
            grid.setColumnStretch(column, 1)
        for row_index in range(2):
            grid.setRowStretch(row_index, 1)
        self.practice_buttons = {}
        targets = ('Top left', 'Plus', 'Top right', 'Minus', 'Reset', 'Clock')
        for index, text in enumerate(targets):
            button = QPushButton(text)
            button.setObjectName('practiceTarget')
            button.setAccessibleName(text)
            button.setMinimumSize(148, 82)
            button.setMaximumWidth(208)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.clicked.connect(lambda _, target=text: self.practice_action(target))
            target_row, target_column = divmod(index, 3)
            grid.addWidget(button, target_row, target_column, Qt.AlignVCenter)
            self.practice_buttons[text] = button
        layout.addWidget(stage, 1)
        layout.addSpacing(4)

        feedback = QHBoxLayout()
        feedback.setSpacing(8)
        feedback.addWidget(label('Counter', 'muted'))
        self.count = label('0', 'practiceCount')
        feedback.addWidget(self.count)
        feedback.addSpacing(8)
        self.practice_result = label('', 'muted', True)
        feedback.addWidget(self.practice_result, 1)
        layout.addLayout(feedback)

        self.practice_command = QLineEdit()
        self.practice_command.setAccessibleName('Type a command')
        self.practice_command.setPlaceholderText('click Plus')
        self.practice_command.returnPressed.connect(self.submit)
        command = label('Type a command')
        command.setBuddy(self.practice_command)
        layout.addWidget(command)
        command_row = QHBoxLayout()
        command_row.addWidget(self.practice_command, 1)
        self.run_practice_command = QPushButton('Run command')
        self.run_practice_command.clicked.connect(self.submit)
        command_row.addWidget(self.run_practice_command)
        layout.addLayout(command_row)

        self.practice_tools_button = QPushButton('More practice tools')
        self.practice_tools_button.setObjectName('advanced')
        layout.addWidget(self.practice_tools_button, 0, Qt.AlignLeft)
        tools = QWidget()
        fields = QVBoxLayout(tools)
        fields.setContentsMargins(0, 0, 0, 0)
        fields.setSpacing(4)
        self.editor = QLineEdit()
        self.editor.setAccessibleName('Dictation field')
        dictation = label('Dictation')
        dictation.setBuddy(self.editor)
        fields.addWidget(dictation)
        fields.addWidget(self.editor)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setAccessibleName('Activity')
        self.log.setMaximumBlockCount(150)
        self.log.setFixedHeight(66)
        self.log.setPlaceholderText('Activity')
        fields.addWidget(self.log)
        layout.addWidget(tools)
        connect_disclosure(self.practice_tools_button, tools)

    def build_settings(self):
        layout=self.page('Settings','Gaze tracking, highlights, and optional actions.')
        frame,box=panel();box.addWidget(label('iPhone','heading'))
        self.phone_transport=SettingsComboBox();self.phone_transport.setAccessibleName('iPhone connection')
        self.phone_transport.addItem('Wi‑Fi · home network','wifi')
        self.phone_transport.addItem('USB · paired connection','usb')
        self.phone_transport.setCurrentIndex(self.phone_transport.findData(self.settings['phone_transport']))
        box.addWidget(self.phone_transport)
        self.wifi_fields=QWidget()
        fields=QVBoxLayout(self.wifi_fields);fields.setContentsMargins(0,0,0,0);fields.setSpacing(8)
        row=QHBoxLayout()
        self.phone_host=QLineEdit(self._saved_phone_host)
        self.phone_host.setPlaceholderText('Found automatically, or enter the address shown on iPhone')
        self.phone_host.setAccessibleName('iPhone Wi‑Fi address')
        self.phone_host.textEdited.connect(lambda _:self.cancel_phone_search())
        row.addWidget(self.phone_host,1)
        self.find_phone=QPushButton('Find iPhone');self.find_phone.clicked.connect(self.start_phone_search)
        row.addWidget(self.find_phone);fields.addLayout(row)
        self.phone_find_status=label('','muted',True);self.phone_find_status.hide()
        fields.addWidget(self.phone_find_status)
        box.addWidget(self.wifi_fields);self.wifi_fields.setVisible(self.phone_transport.currentData()=='wifi')
        self.phone_transport.currentIndexChanged.connect(self.show_wifi_fields)
        box.addWidget(label('Wi‑Fi sends only eye and head data. Use a trusted home network because Mira doesn’t encrypt this connection.','muted',True))
        b=QPushButton('Save connection');b.clicked.connect(self.save_phone_connection);box.addWidget(b,0,Qt.AlignLeft);layout.addWidget(frame)
        self.advanced=QPushButton('Advanced');self.advanced.setObjectName('advanced')
        layout.addWidget(self.advanced)
        self.advanced_panel=QWidget();advanced=QVBoxLayout(self.advanced_panel)
        advanced.setContentsMargins(0,0,0,0);advanced.setSpacing(10)
        layout.addWidget(self.advanced_panel)
        connect_disclosure(self.advanced,self.advanced_panel)
        self.advanced.toggled.connect(self.list_microphones)
        frame,box=panel();box.addWidget(label('Webcam · second angle','heading'))
        box.addWidget(label('If iPhone loses track of your gaze, a webcam you’ve set up can take over. Video stays on your PC and isn’t saved.','muted',True))
        row=QHBoxLayout();self.webcam_device=SettingsComboBox();self.webcam_device.setAccessibleName('Webcam')
        if self.settings.get('webcam_device_id'):
            self.webcam_device.addItem(self.settings.get('webcam_device_name') or 'Saved camera',self.settings['webcam_device_id'])
        else:self.webcam_device.addItem('Find and select a camera','')
        row.addWidget(self.webcam_device,1);self.webcam_find=QPushButton('Find cameras')
        self.webcam_find.clicked.connect(self.find_webcams);row.addWidget(self.webcam_find);box.addLayout(row)
        row=QHBoxLayout();self.webcam_setup=QPushButton('Set up webcam')
        self.webcam_setup.clicked.connect(self.open_webcam_setup);row.addWidget(self.webcam_setup)
        self.webcam_toggle=QPushButton('Connect webcam');self.webcam_toggle.clicked.connect(self.toggle_webcam)
        row.addWidget(self.webcam_toggle);row.addStretch();box.addLayout(row)
        self.webcam_status=label('Webcam off','muted',True);box.addWidget(self.webcam_status)
        box.addWidget(label('Setup uses 9 points and checks 5 new ones. Set up again if you move the webcam. The webcam starts off.','muted',True))
        advanced.addWidget(frame)
        frame,box=panel();box.addWidget(label('Highlight','heading'))
        self.highlight_mode=SettingsComboBox();self.highlight_mode.setAccessibleName('Highlight mode')
        self.highlight_mode.addItem('Follow gaze · dot and area','radial')
        self.highlight_mode.addItem('Magnetic focus · one element','magnetic')
        self.highlight_mode.setCurrentIndex(1 if self.settings.get('highlight_mode')=='magnetic' else 0)
        self.highlight_mode.currentIndexChanged.connect(lambda _:self.update_setting('highlight_mode',self.highlight_mode.currentData()))
        box.addWidget(self.highlight_mode)
        self.highlight_hint=label('','muted',True);box.addWidget(self.highlight_hint)
        self.show_gaze_check=self.add_check(box,'show_gaze','Show gaze dot')
        self.anticipation_check=QCheckBox('Show a possible next target');self.anticipation_check.toggled.connect(self.c.set_anticipation)
        box.addWidget(self.anticipation_check)
        self.quiet_mode_check=self.add_check(box,'quiet_mode','Quiet mode')
        self.quiet_hint=label('Hide gaze cues while control stays on.','muted',True)
        box.addWidget(self.quiet_hint)
        advanced.addWidget(frame)
        frame,box=panel();box.addWidget(label('Optional actions','heading'))
        box.addWidget(label('With the microphone on, say “click” to activate the yellow target. Hold Shift to select with the mouse.','muted',True))
        self.add_check(box,'dwell_enabled','Click by holding your gaze','Keep looking at the same target to click automatically. This is off by default.')
        self.add_slider(box,'dwell_ms','Hold time',700,2500);advanced.addWidget(frame)
        frame,box=panel();box.addWidget(label('Voice and recognition','heading'))
        microphone_form=QFormLayout()
        microphone_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.microphone_picker=SettingsComboBox();self.microphone_picker.setAccessibleName('Microphone')
        self.microphone_picker.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        microphone_form.addRow('Microphone',self.microphone_picker);box.addLayout(microphone_form)
        self.microphone_note=label('','muted',True);box.addWidget(self.microphone_note)
        self._microphone_list_status=''
        self.fill_microphones([], self.settings['microphone'])
        self.add_slider(box,'speech_silence_ms','Pause after speaking',200,800)
        for key,title in (('fast_voice','Use fast recognition for short commands'),('vision','Recognize on-screen text and icons'),('ranker','Use a local model to refine complex commands')):
            self.add_check(box,key,title)
        self.mode=SettingsComboBox();self.mode.addItems(['Local model','Remote API (HTTPS)'])
        self.mode.setCurrentIndex(0 if self.settings['model_mode']=='local' else 1)
        self.model_name=QLineEdit(self.settings['model_name'])
        self.model_url=QLineEdit(self.settings['model_url'])
        model_form=QFormLayout()
        model_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        for name,field in (('Model location',self.mode),('Model name',self.model_name),('Model address',self.model_url)):
            field.setAccessibleName(name)
            model_form.addRow(name,field)
        box.addLayout(model_form)
        box.addWidget(label('A remote API receives your command and the element map. When screen recognition is on, it also receives an image of the window. Use a secure local proxy for API keys.','muted',True))
        b=QPushButton('Save voice and model settings');b.clicked.connect(self.save_connections);box.addWidget(b)
        advanced.addWidget(frame);layout.addStretch()

    def add_check(self,layout,key,text,hint=''):
        check=QCheckBox(text);check.setChecked(bool(self.settings[key]))
        def commit(value):
            if not self.update_setting(key,value):
                with QSignalBlocker(check):
                    check.setChecked(bool(self.settings[key]))
        check.toggled.connect(commit);layout.addWidget(check)
        if hint:layout.addWidget(label(hint,'muted',True))
        return check

    def add_slider(self,layout,key,title,lo,hi):
        row=QHBoxLayout();row.addWidget(label(title),1);value=label(f'{self.settings[key]} ms','metric');row.addWidget(value);layout.addLayout(row)
        slider=QSlider(Qt.Horizontal);slider.setAccessibleName(title);slider.setRange(lo,hi);slider.setValue(self.settings[key])
        slider.setTracking(False)
        def commit(position):
            if not self.update_setting(key,position):
                with QSignalBlocker(slider):
                    slider.setValue(self.settings[key])
            value.setText(f'{slider.value()} ms')
        slider.sliderMoved.connect(lambda position:value.setText(f'{position} ms'))
        slider.valueChanged.connect(commit);layout.addWidget(slider)

    def open_setup(self):
        try:
            if self.setup_window is not None and self.setup_window.isVisible():self.setup_window.raise_();return
            from .setup import open_setup
            self.setup_window=open_setup(self.c,self);self.setup_window.adopted.connect(self.setup_adopted)
        except (OSError,ValueError,RuntimeError) as exc:self.add_message('Setup',str(exc))

    def setup_adopted(self):
        self._profile_stamp = None
        self.navigate(0)
        self.last_paint = 0
        self.refresh()
        self.gaze_btn.setFocus()

    def update_setting(self,key,value):
        if key=='highlight_mode' and (self.c.busy or self.c.output_pending):
            self.add_message('Highlight','Wait for the command to finish');self.last_paint=0;self.refresh();return False
        updated=dict(self.settings,**{key:value})
        try:save(updated)
        except OSError as exc:
            self.add_message('Settings','Couldn’t save: '+str(exc));self.last_paint=0;self.refresh();return False
        self.settings.update(updated)
        if key=='ranker' and not value:
            self.c.stop_ranker()
        if key in ('quiet_mode','show_scene_map'):
            self.last_paint=0;self.refresh()
        if key=='highlight_mode':
            self.c.fusion.reset();self.c.dwell_id=None
            self.c.action_epoch+=1
            from ..core.contracts import Focus
            self.c.focus=Focus(reason='Highlight mode changed')
            self.last_paint=0;self.refresh()
        if key=='vision' and self.c.running:self.add_message('Settings','Pause and start control to apply the new screen recognition setting.')
        if key=='speech_silence_ms' and self.c.microphone:self.add_message('Microphone','The new pause applies the next time you turn on the microphone.')
        return True

    def save_phone_connection(self):
        self.cancel_phone_search()
        try:
            self.c.set_phone_connection(self.phone_transport.currentData(),self.phone_host.text().strip())
            self.phone_host.setText(self.settings['phone_wifi_host'])
            self.add_message('iPhone','Connection saved. You can connect iPhone or start setup now.')
        except (OSError,ValueError,RuntimeError) as exc:self.add_message('iPhone',str(exc))

    def activate_phone(self):
        if self.phone_setup_needed():
            self.open_setup()
        elif self.c.phone and self.c.phone.worker and self.c.phone.worker.is_alive():
            self.open_gaze_practice()
        else:
            self.c.toggle_phone()

    def phone_setup_needed(self):
        return self.c.phone_needs_setup or (
            self.c.phone is None and not self.c.phone_error
            and not self._profile_status.get('available')
        )

    def use_wifi(self):
        self.cancel_phone_search()
        try:
            host=PhoneConnection('wifi',self.settings['phone_wifi_host']).host
        except ValueError:
            host=''
        try:
            self.c.set_phone_connection('wifi',host)
        except (OSError,ValueError,RuntimeError) as exc:
            self.add_message('iPhone',str(exc))
            return
        self.phone_transport.setCurrentIndex(self.phone_transport.findData('wifi'))
        self.last_paint=0;self.refresh()

    def open_phone_settings(self):
        self.navigate(3)
        self.stack.currentWidget().verticalScrollBar().setValue(0)
        self.phone_transport.setFocus()

    def show_wifi_fields(self, index):
        wifi = self.phone_transport.itemData(index) == 'wifi'
        self.wifi_fields.setVisible(wifi)
        if not wifi:
            self.cancel_phone_search()

    def show_phone_search_status(self, message):
        self.phone_find_status.setText(message)
        self.phone_find_status.setVisible(bool(message))

    def cancel_phone_search(self):
        if self._phone_search is not None:
            self._phone_search.set()
            self._phone_search = None
        self.find_phone.setEnabled(True)
        self.show_phone_search_status('')

    def start_phone_search(self):
        if self._phone_search is not None or self.phone_transport.currentData() != 'wifi':
            return
        stop = self._phone_search = threading.Event()
        self.find_phone.setEnabled(False)
        self.show_phone_search_status('Looking for iPhone on your Wi‑Fi…')

        def search():
            try:
                hosts, error = find_phones(stop_event=stop), None
            except Exception as exc:
                hosts, error = [], str(exc) or type(exc).__name__
            if not stop.is_set():
                self.phone_search_finished.emit(stop, hosts, error)

        threading.Thread(target=search, name='iphone-search', daemon=True).start()

    def finish_phone_search(self, stop, hosts, error):
        if stop is not self._phone_search:
            return
        self._phone_search = None
        self.find_phone.setEnabled(True)
        if error is not None:
            self.show_phone_search_status('Couldn’t search for iPhone: ' + error)
        elif not hosts:
            self.show_phone_search_status(WIFI_PHONE_NEXT_STEP)
        else:
            host=hosts[0]
            message=f'Found iPhone at {host}'
            self.phone_host.setText(host)
            try:
                self.c.save_found_phone(host)
            except (OSError, ValueError, RuntimeError) as exc:
                message+=' · use “Save connection” to save the address'
                self.add_message('iPhone', str(exc))
            self.show_phone_search_status(message)
        self.last_paint=0;self.refresh()

    def find_webcams(self):
        try:
            from ..providers.webcam import available_cameras
            devices=available_cameras()
            self.webcam_device.clear()
            for identity,name in devices:self.webcam_device.addItem(name,identity)
            index=self.webcam_device.findData(self.settings.get('webcam_device_id',''))
            if index>=0:self.webcam_device.setCurrentIndex(index)
            if not devices:self.webcam_device.addItem('No cameras found','')
        except Exception as exc:self.add_message('Webcam',str(exc))

    def _select_webcam(self):
        identity=self.webcam_device.currentData()
        if not identity:raise ValueError('Select “Find cameras” and choose a webcam')
        if identity!=self.settings.get('webcam_device_id'):
            self.c.select_webcam(identity,self.webcam_device.currentText())

    def toggle_webcam(self):
        try:
            if self.c.webcam is None:self._select_webcam()
            self.c.toggle_webcam()
        except (OSError,ValueError,RuntimeError) as exc:self.add_message('Webcam',str(exc))

    def open_webcam_setup(self):
        try:
            if self.setup_window is not None and self.setup_window.isVisible():
                self.setup_window.raise_();return
            self._select_webcam()
            from .webcam_setup import open_webcam_setup
            self.setup_window=open_webcam_setup(self.c,self)
        except (OSError,ValueError,RuntimeError) as exc:self.add_message('Webcam',str(exc))

    def fill_microphones(self, devices, choice):
        picker=self.microphone_picker
        picker.clear()
        picker.addItem('System default','')
        name=choice['name'] if isinstance(choice,dict) else choice
        host_api=choice['hostApi'] if isinstance(choice,dict) else None
        index=choice['index'] if isinstance(choice,dict) else None
        matches=microphone_matches(devices,name,host_api,index) if name else []
        selected=matches[0] if len(matches)==1 else None
        names=Counter(device['name'] for device in devices)
        endpoints=Counter((device['name'],device['hostApi']) for device in devices)
        selected_index=0
        for device in devices:
            name_key,api=device['name'],device['hostApi']
            caption=name_key
            if names[name_key]>1:
                caption+=f' · {api}'
            if endpoints[name_key,api]>1:
                caption+=f' · {device["index"]}'
            identity={key:device[key] for key in ('name','hostApi','index')}
            if device is selected:
                selected_index=picker.count()
                if isinstance(choice,str):identity=choice
            picker.addItem(caption,identity)
        if choice and selected is None:
            selected_index=picker.count()
            picker.addItem(f'{name} (saved)',choice)
        picker.setCurrentIndex(selected_index)

    def list_microphones(self, expanded):
        if not expanded or self._microphone_listing:
            return
        self._microphone_listing=True
        self._microphone_list_status='Finding microphones…'
        self.refresh_microphone_note()

        def list_devices():
            try:
                devices=Speech.available_microphones()
            except (OSError,subprocess.SubprocessError,ValueError,KeyError):
                devices=None
            if not self.c.closed:
                self.microphone_list_finished.emit(devices)

        threading.Thread(target=list_devices,name='microphone-list',daemon=True).start()

    def show_microphones(self, devices):
        self._microphone_listing=False
        if devices is not None:
            self.fill_microphones(devices,self.microphone_picker.currentData())
        self._microphone_list_status=(
            'Couldn’t list microphones. Close and reopen Advanced to try again.' if devices is None else
            'No input devices found. Connect a microphone and reopen Advanced.' if not devices else '')
        self.refresh_microphone_note()

    def refresh_microphone_note(self):
        notes='\n'.join(filter(None,(self.c.voice_fallback,self._microphone_list_status)))
        self.microphone_note.setText(notes)
        self.microphone_note.setVisible(bool(notes))

    def save_connections(self):
        updated=dict(self.settings,microphone=self.microphone_picker.currentData(),model_mode='local' if self.mode.currentIndex()==0 else 'remote',
                     model_name=self.model_name.text().strip(),model_url=self.model_url.text().strip())
        try:save(updated)
        except OSError as exc:self.add_message('Settings','Couldn’t save: '+str(exc));return
        self.settings.update(updated);self.add_message('Settings','Saved. The microphone setting applies the next time you turn it on.')

    def navigate(self,index):
        if index != 2 and self.practice_active:
            self.stop_practice()
        self.stack.setCurrentIndex(index)
        for i,b in enumerate(self.nav):b.setChecked(i==index)

    def open_gaze_practice(self):
        self.navigate(2)
        self.start_practice(use_mouse=False)

    def open_mouse_preview(self):
        self.navigate(2)
        self.start_practice(use_mouse=True)

    def start_practice(self, *, use_mouse):
        self.practice_active = True
        with QSignalBlocker(self.pointer):
            self.pointer.setChecked(use_mouse)
        self.refresh_practice_button()
        self.c.set_practice_scope(int(self.winId()), use_mouse=use_mouse)
        if not self.c.running:
            self._practice_started_control = True
            self.c.start()

    def stop_practice(self):
        self.practice_active = False
        with QSignalBlocker(self.pointer):
            self.pointer.setChecked(False)
        if self._practice_started_control and self.c.running:
            self.c.pause()
        self._practice_started_control = False
        self.c.set_practice_scope(0)
        self.refresh_practice_button()

    def practice_has_gaze(self):
        worker = self.c.phone.worker if self.c.phone is not None else None
        phone_ready = worker and worker.is_alive() and getattr(worker, 'frames', 0) > 0 and not self.c.phone_error
        webcam = self.c.webcam
        return bool(phone_ready or (webcam is not None and webcam.active))

    def refresh_practice_button(self):
        text = 'Stop practice' if self.practice_active else (
            'Try eye tracking' if self.practice_has_gaze() else 'Connect iPhone')
        self.practice_button.setText(text)

    def toggle_practice(self):
        if self.practice_active:
            self.stop_practice()
        elif self.practice_has_gaze():
            self.open_gaze_practice()
        else:
            self.navigate(0)
            if self.gaze_btn.isVisible():
                self.gaze_btn.setFocus()

    def toggle_pointer(self, value):
        if value:
            self.open_mouse_preview()
        elif self.practice_active:
            if self.practice_has_gaze():
                self.open_gaze_practice()
            else:
                self.stop_practice()

    def practice_action(self,text):
        if text=='Plus':self.counter+=1
        elif text=='Minus':self.counter-=1
        elif text=='Reset':self.counter=0
        self.count.setText(str(self.counter));self.practice_result.setText('Time: '+time.strftime('%H:%M') if text=='Clock' else 'Clicked: '+text)

    def submit(self):
        text = self.practice_command.text().strip()
        if not text:
            return
        if not self.practice_active:
            self.open_gaze_practice()
        if self.c.scene is not None:
            self.practice_command.clear()
        self.c.command(text)

    def add_message(self,who,text):
        self.log.appendPlainText(time.strftime('%H:%M:%S')+'  '+who+'\n'+text+'\n')
        self.content_status.showMessage(text,5000)

    def refresh_profile(self):
        bounds=self.c.win.bounds()
        stamps=[]
        for name in ('personal-profile.json','gaze-personal.npz','personal-profile.bak.json','gaze-personal.bak.npz','starter-last-result.json'):
            try:
                st=(LOCAL/name).stat();stamps.append((st.st_mtime_ns,st.st_size))
            except OSError:stamps.append(None)
        stamp=(tuple(stamps),tuple(bounds.as_list()))
        if stamp!=self._profile_stamp:
            from ..personalization import ProfileStore
            from ..personalization.history import latest_result
            self._profile_status=ProfileStore(LOCAL).status(bounds);self._profile_stamp=stamp
            self._profile_status['last_result']=latest_result(LOCAL,bounds)
        state=self._profile_status;available=state.get('available',False)
        self.profile_title.setText('Gaze setup ready' if available else 'Set up your gaze profile')
        self.profile_detail.setText('Mira checked five new points in one seated position. Repeat setup if you change how you sit.' if available else
            'Sit comfortably for 9 setup points and 5 new check points.')
        self.profile_setup.setText('Refine setup · about a minute' if available else 'Set up · about a minute')
        report=(state.get('report') if available else state.get('last_result')) or {}
        self.validation_panel.setVisible(bool(report))
        self.no_setup_results.setVisible(not report)
        if not report:
            return
        good = report.get('usable') is True
        self.validation_verdict.setText('Accuracy: Good' if good else 'Accuracy: Fair · try setup again')
        for key,widget in self.metrics.items():
            value=report.get(key);widget.setText('—' if not isinstance(value,(int,float)) else f'{value*100:.0f}%' if key=='coverage' else f'{value:.0f} px')
        source='Saved setup · ' if available else 'Latest attempt · '
        self.validation_hint.setText(source+('ready to refine with clicks.' if good else
                                     'needs refinement. '+' '.join(report.get('problems', [])[:2])))

    def refresh_control_panel(self):
        self.control_panel.setVisible(self.c.running or self.c.output_pending or self.more_controls.isChecked())

    def refresh_sources(self, now):
        c = self.c
        worker = c.phone.worker if c.phone is not None else None
        phone_connected = bool(worker and worker.is_alive())
        has_frames = isinstance(getattr(worker, 'frames', None), int) and worker.frames > 0
        if c.phone_error or (c.phone is not None and not phone_connected):
            phone_state, phone_word = 'hold', 'Error'
        elif c.phone is None:
            phone_state, phone_word = 'off', 'Off'
        elif c.gaze_source_mode == 'iphone' and fresh_gaze(c.gaze, now):
            phone_state, phone_word = 'go', 'Live'
        else:
            phone_state, phone_word = 'standby', 'Waiting'
        gaze_detail = c.phone_error or c.gaze_status
        if c.phone is None and c.running and not c.phone_error:
            gaze_detail = 'Off'
        if c.gaze_fps and c.gaze_source_mode != 'webcam':
            gaze_detail += f' · {c.gaze_fps:.0f} Hz'
        transport = self.settings['phone_transport']
        needs_setup = self.phone_setup_needed()
        self.gaze_value.setToolTip(gaze_detail)
        if c.webcam is not None:
            self.gaze_value.setToolTip(gaze_detail+'\n'+c.webcam_status)
        connection_error=phone_state=='hold' and not needs_setup
        self.phone_settings_button.setVisible(not phone_connected)
        self.use_wifi_button.setVisible(connection_error and transport=='usb')
        self.mouse_preview.setVisible(phone_state != 'go')
        self.disconnect_phone.setVisible(phone_connected)
        connecting = phone_state == 'standby' and not has_frames
        self.disconnect_phone.setText('Cancel connection' if connecting else 'Disconnect iPhone')
        self.gaze_btn.setVisible(phone_state != 'standby')
        self.gaze_btn.setEnabled(not needs_setup or not (c.busy or c.output_pending))
        heading = 'iPhone gaze tracking'
        if needs_setup:
            status = ('Set up again for this screen.' if c.phone_needs_setup else
                      'Requires an iPhone with Face ID and iOS 17 or later.')
            action = 'Set up iPhone'
        elif phone_state == 'hold':
            not_found = isinstance(getattr(worker, 'failure', None), WifiPhoneNotFound)
            heading = 'Can’t find iPhone' if not_found else 'Can’t connect to iPhone'
            status = (WIFI_PHONE_NEXT_STEP if transport == 'wifi' else
                      'Unlock iPhone and connect it by USB to a PC it trusts.')
            action = 'Try again'
        elif phone_state == 'go':
            status = 'Connected over Wi‑Fi' if transport == 'wifi' else 'Connected by USB'
            action = 'Open Practice'
        elif phone_state == 'standby':
            status = 'Looking for iPhone…' if transport == 'wifi' else 'Connecting by USB…'
            if has_frames:
                status = 'Waiting for your gaze. Face the iPhone camera.'
            action = 'Open Practice'
        else:
            status = 'Gaze setup saved. Start tracking in Mira on your iPhone.'
            action = 'Connect iPhone'
        if c.gaze_source_mode == 'webcam':
            status += ' Using webcam.'
        self.gaze_heading.setText(heading)
        self.gaze_value.setText(status)
        self.gaze_btn.setText(action)

        voice_word = {'off': 'Off', 'standby': 'Starting', 'go': 'Listening', 'hold': 'Error'}[c.voice_state]
        voice_detail = c.voice_status
        if voice_detail != voice_word and c.voice_state != 'standby':
            voice_detail = f'{voice_word} · {voice_detail}'
        if c.voice_fallback:
            voice_detail += '\n'+c.voice_fallback
        self.voice_value.setText(voice_detail)
        self.refresh_microphone_note()
        self.voice_btn.setText('Turn off microphone' if c.microphone else 'Turn on microphone')

        for name, state, word in (('iPhone', phone_state, phone_word),
                                  ('Microphone', c.voice_state, voice_word)):
            lamp, caption = self.rail_sources[name]
            lamp.set_state(state, word)
            caption.setText(word)
        self.voice_lamp.set_state(c.voice_state, voice_word)
        return phone_state

    def refresh(self):
        now=time.monotonic();c=self.c
        quiet=bool(self.settings.get('quiet_mode',False))
        selected_until=c.fusion.selection_until(c.focus.target,now,c.attention)
        for overlay in self.overlays:
            overlay.sync(c.focus,c.attention,c.scene,c.running and not quiet,self.settings['show_gaze'],
                         prediction=c.prediction,highlight_mode=self.settings.get('highlight_mode','radial'),
                         selected_until=selected_until)
        if now-self.last_paint<.12:return
        self.last_paint=now;self.refresh_profile()
        saved_host = self.settings.get('phone_wifi_host', '')
        if saved_host != self._saved_phone_host:
            self._saved_phone_host = saved_host
            if not self.phone_host.isModified():
                self.phone_host.setText(saved_host)
        self.webcam_status.setText(c.webcam_status)
        self.webcam_toggle.setText('Disconnect webcam' if c.webcam is not None else 'Connect webcam')
        self.webcam_setup.setEnabled(not c.output_pending and not c.busy)
        self.quiet_mode_check.blockSignals(True);self.quiet_mode_check.setChecked(quiet);self.quiet_mode_check.blockSignals(False)
        self.quiet_hint.setText('Gaze cues are hidden. Click learning and connections follow your settings.' if quiet else
                                'Hide gaze cues while control stays on.')
        show_map=not quiet and bool(self.settings.get('show_scene_map',False))
        self.scene_map_button.setVisible(not quiet)
        self.scene_map_button.setText('Hide screen map' if show_map else 'Show screen map')
        self.scene_map_button.setAccessibleDescription('Expanded' if show_map else 'Collapsed')
        self.focus_panel.setVisible(show_map)
        self.canvas.setMinimumHeight(180 if c.running else 112)
        self.focus_panel.setMaximumHeight(350 if c.running else 146)
        magnetic=self.settings.get('highlight_mode')=='magnetic'
        self.highlight_mode.blockSignals(True);self.highlight_mode.setCurrentIndex(1 if magnetic else 0);self.highlight_mode.blockSignals(False)
        self.highlight_mode.setEnabled(not quiet and not c.busy and not c.output_pending)
        self.show_gaze_check.setEnabled(not quiet and not magnetic);self.anticipation_check.setEnabled(not quiet and not magnetic)
        self.highlight_hint.setText('Quiet mode hides gaze cues.' if quiet else
                                   'Look at another element to move the yellow frame. This mode hides the gaze dot and prediction.' if magnetic else
                                   'The blue dot follows your gaze. Yellow marks the current focus. Purple suggests what you might look at next.')
        self.legend_gaze.setVisible(not magnetic and self.settings['show_gaze'])
        self.profile_setup.setEnabled(not c.output_pending)
        phone_state=self.refresh_sources(now)
        self.refresh_control_panel()
        self.refresh_practice_button()
        self.refresh_learning(phone_state)
        self.run.setText('Pause' if c.running else 'Start')
        target = (current_focus_target(c.scene, c.focus.target, c.attention, now, selected_until)
                  if c.running and not quiet else None)
        confirmation = c.action_confirmation
        if c.running and confirmation and 0 <= now-confirmation[1] < GO_CONFIRMATION_SECONDS:
            call = confirmation[0]
        else:
            call = (f'Looking at {target.label[:85] or target.role}' if target else
                    'Mira is running' if c.running else 'Paused')
        self.target.setText(call)
        focus_reason='Look at your screen' if c.focus.reason=='Waiting for fresh gaze data' else c.focus.reason
        if target and c.focus.ambiguous:
            focus_reason='Several nearby targets · say “next”' if c.voice_state=='go' else 'Your gaze is near several targets.'
        reason = (focus_reason if not quiet and focus_reason != 'Paused' else
                  'Gaze cues are hidden.' if quiet else '') if c.running else ''
        self.reason.setText(reason)
        self.reason.setVisible(bool(reason))
        scene_detail = ''
        if show_map and c.scene and c.running:
            scene_detail = f'{len(c.scene.current_elements(now))} screen elements'
            if c.scene.latency_ms > 0:
                scene_detail += f' · {c.scene.latency_ms:.0f} ms'
        self.latency.setText(scene_detail)
        self.latency.setVisible(bool(self.latency.text()))
        self.status.setText(c.status)
        if show_map:self.canvas.update()

    def refresh_learning(self, phone_state):
        c=self.c
        self.click_learning.blockSignals(True);self.click_learning.setChecked(bool(self.settings.get('click_learning',True)));self.click_learning.blockSignals(False)
        self.anticipation_check.blockSignals(True);self.anticipation_check.setChecked(bool(self.settings.get('anticipation',True)));self.anticipation_check.blockSignals(False)
        self.click_hint.setText(c.click_status)
        progress=c.learning.click_progress
        learned=progress['accepted']
        if not self.click_learning.isChecked():
            click_status='Click learning is off.'
        elif learned:
            count='one' if learned==1 else str(learned)
            click_status=f'Mira has learned from {count} of your clicks.'
        elif phone_state=='go' and c.learning.key is not None:
            click_status='Mira is ready to learn from your clicks.'
        else:
            click_status='Mira learns from your clicks once iPhone is connected.'
        self.click_plain_status.setText(click_status)
        self.learning_value.setText(f'Clicks used to refine gaze: {progress["accepted"]}' if c.learning.key is not None else 'Corrections load with your gaze profile.')
        counts=c.click_counts
        self.click_summary.setText(f'This session · used {counts["accepted"]} of {counts["seen"]} · gaze elsewhere {counts["away"]} · too little data {counts["quality"]}')
        self.learning_model.setText(c.learning.model_status if c.learning.key is not None else 'Connect a working gaze profile to refine it.')
        predictor=c.anticipation
        if not self.anticipation_check.isChecked():
            prediction_status='Prediction is off.'
        elif predictor.error:
            prediction_status='Prediction learning is unavailable.'
        else:
            readiness='Ready.' if predictor.model.learned or predictor.gaze_model.learned else 'Not enough clicks yet.'
            prediction_status='Mira suggests a likely next target in purple and learns from your clicks. '+readiness
        self.prediction_plain_status.setText(prediction_status)
        self.prediction_status.setText(predictor.status)
        self.undo_learning.setEnabled(bool(c.learning.undo))

    def closeEvent(self,event):
        self.cancel_phone_search()
        if self.setup_window is not None:
            try:self.setup_window.close()
            except RuntimeError:pass
        self.refresh_timer.stop();self.c.close()
        for overlay in self.overlays:overlay.close()
        event.accept()
