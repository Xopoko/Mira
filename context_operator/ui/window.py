"""Main window: gaze tracking first, everything else behind Advanced."""
from collections import Counter
import subprocess
import threading
import time
from PySide6.QtCore import Qt, QTimer, QRect, QRectF, QPointF, QSignalBlocker, QSize, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QIcon, QBrush, QRadialGradient, QPainterPath, QFontMetrics
from PySide6.QtWidgets import (QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QStackedWidget, QLineEdit,
    QPlainTextEdit, QSlider, QComboBox, QScrollArea, QSizePolicy,
    QFormLayout, QStatusBar)
from ..core.contracts import GAZE_TTL, SCENE_TTL
from ..core.webcam import fresh as fresh_gaze
from ..foundations.speech.runtime import microphone_matches
from ..providers.models import Speech
from ..providers.phone_connection import PhoneConnection, WIFI_PHONE_NEXT_STEP, WifiPhoneNotFound, find_phones
from ..settings import ASSETS, LOCAL, save
from .overlay import paint_scene_light,paint_magnetic_focus,current_focus_target
from .light_cache import LightCache
from .overview_cache import OverviewCache
from .theme import (ACCENT, CARD, FAINT, GAZE_DOT, GAZE_FOCUS, GAZE_LIGHT, GROUND, LINE, MUTED, OUTLINE,
    RAIL, STATE_COLORS, TEXT, Lamp, NavButton, Pill, SettingsComboBox, Switch, card, connect_disclosure,
    dark_title_bar, keyboard_focus, label, rule, swatch, theme, ui_font)

GO_CONFIRMATION_SECONDS = 2.0
PAGES = ('Overview', 'Calibration', 'Practice', 'Settings')
METRICS = (('median_px', 'Typical error', 'Half of the check samples landed closer to the target than this.'),
           ('p90_px', 'Hard points', 'Nine in ten check samples landed closer than this.'),
           ('coverage', 'Coverage', 'Share of the check samples Mira could place on the screen.'))
# Window shapes inside the pictured monitor, as fractions of it. The iPhone app draws the same picture.
PANES = ((.05, .09, .22, .82), (.31, .09, .64, .16), (.31, .32, .30, .32), (.65, .32, .30, .32), (.31, .71, .64, .20))
DESK_HEIGHT = 182


def paint_desk(painter, lit):
    """A monitor with the iPhone standing under it, centred on x=0 from y=0 down.
    Lit, it shows the gaze light on the monitor."""
    center = 0
    monitor = QRectF(-104, 0, 208, 122)
    glass = QPainterPath()
    glass.addRoundedRect(monitor, 8, 8)

    def pane(unit):
        x, y, w, h = unit
        return QRectF(monitor.x()+x*monitor.width(), monitor.y()+y*monitor.height(), w*monitor.width(), h*monitor.height())

    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(GROUND))
    painter.drawPath(glass)
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(QColor(LINE), 1))
    for unit in PANES:
        painter.drawRoundedRect(pane(unit), 3, 3)
    if lit:
        focused = pane(PANES[2])
        point = QPointF(focused.center().x()+8, focused.center().y()-4)
        radius = monitor.width()*.36

        def light(alpha):
            gradient = QRadialGradient(point, radius)
            for position, strength in ((0, alpha), (1, 0)):
                color = QColor(GAZE_LIGHT)
                color.setAlpha(strength)
                gradient.setColorAt(position, color)
            return QBrush(gradient)

        painter.save()
        painter.setClipPath(glass)
        painter.setPen(Qt.NoPen)
        painter.setBrush(light(72))
        painter.drawEllipse(point, radius, radius)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(light(255), 1.5))
        for unit in PANES:
            painter.drawRoundedRect(pane(unit), 3, 3)
        painter.setPen(QPen(QColor(GAZE_FOCUS), 2))
        painter.drawRoundedRect(focused, 3, 3)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(GAZE_DOT))
        painter.drawEllipse(point, 3, 3)
        painter.restore()
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(QColor(OUTLINE), 2, Qt.SolidLine, Qt.RoundCap))
    painter.drawPath(glass)
    foot = monitor.bottom()+12
    painter.drawLine(QPointF(center, monitor.bottom()+1), QPointF(center, foot))
    painter.drawLine(QPointF(center-30, foot), QPointF(center+30, foot))
    phone = QRectF(center-11, foot+8, 22, 40)
    painter.setPen(QPen(QColor(TEXT), 1.5))
    painter.setBrush(QColor(CARD))
    painter.drawRoundedRect(phone, 6, 6)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(ACCENT if lit else FAINT))
    painter.drawEllipse(QPointF(center, phone.top()+7), 2.5, 2.5)


class PhoneGlyph(QWidget):
    """The iPhone in the device card. Its camera dot repeats the connection state."""
    def __init__(self):
        super().__init__()
        self.state = 'off'
        self.setFixedSize(30, 54)

    def set_state(self, state):
        if state != self.state:
            self.state = state
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(OUTLINE if self.state == 'off' else TEXT), 1.5))
        painter.setBrush(QColor(GROUND))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5), 7, 7)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(STATE_COLORS[self.state]))
        painter.drawEllipse(QPointF(self.width()/2, 9.5), 2.5, 2.5)


class SceneCanvas(QWidget):
    """Numeric element geometry, never a saved screen image."""
    def __init__(self,controller):
        super().__init__();self.controller=controller;self.setMinimumHeight(240)
        self.setAccessibleName('Live preview of screen elements and gaze focus')
        self.overview=OverviewCache();self.light_cache=LightCache(max_pixels=1_000_000)
        self.idle=('','',False)
        # No gaze source can be shown: the picture and its words say why instead of an empty map.
        self.blocked=False

    def paint_message(self,p,text):
        p.setFont(ui_font('BODY',14));p.setPen(QColor(MUTED))
        p.drawText(QRectF(self.rect()).adjusted(24,0,-24,0),Qt.AlignCenter|Qt.TextWordWrap,text)

    def paint_idle(self,p):
        title,text,lit=self.idle
        width=min(440,self.width()-48)
        text_font=ui_font('BODY',14)
        text_height=QFontMetrics(text_font).boundingRect(QRect(0,0,width,400),Qt.AlignHCenter|Qt.TextWordWrap,text).height()
        words=32+text_height
        scale=min(1.5,(self.width()-48)/300,(self.height()-words-60)/DESK_HEIGHT)
        picture=scale>=1
        top=(self.height()-words-(DESK_HEIGHT*scale+28 if picture else 0))/2
        if picture:
            p.save();p.translate(self.width()/2,top);p.scale(scale,scale)
            paint_desk(p,lit)
            p.restore();top+=DESK_HEIGHT*scale+28
        left=(self.width()-width)/2
        p.setPen(QColor(TEXT));p.setFont(ui_font('STRONG',16))
        p.drawText(QRectF(left,top,width,24),Qt.AlignCenter,title)
        p.setPen(QColor(MUTED));p.setFont(text_font)
        p.drawText(QRectF(left,top+32,width,text_height+4),Qt.AlignHCenter|Qt.AlignTop|Qt.TextWordWrap,text)

    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen);p.setBrush(QColor(RAIL));p.drawRoundedRect(QRectF(self.rect()),10,10)
        c=self.controller;scene=c.scene;now=time.monotonic()
        if self.blocked or not c.running:
            self.paint_idle(p);return
        quiet=c.settings.get('quiet_mode',False)
        elements=scene.current_elements(now) if scene and not quiet and 0<=now-scene.captured<SCENE_TTL else ()
        if not elements:
            self.paint_message(p,'Quiet mode is on. Gaze cues are hidden.' if quiet else 'Mira is waiting for the screen map.')
            return
        bounds=scene.bounds;scale=min((self.width()-48)/max(1,bounds.w),(self.height()-40)/max(1,bounds.h))
        ox=(self.width()-bounds.w*scale)/2;oy=(self.height()-bounds.h*scale)/2
        origin=(bounds.x-ox/scale,bounds.y-oy/scale)
        rects=[QRectF(ox+(e.rect.x-bounds.x)*scale,oy+(e.rect.y-bounds.y)*scale,
                     max(3,e.rect.w*scale),max(3,e.rect.h*scale)) for e in elements]
        self.overview.paint(p,rects,self.width(),self.height())
        gaze=c.attention if c.attention and 0<=now-c.attention.captured<GAZE_TTL else None
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
            p.setPen(Qt.NoPen);p.setBrush(QColor(GAZE_DOT))
            p.drawEllipse(QPointF(ox+(gaze.x-bounds.x)*scale,oy+(gaze.y-bounds.y)*scale),2.5,2.5)


class Window(QMainWindow):
    phone_search_finished = Signal(object, object, object)
    microphone_list_finished = Signal(object)

    def __init__(self,controller):
        super().__init__();self.c=controller;self.settings=controller.settings;self.overlays=[];self.setup_window=None
        self.setWindowTitle('Mira');self.resize(1180,780);self.setMinimumSize(940,720)
        self.setWindowIcon(QIcon(str(ASSETS/'icon.png')));self.setStyleSheet(theme())
        self.last_paint=0.;self.counter=0;self.practice_active=False;self._profile_stamp=None;self._profile_status={}
        self._practice_started_control = False
        self._phone_search = None
        self._microphone_listing = False
        self._saved_phone_host = self.settings.get('phone_wifi_host', '')
        self.setting_checks = {}
        self.phone_search_finished.connect(self.finish_phone_search)
        self.microphone_list_finished.connect(self.show_microphones)
        root=QWidget();root.setObjectName('root');self.setCentralWidget(root)
        layout=QHBoxLayout(root);layout.setContentsMargins(0,0,0,0);layout.setSpacing(0)
        rail=QFrame();rail.setObjectName('rail');rail.setFixedWidth(77)
        rl=QVBoxLayout(rail);rl.setContentsMargins(8,16,9,12);rl.setSpacing(4)
        logo=QLabel();logo.setPixmap(self.windowIcon().pixmap(QSize(36,36),self.devicePixelRatioF()))
        logo.setAlignment(Qt.AlignCenter);logo.setAccessibleName('Mira')
        rl.addWidget(logo);rl.addSpacing(14)
        self.stack=QStackedWidget();self.nav=[]
        for index,text in enumerate(PAGES):
            button=NavButton(text)
            button.clicked.connect(lambda _,i=index:self.navigate(i));self.nav.append(button)
            rl.addWidget(button,0,Qt.AlignHCenter)
        rl.addStretch()

        content=QWidget();content_layout=QVBoxLayout(content)
        content_layout.setContentsMargins(0,0,0,0);content_layout.setSpacing(0)
        content_layout.addWidget(self.stack,1)
        self.content_status=QStatusBar(content)
        self.content_status.setSizeGripEnabled(False)
        self.content_status.setContentsMargins(22,0,22,0)
        content_layout.addWidget(self.content_status)
        self.status=label('Ready to connect')
        self.content_status.addWidget(self.status,1)
        self.sources={}
        for name in ('iPhone','Microphone'):
            chip=QWidget();row=QHBoxLayout(chip);row.setContentsMargins(0,0,10,0);row.setSpacing(6)
            lamp=Lamp(name);state=label('Off')
            row.addWidget(lamp);row.addWidget(label(name));row.addWidget(state)
            self.content_status.addPermanentWidget(chip);self.sources[name]=(lamp,state,chip)
        self.content_status.addPermanentWidget(label('Ctrl + Alt + F12 pauses tracking'))
        layout.addWidget(rail);layout.addWidget(content,1)
        self.build_control();self.build_profile();self.build_practice();self.build_settings();self.navigate(0)
        self.c.changed.connect(self.refresh);self.c.message.connect(self.add_message)
        self.refresh_timer=QTimer(self);self.refresh_timer.timeout.connect(self.refresh);self.refresh_timer.start(150)
        dark_title_bar(self);keyboard_focus(self);root.setFocus()
        self.refresh()

    def page(self,title,description='',width=None):
        """A scrolling page, optionally held to a reading width.
        Returns its layout, its title row and the row of actions beside the title."""
        page=QWidget();page.setObjectName('page');outer=QHBoxLayout(page)
        outer.setContentsMargins(28,22,28,20)
        column=QWidget();layout=QVBoxLayout(column)
        layout.setContentsMargins(0,0,0,0);layout.setSpacing(16)
        outer.addWidget(column,100)
        if width:
            column.setMaximumWidth(width);outer.addStretch(1)
        header=QHBoxLayout();header.setSpacing(10)
        heading=QVBoxLayout();heading.setSpacing(4)
        title_row=QHBoxLayout();title_row.setSpacing(12)
        title_row.addWidget(label(title,'title'));heading.addLayout(title_row)
        if description:heading.addWidget(label(description,'muted',True))
        header.addLayout(heading,1)
        layout.addLayout(header)
        area=QScrollArea();area.setWidgetResizable(True);area.setWidget(page);self.stack.addWidget(area)
        return layout,title_row,header

    def build_control(self):
        layout,title_row,header=self.page('Overview')
        self.state_pill=Pill()
        title_row.addWidget(self.state_pill,0,Qt.AlignVCenter);title_row.addStretch()
        self.mouse_preview=QPushButton('Try with mouse')
        self.mouse_preview.clicked.connect(self.open_mouse_preview)
        self.run=QPushButton('Start')
        self.run.clicked.connect(lambda:self.c.pause() if self.c.running else self.c.start())
        self.gaze_btn=QPushButton('Set up iPhone')
        self.gaze_btn.setObjectName('primary')
        self.gaze_btn.clicked.connect(self.activate_phone)
        for button in (self.mouse_preview,self.run,self.gaze_btn):
            header.addWidget(button,0,Qt.AlignBottom)

        columns=QHBoxLayout();columns.setSpacing(16)
        self.focus_panel,box=card();box.setSpacing(12)
        row=QHBoxLayout()
        row.addWidget(label('Live preview','heading'),1)
        self.latency=label('','faint');row.addWidget(self.latency)
        box.addLayout(row)
        self.canvas=SceneCanvas(self.c);box.addWidget(self.canvas,1)
        self.control_panel=QWidget()
        footer=QHBoxLayout(self.control_panel);footer.setContentsMargins(0,0,0,0);footer.setSpacing(16)
        call=QVBoxLayout();call.setSpacing(2)
        self.target=label('Paused','call',True)
        self.reason=label('','muted',True)
        call.addWidget(self.target);call.addWidget(self.reason)
        footer.addLayout(call,1)
        self.legend_gaze=QWidget();gaze_legend=QHBoxLayout(self.legend_gaze)
        gaze_legend.setContentsMargins(0,0,0,0);gaze_legend.setSpacing(6)
        gaze_legend.addWidget(swatch(GAZE_DOT,dot=True));gaze_legend.addWidget(label('Gaze dot','faint'))
        footer.addWidget(self.legend_gaze,0,Qt.AlignTop)
        focus_legend=QHBoxLayout();focus_legend.setSpacing(6)
        focus_legend.addWidget(swatch(GAZE_FOCUS));focus_legend.addWidget(label('Current focus','faint'))
        footer.addLayout(focus_legend);footer.setAlignment(focus_legend,Qt.AlignTop)
        box.addWidget(self.control_panel)
        columns.addWidget(self.focus_panel,1)

        side=QWidget();side.setFixedWidth(304)
        cards=QVBoxLayout(side);cards.setContentsMargins(0,0,0,0);cards.setSpacing(16)
        self.build_phone_card(cards);self.build_accuracy_card(cards);self.build_quick_card(cards)
        cards.addStretch()
        columns.addWidget(side)
        layout.addLayout(columns,1)

        self.more_controls=QPushButton('More controls')
        self.more_controls.setObjectName('advanced')
        layout.addWidget(self.more_controls,0,Qt.AlignLeft)
        frame,box=card()
        connect_disclosure(self.more_controls,frame)
        self.more_controls.toggled.connect(self.refresh_control_panel)
        row=QHBoxLayout();row.setSpacing(12)
        self.voice_lamp=Lamp('Microphone')
        row.addWidget(self.voice_lamp)
        column=QVBoxLayout();column.setSpacing(2)
        column.addWidget(label('Microphone','heading'))
        self.voice_value=label('Off','muted',True)
        column.addWidget(self.voice_value)
        row.addLayout(column,1)
        self.voice_btn=QPushButton('Turn on microphone')
        self.voice_btn.clicked.connect(self.c.toggle_microphone)
        row.addWidget(self.voice_btn)
        box.addLayout(row)
        layout.addWidget(frame)

    def build_phone_card(self,layout):
        frame,box=card();box.setSpacing(12);box.setContentsMargins(18,14,18,14)
        row=QHBoxLayout();row.setSpacing(14)
        self.phone_glyph=PhoneGlyph()
        row.addWidget(self.phone_glyph,0,Qt.AlignTop)
        text=QVBoxLayout();text.setSpacing(3)
        self.gaze_heading=label('iPhone gaze tracking','heading',True)
        self.gaze_value=label('','muted',True)
        text.addWidget(self.gaze_heading);text.addWidget(self.gaze_value);text.addStretch()
        row.addLayout(text,1)
        box.addLayout(row)
        actions=QHBoxLayout();actions.setSpacing(14)
        self.disconnect_phone=QPushButton('Disconnect iPhone')
        self.disconnect_phone.clicked.connect(self.c.toggle_phone)
        actions.addWidget(self.disconnect_phone)
        self.phone_settings_button=QPushButton('Connection settings')
        self.phone_settings_button.setObjectName('link')
        self.phone_settings_button.clicked.connect(self.open_phone_settings)
        actions.addWidget(self.phone_settings_button)
        self.use_wifi_button=QPushButton('Use Wi‑Fi instead')
        self.use_wifi_button.setObjectName('link')
        self.use_wifi_button.clicked.connect(self.use_wifi)
        actions.addWidget(self.use_wifi_button)
        actions.addStretch()
        box.addLayout(actions)
        layout.addWidget(frame)

    def build_accuracy_card(self,layout):
        frame,box=card();box.setSpacing(8);box.setContentsMargins(18,14,18,14)
        row=QHBoxLayout()
        row.addWidget(label('Accuracy','heading'),1)
        details=QPushButton('Details');details.setObjectName('link')
        details.setAccessibleName('Calibration details')
        details.clicked.connect(lambda:self.navigate(1))
        row.addWidget(details)
        box.addLayout(row)
        self.accuracy_note=label('','muted',True)
        box.addWidget(self.accuracy_note)
        self.accuracy_rows=QWidget()
        rows=QVBoxLayout(self.accuracy_rows);rows.setContentsMargins(0,0,0,0);rows.setSpacing(8)
        self.overview_metrics=self.metric_rows(rows)
        box.addWidget(self.accuracy_rows)
        layout.addWidget(frame)

    def build_quick_card(self,layout):
        frame,box=card();box.setSpacing(8);box.setContentsMargins(18,14,18,14)
        box.addWidget(label('Quick settings','heading'))
        self.add_check(box,'show_gaze','Show gaze dot')
        box.addWidget(rule())
        self.quick_magnetic=Switch('Magnetic focus')
        self.quick_magnetic.setToolTip('Frame one element instead of lighting the area around your gaze.')
        self.quick_magnetic.toggled.connect(lambda on:self.update_setting('highlight_mode','magnetic' if on else 'radial'))
        box.addWidget(self.quick_magnetic)
        box.addWidget(rule())
        self.add_check(box,'quiet_mode','Quiet mode').setToolTip('Hide the highlight while tracking stays on.')
        layout.addWidget(frame)

    def metric_rows(self,layout,explain=False):
        """One ruled row per accuracy figure. Its meaning is spelled out below it, or kept in a tooltip."""
        values={}
        for key,title,meaning in METRICS:
            layout.addWidget(rule())
            row=QHBoxLayout();name=label(title,'muted');value=label('—','metric');values[key]=value
            row.addWidget(name,1);row.addWidget(value)
            layout.addLayout(row)
            if explain:layout.addWidget(label(meaning,'faint',True))
            else:name.setToolTip(meaning)
        return values

    def build_profile(self):
        layout,_,header=self.page('Calibration','Gaze setup teaches Mira your eyes in about a minute.',860)
        self.profile_setup=QPushButton('Set up · about a minute')
        self.profile_setup.setObjectName('primary');self.profile_setup.clicked.connect(self.open_setup)
        header.addWidget(self.profile_setup,0,Qt.AlignBottom)
        frame,box=card();self.profile_title=label('Set up your gaze profile','heading',True)
        self.profile_detail=label('','muted',True);box.addWidget(self.profile_title);box.addWidget(self.profile_detail)
        box.addSpacing(4)
        steps=QHBoxLayout();steps.setSpacing(12)
        for number,title,hint in (
            ('1','Connect','Sit comfortably'),('2','Gaze setup','13 points, about a minute'),('3','Check','5 new points'),
        ):
            step=QHBoxLayout();step.setSpacing(9);step.setAlignment(Qt.AlignTop)
            number_label=label(number,'step');number_label.setFixedSize(22,22);number_label.setAlignment(Qt.AlignCenter)
            step.addWidget(number_label,0,Qt.AlignTop)
            caption=QVBoxLayout();caption.setSpacing(2)
            caption.addWidget(label(title,'heading'))
            caption.addWidget(label(hint,'muted'))
            step.addLayout(caption);steps.addLayout(step,1)
        box.addLayout(steps);layout.addWidget(frame)
        self.no_setup_results=label('No setup results yet','muted')
        layout.addWidget(self.no_setup_results)
        self.validation_panel,box=card()
        row=QHBoxLayout()
        self.validation_verdict=label('','heading',True);row.addWidget(self.validation_verdict,1)
        self.validation_toggle=QPushButton('Details');self.validation_toggle.setObjectName('advanced')
        row.addWidget(self.validation_toggle);box.addLayout(row)
        self.validation_details=QWidget();details=QVBoxLayout(self.validation_details)
        details.setContentsMargins(0,0,0,0);details.setSpacing(8)
        box.addWidget(self.validation_details)
        connect_disclosure(self.validation_toggle,self.validation_details)
        self.validation_hint=label('','muted',True)
        details.addWidget(self.validation_hint)
        self.metrics=self.metric_rows(details,explain=True)
        layout.addWidget(self.validation_panel)
        self.learning_advanced=QPushButton('Advanced');self.learning_advanced.setObjectName('advanced')
        layout.addWidget(self.learning_advanced,0,Qt.AlignLeft)
        learning=QWidget();advanced=QVBoxLayout(learning)
        advanced.setContentsMargins(0,0,0,0);advanced.setSpacing(16)
        layout.addWidget(learning)
        connect_disclosure(self.learning_advanced,learning)
        frame,box=card();row=QHBoxLayout();row.addWidget(label('Click learning','heading'),1)
        self.click_details_toggle=QPushButton('Details');self.click_details_toggle.setObjectName('advanced')
        self.click_details_toggle.setAccessibleName('Click learning details')
        row.addWidget(self.click_details_toggle);box.addLayout(row)
        self.click_plain_status=label('','muted',True);box.addWidget(self.click_plain_status)
        row=QHBoxLayout();row.setSpacing(16)
        self.click_learning=self.add_check(row,'click_learning','Refine gaze using clicks',commit=self.c.set_click_learning)
        row.addStretch()
        self.undo_learning=QPushButton('Undo last correction');self.undo_learning.clicked.connect(self.c.undo_learning)
        row.addWidget(self.undo_learning);box.addLayout(row)
        self.click_details=QWidget();details=QVBoxLayout(self.click_details)
        details.setContentsMargins(0,0,0,0);details.setSpacing(6);details.addWidget(rule())
        self.click_hint=label('','muted',True);details.addWidget(self.click_hint)
        self.learning_value=label('','muted',True);details.addWidget(self.learning_value)
        self.click_summary=label('','muted',True);details.addWidget(self.click_summary)
        self.learning_model=label('','muted',True);details.addWidget(self.learning_model)
        box.addWidget(self.click_details);connect_disclosure(self.click_details_toggle,self.click_details)
        advanced.addWidget(frame)
        frame,box=card();row=QHBoxLayout();row.addWidget(label('Next target prediction','heading'),1)
        self.prediction_details_toggle=QPushButton('Details');self.prediction_details_toggle.setObjectName('advanced')
        self.prediction_details_toggle.setAccessibleName('Next target prediction details')
        row.addWidget(self.prediction_details_toggle);box.addLayout(row)
        self.prediction_plain_status=label('','muted',True);box.addWidget(self.prediction_plain_status)
        self.prediction_status=label('','muted',True);box.addWidget(self.prediction_status)
        connect_disclosure(self.prediction_details_toggle,self.prediction_status)
        advanced.addWidget(frame)
        layout.addWidget(label('Run setup again if you move your iPhone or monitor. Mira doesn’t automatically save video or audio.','muted',True));layout.addStretch()

    def build_practice(self):
        layout,_,header = self.page('Practice', 'See how Mira highlights a button as you look at it or move your mouse.')
        self.pointer = Switch('Try with mouse')
        self.pointer.toggled.connect(self.toggle_pointer)
        header.addWidget(self.pointer, 0, Qt.AlignBottom)
        header.addSpacing(8)
        self.practice_button = QPushButton('Connect iPhone')
        self.practice_button.setObjectName('primary')
        self.practice_button.clicked.connect(self.toggle_practice)
        header.addWidget(self.practice_button, 0, Qt.AlignBottom)

        stage = QFrame()
        stage.setObjectName('inset')
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

        feedback = QHBoxLayout()
        feedback.setSpacing(8)
        feedback.addWidget(label('Counter', 'muted'))
        self.count = label('0', 'practiceCount')
        feedback.addWidget(self.count)
        feedback.addSpacing(8)
        self.practice_result = label('', 'muted', True)
        feedback.addWidget(self.practice_result, 1)
        layout.addLayout(feedback)

        self.practice_tools_button = QPushButton('More practice tools')
        self.practice_tools_button.setObjectName('advanced')
        layout.addWidget(self.practice_tools_button, 0, Qt.AlignLeft)
        tools = QWidget()
        fields = QVBoxLayout(tools)
        fields.setContentsMargins(0, 0, 0, 0)
        fields.setSpacing(6)
        self.practice_command = QLineEdit()
        self.practice_command.setAccessibleName('Type a command')
        self.practice_command.setPlaceholderText('click Plus')
        self.practice_command.returnPressed.connect(self.submit)
        command = label('Type a command')
        command.setBuddy(self.practice_command)
        fields.addWidget(command)
        command_row = QHBoxLayout()
        command_row.addWidget(self.practice_command, 1)
        self.run_practice_command = QPushButton('Run command')
        self.run_practice_command.clicked.connect(self.submit)
        command_row.addWidget(self.run_practice_command)
        fields.addLayout(command_row)
        fields.addSpacing(6)
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
        layout,_,_=self.page('Settings',width=860)
        frame,box=card();box.addWidget(label('iPhone','heading'))
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
        frame,box=card();box.addWidget(label('Highlight','heading'))
        self.highlight_mode=SettingsComboBox();self.highlight_mode.setAccessibleName('Highlight mode')
        self.highlight_mode.addItem('Follow gaze · dot and area','radial')
        self.highlight_mode.addItem('Magnetic focus · one element','magnetic')
        self.highlight_mode.setCurrentIndex(1 if self.settings.get('highlight_mode')=='magnetic' else 0)
        self.highlight_mode.currentIndexChanged.connect(lambda _:self.update_setting('highlight_mode',self.highlight_mode.currentData()))
        box.addWidget(self.highlight_mode)
        self.highlight_hint=label('','muted',True);box.addWidget(self.highlight_hint)
        self.show_gaze_check=self.add_check(box,'show_gaze','Show gaze dot')
        self.quiet_mode_check=self.add_check(box,'quiet_mode','Quiet mode')
        self.quiet_hint=label('Hide gaze cues while control stays on.','muted',True)
        box.addWidget(self.quiet_hint)
        layout.addWidget(frame)
        self.advanced=QPushButton('Advanced');self.advanced.setObjectName('advanced')
        layout.addWidget(self.advanced,0,Qt.AlignLeft)
        self.advanced_panel=QWidget();advanced=QVBoxLayout(self.advanced_panel)
        advanced.setContentsMargins(0,0,0,0);advanced.setSpacing(16)
        layout.addWidget(self.advanced_panel)
        connect_disclosure(self.advanced,self.advanced_panel)
        self.advanced.toggled.connect(self.list_microphones)
        frame,box=card();box.addWidget(label('Webcam · second angle','heading'))
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
        frame,box=card();box.addWidget(label('Optional actions','heading'))
        box.addWidget(label('With the microphone on, say “click” to activate the yellow target. Hold Shift to select with the mouse.','muted',True))
        self.add_check(box,'dwell_enabled','Click by holding your gaze','Keep looking at the same target to click automatically. This is off by default.')
        self.add_slider(box,'dwell_ms','Hold time',700,2500)
        self.anticipation_check=self.add_check(box,'anticipation','Show a possible next target',
            'Purple marks what you might look at next. It learns from your clicks.',commit=self.c.set_anticipation)
        advanced.addWidget(frame)
        frame,box=card();box.addWidget(label('Voice and recognition','heading'))
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
        b=QPushButton('Save voice and model settings');b.clicked.connect(self.save_connections);box.addWidget(b,0,Qt.AlignLeft)
        advanced.addWidget(frame)
        layout.addWidget(label('Version 1.3.1','faint'));layout.addStretch()

    def add_check(self,layout,key,text,hint='',commit=None):
        """A switch for one saved setting. Every switch of the same setting stays in step."""
        check=Switch(text);check.setChecked(bool(self.settings[key]))
        def apply(value):
            if commit:commit(value)
            else:self.update_setting(key,value)
            self.sync_checks()
        check.toggled.connect(apply);layout.addWidget(check)
        self.setting_checks.setdefault(key,[]).append(check)
        if hint:layout.addWidget(label(hint,'muted',True))
        return check

    def sync_checks(self):
        for key,checks in self.setting_checks.items():
            for check in checks:
                with QSignalBlocker(check):
                    check.setChecked(bool(self.settings[key]))

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
        if key=='quiet_mode':
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
        self.profile_detail.setText('Mira checked five new points from your usual seat. Repeat setup if you move the iPhone or the screen.' if available else
            'Sit the way you usually do for 13 setup points and 5 new check points.')
        self.profile_setup.setText('Refine setup · about a minute' if available else 'Set up · about a minute')
        report=(state.get('report') if available else state.get('last_result')) or {}
        self.validation_panel.setVisible(bool(report))
        self.no_setup_results.setVisible(not report)
        self.accuracy_rows.setVisible(bool(report))
        if not report:
            self.accuracy_note.setText('Run gaze setup to see how close Mira gets to where you look.')
            return
        good = report.get('usable') is True
        self.validation_verdict.setText('Ready for panels and large buttons' if good else 'Needs another try')
        self.accuracy_note.setText('Best for panels and large buttons.' if good else
                                   'Not accurate enough yet. Run setup again.')
        for key,_,_ in METRICS:
            value=report.get(key)
            text='—' if not isinstance(value,(int,float)) else f'{value*100:.0f}%' if key=='coverage' else f'{value:.0f} px'
            self.metrics[key].setText(text);self.overview_metrics[key].setText(text)
        source='Saved setup · ' if available else 'Latest attempt · '
        self.validation_hint.setText(source+('ready to refine with clicks.' if good else
                                     'needs refinement. '+' '.join(report.get('problems', [])[:2])))

    def refresh_control_panel(self):
        visible=self.c.running or self.c.output_pending or self.more_controls.isChecked()
        self.control_panel.setVisible(visible and not self.canvas.blocked)
        self.run.setVisible(visible)

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
        if needs_setup:
            pill = ('off', 'Not set up')
            idle = (('Set up again for this screen', 'Mira can’t use the saved gaze setup here. Select Set up iPhone to redo it.', True)
                    if c.phone_needs_setup else
                    ('See where you’re looking', 'Stand your iPhone under the screen with the front camera facing you. '
                     'Open Mira on it and tap Start Tracking, then select Set up iPhone.', True))
        elif phone_state == 'hold':
            pill, idle = ('hold', 'Not connected'), (heading, status+' Then select Try again.', False)
        elif phone_state == 'off':
            pill = ('off', 'Ready to connect')
            idle = ('Ready when you are', 'Open Mira on your iPhone and tap Start Tracking, then select Connect iPhone.', True)
        else:
            pill = ('go', 'Tracking') if phone_state == 'go' else ('standby', 'Waiting for your gaze' if has_frames else 'Connecting')
            idle = ('Tracking is paused', 'Select Start to see where you’re looking.', False)
        # The mode outlives a missed frame so a command already begun keeps its source; "tracking" needs a live point.
        if c.gaze_source_mode == 'webcam' and c.gaze is not None:
            status += ' Using webcam.'
            pill = ('go', 'Tracking with webcam')
        self.state_pill.set_state(*pill)
        self.phone_glyph.set_state(phone_state)
        self.canvas.idle = idle
        self.canvas.blocked = phone_state == 'hold' and c.gaze_source_mode != 'webcam' and not c.pointer_mode
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
            lamp, caption, chip = self.sources[name]
            lamp.set_state(state, word)
            caption.setText(word)
            chip.setVisible(name == 'iPhone' or state != 'off')
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
        self.sync_checks()
        self.quiet_hint.setText('Gaze cues are hidden. Click learning and connections follow your settings.' if quiet else
                                'Hide gaze cues while control stays on.')
        magnetic=self.settings.get('highlight_mode')=='magnetic'
        self.highlight_mode.blockSignals(True);self.highlight_mode.setCurrentIndex(1 if magnetic else 0);self.highlight_mode.blockSignals(False)
        for mode in (self.highlight_mode,self.quick_magnetic):
            mode.setEnabled(not quiet and not c.busy and not c.output_pending)
        with QSignalBlocker(self.quick_magnetic):
            self.quick_magnetic.setChecked(magnetic)
        for check in (*self.setting_checks['show_gaze'],self.anticipation_check):
            check.setEnabled(not quiet and not magnetic)
        self.highlight_hint.setText('Quiet mode hides gaze cues.' if quiet else
                                   'Look at another element to move the yellow frame. This mode hides the gaze dot and prediction.' if magnetic else
                                   'The blue dot follows your gaze. Yellow marks the current focus.')
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
        if not quiet and c.scene and c.running:
            scene_detail = f'{len(c.scene.current_elements(now))} screen elements'
            if c.scene.latency_ms > 0:
                scene_detail += f' · {c.scene.latency_ms:.0f} ms'
        self.latency.setText(scene_detail)
        self.status.setText(c.status)
        self.canvas.update()

    def refresh_learning(self, phone_state):
        c=self.c
        self.click_hint.setText(c.click_status)
        learned=c.learning.click_accepted
        if not self.settings.get('click_learning',True):
            click_status='Click learning is off.'
        elif learned:
            count='one' if learned==1 else str(learned)
            click_status=f'Mira has learned from {count} of your clicks.'
        elif phone_state=='go' and c.learning.key is not None:
            click_status='Mira is ready to learn from your clicks.'
        else:
            click_status='Mira learns from your clicks once iPhone is connected.'
        self.click_plain_status.setText(click_status)
        self.learning_value.setText(f'Clicks used to refine gaze: {learned}' if c.learning.key is not None else 'Corrections load with your gaze profile.')
        counts=c.click_counts
        self.click_summary.setText(f'This session · used {counts["accepted"]} of {counts["seen"]} · gaze elsewhere {counts["away"]} · too little data {counts["quality"]}')
        self.learning_model.setText(c.learning.model_status if c.learning.key is not None else 'Connect a working gaze profile to refine it.')
        predictor=c.anticipation
        if not self.settings.get('anticipation',True):
            prediction_status='Prediction is off.'
        elif predictor.error:
            prediction_status='Prediction learning is unavailable.'
        else:
            prediction_status='Mira suggests a likely next target in purple and learns from your clicks.'
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
