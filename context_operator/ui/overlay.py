"""Radial light reveals only nearby parts of recognized screen elements."""
import math
import time
from PySide6.QtCore import Qt,QTimer,QRectF,QPointF
from PySide6.QtGui import QPainter,QColor,QPen,QBrush,QRadialGradient,QPainterPath,QRegion
from PySide6.QtWidgets import QWidget
from ..core.contracts import GAZE_TTL,MAP_TTL,SCENE_TTL
from .light_cache import LIGHT_CACHE,tile_bounds

class FadeStrengths:
    def __init__(self,born,now):self.born=born;self.now=now
    def get(self,identity,default=1.):
        stamp=self.born.get(identity)
        return default if stamp is None else max(0.,1-math.exp(-(self.now-stamp)/.065))


def prediction_fresh(prediction,now):
    return prediction is not None and prediction.is_fresh(now)


def current_focus_target(scene,target,attention,now,selected_until=0.):
    """Resolve the target while its explicit selection or gaze is fresh."""
    if scene is None:return None
    if selected_until:
        if now>=selected_until:return None
    elif attention is None or not 0<=now-attention.captured<GAZE_TTL:
        return None
    return scene.current_target(target,now)


def focus_ink(target,origin=(0,0),scale=1.):
    if target is None:return QRegion()
    r=target.rect
    return QRegion(QRectF((r.x-origin[0])*scale,(r.y-origin[1])*scale,r.w*scale,r.h*scale)
                   .adjusted(-18,-18,18,18).toAlignedRect())


def paint_magnetic_focus(painter,scene,target,now,origin=(0,0),scale=1.,strength=1.):
    """One complete object contour, fixed to its geometry rather than eye position."""
    target=scene.current_target(target,now) if scene is not None else None
    if target is None:return
    identity=target.identity
    groups=[(w.elements,w.visible,w.captured) for w in scene.windows] if scene.windows else [(scene.elements,(),scene.captured)]
    def project(r):return QRectF((r.x-origin[0])*scale,(r.y-origin[1])*scale,r.w*scale,r.h*scale)
    for elements,visible,stamp in groups:
        if not 0<=now-stamp<(MAP_TTL if scene.windows else SCENE_TTL):continue
        if not any(e.identity==identity for e in elements):continue
        painter.save();painter.setRenderHint(QPainter.Antialiasing)
        if visible:
            clip=QPainterPath()
            for part in visible:clip.addRect(project(part))
            painter.setClipPath(clip,Qt.IntersectClip)
        rect=project(target.rect);corner=min(8.,rect.height()/4,rect.width()/4)
        def tint(color,alpha):
            value=QColor(color);value.setAlpha(round(alpha*max(0.,min(1.,strength))));return value
        painter.setPen(Qt.NoPen);painter.setBrush(tint('#f5cd69',17));painter.drawRoundedRect(rect,corner,corner)
        painter.setBrush(Qt.NoBrush)
        for width,alpha in ((28,8),(18,13),(10,24)):
            painter.setPen(QPen(tint('#78caff',alpha),width));painter.drawRoundedRect(rect,corner,corner)
        for width,alpha in ((7,32),(2.2,235)):
            painter.setPen(QPen(tint('#f5cd69',alpha),width));painter.drawRoundedRect(rect,corner,corner)
        painter.restore()
        return

def ink_region(center,radius,prediction,origin=(0,0),scale=1.):
    """Logical widget damage for the current light and prediction."""
    result=QRegion()
    if center:
        x=(center.x()-origin[0])*scale;y=(center.y()-origin[1])*scale;r=radius*scale+3
        result|=QRegion(QRectF(x-r,y-r,2*r,2*r).toAlignedRect())
    if prediction:
        r=prediction.target.rect
        result|=QRegion(QRectF((r.x-origin[0])*scale,(r.y-origin[1])*scale,r.w*scale,r.h*scale).adjusted(-16,-16,16,16).toAlignedRect())
    return result


def light_brush(center,radius,color,alpha):
    gradient=QRadialGradient(center,max(1.,radius))
    for position,strength in ((0,1.),(.25,.94),(.6,.45),(.85,.10),(1.,0.)):
        c=QColor(color);c.setAlpha(round(min(255,alpha)*strength));gradient.setColorAt(position,c)
    return QBrush(gradient)



def paint_scene_light(painter,scene,center,radius,now,selected=None,energy=0.,origin=(0,0),scale=1.,strengths=None,prediction=None,cache=None):
    """One light field, clipped to the currently exposed parts of each window."""
    def project(r):return QRectF((r.x-origin[0])*scale,(r.y-origin[1])*scale,r.w*scale,r.h*scale)
    point=QPointF((center.x()-origin[0])*scale,(center.y()-origin[1])*scale) if center else None
    forecast=scene.current_target(prediction.target,now) if prediction_fresh(prediction,now) else None
    tile=tile_bounds(point,radius*scale) if point else None
    if tile:
        left,top=tile.left()/scale+origin[0]-15,tile.top()/scale+origin[1]-15
        right,bottom=tile.right()/scale+origin[0]+15,tile.bottom()/scale+origin[1]+15
    groups=[(w.elements,w.visible,w.captured) for w in scene.windows] if scene.windows else [(scene.elements,(),scene.captured)]
    for elements,visible,stamp in groups:
        if not 0<=now-stamp<(MAP_TTL if scene.windows else SCENE_TTL):continue
        painter.save()
        if visible:
            clip=QPainterPath()
            for part in visible:clip.addRect(project(part))
            painter.setClipPath(clip,Qt.IntersectClip)
        if forecast is not None and any(e.identity==forecast.identity for e in elements):
            paint_prediction(painter,project(forecast.rect),prediction.strength)
        if point:
            nearby=[(e.id,project(e.rect)) for e in elements if e.enabled and 0<=now-(e.captured or stamp)<(MAP_TTL if scene.windows else SCENE_TTL)
                    and e.rect.x+e.rect.w>=left and e.rect.x<=right and e.rect.y+e.rect.h>=top and e.rect.y<=bottom]
            (cache or LIGHT_CACHE).paint(painter,nearby,point,radius*scale,selected,energy,strengths)
        painter.restore()


def paint_prediction(painter,rect,strength=1.):
    """Visible violet outline, offset so the yellow focus cannot cover it."""
    painter.save();painter.setRenderHint(QPainter.Antialiasing)
    rect=rect.adjusted(-3,-3,3,3)
    radius=max(56.,min(260.,math.hypot(rect.width(),rect.height())*.9))
    gain=.82+.18*min(1.,max(0.,strength))
    painter.setPen(Qt.NoPen);painter.setBrush(light_brush(rect.center(),radius,'#bb86ff',10*gain))
    painter.drawRoundedRect(rect,7,7)
    painter.setBrush(Qt.NoBrush)
    for width,alpha in ((15,20),(8,45),(2.8,210)):
        painter.setPen(QPen(light_brush(rect.center(),radius,'#bb86ff',alpha*gain),width))
        painter.drawRoundedRect(rect,7,7)
    painter.restore()


class FocusOverlay(QWidget):
    def __init__(self,screen,win):
        super().__init__(None,Qt.Tool|Qt.FramelessWindowHint|Qt.WindowStaysOnTopHint|Qt.WindowTransparentForInput|Qt.WindowDoesNotAcceptFocus)
        self.setWindowTitle('Focus layer')
        self.setAttribute(Qt.WA_TranslucentBackground);self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.win=win;self.setGeometry(screen.geometry())
        self.attention=None;self.center=None;self.previous=None
        self.prediction=None
        self.highlight_mode='radial';self.focus_target=None;self.selected_until=0.
        self.magnet_since=0.;self.magnetic_dirty=False
        self.scene=None;self.born={}
        self.ink=QRegion()
        self.show_point=False;self.energy=0.;self.last_tick=time.monotonic();self.show()
        handle=int(self.winId())
        win.u.SetWindowLongW(handle,-20,win.u.GetWindowLongW(handle,-20)|0x80000|0x20|0x08000000)
        win.u.SetWindowDisplayAffinity(handle,0x11)
        self.timer=QTimer(self);self.timer.timeout.connect(self.animate);self.timer.start(16)

    def sync(self,focus,attention,scene,enabled=True,show_gaze=True,prediction=None,highlight_mode='radial',*,selected_until=0.):
        now=time.monotonic()
        self.attention=attention if enabled and attention and 0<=now-attention.captured<GAZE_TTL else None
        self.show_point=show_gaze and self.attention is not None and self.attention.source!='mouse'
        self.selected_until=selected_until if enabled else 0.
        next_scene=scene if enabled and scene and 0<=now-scene.captured<SCENE_TTL else None
        target=current_focus_target(next_scene,focus.target,self.attention,now,self.selected_until)
        if (highlight_mode!=self.highlight_mode or
                (target.identity if target else None)!=(self.focus_target.identity if self.focus_target else None)):
            self.magnet_since=now
        self.highlight_mode=highlight_mode
        self.focus_target=target
        if self.scene is not next_scene:
            self.magnetic_dirty=True
            self.scene=next_scene
            ids={e.id for e in next_scene.elements} if next_scene else set()
            self.born={identity:self.born.get(identity,now) for identity in ids}
        self.prediction=prediction if (next_scene and 0<=now-scene.captured<.8 and prediction_fresh(prediction,now)
                                      and next_scene.current_target(prediction.target,now)) else None
        if highlight_mode=='magnetic':
            self.prediction=None;self.show_point=False;self.center=None;self.previous=None
        if self.attention is None:self.center=None;self.previous=None;self.energy=0.

    def animate(self):
        now=time.monotonic();dt=min(.1,max(.001,now-self.last_tick));self.last_tick=now
        attention=self.attention
        magnetic=self.highlight_mode=='magnetic'
        if magnetic:
            self.center=None
            target=current_focus_target(self.scene,self.focus_target,attention,now,self.selected_until)
            bounds=self.win.rect(int(self.winId()));dpr=self.devicePixelRatioF()
            current=focus_ink(target,(bounds.x,bounds.y),1/dpr).intersected(QRegion(self.rect()))
            if current==self.ink and not self.magnetic_dirty and (current.isEmpty() or now-self.magnet_since>=.4):return
            dirty=self.ink.united(current);self.ink=current
            self.magnetic_dirty=False
            if not dirty.isEmpty():self.update(dirty)
            return
        if attention is None or not 0<=now-attention.captured<GAZE_TTL:
            self.center=None
        else:
            point=QPointF(attention.x,attention.y)
            speed=math.hypot(point.x()-self.previous.x(),point.y()-self.previous.y())/dt if self.previous else 0.
            self.previous=point
            self.energy=max(self.energy*math.exp(-dt/.17),min(1.,speed/1600))
            if self.center is None or attention.source=='mouse':self.center=point
            else:self.center+=(point-self.center)*(1-math.exp(-dt/.035))
        forecast=self.prediction if prediction_fresh(self.prediction,now) else None
        target=current_focus_target(self.scene,self.focus_target,None,now,self.selected_until)
        if self.center is None and forecast is None and target is None and self.ink.isEmpty():return
        bounds=self.win.rect(int(self.winId()));dpr=self.devicePixelRatioF()
        origin=(bounds.x,bounds.y);scale=1/dpr
        current=ink_region(self.center,attention.radius if attention else 230.,forecast,origin,scale)
        current=(current|focus_ink(target,origin,scale)).intersected(QRegion(self.rect()))
        dirty=self.ink.united(current);self.ink=current
        if not dirty.isEmpty():self.update(dirty)

    def paintEvent(self,event):
        painter=QPainter(self)
        # Translucent native backing stores must erase the old dirty pixels too,
        # including the final repaint after pause/lost attention. SourceOver with
        # transparent pixels would only accumulate the previous light.
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(event.rect(),Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        now=time.monotonic()
        target=current_focus_target(self.scene,self.focus_target,self.attention,now,self.selected_until)
        if self.highlight_mode=='magnetic':
            bounds=self.win.rect(int(self.winId()));dpr=self.devicePixelRatioF()
            age=max(0.,now-self.magnet_since)
            strength=1. if age>=.4 else .6+.4*(1-math.exp(-age/.075))
            paint_magnetic_focus(painter,self.scene,target,now,(bounds.x,bounds.y),1/dpr,strength)
            return
        forecast=self.prediction if prediction_fresh(self.prediction,now) else None
        explicit=target if self.selected_until else None
        attention=self.attention
        center=self.center if attention is not None and 0<=now-attention.captured<GAZE_TTL else None
        if center is None and forecast is None and explicit is None:return
        painter.setRenderHint(QPainter.Antialiasing)
        bounds=self.win.rect(int(self.winId()));dpr=self.devicePixelRatioF()
        if self.scene:
            paint_scene_light(painter,self.scene,center,attention.radius if attention else 230.,now,
                              target.id if target and explicit is None else None,self.energy,(bounds.x,bounds.y),1/dpr,
                              FadeStrengths(self.born,now),forecast)
        if explicit is not None:
            paint_magnetic_focus(painter,self.scene,explicit,now,(bounds.x,bounds.y),1/dpr)
        if center is not None and self.show_point:
            center=QPointF((center.x()-bounds.x)/dpr,(center.y()-bounds.y)/dpr)
            painter.setPen(Qt.NoPen);painter.setBrush(QColor(134,221,255,190));painter.drawEllipse(center,2.5,2.5)
