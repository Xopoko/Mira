"""Bounded RAM cache of painted UI geometry, never screen pixels."""
import math
from collections import OrderedDict
from PySide6.QtCore import Qt,QRectF
from PySide6.QtGui import QImage,QPainter,QColor,QPen,QBrush,QRadialGradient

def tile_bounds(center,radius):
    # The miniature overview needs small tiles too. A fixed 128 px grid made its
    # ~40 px light allocate and blend a 384 px square on every frame/window.
    step=float(max(16,min(128,2**math.floor(math.log2(max(1.,radius/2))))))
    margin=math.ceil(radius/step)*step
    return QRectF(math.floor(center.x()/step)*step-margin,math.floor(center.y()/step)*step-margin,
                  margin*2+step,margin*2+step)

class LightCache:
    def __init__(self,max_pixels=8_000_000):
        self.max_pixels=max_pixels;self.pixels=0;self.images=OrderedDict();self.builds=0
        self.sprites=OrderedDict()

    def _shape(self,p,rect,primary,strength,dpr,clip):
        """Nine-slice a small cached rounded frame; big panes cost only visible pixels."""
        w,h=rect.width(),rect.height();sw,sh=min(w,64.),min(h,64.)
        key=(sw,sh,primary,strength,dpr);sprite=self.sprites.get(key)
        if sprite is None:
            sprite=QImage(math.ceil((sw+16)*dpr),math.ceil((sh+16)*dpr),QImage.Format_ARGB32_Premultiplied)
            sprite.setDevicePixelRatio(dpr);sprite.fill(0);q=QPainter(sprite);q.setRenderHint(QPainter.Antialiasing)
            color=QColor('#f5cd69' if primary else '#78caff');gain=(1. if primary else .82)*strength
            def tint(alpha):
                c=QColor(color);c.setAlpha(round(alpha*gain));return c
            body=QRectF(8,8,sw,sh);q.setPen(Qt.NoPen);q.setBrush(tint(42 if primary else 26));q.drawRoundedRect(body,6,6);q.setBrush(Qt.NoBrush)
            for width,a in ((13,17),(7,36),(2.4 if primary else 1.7,250)):
                q.setPen(QPen(tint(a),width));q.drawRoundedRect(body,6,6)
            q.end();self.sprites[key]=sprite
            while len(self.sprites)>96:self.sprites.popitem(last=False)
        else:self.sprites.move_to_end(key)
        target=rect.adjusted(-8,-8,8,8)
        # Each segment: destination offset/size, source offset/size (logical).
        def segments(size,small):
            return ((0,size+16,0,small+16),) if size==small else ((0,16,0,16),(16,size-16,16,small-16),(size,16,small,16))
        for dx,dw,sx,pw in segments(w,sw):
            for dy,dh,sy,ph in segments(h,sh):
                dest=QRectF(target.x()+dx,target.y()+dy,dw,dh)
                if dest.intersects(clip):p.drawImage(dest,sprite,QRectF(sx*dpr,sy*dpr,pw*dpr,ph*dpr))

    def paint(self,painter,elements,center,radius,selected,energy,strengths):
        # Snap the backing tile, not the light: movement remains subpixel smooth.
        bounds=tile_bounds(center,radius)
        area=QRectF(center.x()-radius,center.y()-radius,radius*2,radius*2)
        dpr=painter.device().devicePixelRatioF()
        shapes=tuple((identity==selected,rect.x()-bounds.x(),rect.y()-bounds.y(),rect.width(),rect.height(),
                      round((strengths.get(identity,1.) if strengths is not None else 1.)*16)/16)
                     for identity,rect in elements if rect.adjusted(-8,-8,8,8).intersects(bounds))
        if not shapes:return
        key=(bounds.width(),bounds.height(),dpr,shapes)
        image=self.images.get(key)
        if image is None:
            image=QImage(math.ceil(bounds.width()*dpr),math.ceil(bounds.height()*dpr),QImage.Format_ARGB32_Premultiplied)
            image.setDevicePixelRatio(dpr);image.fill(0);p=QPainter(image);p.setRenderHint(QPainter.Antialiasing)
            clip=QRectF(0,0,bounds.width(),bounds.height())
            flat=[0.,0.,0.,0.]
            def flush_flat():
                if flat[3]:
                    p.fillRect(clip,QColor.fromRgbF(*(v/flat[3] for v in flat[:3]),flat[3]))
                    flat[:]=[0.,0.,0.,0.]
            for primary,x,y,w,h,strength in shapes:
                rect=QRectF(x,y,w,h)
                if rect.adjusted(8,8,-8,-8).contains(clip):
                    # Deep accessibility trees expose nested large containers.
                    # Combine consecutive uniform interiors analytically instead
                    # of blending the same tile-sized fill thirty times.
                    color=QColor('#f5cd69' if primary else '#78caff')
                    a=round((42 if primary else 26)*(1. if primary else .82)*strength)/255
                    flat[:]=[c*a+v*(1-a) for c,v in zip((color.redF(),color.greenF(),color.blueF(),1.),flat)]
                    continue
                flush_flat();self._shape(p,rect,primary,strength,dpr,clip)
            flush_flat()
            p.end();self.builds+=1
            cost=image.width()*image.height()
            if cost<=self.max_pixels:
                while self.images and (self.pixels+cost>self.max_pixels or len(self.images)>=16):
                    _,old=self.images.popitem(last=False);self.pixels-=old.width()*old.height()
                self.images[key]=image;self.pixels+=cost
        else:self.images.move_to_end(key)
        # One moving gradient for the whole tile, instead of four per control.
        lit=image.copy();p=QPainter(lit);p.setCompositionMode(QPainter.CompositionMode_DestinationIn)
        gradient=QRadialGradient(center-bounds.topLeft(),max(1.,radius));boost=1+min(1.,max(0.,energy))*.3
        for position,strength in ((0,1.),(.25,.94),(.6,.45),(.85,.10),(1.,0.)):
            gradient.setColorAt(position,QColor(255,255,255,round(255*min(1.,strength*boost))))
        p.fillRect(QRectF(0,0,bounds.width(),bounds.height()),QBrush(gradient));p.end()
        painter.save();painter.setClipRect(area.adjusted(-1,-1,1,1),Qt.IntersectClip)
        painter.drawImage(bounds.topLeft(),lit);painter.restore()

LIGHT_CACHE=LightCache()
