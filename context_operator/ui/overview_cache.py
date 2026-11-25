"""Small cached geometric overview; it contains no captured screen pixels."""
import math
from collections import OrderedDict
from PySide6.QtCore import Qt,QRectF,QPointF
from PySide6.QtGui import QImage,QPainter,QPen,QColor


class OverviewCache:
    def __init__(self,max_pixels=2_000_000):
        self.max_pixels=max_pixels;self.pixels=0;self.images=OrderedDict();self.sprites=OrderedDict();self.builds=0

    def paint(self,painter,rects,width,height):
        dpr=painter.device().devicePixelRatioF()
        key=(width,height,dpr,tuple((r.x(),r.y(),r.width(),r.height()) for r in rects))
        image=self.images.get(key)
        if image is None:
            image=QImage(math.ceil(width*dpr),math.ceil(height*dpr),QImage.Format_ARGB32_Premultiplied)
            image.setDevicePixelRatio(dpr);image.fill(0);p=QPainter(image)
            for rect in rects:self._shape(p,rect,dpr)
            p.end();self.builds+=1;cost=image.width()*image.height()
            if cost<=self.max_pixels:
                while self.images and (self.pixels+cost>self.max_pixels or len(self.images)>=3):
                    _,old=self.images.popitem(last=False);self.pixels-=old.width()*old.height()
                self.images[key]=image;self.pixels+=cost
        else:self.images.move_to_end(key)
        painter.drawImage(QPointF(0,0),image)

    def _shape(self,p,rect,dpr):
        w,h=rect.width(),rect.height();sw,sh=min(w,20.),min(h,20.)
        key=(sw,sh,dpr);sprite=self.sprites.get(key)
        if sprite is None:
            sprite=QImage(math.ceil((sw+4)*dpr),math.ceil((sh+4)*dpr),QImage.Format_ARGB32_Premultiplied)
            sprite.setDevicePixelRatio(dpr);sprite.fill(0)
            q=QPainter(sprite);q.setRenderHint(QPainter.Antialiasing)
            q.setPen(QPen(QColor('#4A515C'),.7));q.setBrush(QColor(58,64,74,70))
            q.drawRoundedRect(QRectF(2,2,sw,sh),3,3);q.end()
            self.sprites[key]=sprite
            while len(self.sprites)>256:self.sprites.popitem(last=False)
        else:self.sprites.move_to_end(key)
        def segments(size,small):
            return ((0,size+4,0,small+4),) if size==small else ((0,6,0,6),(6,size-8,6,small-8),(size-2,6,small-2,6))
        for dx,dw,sx,pw in segments(w,sw):
            for dy,dh,sy,ph in segments(h,sh):
                p.drawImage(QRectF(rect.x()-2+dx,rect.y()-2+dy,dw,dh),sprite,QRectF(sx*dpr,sy*dpr,pw*dpr,ph*dpr))
