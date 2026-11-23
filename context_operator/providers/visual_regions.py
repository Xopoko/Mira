"""Verify each recognized object against its own pixels, not unrelated animation."""
from dataclasses import dataclass
import hashlib,math
from ..core.contracts import Rect

def digest(image):return hashlib.blake2s(image.tobytes(),digest_size=12).digest()

@dataclass(frozen=True)
class VisualFrame:
    owner: tuple
    rect: Rect
    pixels: bytes

@dataclass(frozen=True)
class VisualRegion:
    rect: Rect
    context: Rect
    pixels: bytes
    label: str
    source: str

def crop(image,frame,rect):
    if (rect.x<frame.x or rect.y<frame.y or rect.x+rect.w>frame.x+frame.w
            or rect.y+rect.h>frame.y+frame.h):return None
    return image.crop((int(rect.x-frame.x),int(rect.y-frame.y),int(rect.x+rect.w-frame.x),int(rect.y+rect.h-frame.y)))

def regions(frame,image,result):
    found=[]
    for item in result.get('items',()):
        x,y,w,h=item['rect'];left,top=math.floor(x),math.floor(y)
        right,bottom=math.ceil(x+w),math.ceil(y+h)
        if left<0 or top<0 or right>image.width or bottom>image.height or right<=left or bottom<=top:continue
        rect=Rect(frame.rect.x+left,frame.rect.y+top,right-left,bottom-top)
        l,t=max(0,left-8),max(0,top-8);r,b=min(image.width,right+8),min(image.height,bottom+8)
        context=Rect(frame.rect.x+l,frame.rect.y+t,r-l,b-t)
        found.append(VisualRegion(rect,context,digest(crop(image,frame.rect,context)),item.get('label',''),item['source']))
    return tuple(found)

def verified(regions,image,frame):
    result=[]
    for region in regions:
        patch=crop(image,frame,region.context)
        if patch is not None and digest(patch)==region.pixels:result.append(region)
    return tuple(result)


def same_object(a,b):
    """Detector edge jitter across crops must not create another target."""
    if a.source!=b.source:return False
    a,b=a.rect,b.rect
    # +/-3 px around the true edge is a 6 px span between two predictions.
    tolerance=max(6.,min(12.,min(a.w,a.h,b.w,b.h)*.12))
    if max(abs(a.x-b.x),abs(a.y-b.y),abs(a.x+a.w-b.x-b.w),abs(a.y+a.h-b.y-b.h))>tolerance:return False
    intersection=max(0.,min(a.x+a.w,b.x+b.w)-max(a.x,b.x))*max(0.,min(a.y+a.h,b.y+b.h)-max(a.y,b.y))
    small,large=sorted((a.w*a.h,b.w*b.h))
    # Equal small icons can have low IoU after diagonal jitter. Edge proximity
    # plus comparable size separates them from a genuinely nested small target.
    return small>0 and small/large>=.7 and intersection/small>=.35


def unique_verified(candidates,image,frame):
    """Newest verified region wins; don't repeatedly hash its older duplicates."""
    result=[];buckets={}
    for region in candidates:
        x,y=region.rect.center;cell=(math.floor(x/64),math.floor(y/64))
        nearby=(other for dx in (-1,0,1) for dy in (-1,0,1)
                for other in buckets.get((region.source,cell[0]+dx,cell[1]+dy),()))
        if any(same_object(region,other) for other in nearby):continue
        if not verified((region,),image,frame):continue
        result.append(region)
        buckets.setdefault((region.source,*cell),[]).append(region)
    return tuple(result)
