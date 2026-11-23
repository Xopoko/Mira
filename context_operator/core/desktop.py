"""Visible desktop geometry and independently aged window maps, in physical pixels."""
from dataclasses import replace
from .contracts import MAP_TTL, Rect, Scene

def intersection(a,b):
    x,y=max(a.x,b.x),max(a.y,b.y)
    right,bottom=min(a.x+a.w,b.x+b.w),min(a.y+a.h,b.y+b.h)
    return Rect(x,y,right-x,bottom-y) if right>x and bottom>y else None

def subtract(rect,cover):
    cut=intersection(rect,cover)
    if cut is None:return (rect,)
    pieces=(Rect(rect.x,rect.y,rect.w,cut.y-rect.y),
            Rect(rect.x,cut.y+cut.h,rect.w,rect.y+rect.h-cut.y-cut.h),
            Rect(rect.x,cut.y,cut.x-rect.x,cut.h),
            Rect(cut.x+cut.w,cut.y,rect.x+rect.w-cut.x-cut.w,cut.h))
    return tuple(p for p in pieces if p.w>0 and p.h>0)

def visible_regions(rect,occluders,bounds):
    clipped=intersection(rect,bounds)
    parts=(clipped,) if clipped else ()
    for cover in occluders:
        parts=tuple(piece for part in parts for piece in subtract(part,cover))
        if not parts:break
    return parts

def compose(sequence,now,bounds,surfaces,maps,primary=0,latency_ms=0):
    windows=[];elements=[]
    for surface in surfaces:
        cached=maps.get(surface.window)
        if (cached is None or (cached.pid,cached.rect)!=(surface.pid,surface.rect)
                or not 0<=now-cached.captured<MAP_TTL):continue
        window=replace(cached,visible=surface.visible,title=surface.title)
        windows.append(window)
        # Partial outlines may show, but the action's center must remain visible.
        elements.extend(e for e in window.elements if any(r.contains(*e.rect.center) for r in window.visible))
    owner=next((w for w in windows if w.window==primary),None)
    return Scene(sequence,now,bounds,tuple(elements),primary,owner.title if owner else '',latency_ms,
                 owner.rect if owner else None,tuple(windows))
