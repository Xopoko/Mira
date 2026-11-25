"""Provider-neutral contracts. Coordinates are physical virtual-desktop pixels."""
from dataclasses import dataclass
import math

TARGET_ACTIONS = frozenset({'click', 'double_click', 'right_click', 'move', 'scroll'})
WINDOWS_ACTIONS = TARGET_ACTIONS | {'type_text', 'press_keys'}
SCROLL_DIRECTIONS = ('up', 'down')
SCENE_TTL = 3.
MAP_TTL = 1.25
GAZE_TTL = .2

@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self): return self.x + self.w/2, self.y + self.h/2
    def contains(self, x, y): return self.x <= x < self.x+self.w and self.y <= y < self.y+self.h
    def distance(self, x, y):
        return math.hypot(max(self.x-x, 0, x-self.x-self.w), max(self.y-y, 0, y-self.y-self.h))
    def as_list(self): return [self.x,self.y,self.w,self.h]

@dataclass(frozen=True)
class Element:
    id: str
    label: str
    role: str
    rect: Rect
    window: int = 0
    source: str = 'uia'
    enabled: bool = True
    pid: int = 0
    visual_hash: str = ''
    captured: float = 0.

    @property
    def identity(self):
        """The same control across scene refreshes, including verified visual content."""
        return (self.id, self.window, self.pid, self.rect, self.label, self.role,
                self.source, self.visual_hash)

@dataclass(frozen=True)
class WindowMap:
    window: int
    pid: int
    rect: Rect
    captured: float
    elements: tuple[Element, ...] = ()
    visible: tuple[Rect, ...] = ()
    title: str = ''
    blocked_capture: bool = False

@dataclass(frozen=True)
class Scene:
    id: int
    captured: float
    bounds: Rect
    elements: tuple[Element, ...]
    window: int = 0
    title: str = ''
    latency_ms: float = 0
    window_rect: Rect | None = None
    windows: tuple[WindowMap, ...] = ()

    def window_map(self,window):
        return next((w for w in self.windows if w.window==window),None)

    def current_elements(self, now, max_age=SCENE_TTL):
        if not 0 <= now - self.captured < max_age:
            return ()
        if not self.windows:
            return tuple(e for e in self.elements if 0 <= now - (e.captured or self.captured) < max_age)

        max_age = min(max_age, MAP_TTL)
        windows = {w.window: w for w in self.windows if 0 <= now - w.captured < max_age}
        current = []
        for element in self.elements:
            window = windows.get(element.window)
            if window is None or window.pid != element.pid:
                continue
            if window.blocked_capture and element.source != 'uia':
                continue
            if 0 <= now - (element.captured or window.captured) < max_age:
                current.append(element)
        return tuple(current)

    def current_target(self, element: Element | None, now) -> Element | None:
        if element is None:
            return None
        identity = element.identity
        return next((e for e in self.current_elements(now)
                     if e.enabled and e.rect.w > 0 and e.rect.h > 0 and e.identity == identity), None)

@dataclass(frozen=True)
class Gaze:
    x: float
    y: float
    captured: float
    radius: float = 230
    source: str = 'iphone'

@dataclass(frozen=True)
class GazeMeasurement:
    """Synchronized numeric eye/head features and the unsmoothed base prediction."""
    captured: float
    features: tuple[float, ...]
    point: tuple[float, float]
    supported: bool = True

@dataclass(frozen=True)
class RankedTarget:
    element: Element
    score: float
    distance: float
    reason: str

@dataclass(frozen=True)
class Focus:
    target: Element | None = None
    alternatives: tuple[RankedTarget, ...] = ()
    ambiguous: bool = False
    reason: str = 'Look at an element'

@dataclass(frozen=True)
class Intent:
    action: str
    text: str = ''
    direction: str = ''
    target_id: str = ''
    keys: tuple[str, ...] = ()
    amount: int = 3
    reply: str = ''
