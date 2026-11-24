"""One COM-thread-owned reader: controls and public text, without editor values."""
import hashlib,re,time
from ..core.contracts import Element,Rect,WindowMap
from .windows import Windows

ROLES={'Button','Edit','Hyperlink','MenuItem','CheckBox','RadioButton','ComboBox','ListItem',
       'TabItem','TreeItem','Slider','Spinner','SplitButton','DataItem','Text'}

def distinct_elements(elements):
    # Some accessibility providers repeat one control through multiple tree paths.
    elements=tuple({(e.role,e.rect,e.label):e for e in elements}.values())
    # Public text within a button/editor is its caption, not another competing target.
    controls=[e for e in elements if e.role not in ('Text','ListItem','TreeItem','DataItem')]
    return tuple(e for e in elements if e.role!='Text' or not any(
        c.rect.x<=e.rect.x and c.rect.y<=e.rect.y and c.rect.x+c.rect.w>=e.rect.x+e.rect.w
        and c.rect.y+c.rect.h>=e.rect.y+e.rect.h for c in controls))

def protected(title):
    # Include the localized Windows Security title in the capture exclusion.
    return bool(re.search(
        r'1password|bitwarden|keepass|windows security'
        r'|\u0431\u0435\u0437\u043e\u043f\u0430\u0441\u043d\u043e\u0441\u0442\u044c windows',
        title, re.I,
    ))

class WindowReader:
    def __init__(self):
        import comtypes.client
        import uiautomation as auto
        self.win=Windows();self.names=auto.ControlTypeNames
        core=comtypes.client.GetModule('UIAutomationCore.dll')
        self.uia=comtypes.client.CreateObject(core.CUIAutomation8,interface=core.IUIAutomation2)
        self.uia.ConnectionTimeout=500
        self.uia.TransactionTimeout=1000
        self.cache=self.uia.CreateCacheRequest();self.cache.TreeScope=1
        self.cache.AutomationElementMode=0  # Native AutomationElementMode_None: cached properties only.
        for prop in (30001,30003,30005,30010,30011,30019,30022):self.cache.AddProperty(prop)
        conditions=[self.uia.CreatePropertyCondition(30003,key) for key,name in self.names.items() if name.replace('Control','') in ROLES]
        condition=conditions[0]
        for other in conditions[1:]:condition=self.uia.CreateOrCondition(condition,other)
        for prop,value in ((30019,False),(30022,False),(30010,True)):
            condition=self.uia.CreateAndCondition(condition,self.uia.CreatePropertyCondition(prop,value))
        self.condition=condition
        self.secret_condition=self.uia.CreateAndCondition(self.uia.CreatePropertyCondition(30019,True),self.uia.CreatePropertyCondition(30022,False))

    def scan(self,surface):
        started=time.monotonic();hwnd=surface.window
        if protected(surface.title):return WindowMap(hwnd,surface.pid,surface.rect,started,blocked_capture=True)
        if self.win.pid(hwnd)!=surface.pid or self.win.rect(hwnd)!=surface.rect:raise RuntimeError('Window changed before scan')
        root=self.uia.ElementFromHandle(hwnd)
        started=time.monotonic()  # Earliest time of the actual element snapshot, not connection setup.
        # Keep password elements out of the cache request, including their names.
        blocked=bool(root.FindFirst(4,self.secret_condition))
        controls=root.FindAllBuildCache(4,self.condition,self.cache)
        count=controls.Length if controls else 0
        blocked=blocked or count>4000  # An incomplete scan cannot clear the capture guard.
        elements=[]
        for i in range(min(count,4000)):
            try:
                control=controls.GetElement(i)
                if control.CachedIsPassword:
                    blocked=True;continue
                if control.CachedIsOffscreen or not control.CachedIsEnabled:continue
                role=self.names[control.CachedControlType].replace('Control','')
                r=control.CachedBoundingRectangle;rect=Rect(r.left,r.top,r.right-r.left,r.bottom-r.top)
                if rect.w<3 or rect.h<3:continue
                label=control.CachedName[:180]
                if role=='Text' and not label.strip():continue
                key=f'{hwnd}:{control.CachedAutomationId}:{role}:{rect.as_list()}:{label}'
                identity=hashlib.blake2s(key.encode(),digest_size=10).hexdigest()
                elements.append(Element(identity,label or role,role,rect,hwnd,'uia',True,surface.pid,captured=started))
            except Exception:
                blocked=True
        if (not self.win.u.IsWindowVisible(hwnd) or self.win.u.IsIconic(hwnd)
                or self.win.pid(hwnd)!=surface.pid or self.win.rect(hwnd)!=surface.rect):
            raise RuntimeError('Window changed during scan')
        return WindowMap(hwnd,surface.pid,surface.rect,started,distinct_elements(elements),surface.visible,surface.title,blocked)
