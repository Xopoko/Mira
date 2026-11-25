"""Bounded text/spatial target matching. Scores are heuristics, not ML probabilities."""
import math
import re
from difflib import SequenceMatcher
from .contracts import GAZE_TTL

# Preferred spellings and stop words for English and Russian transcripts.
ALIASES = (
    (r'\b(?:chat\s*gpt|\u0447\u0430\u0442\s*(?:\u0434\u0436\u0438\s*\u043f\u0438\s*\u0442\u0438|\u0434\u0436\u0438\u043f\u0438\u0442\u0438|\u0433\u043f\u0442))\b', 'chatgpt'),
    (r'\b(?:new\s+tab|\u043d\u044c\u044e\s+\u0442[\u044d\u0435]\u0431|\u043d\u043e\u0432\w*\s+\u0432\u043a\u043b\u0430\u0434\w*)\b', 'newtab'),
    (r'\b(?:plus|\u043f\u043b\u044e\u0441)\b', 'plus'),
    (r'\b(?:minus|\u043c\u0438\u043d\u0443\u0441)\b', 'minus'),
)
FILLER = (
    r'\b(?:please|this|that|here|the|on|a|it|button|tab|link'
    r'|\u043f\u043e\u0436\u0430\u043b\u0443\u0439\u0441\u0442\u0430|\u0434\u0430\u0432\u0430\u0439|\u043d\u0443|\u0432\u043e\u0442|\u044d\u0442\u043e|\u044d\u0442\u043e\u0442|\u044d\u0442\u0443|\u044d\u0442\u0430'
    r'|\u0437\u0434\u0435\u0441\u044c|\u0441\u044e\u0434\u0430|\u0442\u0443\u0442|\u0442\u0430\u043c|\u043d\u0430|\u043f\u043e|\u043c\u043d\u0435'
    r'|\u043a\u043d\u043e\u043f\w*|\u0432\u043a\u043b\u0430\u0434\u043a\w*|\u0441\u0441\u044b\u043b\u043a\w*)\b'
)
VERBS = (
    r'\b(?:click|tap|press|hit|open|select|choose|highlight|switch|double|right|twice|to|go|focus|point'
    r'|\u043d\u0430\u0436\w*|\u0436\u043c\u0438|\u043a\u043b\u0438\u043a\w*|\u0442\u044b\u043a\w*|\u0442\u0430\u043f\w*|\u0449\u0451\u043b\u043a\w*|\u0449\u0435\u043b\u043a\w*'
    r'|\u043e\u0442\u043a\u0440\w*|\u0432\u044b\u0431\w*|\u0432\u044b\u0434\u0435\u043b\w*|\u043f\u043e\u0434\u0441\u0432\u0435\u0442\w*|\u043f\u043e\u043a\u0430\u0436\u0438'
    r'|\u043d\u0430\u0432\u0435\u0434\w*|\u0441\u0444\u043e\u043a\u0443\u0441\w*|\u043f\u0435\u0440\u0435\u0439\w*|\u043f\u0435\u0440\u0435\u043a\u043b\u044e\u0447\w*'
    r'|\u0434\u0432\u0430\u0436\u0434\u044b|\u0434\u0432\u0430|\u0440\u0430\u0437\u0430|\u0434\u0432\u043e\u0439\w*|\u043f\u0440\u0430\u0432\w*|\u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\w*|\u043c\u0435\u043d\u044e)\b'
)

def normalize(text):
    text=text.casefold().replace('\u0451','\u0435')
    for pattern,value in ALIASES:text=re.sub(pattern,value,text)
    return ' '.join(re.findall(r'\w+',text))

def target_query(text):
    return ' '.join(re.sub(FILLER,' ',re.sub(VERBS,' ',normalize(text))).split())

def label_text(text):return ' '.join(re.sub(FILLER,' ',normalize(text)).split())

def similarity(query,label):
    if not query or not label:return 0.
    q=set(query.split());l=set(label.split())
    coverage=len(q&l)/len(q)
    return max(coverage,SequenceMatcher(None,query,label).ratio()*.9)

def ground(query,scene,gaze,now,max_gaze_age=GAZE_TTL):
    """Return (target, ambiguous); a missing match can go to the contextual model."""
    valid=gaze is not None and 0<=now-gaze.captured<max_gaze_age
    ranked=[]
    for e in scene.current_elements(now):
        if not e.enabled:continue
        lexical=similarity(query,label_text(e.label))
        if lexical<.74:continue
        spatial=math.exp(-.5*(math.dist(e.rect.center,(gaze.x,gaze.y))/max(60,gaze.radius))**2) if valid else 0.
        ranked.append((lexical+.32*spatial,e))
    ranked.sort(key=lambda pair:pair[0],reverse=True)
    if not ranked:return None,False
    ambiguous=len(ranked)>1 and ranked[0][0]-ranked[1][0]<.10
    return ranked[0][1],ambiguous
