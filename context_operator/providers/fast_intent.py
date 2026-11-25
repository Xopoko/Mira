"""Small trained CPU classifier + explicit lexical/spatial grounding; no generation."""
from collections import Counter
from pathlib import Path
import json
import re
import time
import numpy as np
from ..core.contracts import Intent, GAZE_TTL, SCENE_TTL
from ..core.grounding import normalize,target_query,label_text,ground

MODEL=Path(__file__).resolve().parents[1]/'assets/intent-model.npz'
VOCAB=MODEL.with_suffix('.json')

# The guard and the target placeholder must match the bilingual training vocabulary.
CONTEXT_ONLY = re.compile(
    r'\b(?:dont|not|never|then|instead|why|said|if'
    r'|\u043d\u0435|\u043d\u0435\u0442|\u043d\u0435\u043b\u044c\u0437\u044f'
    r'|\u0435\u0441\u043b\u0438|\u043f\u043e\u0442\u043e\u043c'
    r'|\u0437\u0430\u0442\u0435\u043c|\u043a\u043e\u0433\u0434\u0430'
    r'|\u0441\u043a\u0430\u0437\u0430\u043b|\u0433\u043e\u0432\u043e\u0440\u0438\u043b'
    r'|\u043f\u043e\u0447\u0435\u043c\u0443|\u0437\u0430\u0447\u0435\u043c'
    r'|\u043a\u0430\u043a|\u0447\u0442\u043e|\u043e\u0442\u043c\u0435\u043d\u0438'
    r')\b'
)
TARGET_PLACEHOLDER = '\u0446\u0435\u043b\u044c'

def features(text):
    counts=Counter()
    for word in text.lower().split():
        word=' '+word+' '
        for n in range(2,6):
            for i in range(max(1,len(word)-n+1)):counts[word[i:i+n]]+=1
            if len(word)<=n:break
    return counts

class FastIntent:
    def __init__(self):
        metadata=json.loads(VOCAB.read_text(encoding='utf-8'))
        self.vocabulary=metadata['vocabulary'];self.classes=metadata['classes']
        with np.load(MODEL,allow_pickle=False) as values:
            self.idf=values['idf'];self.weights=values['weights'];self.bias=values['bias']
    def classify(self,text):
        counts=features(normalize(text));items=[(self.vocabulary[k],v) for k,v in counts.items() if k in self.vocabulary]
        if not items:return 'other',0.,0.
        indices=np.array([i for i,v in items]);values=np.array([v for i,v in items])*self.idf[indices]
        values/=max(np.linalg.norm(values),1e-12)
        logits=self.weights[:,indices]@values+self.bias
        exp=np.exp(logits-logits.max());p=exp/exp.sum();order=np.argsort(p)
        return self.classes[order[-1]],float(p[order[-1]]),float(p[order[-1]]-p[order[-2]])
    def interpret(self,text,scene,focus,gaze,now=None,max_gaze_age=GAZE_TTL,allow_selected=False):
        now=time.monotonic() if now is None else now
        if scene is None or not 0<=now-scene.captured<SCENE_TTL:return None
        t=normalize(text)
        # Multi-step, quoted, negated or narrated language needs whole-sentence interpretation.
        if (len(t.split()) > 12 or CONTEXT_ONLY.search(t)
                or any(c in text for c in '«»"?;')):
            return None
        query=target_query(t)
        target,ambiguous=ground(query,scene,gaze,now,max_gaze_age) if query else (None,False)
        # A bare visible name changes the highlighted target; it never dispatches a click.
        if target and label_text(t)==query:
            action='select';confidence=1.
        else:
            masked=t
            if query and query in masked:masked=masked.replace(query, TARGET_PLACEHOLDER)
            action,confidence,margin=self.classify(masked)
            if action=='other' or confidence<.78 or margin<.45:return None
        if query and target is None:return None
        if ambiguous:return Intent('clarify',reply='Several elements match. Look at the one you want and say “click”.')
        if target is None:
            if not allow_selected and not (gaze and 0<=now-gaze.captured<max_gaze_age):
                return Intent('clarify',reply='Look at the element again or say its name.')
            target=focus.target
            if target is None:return Intent('clarify',reply='Look at an element or say its name.')
        if not any(e.id==target.id for e in scene.elements):return None
        return Intent(action,target_id=target.id)
