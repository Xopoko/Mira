"""Reproducible small command classifier; authored examples, no user recordings.

Training uses sklearn. Runtime loads only numeric NPZ + vocabulary JSON, no pickle.
General language and uncertain predictions stay with the context model.
"""
from pathlib import Path
import sys,json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from context_operator.core.grounding import normalize

# The shipped model uses both English and Russian synthetic phrases.
PHRASES = {
    'click': [
        '\u043d\u0430\u0436\u043c\u0438',
        '\u043d\u0430\u0436\u0430\u0442\u044c',
        '\u0436\u043c\u0438',
        '\u043a\u043b\u0438\u043a\u043d\u0438',
        '\u043a\u043b\u0438\u043a',
        '\u0442\u044b\u043a\u043d\u0438',
        '\u0442\u0430\u043f\u043d\u0438',
        '\u0449\u0435\u043b\u043a\u043d\u0438',
        '\u043e\u0442\u043a\u0440\u043e\u0439',
        '\u043e\u0442\u043a\u0440\u044b\u0442\u044c',
        '\u043f\u0435\u0440\u0435\u0439\u0434\u0438',
        '\u043f\u0435\u0440\u0435\u043a\u043b\u044e\u0447\u0438\u0441\u044c',
        'click',
        'tap',
        'press',
        'open',
        'hit',
        'switch to',
        'go to',
    ],
    'select': [
        '\u0432\u044b\u0434\u0435\u043b\u0438',
        '\u0432\u044b\u0431\u0435\u0440\u0438',
        '\u043f\u043e\u043a\u0430\u0436\u0438',
        '\u043f\u043e\u0434\u0441\u0432\u0435\u0442\u0438',
        '\u0432\u044b\u0434\u0435\u043b\u0438\u0442\u044c',
        '\u0432\u044b\u0431\u0440\u0430\u0442\u044c',
        '\u043d\u0430\u0432\u0435\u0434\u0438\u0441\u044c',
        '\u0441\u0444\u043e\u043a\u0443\u0441\u0438\u0440\u0443\u0439\u0441\u044f',
        'select',
        'choose',
        'highlight',
        'focus on',
        'point to',
    ],
    'double_click': [
        '\u0434\u0432\u043e\u0439\u043d\u043e\u0439 \u043a\u043b\u0438\u043a',
        '\u043d\u0430\u0436\u043c\u0438 \u0434\u0432\u0430 \u0440\u0430\u0437\u0430',
        '\u043d\u0430\u0436\u043c\u0438 \u0434\u0432\u0430\u0436\u0434\u044b',
        '\u043a\u043b\u0438\u043a\u043d\u0438 \u0434\u0432\u0430\u0436\u0434\u044b',
        '\u0434\u0432\u043e\u0439\u043d\u043e\u0435 '
        '\u043d\u0430\u0436\u0430\u0442\u0438\u0435',
        '\u0449\u0435\u043b\u043a\u043d\u0438 \u0434\u0432\u0430 \u0440\u0430\u0437\u0430',
        'double click',
        'click twice',
        'tap twice',
    ],
    'right_click': [
        '\u043f\u0440\u0430\u0432\u044b\u0439 \u043a\u043b\u0438\u043a',
        '\u043d\u0430\u0436\u043c\u0438 \u043f\u0440\u0430\u0432\u043e\u0439 '
        '\u043a\u043d\u043e\u043f\u043a\u043e\u0439',
        '\u043a\u043b\u0438\u043a\u043d\u0438 \u043f\u0440\u0430\u0432\u043e\u0439',
        '\u043e\u0442\u043a\u0440\u043e\u0439 '
        '\u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u043d\u043e\u0435 '
        '\u043c\u0435\u043d\u044e',
        '\u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u043d\u043e\u0435 '
        '\u043c\u0435\u043d\u044e',
        '\u043f\u043e\u043a\u0430\u0436\u0438 \u043c\u0435\u043d\u044e',
        'right click',
        'open context menu',
        'context menu',
    ],
    'other': [
        '\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438 \u043e\u0431 '
        '\u044d\u0442\u043e\u043c',
        '\u0447\u0442\u043e \u044d\u0442\u043e \u0442\u0430\u043a\u043e\u0435',
        '\u043a\u0430\u043a \u043d\u0430\u0436\u0430\u0442\u044c '
        '\u043a\u043d\u043e\u043f\u043a\u0443',
        '\u043f\u043e\u0447\u0435\u043c\u0443 \u043d\u0435 '
        '\u0440\u0430\u0431\u043e\u0442\u0430\u0435\u0442',
        '\u043d\u0435 \u043d\u0430\u0436\u0438\u043c\u0430\u0439',
        '\u043d\u0435 \u043e\u0442\u043a\u0440\u044b\u0432\u0430\u0439',
        '\u0437\u0430\u043a\u0440\u043e\u0439',
        '\u0437\u0430\u043a\u0440\u043e\u0439 \u043e\u043a\u043d\u043e',
        '\u0443\u0434\u0430\u043b\u0438',
        '\u043d\u0430\u043f\u0438\u0448\u0438 \u0442\u0435\u043a\u0441\u0442',
        '\u043f\u0440\u043e\u043a\u0440\u0443\u0442\u0438 \u0432\u043d\u0438\u0437',
        '\u0441\u0434\u0435\u043b\u0430\u0439 \u0448\u0440\u0438\u0444\u0442 '
        '\u043a\u0440\u0443\u043f\u043d\u0435\u0435',
        '\u043f\u043e\u0438\u0449\u0438 '
        '\u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u044e',
        '\u0441\u043d\u0430\u0447\u0430\u043b\u0430 \u043e\u0442\u043a\u0440\u043e\u0439 '
        '\u043f\u043e\u0442\u043e\u043c \u0437\u0430\u043a\u0440\u043e\u0439',
        '\u0435\u0441\u043b\u0438 \u0433\u043e\u0442\u043e\u0432\u043e '
        '\u043d\u0430\u0436\u043c\u0438',
        '\u043e\u043d \u0441\u043a\u0430\u0437\u0430\u043b \u043d\u0430\u0436\u043c\u0438',
        '\u044f \u0441\u043c\u043e\u0442\u0440\u044e \u043d\u0430 '
        '\u044d\u043a\u0440\u0430\u043d',
        'what is this',
        'do not click',
        'close window',
        'delete this',
        'type hello',
        'scroll down',
        'explain this',
        'he said click',
        'click then type',
    ],
}

PREFIXES = (
    '',
    '\u043f\u043e\u0436\u0430\u043b\u0443\u0439\u0441\u0442\u0430 ',
    '\u043d\u0443 ',
    '\u0434\u0430\u0432\u0430\u0439 ',
    'please ',
)
SUFFIXES = (
    '',
    ' \u0441\u044e\u0434\u0430',
    ' \u044d\u0442\u043e',
    ' \u0446\u0435\u043b\u044c',
    ' \u043a\u043d\u043e\u043f\u043a\u0443 \u0446\u0435\u043b\u044c',
    ' \u0432\u043a\u043b\u0430\u0434\u043a\u0443 \u0446\u0435\u043b\u044c',
    ' this',
    ' target',
    ' \u043f\u043e\u0436\u0430\u043b\u0443\u0439\u0441\u0442\u0430',
)


def train():
    texts=[];labels=[]
    for label,phrases in PHRASES.items():
        for phrase in phrases:
            for prefix in PREFIXES:
                for suffix in SUFFIXES:
                    texts.append(normalize(prefix+phrase+suffix));labels.append(label)
    vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),dtype=np.float64)
    x=vectorizer.fit_transform(texts)
    model=LogisticRegression(C=18,max_iter=600,random_state=42).fit(x,labels)
    folder=Path(__file__).resolve().parents[1]/'context_operator/assets';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'intent-model.npz',idf=vectorizer.idf_,weights=model.coef_,bias=model.intercept_)
    (folder/'intent-model.json').write_text(json.dumps({'version':1,'training_examples':len(texts),'classes':model.classes_.tolist(),'vocabulary':vectorizer.vocabulary_}),encoding='utf-8')
    # Independent implementation parity, not a quality evaluation.
    from context_operator.providers.fast_intent import FastIntent
    fast=FastIntent()
    probes = (
        '\u043d\u0443 \u043f\u043e\u0436\u0430\u043b\u0443\u0439\u0441\u0442\u0430 \u043a\u043b\u0438\u043a\u043d\u0438 \u0432\u043e\u0442 \u0441\u044e\u0434\u0430',
        '\u0432\u044b\u0434\u0435\u043b\u0438 \u0446\u0435\u043b\u044c',
        'right click this',
        '\u043d\u0435 \u043e\u0442\u043a\u0440\u044b\u0432\u0430\u0439 \u0446\u0435\u043b\u044c',
    )
    for t in probes:
        expected=model.predict_proba(vectorizer.transform([normalize(t)]))[0]
        action,score,_=fast.classify(t)
        assert action==model.classes_[expected.argmax()] and abs(score-expected.max())<1e-9,(t,action,score,expected)
    print(json.dumps({'training_examples':len(texts),'features':len(vectorizer.vocabulary_),'bytes':(folder/'intent-model.npz').stat().st_size,'parity':'passed'}))
if __name__=='__main__':train()
