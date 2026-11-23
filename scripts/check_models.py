"""Real local inference against synthetic content. No desktop, microphone or input."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw,ImageFont
from context_operator.settings import load,LOCAL
from context_operator.core.contracts import Element,Focus,Rect,Scene
from context_operator.providers.models import Ranker,ContextModel
from context_operator.foundations.perception.runtime import PerceptionClient

def main():
    settings=load()
    if settings['model_mode']!='local':
        print('This check requires the local model mode.',file=sys.stderr)
        return 1
    LOCAL.mkdir(exist_ok=True)
    report={};elements=(Element('plus','Plus','Button',Rect(60,80,210,80)),Element('minus','Minus','Button',Rect(310,80,210,80)),Element('field','Dictation field','Edit',Rect(60,210,460,65)))
    scene=Scene(1,time.monotonic(),Rect(0,0,600,350),elements,title='Model test scene')
    focus=Focus(elements[0],(),False,'Test',1)
    image=Image.new('RGB',(600,350),'#142034');draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',28)
    for el in elements:
        r=el.rect;draw.rounded_rectangle((r.x,r.y,r.x+r.w,r.y+r.h),radius=12,fill='#9fead3');draw.text((r.x+18,r.y+18),el.label,font=font,fill='#082f34')
    image.save(LOCAL/'synthetic-scene.png')
    ranker=Ranker();started=time.monotonic()
    try:
        scores=ranker.rank('Click the plus button',list(elements))
        top=max(scores,key=scores.get,default=None)
        passed=(scores.keys()=={element.id for element in elements}
                and all(scores['plus']>scores[element] for element in ('minus','field')))
        report['ranker']={'scores':scores,'top':top,'passed':passed,'elapsed_ms':round((time.monotonic()-started)*1000)}
    except Exception as exc:report['ranker']={'error':str(exc),'passed':False}
    finally:ranker.close()
    print(json.dumps({'ranker':report['ranker']},ensure_ascii=False),flush=True)
    perception=PerceptionClient();started=time.monotonic()
    try:
        result=perception.observe(image)
        labels={' '.join(item['label'].casefold().split()) for item in result['items'] if item['source']=='ocr'}
        report['perception']={'items':result['items'],'timings':result['timings'],'passed':{'plus','minus'}<=labels,
                              'elapsed_ms':round((time.monotonic()-started)*1000)}
    except Exception as exc:report['perception']={'error':str(exc),'passed':False}
    finally:perception.close()
    print(json.dumps({'perception':report['perception']},ensure_ascii=False),flush=True)
    model=ContextModel(settings);results=[]
    cases=(
        ('Could you click the button labeled Plus?','click','plus',None),
        ('Click Minus instead of Plus','click','minus',None),
        ('In the dictation field, type: Hello, world!','type_text','field','Hello, world!'),
    )
    for text,action,target,dictation in cases:
        started=time.monotonic()
        try:
            decision=model.interpret(text,scene,focus,image)
            passed=(decision.action==action and decision.target_id==target
                    and (dictation is None or decision.text==dictation))
            results.append(dict(request=text,action=decision.action,target=decision.target_id,text=decision.text,
                                passed=passed,elapsed_ms=round((time.monotonic()-started)*1000)))
        except Exception as exc:results.append(dict(request=text,error=str(exc),passed=False))
        print(json.dumps(results[-1],ensure_ascii=False),flush=True)
    report['intent']=results
    report['passed']=report['ranker']['passed'] and report['perception']['passed'] and all(case['passed'] for case in results)
    (LOCAL/'model-check.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
