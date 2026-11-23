"""Own nonactivating Qt fixture + synthetic WAV. No microphone, camera, native input or LLM."""
from pathlib import Path
import os,sys,time,json,threading,subprocess,statistics
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QPushButton,QLabel,QTabWidget
from PySide6.QtCore import QObject,Signal,QTimer,Qt
from context_operator.providers.windows import ScreenProvider,WindowsOutput,Windows
from context_operator.providers.fast_intent import FastIntent
from context_operator.core.contracts import *
from context_operator.settings import LOCAL,ROOT

app=QApplication([]);w=QWidget(None,Qt.Tool|Qt.WindowDoesNotAcceptFocus|Qt.WindowStaysOnTopHint)
w.setAttribute(Qt.WA_ShowWithoutActivating);w.setWindowTitle('Context Operator · reactive fixture');w.resize(520,270);w.move(50,50)
layout=QVBoxLayout(w);tabs=QTabWidget();tabs.addTab(QWidget(),'ChatGPT');tabs.addTab(QWidget(),'New Tab');layout.addWidget(tabs)
button=QPushButton('Plus');button.setAccessibleName('Plus');layout.addWidget(button)
count=QLabel('0');layout.addWidget(count);button.clicked.connect(lambda:count.setText(str(int(count.text())+1)))
class Bridge(QObject):done=Signal(object)
bridge=Bridge();exit_code=1
def finish(report):
    global exit_code
    report.update(counter=count.text(),tab=tabs.currentIndex())
    report['passed']=count.text()=='1' and tabs.currentIndex()==1 and 'error' not in report
    report['scope']='own nonactivating fixture; synthetic WAV; no native mouse/keyboard or Qwen'
    (LOCAL/'reactive-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True),flush=True)
    exit_code=0 if report['passed'] else 1;w.close();app.quit()
bridge.done.connect(finish)
def work():
    import uiautomation as auto
    report={'stages':[]}
    try:
        args=[sys.executable,'-u',str(ROOT/'context_operator/foundations/speech/runtime.py'),'--model-dir',str(LOCAL/'models/parakeet'),'--vad-model',str(LOCAL/'models/silero_vad.onnx'),'--wav',str(LOCAL/'synthetic-command.wav'),'--silence-ms','280','--threads','4']
        p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW)
        events=[json.loads(line) for line in p.stdout];p.wait(timeout=5);p.stdin.close()
        final=next((e for e in events if e.get('type')=='final'),None)
        if final is None:raise RuntimeError('Synthetic speech failed: '+str([e.get('type') for e in events]))
        report['speech']={k:final[k] for k in ('text','asrMs','afterSpeechMs','audioMs')}
        with auto.UIAutomationInitializerInThread():
            screen=ScreenProvider();screen.practice_window=int(w.winId());output=WindowsOutput();router=FastIntent()
            output.win.activate=lambda _:(_ for _ in ()).throw(RuntimeError('Native activation forbidden'))
            output.win.send=lambda _:(_ for _ in ()).throw(RuntimeError('Native input forbidden'))
            for text in ('open New Tab',final['text']):
                scene=screen.observe()
                if any(e.pid!=os.getpid() for e in scene.elements):raise RuntimeError('Foreign fixture target')
                if not report['stages']:
                    report['inactive_window']=screen.win.foreground()!=scene.window
                    report['inactive_capture']=screen.capture(scene) is not None
                    if not report['inactive_window'] or not report['inactive_capture']:raise RuntimeError('Inactive fixture capture was not established')
                started=time.perf_counter();intent=router.interpret(text,scene,Focus(),None)
                classified=time.perf_counter()
                if intent is None:raise RuntimeError('Fast route declined fixture request')
                target=next(e for e in scene.elements if e.id==intent.target_id)
                receipt=output.execute(intent,target,scene)
                report['stages'].append({'request':text,'action':intent.action,'target':target.label,'route':receipt.get('route'),'interpret_ms':round((classified-started)*1000,3),'interpret_and_dispatch_ms':round((time.perf_counter()-started)*1000,3)})
                time.sleep(.12)
            # Repeated CPU timing on a bounded synthetic 200-element map, no output.
            elements=tuple(Element(str(i),'Synthetic element '+str(i),'Button',Rect(i%20*45,i//20*35,40,30),1) for i in range(199))+(Element('chat','ChatGPT','TabItem',Rect(0,0,100,40),1),)
            now=time.monotonic();scene=Scene(1,now,Rect(0,0,1000,700),elements,1)
            timings=[]
            for _ in range(120):
                started=time.perf_counter();result=router.interpret('open ChatGPT',scene,Focus(),Gaze(40,20,now),now)
                if result is None or result.target_id!='chat':raise RuntimeError('Benchmark target mismatch')
                timings.append((time.perf_counter()-started)*1000)
            report['cpu_200_elements']={'samples':len(timings),'median_ms':round(statistics.median(timings),3),'p95_ms':round(sorted(timings)[int(len(timings)*.95)],3)}
    except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)
    bridge.done.emit(report)
w.show();QTimer.singleShot(300,lambda:threading.Thread(target=work,daemon=True).start())
QTimer.singleShot(60000,lambda:finish({'error':'fixture_timeout'}))
app.exec();sys.exit(exit_code)
