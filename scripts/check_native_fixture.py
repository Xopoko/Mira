"""Bounded integration test on this process's own disposable Qt controls.

Never activates a window, sends mouse/keyboard events or touches another application.
Exercises real Windows UI Automation Invoke/Value patterns through the product adapter.
"""
from pathlib import Path
import os,sys,time,json,threading
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QPushButton,QLineEdit,QLabel
from PySide6.QtCore import QObject,Signal,QTimer,Qt
from context_operator.providers.windows import ScreenProvider,WindowsOutput
from context_operator.providers.models import ContextModel
from context_operator.settings import load,LOCAL
from context_operator.core.contracts import Intent,Focus

app=QApplication([])
w=QWidget(None,Qt.Tool|Qt.WindowDoesNotAcceptFocus)
w.setAttribute(Qt.WA_ShowWithoutActivating)
w.setWindowTitle('Context Operator · isolated verification')
w.resize(550,270);w.move(100,100)
layout=QVBoxLayout(w);title=QLabel('UI Automation fixture');layout.addWidget(title)
button=QPushButton('Plus');button.setAccessibleName('Plus');layout.addWidget(button)
count=QLabel('0');layout.addWidget(count)
editor=QLineEdit();editor.setAccessibleName('Dictation field');layout.addWidget(editor)
button.clicked.connect(lambda:count.setText(str(int(count.text())+1)))
class Bridge(QObject):done=Signal(object)
bridge=Bridge();exit_code=1

def finish(report):
    global exit_code
    report.update(counter=count.text(),text=editor.text())
    report['passed']=count.text()=='1' and editor.text()=='Context works'
    report['input_scope']='same_process_disposable_fixture_uia_only'
    (LOCAL/'native-fixture-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    w.grab().save(str(LOCAL/'native-fixture-render.png'))
    print(json.dumps(report,ensure_ascii=True),flush=True)
    exit_code=0 if report['passed'] else 1
    w.close();app.quit()

bridge.done.connect(finish)
def work():
    import uiautomation as auto
    report={'stages':[]}
    try:
        with auto.UIAutomationInitializerInThread():
            screen=ScreenProvider();screen.practice_window=int(w.winId());output=WindowsOutput()
            # Make an accidental native-coordinate fallback impossible in this test.
            output.win.activate=lambda _:(_ for _ in ()).throw(RuntimeError('Native activation forbidden in fixture'))
            output.win.send=lambda _:(_ for _ in ()).throw(RuntimeError('Native input forbidden in fixture'))
            for query in ('Click the plus button','In the dictation field, type: Context works'):
                scene=screen.observe()
                if not scene.window:raise RuntimeError('Fixture observation unavailable')
                if any(e.pid!=os.getpid() for e in scene.elements):raise RuntimeError('Unexpected process in isolated fixture')
                started=time.monotonic()
                decision=ContextModel(load()).interpret(query,scene,Focus())
                # Refresh after model latency and bind the same observed element.
                fresh=screen.observe();target=next((e for e in fresh.elements if e.id==decision.target_id),None)
                if target is None:raise RuntimeError('Fixture target missing')
                receipt=output.execute(decision,target,fresh)
                report['stages'].append({'action':decision.action,'target':target.label,'route':receipt.get('route'),'ms':round((time.monotonic()-started)*1000)})
                time.sleep(.15)
    except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)
    bridge.done.emit(report)
w.show()
QTimer.singleShot(300,lambda:threading.Thread(target=work,daemon=True).start())
QTimer.singleShot(90000,lambda:finish({'error':'fixture_timeout'}))
app.exec()
sys.exit(exit_code)
