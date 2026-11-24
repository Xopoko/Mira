"""Offscreen real Qt screens and interactions, with no hardware or desktop input."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QPushButton
from PySide6.QtGui import QFontDatabase
from context_operator.controller import Controller
from context_operator.core.contracts import Gaze, Intent
from context_operator.settings import DEFAULTS, LOCAL, load
from context_operator.ui.window import Window
from context_operator.ui.setup import SetupWindow
from context_operator.personalization import ProfileStore
from context_operator.foundations.gaze.arkit import Sample
import numpy as np

app=QApplication.instance() or QApplication([])
for font in ('segoeui.ttf','seguisb.ttf','segoeuib.ttf'):
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+font)
c=Controller(dict(DEFAULTS,vision=False,ranker=False,anticipation=False,click_learning=False))
c.timer.stop()
window=Window(c)
window.refresh_timer.stop()
out=LOCAL/'evidence/desktop-ui'
out.mkdir(parents=True,exist_ok=True)
window.show()
app.processEvents()
results={}
for i,name in enumerate(('overview','profile','practice','settings')):
    window.navigate(i)
    app.processEvents()
    window.grab().save(str(out/f'{name}.png'))
    area=window.stack.widget(i)
    results[name]={'viewport':[window.width(),window.height()],
                   'horizontal_overflow':area.horizontalScrollBar().maximum(),
                   'vertical_scroll':area.verticalScrollBar().maximum()}

window.navigate(2)
window.resize(window.minimumSize())
app.processEvents()
practice = window.stack.currentWidget()
assert practice.horizontalScrollBar().maximum() == 0
assert practice.verticalScrollBar().maximum() == 0
window.practice_buttons['Top left'].setFocus()
QTest.keyClick(window.practice_buttons['Top left'], Qt.Key_Tab)
assert window.practice_buttons['Plus'].hasFocus()
window.resize(1220, 900)
app.processEvents()
assert not window.editor.isVisible() and not window.log.isVisible()
window.practice_buttons['Clock'].setFocus()
QTest.keyClick(window.practice_buttons['Clock'],Qt.Key_Tab)
assert window.practice_command.hasFocus()
window.practice_command.setText('click')
QTest.keyClick(window.practice_command,Qt.Key_A,Qt.ControlModifier)
assert window.practice_command.selectedText()=='click'
field=window.practice_command.grab().toImage()
assert field.pixelColor(field.width()//2,0).name()=='#3d6bff'
assert any(field.pixelColor(x,field.height()//2).name()=='#2c4fd1' for x in range(10,45))
window.practice_command.clear()
window.practice_tools_button.click()
assert window.editor.isVisible() and window.log.isVisible()
window.practice_tools_button.click()
assert not window.editor.isVisible() and not window.log.isVisible()

with patch('context_operator.ui.window.time.monotonic',return_value=100.):
    window.last_paint=0;window.refresh()
    assert list(window.rail_sources)==['iPhone','Microphone']
    assert window.focus_panel.maximumHeight()<180
    c.running=True;c.microphone=True;c.voice_state='standby'
    c.voice_status='Loading speech recognition'
    window.last_paint=0;window.refresh()
    assert window.rail_sources['Microphone'][0].state=='standby'
    assert window.focus_panel.maximumHeight()>180
    c._speech({'type':'ready'},c.voice_generation)
    window.last_paint=0;window.refresh()
    assert window.rail_sources['Microphone'][0].state=='go'
    c._speech({'type':'ready','device':{'fallbackFrom':'USB microphone'}},c.voice_generation)
    c._speech({'type':'speech_start','utteranceId':1},c.voice_generation)
    window.last_paint=0;window.refresh()
    fallback='Using the default microphone because USB microphone isn’t connected.'
    assert fallback in window.voice_value.text() and fallback in window.microphone_note.text()
    c._speech({'type':'error','message':'Recognition failed'},c.voice_generation)
    c._speech({'type':'stopped'},c.voice_generation)
    window.last_paint=0;window.refresh()
    assert window.rail_sources['Microphone'][0].state=='hold'
    assert window.voice_value.text()=='Error · Recognition failed'
    window.last_paint=0
    c._output({'target':'Plus'},Intent('click'),'',c.generation,99.9)
    assert window.target.text()=='Click sent to Windows'
with patch('context_operator.ui.window.time.monotonic',return_value=103.):
    window.last_paint=0;window.refresh()
    assert window.target.text()=='Mira is running'
    window.last_paint=0
    c._output(None,Intent('click'),'Test dispatch error',c.generation,102.9)
    assert window.target.text()=='Mira is running'
c.pause()
window.last_paint=0;window.refresh()
results['keyboard_focus_selection_and_source_feedback']='passed'

window.navigate(2)
assert window.practice_button.text() == 'Connect iPhone'
window.practice_button.click()
assert window.stack.currentIndex() == 0 and not c.running
window.navigate(2)
window.practice_buttons['Plus'].click()
assert window.count.text()=='1'
window.practice_buttons['Reset'].click()
assert window.count.text()=='0'

def start_preview():
    assert c.practice_window == int(window.winId())
    c.running = True

with patch.object(c, 'start', side_effect=start_preview) as start, patch.object(c, 'command') as command:
    stale_scene = object()
    c.scene = stale_scene
    previous_epoch = c.action_epoch
    window.open_mouse_preview()
    start.assert_called_once_with()
    assert window.practice_active and window.pointer.isChecked() and c.pointer_mode
    c._scene(stale_scene, c.generation, previous_epoch)
    assert c.scene is None

    window.practice_command.setText('click Plus')
    window.run_practice_command.click()
    command.assert_called_once_with('click Plus')
    assert window.practice_command.text() == 'click Plus'
    c.scene = object()
    window.practice_command.setText('click Minus')
    QTest.keyClick(window.practice_command,Qt.Key_Return)
    command.assert_called_with('click Minus')
    assert not window.practice_command.text()

    window.navigate(0)
    assert not window.practice_active and not c.running and not c.pointer_mode and c.practice_window == 0

    c.running = True
    c.phone = phone = SimpleNamespace(worker=SimpleNamespace(frames=1, is_alive=lambda: True))
    window.last_paint = 0
    window.refresh()
    assert window.practice_button.text() == 'Try eye tracking'
    window.navigate(2)
    window.practice_button.click()
    assert window.practice_active and not c.pointer_mode
    window.open_mouse_preview()
    window.open_gaze_practice()
    assert c.phone is phone and window.practice_active and not window.pointer.isChecked() and not c.pointer_mode
    c.phone = None
    window.navigate(0)
    assert c.running and c.practice_window == 0
    c.pause()
window.navigate(3)
assert window.advanced.accessibleDescription()=='Collapsed'
devices=[{'name':'USB microphone','hostApi':'Windows WASAPI','index':4},
         {'name':'USB microphone','hostApi':'MME','index':1},
         {'name':'Headset','hostApi':'Windows WASAPI','index':5}]
with patch('context_operator.ui.window.Speech.available_microphones',return_value=devices):
    window.advanced.click()
    for _ in range(30):
        if not window._microphone_listing:break
        QTest.qWait(10)
    assert not window._microphone_listing
picker=window.microphone_picker
assert picker.itemText(0)=='System default' and picker.currentData()==''
assert picker.itemText(1)=='USB microphone · Windows WASAPI'
assert picker.itemText(2)=='USB microphone · MME'
with patch('context_operator.ui.window.save') as save:
    picker.setCurrentIndex(2)
    window.save_connections()
    assert save.call_args.args[0]['microphone']==devices[1]
    picker.setCurrentIndex(0)
    window.save_connections()
    assert save.call_args.args[0]['microphone']==''
assert window.advanced_panel.isVisible()
assert window.advanced.accessibleDescription()=='Expanded'
assert all(combo.accessibleName() for combo in window.findChildren(QComboBox))
app.processEvents()
window.grab().save(str(out/'settings-expanded.png'))
for i in range(4):
    window.resize(960,680);window.navigate(i);app.processEvents()
    area=window.stack.widget(i)
    assert area.horizontalScrollBar().maximum()==0,(i,area.horizontalScrollBar().maximum())
    window.grab().save(str(out/f'compact-{i}.png'))
texts=[b.text() for b in window.findChildren(QPushButton)]
assert not any(word in text for text in texts for word in ('MobileGaze','UniGaze','hybrid','experiment'))
assert [b.text() for b in window.nav]==['Overview','Profile','Practice','Settings']
results['interactions']='passed'
with patch('context_operator.personalization.ProfileStore.status',return_value={}),patch(
        'context_operator.personalization.history.latest_result',return_value=None):
    window._profile_stamp=None
    window.refresh_profile()
    assert window.no_setup_results.text()=='No setup results yet' and not window.no_setup_results.isHidden()
    assert window.validation_panel.isHidden()
# A failed retry must not replace the accepted profile's displayed metrics.
accepted={'quality':'starter','usable':True,'passed':False,'median_px':276.,
          'p90_px':422.,'jitter_px':56.,'coverage':.9667}
with patch('context_operator.personalization.ProfileStore.status',return_value={
        'available':True,'report':accepted}),patch(
        'context_operator.personalization.history.latest_result',return_value={
            'median_px':999.,'p90_px':999.,'jitter_px':999.,'coverage':.2,
            'usable':False,'problems':['The second attempt failed.']}):
    window._profile_stamp=None
    window.refresh_profile()
    assert window.validation_verdict.text()=='Accuracy: Good'
    assert window.metrics['median_px'].text()=='276 px'
    assert 'Saved setup' in window.validation_hint.text()
    window.navigate(1);app.processEvents()
    assert window.validation_details.isHidden()
    window.validation_toggle.click()
    assert window.validation_details.isVisible()
    window.grab().save(str(out/'profile-accepted-after-failed-retry.png'))
results['accepted_profile_after_failed_retry']='passed'
with patch.dict(c.settings,click_learning=True,anticipation=True), \
     patch.object(c.learning,'click_accepted',0), \
     patch.object(c.anticipation.model,'learned',0), patch.object(c.anticipation.gaze_model,'learned',0), \
     patch.object(c.anticipation.model,'evaluated',0), patch.object(c.anticipation.gaze_model,'evaluated',0), \
     patch.object(c.anticipation,'error',''):
    window.navigate(1);app.processEvents()
    window.refresh_learning('off')
    assert window.click_plain_status.text()=='Mira learns from your clicks once iPhone is connected.'
    assert window.prediction_plain_status.text().endswith('Not enough clicks yet.')
    assert window.click_details.isHidden() and window.prediction_status.isHidden()
    window.click_details_toggle.click();window.prediction_details_toggle.click()
    assert window.click_summary.isVisible() and window.prediction_status.isVisible()
    assert window.click_summary.text().startswith('This session')
    assert 'training clicks' in window.prediction_status.text()
    c.learning.click_accepted=3;c.anticipation.gaze_model.learned=1
    window.refresh_learning('go')
    assert window.click_plain_status.text()=='Mira has learned from 3 of your clicks.'
    assert window.prediction_plain_status.text().endswith('Ready.')
    c.settings['click_learning']=False;c.settings['anticipation']=False
    window.refresh_learning('go')
    assert window.click_plain_status.text()=='Click learning is off.'
    assert window.prediction_plain_status.text()=='Prediction is off.'
    window.click_details_toggle.click();window.prediction_details_toggle.click()
results['profile_plain_status_and_microphone_picker']='passed'
with tempfile.TemporaryDirectory() as folder, \
     patch('context_operator.settings.LOCAL',Path(folder)), \
     patch('context_operator.personalization.ProfileStore.status',return_value={}) as profile, \
     patch('context_operator.personalization.history.latest_result',return_value={'usable':False}):
    c.history.clear()
    window.navigate(0);window._profile_stamp=None
    window.last_paint=0;window.refresh()
    assert window.gaze_btn.isVisible() and window.gaze_btn.text()=='Set up iPhone'
    assert window.mouse_preview.isVisible() and window.mouse_preview.text()=='Try with mouse'
    assert window.phone_settings_button.isVisible()
    assert not window.voice_btn.isVisible() and not window.run.isVisible()
    with patch.object(window,'open_setup') as open_setup:
        window.gaze_btn.click()
        open_setup.assert_called_once_with()
    assert window.validation_verdict.text()=='Accuracy: Fair · try setup again'
    assert not window.scene_map_button.isVisible()
    window.more_controls.click()
    assert window.run.isVisible() and window.voice_btn.isVisible()
    assert window.scene_map_button.isVisible() and window.scene_map_button.text()=='Show screen map'
    window.scene_map_button.click()
    assert load()['show_scene_map'] and window.focus_panel.isVisible()
    window.scene_map_button.click()
    assert not load()['show_scene_map'] and not window.focus_panel.isVisible()
    window.more_controls.click()

    profile.return_value={'available':True,'report':accepted}
    window.navigate(1)
    window.setup_adopted()
    assert window.stack.currentIndex()==0 and window.gaze_btn.text()=='Connect iPhone'
    assert window.gaze_btn.hasFocus() and not window.run.isVisible()
    with patch.object(c,'toggle_phone') as connect_phone:
        window.gaze_btn.click()
        connect_phone.assert_called_once_with()

    c.running=True;c.phone=SimpleNamespace(worker=SimpleNamespace(is_alive=lambda:True),stop=Mock())
    window._profile_stamp=None;window.last_paint=0;window.refresh()
    assert not window.gaze_btn.isVisible() and window.disconnect_phone.text()=='Cancel connection'
    c.gaze_source_mode='iphone';c.gaze=Gaze(0,0,time.monotonic())
    window.last_paint=0;window.refresh()
    assert window.gaze_btn.isVisible() and window.gaze_btn.text()=='Open Practice'
    assert window.run.isVisible() and window.run.text()=='Pause'
    assert window.disconnect_phone.isVisible() and not window.voice_btn.isVisible()
    app.processEvents()
    window.grab().save(str(out/'overview-connected.png'))
    window.gaze_btn.click()
    assert window.practice_active and not c.pointer_mode and not c.microphone and not c.history
    window.navigate(0)
    assert c.running and c.phone is not None and not window.practice_active
    c.pause()
results['first_run_verdict_and_saved_map']='passed'
window.close()
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder)
    worker=Mock()
    worker.is_alive.return_value=True
    controller=SimpleNamespace(win=c.win,settings=dict(DEFAULTS))
    setup=SetupWindow(controller,archive=root/'session.json',store=ProfileStore(root),
                      worker_factory=lambda *a,**k:worker)
    setup.timer.stop();setup.resize(1220,900);setup.show();app.processEvents()
    setup.grab().save(str(out/'setup-welcome.png'))
    worker.latest.return_value=Sample(error='Could not find iPhone')
    setup.worker=worker
    setup.refresh()
    assert 'Connection settings' in setup.status.text() and not setup.begin_button.isEnabled()
    now=time.monotonic()
    def frame(seq,stamp):
        return Sample(seq=seq,captured=stamp,features=np.zeros(22),
                      eye_origins=np.zeros((2,3)))
    worker.latest.return_value=frame(2,now)
    setup.worker=worker;setup.running=True
    setup.session.frames=[frame(1,now-.4),frame(2,now)]
    setup.refresh();app.processEvents()
    setup.grab().save(str(out/'setup-recording.png'))
    assert setup.begin_button.isHidden()
    setup.running=False
    setup.refresh();app.processEvents()
    setup.grab().save(str(out/'setup-paused.png'))
    setup.session.phase='result'
    setup.candidate=Mock()
    setup.session.candidate_key='a'*64
    setup.session.report={'passed':False,'usable':True,'quality':'starter','median_px':62.,'p90_px':131.,'bias_px':[4.,-9.],
                          'jitter_px':17.,'coverage':.98}
    setup.refresh();app.processEvents()
    setup.grab().save(str(out/'setup-result-example.png'))
    assert not setup.apply_button.isHidden()
    setup.session.report.update({'usable':False,
        'problems':['High error on new points.']})
    setup.refresh();app.processEvents()
    setup.grab().save(str(out/'setup-failed-example.png'))
    assert not setup.retry_button.isHidden()
    assert setup.apply_button.isHidden()
    worker.is_alive.return_value=False
    setup.close()
print(json.dumps(results,ensure_ascii=False))
