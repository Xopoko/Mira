"""Exercise quiet mode in real offscreen Qt; no phone, desktop capture or input."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication,QCheckBox,QSlider
from PySide6.QtGui import QImage,QFontDatabase
from PySide6.QtTest import QTest
from context_operator.settings import DEFAULTS,LOCAL,load
from context_operator.controller import Controller
from context_operator.ui.window import Window
from context_operator.core.contracts import Element,Rect,Scene,Gaze,Focus


def render(widget):
    image=QImage(widget.size(),QImage.Format_ARGB32_Premultiplied);image.fill(0)
    widget.render(image)
    return image


def ink(image):
    return sum(bytes(image.constBits())[3::4])


app=QApplication.instance() or QApplication([])
for font in ('segoeui.ttf','seguisb.ttf','segoeuib.ttf'):
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+font)
out=LOCAL/'evidence/quiet-mode';out.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory() as folder, \
     patch('context_operator.settings.LOCAL',Path(folder)), \
     patch('context_operator.controller.LOCAL',Path(folder)), \
     patch.object(Window,'refresh_profile'), \
     patch('context_operator.ui.window.time.monotonic',return_value=10.1):
    c=Controller(dict(DEFAULTS,vision=False,ranker=False,click_learning=True,anticipation=True,
                      game_gaze_enabled=True,dwell_enabled=True))
    c.timer.stop();window=Window(c);window.refresh_timer.stop();window.show()
    bounds=Rect(0,0,800,600)
    win=Mock();win.rect.return_value=bounds;win.u.GetWindowLongW.return_value=0
    screen=app.primaryScreen()
    with patch.object(c,'win',win):window.add_overlay(screen)
    overlay=window.overlays[screen];overlay.timer.stop();overlay.resize(800,600)
    target=Element('target','Sample element','Button',Rect(150,150,120,60),1,pid=2,captured=10.)
    forecast=Element('next','Next element','Button',Rect(380,150,120,60),1,pid=2,captured=10.)
    scene=Scene(1,10.,bounds,(target,forecast),1)
    gaze=Gaze(210,180,10.,230.)
    focus=Focus(target=target)
    prediction=Mock(target=forecast,strength=1.)
    prediction.is_fresh.return_value=True
    c.running=True;c.scene=scene;c.gaze=c.attention=gaze;c.focus=focus;c.prediction=prediction
    c.pause=Mock(wraps=c.pause)
    original={key:c.settings[key] for key in ('click_learning','anticipation','game_gaze_enabled','show_gaze')}
    window.navigate(3)
    with patch('context_operator.ui.window.Speech.available_microphones',return_value=[]):
        window.advanced.click()
        for _ in range(20):
            if not window._microphone_listing:break
            QTest.qWait(10)
        assert not window._microphone_listing
    assert window.quiet_mode_check.isVisible()
    try:
        for mode in ('radial','magnetic'):
            window.update_setting('highlight_mode',mode)
            c.focus=focus
            epoch=c.action_epoch
            window.last_paint=0;window.refresh();overlay.animate();app.processEvents()
            assert ink(render(overlay))>0,mode
            window.navigate(3);app.processEvents()
            window.quiet_mode_check.click();app.processEvents();overlay.animate()
            assert load()['quiet_mode'] is True
            assert c.running and c.focus is focus and c.action_epoch==epoch
            assert {key:c.settings[key] for key in original}==original
            assert c.settings['highlight_mode']==mode
            assert not window.highlight_mode.isEnabled() and not window.latency.text()
            assert all(check.isChecked() for check in window.setting_checks['quiet_mode'])
            window.navigate(2);app.processEvents()
            assert window.isVisible() and window.practice_button.isVisible()
            assert overlay.ink.isEmpty() and ink(render(overlay))==0,mode
            assert overlay.center is None and overlay.prediction is None and overlay.focus_target is None
            with patch('context_operator.ui.window.paint_scene_light') as radial, \
                 patch('context_operator.ui.window.paint_magnetic_focus') as magnetic:
                render(window.canvas)
                radial.assert_not_called();magnetic.assert_not_called()
            if mode=='radial':
                window.navigate(0);app.processEvents()
                window.grab().save(str(out/'quiet-overview.png'))
                window.navigate(3);app.processEvents()
                window.grab().save(str(out/'quiet-settings.png'))
                window.navigate(1);app.processEvents()
                assert window.click_learning.isChecked() and window.click_learning.isEnabled()
            window.navigate(3);app.processEvents()
            with patch('context_operator.ui.window.save',side_effect=OSError('test disk failure')):
                window.quiet_mode_check.click();app.processEvents()
            assert c.settings['quiet_mode'] and window.quiet_mode_check.isChecked()
            assert load()['quiet_mode'] is True and ink(render(overlay))==0
            window.quiet_mode_check.click();app.processEvents();overlay.animate()
            assert load()['quiet_mode'] is False
            assert not any(check.isChecked() for check in window.setting_checks['quiet_mode']) and ink(render(overlay))>0
        window.resize(window.minimumSize());window.navigate(3);app.processEvents()
        dwell=next(check for check in window.findChildren(QCheckBox) if check.text()=='Click by holding your gaze')
        slider=next(item for item in window.findChildren(QSlider) if item.accessibleName()=='Hold time')
        with patch('context_operator.ui.window.save',side_effect=OSError('test disk failure')) as save:
            dwell.click()
            assert dwell.isChecked() and c.settings['dwell_enabled']
            slider.setSliderDown(True);slider.setSliderPosition(900)
            save.assert_called_once()
            slider.setSliderDown(False)
            assert save.call_count==2 and slider.value()==c.settings['dwell_ms']==1200
        status=window.content_status
        rail=window.nav[3].parentWidget()
        assert rail.height()==window.centralWidget().height()==window.height()
        assert status.isVisible() and status.parentWidget() is window.stack.parentWidget()
        assert status.geometry().top()>window.stack.geometry().bottom()
        assert 'test disk failure' in status.currentMessage()
        # Restart uses the persisted preference while retaining the normal UI.
        window.navigate(3)
        window.quiet_mode_check.click();app.processEvents()
        restored=Controller(load());restored.timer.stop()
        reopened=Window(restored);reopened.refresh_timer.stop();reopened.show();app.processEvents()
        assert all(check.isChecked() for check in reopened.setting_checks['quiet_mode'])
        assert reopened.isVisible() and reopened.nav[3].isVisible()
        reopened.close()
        c.pause.assert_not_called()
    finally:
        window.close()
result={'qt_quiet_toggle_both_modes':'passed','overlay_pixels_cleared':'passed',
        'main_window_and_navigation':'passed','save_reload_and_failed_save':'passed',
        'focus_learning_export_settings_preserved':'passed','physical_eye_accuracy_tested':False}
(out/'quiet-ui-check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
