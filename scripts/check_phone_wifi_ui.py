"""Exercise Wi-Fi discovery and its error states with a fake search and no hardware."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
import sys
from threading import Event
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont,QFontDatabase
from context_operator.controller import Controller
from context_operator.providers.phone_connection import WIFI_PHONE_NEXT_STEP, WifiPhoneNotFound
from context_operator.settings import DEFAULTS
from context_operator.ui.window import Window

app=QApplication([])
for name in ('SegUIVar.ttf','segoeui.ttf','segoeuib.ttf','seguisb.ttf','seguisym.ttf'):
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)
app.setFont(QFont('Segoe UI',10))
c=Controller(dict(DEFAULTS,vision=False,ranker=False,anticipation=False,phone_transport='wifi'))
c.timer.stop()
failure=WifiPhoneNotFound()
c.phone=SimpleNamespace(worker=Mock(failure=failure,is_alive=lambda:False),stop=Mock())
c.running=True;c.gaze_status=f'iPhone: {failure}'
w=Window(c);w.refresh_timer.stop();w.show();app.processEvents()
review=Path('.impeccable/review');review.mkdir(parents=True,exist_ok=True)
assert w.gaze_heading.text()=='Can’t find iPhone'
assert w.gaze_value.text()==WIFI_PHONE_NEXT_STEP and w.gaze_btn.text()=='Try again'
assert w.sources['iPhone'][0].state=='hold' and w.state_pill.word=='Not connected'
assert w.canvas.blocked and w.canvas.idle[0]=='Can’t find iPhone' and w.control_panel.isHidden()
assert w.phone_settings_button.isVisible()
assert w.grab().save(str(review/'windows-overview-phone-error.png'))
w.phone_settings_button.click();app.processEvents()
assert w.stack.currentIndex()==3 and w.stack.currentWidget().verticalScrollBar().value()==0
assert w.find_phone.isVisible() and not w.phone_host.text()
assert w.phone_transport.itemData(0)=='wifi' and w.phone_transport.itemData(1)=='usb'

def wait_for_search():
    deadline=time.monotonic()+2
    while not w.find_phone.isEnabled() and time.monotonic()<deadline:
        QTest.qWait(10)
    assert w.find_phone.isEnabled()

released=Event()
def fake_search(**_):
    assert released.wait(1)
    return ['192.168.2.42']

with patch('context_operator.ui.window.find_phones',side_effect=fake_search), \
     patch('context_operator.settings.save') as save:
    w.find_phone.click()
    assert w.phone_find_status.text()=='Looking for iPhone on your Wi‑Fi…'
    assert not w.find_phone.isEnabled()
    released.set()
    wait_for_search()
    assert w.phone_find_status.text()=='Found iPhone at 192.168.2.42'
    assert w.phone_host.text()==c.settings['phone_wifi_host']=='192.168.2.42'
    save.assert_called_once()
    assert c.running
    c.phone.stop.assert_not_called()
    assert w.grab().save(str(review/'windows-settings-find-iphone.png'))
    c.save_found_phone('192.168.2.43')
    w.last_paint=0;w.refresh()
    assert w.phone_host.text()=='192.168.2.43'

with patch('context_operator.ui.window.find_phones',return_value=[]), \
     patch('context_operator.settings.save') as save:
    w.find_phone.click();wait_for_search()
    assert w.phone_find_status.text()==WIFI_PHONE_NEXT_STEP
    save.assert_not_called()
    w.phone_host.setText('192.168.2.44')
    w.save_phone_connection()
    assert c.settings['phone_wifi_host']=='192.168.2.44'
    save.assert_called_once()
with patch('context_operator.settings.save') as save:
    w.phone_transport.setCurrentIndex(w.phone_transport.findData('usb'))
    w.save_phone_connection()
    assert c.settings['phone_transport']=='usb'
    c.phone_error='Pair iPhone first'
    w.navigate(0);w.last_paint=0;w.refresh()
    assert w.use_wifi_button.isVisible() and w.gaze_btn.text()=='Try again'
    w.use_wifi_button.click()
    assert c.settings['phone_transport']==w.phone_transport.currentData()=='wifi'
    assert save.call_count==2
c.phone_error='Set up your gaze first';c.phone_needs_setup=True
w.last_paint=0;w.refresh()
assert w.gaze_btn.text()=='Set up iPhone' and w.phone_settings_button.isVisible()
with patch.object(w,'open_setup') as setup:
    w.gaze_btn.click()
    setup.assert_called_once()
w.close()
print('Rendered Wi-Fi error and discovery states and verified manual saving, using a fake search.')
