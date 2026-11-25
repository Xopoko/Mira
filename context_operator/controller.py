"""One controller binds independent providers. Slow inference never blocks gaze rendering."""
from collections import deque
from dataclasses import replace
import hashlib
import threading
import time
from PySide6.QtCore import QObject, Signal, QTimer
from .core.contracts import Focus, Gaze, GazeMeasurement, Element, Rect, TARGET_ACTIONS, GAZE_TTL, MAP_TTL
from .core.fusion import Fusion
from .core.webcam import GazeArbitrator
from .core.commands import quick_intent
from .providers.windows import WindowsOutput, Windows
from .providers.desktop import ScreenProvider
from .providers.models import ContextModel, Ranker, Speech
from .providers.gaze import PhoneGaze, learning_path as phone_learning_path
from .personalization import ProfileUnavailable
from .providers.phone_connection import PhoneConnection
from .providers.fast_intent import FastIntent
from .providers.learning import GazeLearning
from .providers.clicks import MouseClicks
from .core.click_learning import ClickHistory,PointerMotion
from .providers.anticipation import IntentLearning
from .settings import LOCAL

class Controller(QObject):
    changed=Signal()
    message=Signal(str,str)
    received_scene=Signal(object,int,int)
    received_vision=Signal(object,int)
    received_speech=Signal(object,int)
    received_decision=Signal(object,object,int,float,int)
    failed=Signal(str,int)
    received_output=Signal(object,object,str,int)

    def __init__(self,settings,*,output=None,screen_factory=ScreenProvider,model_factory=ContextModel,click_source=None,game_gaze_factory=None,webcam_factory=None):
        super().__init__()
        self.settings=settings;self.win=Windows();self.output=output or WindowsOutput()
        self.screen_factory=screen_factory;self.model_factory=model_factory
        self.game_gaze_factory=game_gaze_factory;self.game_gaze_publisher=None;self.game_gaze_unavailable=False
        self.game_gaze_retry_at=0.
        self.webcam_factory=webcam_factory;self.webcam=None;self.webcam_status='Webcam off'
        self.gaze_arbitrator=GazeArbitrator();self.gaze_source_mode='none'
        self.gaze_source_epoch=0;self.revoked_utterances=set()
        self.fusion=Fusion(settings['stability_ms']/1000)
        self.scene=None;self.focus=Focus();self.gaze=None;self.attention=None;self.mouse_control=False;self.phone=None
        self.running=False;self.pointer_mode=False;self.microphone=False;self.busy=False
        self.generation=0;self.scene_stop=threading.Event();self.closed=False
        self.speech=Speech(self.received_speech.emit);self.voice_generation=0
        self.ranker=None
        self._ranker_warmup=None
        self._ranker_shutdown=None
        self.action_epoch=0;self.output_pending=False
        self.fast=FastIntent()
        self.learning=GazeLearning(LOCAL/'gaze-adaptation.json')
        self.clicks=click_source if click_source is not None else MouseClicks()
        self.click_history=ClickHistory();self.click_status='Connect iPhone to learn from clicks'
        self.click_counts={'seen':0,'accepted':0,'away':0,'quality':0}
        self.anticipation=IntentLearning(LOCAL/'intent-prediction.json');self.prediction=None
        self.history=deque(maxlen=6)
        self.vision_elements=();self.vision_at=0.
        self.vision_sources={}
        self.scene_provider=None;self.practice_window=0
        self.setup_close=None
        self.status='Ready to connect';self.gaze_status='Off';self.gaze_fps=0.
        self.phone_error='';self.phone_needs_setup=False
        self._reported_wifi_connection=None
        self.voice_status='Off';self.voice_state='off';self.voice_fallback=''
        self.last_transcript=''
        self.action_confirmation: tuple[str, float] | None = None
        self.speech_focus={};self.dwell_id=None;self.dwell_since=0.;self.dwell_fired=None
        self.received_scene.connect(self._scene)
        self.received_vision.connect(self._vision)
        self.received_speech.connect(self._speech)
        self.received_decision.connect(self._decision)
        self.failed.connect(self._failure)
        self.received_output.connect(self._output)
        self.timer=QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(25)

    def _failure(self,error,generation):
        if generation!=self.generation:return
        self.busy=self.output_pending;self.status=error;self.message.emit('Notice',error);self.changed.emit()

    def start(self):
        if self.running:return
        if self.setup_close:
            close,self.setup_close=self.setup_close,None;close()
        self.running=True;self.generation+=1;generation=self.generation
        self._start_clicks()
        self.scene_stop=threading.Event();stop=self.scene_stop
        self.status='Recognizing screen elements'
        def scan():
            import uiautomation as auto
            provider=None
            try:
                with auto.UIAutomationInitializerInThread():
                    provider=self.screen_factory();self.scene_provider=provider
                    while not stop.is_set():
                        epoch=self.action_epoch
                        provider.practice_window=self.practice_window
                        provider.attention=self.attention
                        scene=provider.observe()
                        if not stop.is_set():self.received_scene.emit(scene,generation,epoch)
                        stop.wait(.06)
            except Exception as exc:
                self.failed.emit('Screen map: '+str(exc),generation)
            finally:
                if provider is not None:provider.close()
                if self.scene_provider is provider:self.scene_provider=None
        threading.Thread(target=scan,daemon=True,name='screen-perception').start()
        if self.settings['vision']:self._start_vision(stop,generation)
        self.changed.emit()

    def set_practice_scope(self, window, *, use_mouse=False):
        if (window, use_mouse) == (self.practice_window, self.pointer_mode):
            return
        self.practice_window = window
        self.pointer_mode = use_mouse
        self._revoke_gaze_target()
        self.scene = None
        self.vision_elements = ()
        self.vision_sources = {}
        self.action_confirmation = None

    def _ranker_for_command(self):
        if self.ranker is None:
            if self._ranker_shutdown is not None and not self._ranker_shutdown.is_set():
                return None
            self.ranker = Ranker()
        ranker = self.ranker
        if ranker.is_ready:
            return ranker
        if self._ranker_warmup is None or not self._ranker_warmup.is_alive():
            def warm():
                try:
                    ranker.warm()
                except Exception as exc:
                    if ranker is self.ranker:
                        self.message.emit('Target matching', str(exc))
            self._ranker_warmup = threading.Thread(target=warm, daemon=True, name='ranker-warmup')
            self._ranker_warmup.start()
        return None

    def stop_ranker(self):
        ranker, self.ranker = self.ranker, None
        self._ranker_warmup = None
        if ranker is not None:
            self._ranker_shutdown = ranker.close()

    def _start_vision(self,stop,generation):
        def perceive():
            def unavailable(reason):
                if not stop.is_set():self.message.emit('Screen recognition','Unavailable: '+reason)
            cache=None
            try:
                from .foundations.perception.runtime import PerceptionClient
                from .providers.visual_cache import VisualCache
                from .providers.visual_regions import verified,crop
                cache=VisualCache(stop,PerceptionClient,unavailable)
                while not stop.wait(.1):
                    epoch=self.action_epoch
                    scene=self.scene;provider=self.scene_provider
                    if scene is None or provider is None:continue
                    region=None;gaze=self.attention
                    scene=provider.vision_scene(scene,gaze,time.monotonic())
                    if scene is None:continue
                    if gaze and 0<=time.monotonic()-gaze.captured<GAZE_TTL:
                        region=Rect(round(gaze.x/160)*160-480,round(gaze.y/160)*160-320,960,640)
                    captured=provider.capture(scene,region)
                    if captured is None:continue
                    image,rect=captured
                    if image.width>1600 or image.height>1000:
                        # Full-window recognition is useful on a spoken request; continuous gaze uses a region.
                        continue
                    window=scene.window_map(scene.window)
                    owner=(scene.window,window.pid if window else 0,scene.window_rect,epoch)
                    result=cache.lookup_frame(owner,rect,image)
                    current_capture=provider.capture(scene,rect)
                    if current_capture is None:continue
                    live_image,live_rect=current_capture
                    verified_at=time.monotonic()
                    elements=[]
                    for item in verified(result,live_image,live_rect):
                        r=item.rect
                        identity=hashlib.blake2s(f'{scene.window}:{r.as_list()}:{item.label}'.encode(),digest_size=10).hexdigest()
                        patch=crop(live_image,live_rect,r)
                        digest=hashlib.blake2s(patch.tobytes(),digest_size=12).hexdigest()
                        elements.append(Element(identity,item.label or 'Icon','Text' if item.source=='ocr' else 'Icon',r,scene.window,item.source,True,window.pid if window else self.win.pid(scene.window),digest,verified_at))
                    if not stop.is_set():self.received_vision.emit((scene,verified_at,tuple(elements),epoch),generation)
            except Exception as exc:
                unavailable(str(exc) or type(exc).__name__)
            finally:
                if cache is not None:cache.close()
        threading.Thread(target=perceive,daemon=True,name='visual-perception').start()

    def pause(self):
        # Revoke queued effects before stopping providers, which may wait for workers.
        self.running=False;self.generation+=1;self._close_game_gaze();self.scene_stop.set();self.busy=self.output_pending
        self.action_confirmation=None
        self.action_epoch+=1;self.voice_generation=-1
        self.stop_ranker()
        self.clicks.stop();self.click_history.clear()
        self.anticipation.reset();self.prediction=None
        self.click_status='Click learning is paused'
        self.attention=None;self.mouse_control=False
        self.focus=Focus(reason='Paused');self.fusion.reset()
        self.vision_elements=();self.scene=None
        self.dwell_id=self.dwell_fired=None
        self.speech.stop();self.microphone=False;self.voice_status='Off';self.voice_state='off';self.voice_fallback=''
        self.speech_focus.clear();self.revoked_utterances.clear()
        if self.phone:self.phone.stop();self.phone=None
        self._stop_webcam()
        if self.setup_close:
            close,self.setup_close=self.setup_close,None
            try:close()
            except Exception as exc:self.message.emit('Gaze setup','Couldn’t close setup: '+str(exc))
        self.gaze=None;self.gaze_source_mode='none';self.gaze_status='Off';self.gaze_fps=0;self.phone_error='';self.phone_needs_setup=False
        self._reported_wifi_connection=None
        self.status='Paused';self.changed.emit()

    def toggle_phone(self):
        phone=self.phone
        self.phone_error='';self.phone_needs_setup=False
        self._reported_wifi_connection=None
        if phone is not None and self.gaze_source_mode in ('none','iphone'):
            self._revoke_gaze_target()
        else:
            self.click_history.clear();self.anticipation.reset();self.prediction=None
        self.phone=None
        self.gaze_fps=0.
        self.clicks.stop()
        if phone and phone.worker and phone.worker.is_alive():
            if self.gaze_source_mode=='none':self.gaze_status='Off'
            phone.stop()
        else:
            new_phone=None
            try:
                if phone:phone.stop()
                new_phone=PhoneGaze(self.win.bounds(),connection=PhoneConnection.from_settings(self.settings))
                self.phone=new_phone
                path=phone_learning_path(new_phone.profile_key,LOCAL)
                if self.learning.path!=path:self.learning=GazeLearning(path)
                new_phone.start();self.pointer_mode=False
                self.learning.bind(new_phone.profile_key,self.win.bounds(),new_phone.feature_scale)
                self.learning.trainer.set_seed(new_phone.profile)
                if not self.running:self.start()
                self.gaze_status='Connecting to iPhone'
            except Exception as exc:
                if new_phone:
                    try:new_phone.stop()
                    except Exception:pass
                self.phone=None
                error=str(exc) or 'Couldn’t connect to iPhone'
                self.phone_error=error;self.phone_needs_setup=isinstance(exc,ProfileUnavailable)
                self.message.emit('Gaze',error);self.status=error
        self._start_clicks();self.changed.emit()

    def _revoke_gaze_target(self):
        self.action_epoch+=1;self.gaze_source_epoch+=1
        self.gaze_source_mode='none'
        self.revoked_utterances.update(self.speech_focus)
        if len(self.revoked_utterances)>32:self.revoked_utterances=set(self.speech_focus)
        self.speech_focus.clear()
        self.focus=Focus();self.fusion.reset();self.attention=None;self.gaze=None
        self.dwell_id=self.dwell_fired=None
        self.click_history.clear()
        self.anticipation.reset();self.prediction=None

    def _stop_webcam(self):
        webcam,self.webcam=self.webcam,None
        self.gaze_arbitrator.reset()
        if self.gaze_source_mode=='webcam':
            self._revoke_gaze_target();self.gaze_status='Waiting for fresh gaze data'
        self.webcam_status='Webcam off'
        if webcam is not None:
            try:webcam.stop()
            except Exception as exc:self.webcam_status='Off · '+type(exc).__name__

    def _webcam_failed(self,webcam,reason):
        if webcam is not self.webcam:return
        self.webcam_status=reason
        if self.gaze_source_mode=='webcam':
            self._revoke_gaze_target();self.gaze_status='Waiting for fresh gaze data'

    def select_webcam(self,device_id,name):
        if self.busy or self.output_pending:raise RuntimeError('Wait for the command to finish')
        if not isinstance(device_id,str) or not device_id or len(device_id)>512:
            raise ValueError('Choose a webcam from the list')
        from .settings import save
        updated=dict(self.settings,webcam_device_id=device_id,webcam_device_name=str(name)[:200])
        save(updated)
        self._stop_webcam();self.settings.update(updated);self.changed.emit()

    def toggle_webcam(self):
        if self.webcam is not None:
            self._stop_webcam();self.changed.emit();return
        if self.setup_close or self.busy or self.output_pending:
            self.message.emit('Webcam','Finish setup or wait for the current command');return
        webcam=None
        try:
            if self.webcam_factory is None:
                from .providers.webcam import WebcamGaze
                self.webcam_factory=WebcamGaze
            webcam=self.webcam_factory(self.win.bounds(),self.settings.get('webcam_device_id',''))
            webcam.failed.connect(self._webcam_failed)
            webcam.start();self.webcam=webcam
            self.gaze_arbitrator.reset();self.webcam_status='Connecting to webcam'
            if not self.running:self.start()
        except Exception as exc:
            if webcam is not None:webcam.stop()
            self.webcam=None
            self.webcam_status=('Set up the webcam in Settings first' if isinstance(exc,FileNotFoundError) else str(exc))
            self.message.emit('Webcam',self.webcam_status)
        self.changed.emit()

    def _select_gaze_source(self,now):
        primary=self.gaze
        secondary=None
        if self.webcam is not None:
            if self.webcam.bounds!=self.win.bounds():
                self._stop_webcam();self.webcam_status='The screen changed. Set up the webcam again.'
            else:
                secondary,self.webcam_status=self.webcam.poll(now)
                if not self.webcam.active:
                    secondary=None
                    self._webcam_failed(self.webcam,self.webcam_status)
        choice=self.gaze_arbitrator.choose(primary,secondary,now)
        # A missing frame clears live gaze without changing the source of latched commands.
        if choice.mode!='none' and choice.mode!=self.gaze_source_mode:
            if self.gaze_source_mode!='none' or choice.mode=='webcam':self._revoke_gaze_target()
            self.gaze_source_mode=choice.mode
        self.gaze=choice.gaze
        if choice.mode=='webcam':
            self.gaze_status='Webcam · tracking gaze';self.prediction=None
        elif choice.mode=='none' and self.phone is None:self.gaze_status='Waiting for fresh gaze data'
        elif choice.corroborated is True:self.webcam_status='Webcam · matches iPhone gaze'
        elif choice.corroborated is False:self.webcam_status='Using iPhone · gaze readings differ'

    def _start_clicks(self):
        teach=(self.phone and self.learning.key is not None
                and self.settings.get('click_learning',True))
        if self.running and (teach or self.settings['anticipation']):
            self.clicks.start()
            if teach:self.click_status='Learning from clicks near where you just looked'

    def set_click_learning(self,enabled):
        from .settings import save
        updated=dict(self.settings,click_learning=bool(enabled))
        try:save(updated)
        except OSError as exc:self.message.emit('Click learning',str(exc));self.changed.emit();return
        self.settings.update(updated);self.clicks.stop();self.click_history.clear()
        self.anticipation.reset();self.prediction=None
        self.click_status='Click learning is off' if not enabled else 'Connect iPhone to learn from clicks'
        self._start_clicks();self.changed.emit()

    def set_anticipation(self,enabled):
        from .settings import save
        updated=dict(self.settings,anticipation=bool(enabled))
        try:save(updated)
        except OSError as exc:self.message.emit('Prediction',str(exc));self.changed.emit();return
        self.settings.update(updated);self.anticipation.reset();self.prediction=None
        self.clicks.stop();self.click_history.clear();self._start_clicks();self.changed.emit()

    def _predict_intention(self,events,now):
        motions=self.clicks.drain_motion()
        self.prediction=None
        if not self.settings['anticipation']:return
        stream=sorted([*motions,*events],key=lambda e:e.captured if isinstance(e,PointerMotion) else e.released)
        for event in stream:
            if isinstance(event,PointerMotion):self.anticipation.observe(event)
            else:self.anticipation.feedback(event,now)
        measurement=self.phone.measurement if self.phone else None
        supported=(not self.phone or isinstance(measurement,GazeMeasurement) and
                   measurement.supported and 0<=now-measurement.captured<GAZE_TTL)
        self.prediction=self.anticipation.predict(self.scene,self.gaze,now,gaze_supported=supported)

    def _learn_clicks(self,measurement,now,events):
        if self.pointer_mode:
            self.click_history.clear()
            return
        if (not self.phone or self.learning.key is None
                or not self.settings.get('click_learning',True)):
            self.click_status=('Click learning is off' if not self.settings.get('click_learning',True) else
                               'Connect iPhone to start learning from clicks' if not self.phone else
                               'Complete the short gaze setup to start click learning')
            self.click_history.clear();return
        self.click_history.observe(measurement,now,self.phone.invalid_captured)
        if self.clicks.error:self.click_status='Can’t read clicks: '+self.clicks.error
        # Gate against the uncorrected source and its real display radius. The
        # click learner and the intention predictor cannot expand their own labels.
        radius=self.gaze.radius if self.gaze is not None else self.phone.validation_radius
        changed=False
        for click in events:
            self.click_counts['seen']+=1
            match=self.click_history.match(click,now,self.win.bounds(),radius=radius)
            if match is None:
                category='away' if self.click_history.last_rejection_code=='away' else 'quality'
                self.click_counts[category]+=1
                self.click_status='Click skipped · '+(self.click_history.last_rejection or 'no recent gaze before the click');continue
            try:
                if self.learning.learn_click(match.sample,(click.x,click.y),weight=match.weight,limit=match.limit):
                    self.click_counts['accepted']+=1
                    changed=True;self.click_status='Click used to refine gaze'
                else:
                    result=self.learning.last_click_result
                    self.click_counts['away' if result=='away' else 'quality']+=1
                    self.click_status='Click skipped · '+{
                        'conflict':'it disagrees with earlier clicks here',
                        'away':'too far from the corrected gaze',
                    }.get(result,'the gaze reading wasn’t usable')
            except OSError as exc:
                self.settings['click_learning']=False;self.click_history.clear()
                if not self.settings['anticipation']:self.clicks.stop()
                self.click_status='Click learning stopped because Mira couldn’t save the correction'
                self.message.emit('Click learning',str(exc));break
        if changed:self.fusion.reset();self.dwell_id=None

    def undo_learning(self):
        try:self.learning.undo_last()
        except OSError as exc:
            self.message.emit('Click learning','Couldn’t undo the correction: '+str(exc));return
        self.click_history.clear();self.clicks.drain();self.fusion.reset()
        self.click_status='Last correction undone';self.changed.emit()

    def set_phone_connection(self,transport,host):
        if self.output_pending or self.busy:raise RuntimeError('Wait for the command to finish')
        connection=PhoneConnection(transport,host)
        from .settings import save
        updated=dict(self.settings,phone_transport=connection.transport,phone_wifi_host=connection.host)
        save(updated)
        self.settings.update(updated)
        self.pause()
        self.gaze_status='Connection saved · select “Connect iPhone”'
        self.changed.emit()

    def save_found_phone(self, host):
        if self.settings.get('phone_transport') != 'wifi':
            self.set_phone_connection('wifi', host)
            return
        connection = PhoneConnection('wifi', host)
        if connection.host == self.settings.get('phone_wifi_host', ''):
            return
        from .settings import save
        updated = dict(self.settings, phone_wifi_host=connection.host)
        save(updated)
        self.settings.update(updated)
        self.changed.emit()

    def _remember_discovered_phone(self, phone):
        connection = phone.connection
        if (not isinstance(connection, PhoneConnection) or connection.transport != 'wifi'
                or not connection.found_host or self._reported_wifi_connection is connection):
            return
        self._reported_wifi_connection = connection
        if (self.settings.get('phone_transport') != 'wifi'
                or self.settings.get('phone_wifi_host', '') != connection.host):
            return
        try:
            self.save_found_phone(connection.found_host)
        except OSError as exc:
            self.message.emit('iPhone', 'Couldn’t save the address found on Wi‑Fi: '+str(exc))

    def activate_profile(self):
        if self.output_pending or self.busy:raise RuntimeError('Wait for the command to finish')
        phone=PhoneGaze(self.win.bounds())
        path=phone_learning_path(phone.profile_key,LOCAL)
        learning=GazeLearning(path);learning.bind(phone.profile_key,self.win.bounds(),phone.feature_scale)
        learning.trainer.set_seed(phone.profile)
        setup_close=self.setup_close;self.setup_close=None
        try:
            self.pause()
        except Exception:
            self.setup_close=setup_close
            raise
        self.learning=learning
        self.status='Gaze profile selected. Connect iPhone.'
        if setup_close:
            close=setup_close
            try:close()
            except Exception as exc:self.status='Gaze profile selected. Couldn’t close setup: '+str(exc)
        self.changed.emit()

    def toggle_microphone(self):
        self.speech_focus.clear();self.revoked_utterances.clear()
        self.voice_fallback=''
        if self.microphone:
            self.speech.stop();self.voice_generation=-1;self.microphone=False;self.voice_status='Off';self.voice_state='off'
        else:
            if not self.running:self.start()
            self.microphone=True;self.voice_status='Loading speech recognition'
            self.voice_state='standby'
            try:self.voice_generation=self.speech.start(self.settings['microphone'],self.settings.get('speech_silence_ms',280))
            except Exception as exc:
                self.microphone=False
                self.voice_status=str(exc) or 'Couldn’t start speech recognition'
                self.voice_state='hold'
        self.changed.emit()

    def _scene(self,scene,generation,epoch):
        if generation!=self.generation or not self.running or epoch!=self.action_epoch:return
        now=time.monotonic();extras=[]
        for e in self.vision_elements:
            if not 0<=now-(e.captured or self.vision_at)<MAP_TTL:continue
            window=scene.window_map(e.window)
            if scene.windows:
                if not window or self.vision_sources.get(e.window)!=self._window_signature(scene,e.window):continue
                if not any(r.contains(*e.rect.center) for r in window.visible):continue
            elif (e.window!=scene.window or not scene.window_rect or
                  self.vision_sources.get(e.window)!=self._window_signature(scene,e.window)):continue
            if not any(self._covers_visual(u,e) for u in scene.elements):extras.append(e)
        self.vision_elements=tuple(extras)
        self.vision_sources={h:s for h,s in self.vision_sources.items() if any(e.window==h for e in extras)}
        if extras:
            scene=replace(scene,elements=scene.elements+tuple(extras),windows=tuple(replace(w,elements=w.elements+tuple(e for e in extras if e.window==w.window)) for w in scene.windows))
        self.scene=scene
        if self.settings['anticipation']:self.anticipation.observe_scene(scene,time.monotonic())
        if not self.busy:
            error=self.scene_provider.last_error if self.scene_provider is not None else ''
            if not scene.elements and error:self.status='Can’t read screen elements ('+error+')'
            else:self.status='Look and speak' if self.gaze else 'Ready · connect iPhone or name an element'
        self.changed.emit()

    def _vision(self,event,generation):
        source,started,elements,epoch=event
        if generation!=self.generation or not self.scene:return
        if epoch!=self.action_epoch:return
        if time.monotonic()-started>MAP_TTL:return
        if self._window_signature(source,source.window)!=self._window_signature(self.scene,source.window):return
        self.vision_elements=tuple(e for e in self.vision_elements if e.window!=source.window)+elements
        self.vision_sources[source.window]=self._window_signature(source,source.window);self.vision_at=started

    @staticmethod
    def _covers_visual(u,e):
        if u.source!='uia' or u.window!=e.window or e.rect.w<=0 or e.rect.h<=0:return False
        width=max(0,min(u.rect.x+u.rect.w,e.rect.x+e.rect.w)-max(u.rect.x,e.rect.x))
        height=max(0,min(u.rect.y+u.rect.h,e.rect.y+e.rect.h)-max(u.rect.y,e.rect.y))
        if width*height<e.rect.w*e.rect.h*.9:return False
        # Containment alone does not make a big editor/list the same object as
        # a small send icon. Nor does one UIA word replace a whole OCR line.
        return ((e.role=='Text' and u.role in ('Edit','Text'))
                or u.rect.w*u.rect.h<=e.rect.w*e.rect.h*4)

    @staticmethod
    def _window_signature(scene,hwnd):
        window=scene.window_map(hwnd)
        # Each visual region is pixel-verified independently. A changing caption
        # elsewhere in this window must not invalidate a stationary button.
        if window:return (window.pid,window.rect,window.blocked_capture)
        if scene.windows or scene.window!=hwnd:return None
        return (None,scene.window_rect,frozenset(e.id for e in scene.elements if e.window==hwnd and e.source=='uia'))

    def _speech(self,event,generation):
        if generation!=self.voice_generation or not self.microphone or not self.running:return
        kind=event.get('type')
        if kind=='speech_start':
            now=time.monotonic();gaze=self.attention
            if not (gaze and 0<=now-gaze.captured<GAZE_TTL):gaze=None
            focus=self.focus
            if focus.target and (not self.scene or not self.scene.current_target(focus.target,now)):
                focus=Focus()
            self.speech_focus[event.get('utteranceId')]=(focus,gaze)
            if len(self.speech_focus)>8:self.speech_focus.pop(next(iter(self.speech_focus)))
            self.voice_status='Listening…';self.voice_state='go'
        elif kind=='final':
            self.voice_state='go'
            if event.get('utteranceId') in self.revoked_utterances:
                self.revoked_utterances.discard(event.get('utteranceId'))
                self.voice_status='The gaze source changed. Repeat your command.';return
            self.voice_status=f'After speech {event.get("afterSpeechMs",event.get("asrMs",0))} ms · recognition {event.get("asrMs",0)} ms'
            focus,gaze=self.speech_focus.pop(event.get('utteranceId'),(Focus(),None))
            self.command(event['text'],focus,gaze)
        elif kind=='ready':
            self.voice_status='Listening';self.voice_state='go'
            missing=(event.get('device') or {}).get('fallbackFrom')
            if missing:
                self.voice_fallback=f'Using the default microphone because {missing} isn’t connected.'
        elif kind=='error':
            self.voice_status=event.get('message') or 'Microphone error';self.voice_state='hold'
            self.message.emit('Microphone',self.voice_status)
        elif kind=='stopped':
            self.microphone=False
            self.voice_fallback=''
            if self.voice_state!='hold':
                self.voice_status='Speech recognition stopped';self.voice_state='hold'
        self.changed.emit()

    def tick(self):
        if self.closed:return
        if self.win.emergency() and (self.running or self.microphone or self.phone or self.setup_close):
            self.pause();self.message.emit('Stop','Capture and control stopped');return
        now=time.monotonic()
        try:self.learning.poll_training()
        except OSError as exc:self.message.emit('Click learning','Couldn’t save the model: '+str(exc))
        if not self.running:
            self._publish_game_gaze(now,running=False)
            return
        phone=self.phone
        if phone:
            self._remember_discovered_phone(phone)
            self.gaze,self.gaze_status,self.gaze_fps=phone.poll(self.settings['radius'])
            if phone.worker is not None and not phone.worker.is_alive():
                if self.gaze_source_mode=='iphone':self._revoke_gaze_target()
                self.gaze=None;phone=None
        else:self.gaze=None
        raw_gaze=self.gaze
        measurement=phone.measurement if phone else None
        click_events=self.clicks.drain()
        self._learn_clicks(measurement,now,click_events)
        if phone:self.gaze=self.learning.correct(raw_gaze,measurement,now,self.settings['radius'])
        self._predict_intention(click_events,now)
        if raw_gaze is None and self.gaze is not None:self.gaze_status='TrueDepth · tracking gaze in a known position'
        self._select_gaze_source(now)
        mouse_control=bool(self.win.shift_down() or self.pointer_mode)
        if mouse_control!=self.mouse_control:
            self.fusion.reset();self.dwell_id=None
        self.mouse_control=mouse_control
        self._publish_game_gaze(now,desktop=self.win.bounds(),running=True)
        self.attention=Gaze(*self.win.cursor(),now,self.settings['radius'],'mouse') if mouse_control else self.gaze
        self.fusion.stability=self.settings['stability_ms']/1000
        self.focus=(self.fusion.choose_pointer(self.scene,self.attention,now,self.win.at(self.attention.x,self.attention.y))
                    if mouse_control else self.fusion.choose(self.scene,self.attention,now,
                                                            magnetic=self.settings.get('highlight_mode')=='magnetic'))
        target=self.focus.target
        target_key=(target.window,target.rect) if target else None
        # A transient empty scene after dispatch or a blink does not re-arm dwell.
        if target_key and self.dwell_fired and target_key!=self.dwell_fired:self.dwell_fired=None
        if (self.settings.get('dwell_enabled') and not self.mouse_control and target and not self.focus.ambiguous
                and self.attention and 0<=now-self.attention.captured<GAZE_TTL
                and not self.fusion.selection_until(target,now,self.attention) and not self.busy and not self.output_pending):
            if target.identity!=self.dwell_id:self.dwell_id=target.identity;self.dwell_since=now
            elif self.dwell_fired!=target_key and now-self.dwell_since>self.settings['dwell_ms']/1000:
                self.dwell_fired=target_key;self.command('click')
        else:self.dwell_id=None
        self.changed.emit()

    def _close_game_gaze(self):
        publisher,self.game_gaze_publisher=self.game_gaze_publisher,None
        if publisher:
            try:publisher.close()
            except Exception as exc:
                if not self.game_gaze_unavailable:
                    self.message.emit('Game gaze feed','Unavailable: '+(str(exc) or type(exc).__name__))

    def _publish_game_gaze(self,now,*,desktop=None,running=False):
        if not self.settings.get('game_gaze_enabled',False):
            self._close_game_gaze();self.game_gaze_unavailable=False
            return
        if not running:
            self._close_game_gaze()
            return
        if self.game_gaze_unavailable and now < self.game_gaze_retry_at:return
        phone=self.phone
        try:
            if self.game_gaze_publisher is None:
                if self.game_gaze_factory is None:
                    from .providers.game_gaze import GameGazePublisher
                    self.game_gaze_factory=GameGazePublisher
                self.game_gaze_publisher=self.game_gaze_factory()
            _,written=self.game_gaze_publisher.publish(
                self.gaze,phone.measurement if phone else None,desktop,now,
                phone_present=bool(phone),
                pointer_mode=self.pointer_mode,
                mouse_control=self.mouse_control,
                setup=bool(self.setup_close),
                running=bool(running),
            )
            if written:self.game_gaze_unavailable=False
        except Exception as exc:
            if not self.game_gaze_unavailable:
                self.message.emit('Game gaze feed','Unavailable: '+(str(exc) or type(exc).__name__))
            self.game_gaze_unavailable=True
            self.game_gaze_retry_at=now+2.
            self._close_game_gaze()

    def command(self,text,latched=None,latched_gaze=None):
        text=text.strip()
        if not text:return
        quick=quick_intent(text)
        if quick and quick.action=='pause':self.pause();return
        if quick and quick.action=='resume':self.start();return
        if not self.running:self.start();self.message.emit('Screen','Wait for the screen map, then repeat your command');return
        if self.busy or self.output_pending:
            self.message.emit('Command','Mira is still processing the previous command. Say “stop” to cancel the wait. An action that already started may finish.');return
        if not self.scene or not self.scene.elements:self.message.emit('Screen','Switch to the app you want to control so Mira can find its elements');return
        self.last_transcript=text;self.message.emit('You',text)
        # An empty onset snapshot is still a snapshot, never permission to retarget later.
        focus=latched if latched is not None else self.focus
        started=time.monotonic();generation=self.generation;source_epoch=self.gaze_source_epoch
        gaze=latched_gaze if latched is not None or latched_gaze is not None else self.attention
        max_gaze_age=15 if latched_gaze is not None else GAZE_TTL
        if focus.target and not self.scene.current_target(focus.target,started):
            focus=Focus()
        selected=bool(self.fusion.selection_until(focus.target,started,gaze))
        if quick:
            if quick.action=='select':
                target=self.fusion.cycle(focus,started,quick.direction)
                self.status='Selected: '+target.label if target else 'No nearby element';return
            if quick.action in TARGET_ACTIONS and not selected and not (gaze and 0<=started-gaze.captured<max_gaze_age):
                self.message.emit('Gaze','Look at the element again or say its name.');return
            if quick.action in TARGET_ACTIONS and not focus.target:
                self.message.emit('Target','Look at an element or say its name.');return
            if focus.target and quick.action!='type_text':quick=replace(quick,target_id=focus.target.id)
            self._decision(quick,self.scene,generation,started);return
        if self.settings.get('fast_voice',True):
            intent=self.fast.interpret(text,self.scene,focus,gaze,max_gaze_age=max_gaze_age,allow_selected=selected)
            if intent:
                self._decision(intent,self.scene,generation,started);return
        self.busy=True;self.status='Matching your command…'
        scene=self.scene
        ranker = self._ranker_for_command() if self.settings['ranker'] and scene.current_elements(started) else None
        def infer():
            try:
                learned={}
                if ranker is not None:
                    try:
                        shortlist=Fusion(0).choose(scene,gaze,started,text)
                        candidates=[row.element for row in shortlist.alternatives]
                        learned=ranker.rank(text,candidates,scene.title,wait_for_ready=False)
                    except Exception as exc:
                        if ranker is self.ranker:
                            self.message.emit('Target matching', str(exc))
                if generation!=self.generation:return
                provider=self.scene_provider;image=None
                if self.settings['vision'] and provider:
                    capture=provider.capture(scene)
                    if capture:image=capture[0]
                decision=self.model_factory(self.settings).interpret(text,scene,focus,image,self.history,learned)
                self.received_decision.emit(decision,scene,generation,started,source_epoch)
            except Exception as exc:self.failed.emit(str(exc),generation)
        threading.Thread(target=infer,daemon=True,name='context-command').start()
        self.changed.emit()

    def _decision(self,intent,source,generation,started,source_epoch=None):
        if generation!=self.generation or not self.running:return
        if self.output_pending:return
        self.busy=False
        if source_epoch is not None and source_epoch!=self.gaze_source_epoch:
            self.status='The gaze source changed. Repeat your command.';self.changed.emit();return
        if intent.action=='pause':self.pause();return
        if intent.action=='resume':self.start();return
        if intent.action=='clarify':self.message.emit('Mira',intent.reply or 'Say which element you mean');return
        current=self.scene
        if not current:
            self.message.emit('Screen','The window changed. Repeat your command for the new target.');return
        original=next((e for e in source.current_elements(started) if e.enabled and e.id==intent.target_id),None)
        target=current.current_target(original,time.monotonic())
        if intent.target_id and target is None:
            self.message.emit('Target','The element moved or changed. Look at it again.');return
        if target and (source.windows or current.windows):
            before=source.window_map(target.window);after=current.window_map(target.window)
            if before is None or after is None or (before.pid,before.rect)!=(after.pid,after.rect):
                self.message.emit('Target','The target window changed. Look at it again.');return
        elif current.window!=source.window or current.window_rect!=source.window_rect:
            self.message.emit('Screen','The window changed. Repeat your command for the new target.');return
        if intent.action=='select':
            if target:
                self.fusion.select(target,time.monotonic(),3.)
                self.status='Selected: '+target.label;self.changed.emit()
            return
        self.busy=True;self.output_pending=True
        self.action_epoch+=1;self.vision_elements=();self.vision_at=0
        dispatch_epoch=self.gaze_source_epoch
        def dispatch():
            try:
                if generation!=self.generation or not self.running or dispatch_epoch!=self.gaze_source_epoch:raise RuntimeError('Control or its input source changed before Mira could send the action')
                if isinstance(self.output,WindowsOutput):self.output.set_guard(lambda:generation==self.generation and self.running and dispatch_epoch==self.gaze_source_epoch)
                outcome=self.output.execute(intent,target,current)
                self.received_output.emit(outcome,intent,'',generation)
            except Exception as exc:self.received_output.emit(None,intent,str(exc),generation)
        threading.Thread(target=dispatch,daemon=True,name='windows-output').start()

    def _output(self,outcome,intent,error,generation):
        late=generation!=self.generation
        self.busy=False;self.output_pending=False
        # A failed native dispatch can be partial too: no target survives for automatic replay.
        self.action_epoch+=1;self.vision_elements=();self.vision_at=0
        self.scene=None;self.focus=Focus(reason='Checking the new screen');self.fusion.reset()
        self.action_confirmation=None
        if outcome is not None and not error:
            completed_at=time.monotonic()
            labels = {'click': 'Click', 'double_click': 'Double-click',
                      'right_click': 'Right-click', 'scroll': 'Scroll',
                      'type_text': 'Text', 'press_keys': 'Keys', 'move': 'Pointer move'}
            action=labels.get(intent.action,intent.action)
            self.status=f'Sent to Windows · {action} · {outcome["target"]}'
            if not late:
                self.action_confirmation=(f'{action} sent to Windows',completed_at)
            title = 'Windows reported the action after control stopped' if late else 'Action'
            self.message.emit(title,self.status)
            self.history.append({'request':self.last_transcript,'action':intent.action,'target':outcome['target']})
        else:
            error = error or 'Mira couldn’t confirm whether Windows received the action.'
            self.status=error
            self.message.emit('Check whether the action went through',error+' Check before trying again.')
        if late and not self.running:self.status='Paused'
        self.changed.emit()

    def close(self):
        self.closed=True;self.timer.stop();self.pause()
