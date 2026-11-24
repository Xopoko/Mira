from pathlib import Path
import json
import os

ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path(__file__).resolve().parent/'assets'
LOCAL=(ROOT/'.local' if (ROOT/'launch.pyw').is_file() else
       Path(os.environ.get('LOCALAPPDATA') or Path.home()/'AppData'/'Local')/'Mira')
DEFAULTS={'radius':230,'stability_ms':110,'microphone':'',
          'model_url':'http://127.0.0.1:18080/v1','model_name':'qwen3.8-local',
          'model_mode':'local','vision':False,'ranker':False,'show_gaze':True,'highlight_mode':'radial','quiet_mode':False,
          'dwell_enabled':False,'dwell_ms':1200,'fast_voice':True,'speech_silence_ms':280,
          'click_learning':True,'anticipation':True,
          'phone_transport':'wifi','phone_wifi_host':'',
          'show_scene_map':False,
          'webcam_device_id':'','webcam_device_name':'',
          'game_gaze_enabled':False}

def load():
    settings=dict(DEFAULTS)
    try:
        saved=json.loads((LOCAL/'settings.json').read_text(encoding='utf-8'))
        if isinstance(saved,dict):settings.update({key:saved[key] for key in DEFAULTS if key in saved})
    except (OSError,ValueError):pass
    if settings.get('highlight_mode') not in ('radial','magnetic'):settings['highlight_mode']='radial'
    return settings

def save(settings):
    LOCAL.mkdir(parents=True,exist_ok=True)
    target=LOCAL/'settings.json'
    temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps({k:settings[k] for k in DEFAULTS},ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,target)
