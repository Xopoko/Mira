from pathlib import Path
import json
import os

ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path(__file__).resolve().parent/'assets'
LOCAL=(ROOT/'.local' if (ROOT/'launch.pyw').is_file() else
       Path(os.environ.get('LOCALAPPDATA') or Path.home()/'AppData'/'Local')/'Mira')
DEFAULTS={'radius':230,'stability_ms':110,'microphone':'',
          'model_url':'http://127.0.0.1:18080/v1','model_name':'qwen3.8-local',
          'model_mode':'local','vision':True,'ranker':True,'show_gaze':True,'highlight_mode':'radial','quiet_mode':False,
          'dwell_enabled':False,'dwell_ms':1200,'fast_voice':True,'speech_silence_ms':280,
          'gaze_mode':'truedepth','truedepth_profile':'stable','click_learning':True,'anticipation':True,
          'phone_transport':'wifi','phone_wifi_host':'',
          'first_run_done':False,'show_scene_map':False,
          'webcam_device_id':'','webcam_device_name':'',
          'game_gaze_enabled':False}

def load():
    try:settings={**DEFAULTS,**json.loads((LOCAL/'settings.json').read_text(encoding='utf-8'))}
    except (OSError,ValueError):settings=dict(DEFAULTS)
    # RGB experiments are retired. Normalize old selections in memory and let
    # the next ordinary settings save persist the migration.
    if settings.get('gaze_mode') in ('mobilegaze','unigaze'):
        settings['gaze_mode']='truedepth'
    settings.pop('hybrid_enabled',None);settings.pop('hybrid_use',None)
    settings.pop('spatial_screen',None)
    if settings.get('highlight_mode') not in ('radial','magnetic'):settings['highlight_mode']='radial'
    for key in ('button_vk','button_name','button_enabled','button_device_id',
                'button_scan_code','button_extended'):
        settings.pop(key,None)
    return settings

def save(settings):
    LOCAL.mkdir(parents=True,exist_ok=True)
    target=LOCAL/'settings.json'
    temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps({k:settings[k] for k in DEFAULTS},ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,target)
