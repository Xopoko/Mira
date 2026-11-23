from pathlib import Path
import traceback
try:
    from context_operator.app import main
    raise SystemExit(main())
except Exception:
    local=Path(__file__).resolve().parent/'.local'
    local.mkdir(exist_ok=True)
    (local/'startup-error.log').write_text(traceback.format_exc(),encoding='utf-8')
    import ctypes
    ctypes.windll.user32.MessageBoxW(None,'Mira couldn’t start. Details are saved in .local/startup-error.log.','Mira',0x10)
    raise SystemExit(1)
