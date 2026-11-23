"""Bounded read-only desktop mapping measurement; never prints names or stores pixels."""
import json,time
from pathlib import Path
import numpy as np
from context_operator.providers.desktop import ScreenProvider

provider=ScreenProvider();samples=[];peak_windows=0;peak_elements=0;per_window={};first_seen={}
start=time.monotonic()
try:
    while time.monotonic()-start<5:
        tick=time.perf_counter();scene=provider.observe();samples.append((time.perf_counter()-tick)*1000)
        peak_windows=max(peak_windows,len(scene.windows));peak_elements=max(peak_elements,len(scene.elements))
        for window in scene.windows:
            first_seen.setdefault(window.window,(time.monotonic()-start)*1000)
            per_window[window.window]=len(window.elements)
        time.sleep(.06)
finally:provider.close()
result=dict(scope='read-only current visible desktop; no capture, activation or input',
            observations=len(samples),compose_median_ms=float(np.median(samples)),compose_p95_ms=float(np.percentile(samples,95)),
            peak_windows=peak_windows,peak_elements=peak_elements,window_element_counts=sorted(per_window.values()),
            first_window_ms=min(first_seen.values()) if first_seen else None,
            all_observed_windows_ms=max(first_seen.values()) if first_seen else None,last_reader_error=provider.last_error,
            stopped_workers=sum(not t.is_alive() for t in provider.threads))
out=Path('.local/evidence/desktop');out.mkdir(parents=True,exist_ok=True)
(out/'timing.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
