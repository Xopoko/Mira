# Overlay rendering

The overlay caches the rasterized outlines and fills near the gaze point and
applies a moving gradient to that region. Dirty rectangles include both the
previous and current highlight, so an expired target or prediction can be
cleared even when tracking stops. A separate cache serves the overview. Both
caches hold drawn geometry, without application screenshots.

The overlay checks observation age and occlusion before using cached geometry.
Visual recognition results with nearly identical bounds can share an outline
when they describe the same object in the same window.

## Benchmark

Run the synthetic Qt raster benchmark from the repository root.

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_overlay
.\.venv\Scripts\python.exe -m scripts.benchmark_overlay --churn
```

The benchmark paints the production overlay to a 5120 × 1440 image. It reports
median, 95th-percentile and maximum frame times for 230, 1,030 and 3,030
elements. The churn mode changes the target, scene geometry and gaze point on
every frame. Use `--save <path>` to save the JSON output. The figures measure
rasterization and in-memory clearing on the machine running the benchmark.
Windows composition and tracking latency require separate live measurements.
