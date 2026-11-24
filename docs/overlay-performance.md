# Overlay rendering

The overlay caches the rasterized outlines and fills near the gaze point and
applies a moving gradient to that region. Dirty rectangles include both the
previous and current highlight, so an expired target or prediction can be
cleared even when tracking stops. A separate cache serves the overview. Both
caches hold drawn geometry, without application screenshots.

The overlay checks observation age and occlusion before using cached geometry.
Visual recognition results from the same source with nearly identical bounds
in the same window share an outline.

## Benchmark

Run the synthetic Qt raster benchmark from the repository root.

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_overlay
.\.venv\Scripts\python.exe -m scripts.benchmark_overlay --churn
```

The benchmark calls the overlay's radial painter on an offscreen 5120 × 1440
image and clears the whole image each frame. It reports
median, 95th-percentile and maximum frame times for 230, 1,030 and 3,030
elements. The churn mode changes the target, scene geometry and gaze point on
every frame. Use `--save <path>` to save the JSON output. The figures measure
rasterization and in-memory clearing on the machine running the benchmark.
Windows composition and tracking latency require separate live measurements.
