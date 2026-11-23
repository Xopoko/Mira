"""Explicit physical display presets; never inferred from pixel resolution."""

SCREEN_PRESETS = {
    'flat': ('Flat screen · measured during setup', None),
    'odyssey-g9-49-1000r': ('Samsung Odyssey G9 · 49″ · 1000R', {
        'kind': 'cylinder', 'width_m': 1.191936, 'height_m': .335232, 'radius_m': 1.0,
    }),
}


def screen_geometry(preset):
    if preset not in SCREEN_PRESETS:
        raise ValueError('Choose the shape of your monitor in gaze settings')
    value=SCREEN_PRESETS[preset][1]
    return dict(value) if value is not None else None
