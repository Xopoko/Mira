"""Low-latency explicit controls. Unmatched phrases continue to the intent classifier."""
import re

from .contracts import Intent


# Unicode escapes preserve the Russian command vocabulary in an English source file.
DIRECTIONS = {
    'left': ('move left', '\u043b\u0435\u0432\u0435\u0435', '\u0441\u043b\u0435\u0432\u0430'),
    'right': ('move right', '\u043f\u0440\u0430\u0432\u0435\u0435', '\u0441\u043f\u0440\u0430\u0432\u0430'),
    'up': ('move up', '\u0432\u044b\u0448\u0435', '\u0441\u0432\u0435\u0440\u0445\u0443'),
    'down': ('move down', '\u043d\u0438\u0436\u0435', '\u0441\u043d\u0438\u0437\u0443'),
}
DICTATION = (
    r'^(?:type|\u043d\u0430\u043f\u0438\u0448\u0438|\u043d\u0430\u043f\u0435\u0447\u0430\u0442\u0430\u0439|\u0432\u0432\u0435\u0434\u0438|\u0434\u0438\u043a\u0442\u0443\u0439)\s*[: ]\s*(.+)$'
)
SCROLL = r'(?:scroll|\u043f\u0440\u043e\u043a\u0440\u0443\u0442\u0438|\u043b\u0438\u0441\u0442\u0430\u0439|\u0441\u043a\u0440\u043e\u043b\u043b)\s+'
CHORD = r'(?:(?:press|\u043d\u0430\u0436\u043c\u0438)\s+)?((?:ctrl|control|alt|shift|win)[+ ]\S+(?:[+]\S+)*)'


def quick_intent(text):
    t = text.strip().strip('.!?').casefold()
    if t in ('stop', 'pause', '\u0441\u0442\u043e\u043f', '\u043e\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0441\u044c', '\u043f\u0430\u0443\u0437\u0430'):
        return Intent('pause')
    if t in ('resume', '\u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0430\u0439', '\u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0438\u0442\u044c', '\u0432\u043e\u0437\u043e\u0431\u043d\u043e\u0432\u0438'):
        return Intent('resume')
    if t in (
        'click', '\u043d\u0430\u0436\u043c\u0438', '\u043d\u0430\u0436\u0430\u0442\u044c', '\u043a\u043b\u0438\u043a', '\u043a\u043b\u0438\u043a\u043d\u0438', '\u0441\u044e\u0434\u0430',
        '\u0432\u043e\u0442 \u0441\u044e\u0434\u0430', '\u043d\u0430\u0436\u043c\u0438 \u0441\u044e\u0434\u0430', '\u043d\u0430\u0436\u043c\u0438 \u044d\u0442\u043e',
    ):
        return Intent('click')
    if t in ('double click', '\u0434\u0432\u043e\u0439\u043d\u043e\u0439 \u043a\u043b\u0438\u043a', '\u0434\u0432\u0430\u0436\u0434\u044b'):
        return Intent('double_click')
    if t in ('right click', '\u043f\u0440\u0430\u0432\u044b\u0439 \u043a\u043b\u0438\u043a', '\u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u043d\u043e\u0435 \u043c\u0435\u043d\u044e'):
        return Intent('right_click')
    if t in ('next', '\u0434\u0440\u0443\u0433\u043e\u0439', '\u0434\u0440\u0443\u0433\u0430\u044f', '\u0434\u0440\u0443\u0433\u043e\u0435', '\u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0439', '\u043d\u0435 \u044d\u0442\u043e\u0442'):
        return Intent('select', direction='next')
    for direction, variants in DIRECTIONS.items():
        if t in variants:
            return Intent('select', direction=direction)
    if t in ('back', '\u043d\u0430\u0437\u0430\u0434'):
        return Intent('press_keys', keys=('Alt', 'Left'))
    if t in ('escape', '\u043e\u0442\u043c\u0435\u043d\u0430', '\u044d\u0441\u043a\u0435\u0439\u043f'):
        return Intent('press_keys', keys=('Escape',))
    if t in ('enter', '\u044d\u043d\u0442\u0435\u0440', '\u0432\u0432\u043e\u0434'):
        return Intent('press_keys', keys=('Enter',))
    match = re.match(DICTATION, text, re.I | re.S)
    if match:
        return Intent('type_text', text=match[1])
    if re.fullmatch(SCROLL + r'(?:down|\u0432\u043d\u0438\u0437)', t):
        return Intent('scroll', direction='down')
    if re.fullmatch(SCROLL + r'(?:up|\u0432\u0432\u0435\u0440\u0445)', t):
        return Intent('scroll', direction='up')
    match = re.fullmatch(CHORD, t)
    if match:
        keys = tuple(x.title().replace('Control', 'Ctrl') for x in re.split(r'[+ ]', match[1]))
        return Intent('press_keys', keys=keys)
    return None
