"""Locate optional local model integrations without loading any model."""

import os
from pathlib import Path

from ..settings import LOCAL, ROOT


_configured_owner = os.environ.get('MIRA_LOCAL_MODELS_ROOT')
OWNER = Path(_configured_owner).expanduser().resolve() if _configured_owner else ROOT.parent / 'LocalModels'


def owner_is_absent():
    try:
        OWNER.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        pass  # An unreadable installation must be handled by its policy.
    return False


STANDALONE = (not _configured_owner and os.environ.get('MIRA_ENABLE_STANDALONE_MODELS') == '1'
              and owner_is_absent())
RUNTIME_ROOT = LOCAL if STANDALONE else OWNER / 'local'
