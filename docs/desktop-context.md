# Desktop map

Mira combines observations from visible windows into one desktop map. The blue
highlight can span applications while the yellow outline marks a single action
target. All coordinates are in physical pixels on the virtual desktop, which
can have a negative origin on multi-monitor setups.

## Windows UI Automation

Three independent COM MTA workers read UI Automation properties in batches with
`FindAllBuildCache`. Query scheduling favors windows near the point of
attention. Each window keeps its own observation time and identity. A result
expires after 1.25 seconds and is discarded if its process or bounds have
changed.
Covered fragments are clipped out. The center of an element must be visible
before it can be used for a click.

Accessibility coverage depends on the target app. UI Automation connection and
transaction timeouts bound individual calls. When Windows or an application
responds slowly, outdated elements expire while the remaining workers continue
reading other windows.

## Optional visual recognition

When configured, visual recognition supplements the map in visible window
regions. Before taking a crop, Mira checks the window identity, geometry and
occlusion. It skips windows whose titles match its protected-window list. It
also blocks captures of a window when the UI Automation scan reports a visible
password field there or cannot complete that check. This relies on what the
application reports to UI Automation.

Recognized objects can be reused while their source pixels and a small padding
region still match. The in-memory cache stores hashes and recognition results.
It is scoped by window identity, geometry and command epoch.

Run the desktop map unit tests from the repository root.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_desktop tests.test_uia_reader tests.test_visual_regions
```

See [UI Automation caching](https://learn.microsoft.com/windows/win32/winauto/uiauto-cachingforclients),
[threading](https://learn.microsoft.com/windows/win32/winauto/uiauto-threading)
and [transaction timeouts](https://learn.microsoft.com/windows/win32/api/uiautomationclient/nf-uiautomationclient-iuiautomation2-put_transactiontimeout)
in Microsoft's documentation.
