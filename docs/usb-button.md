# Confirm actions with a USB button

1. Open **Settings → Confirm actions → Bind button**.
2. With the binding dialog active, press and release **Enter** on your main
   keyboard.
3. Press and release the separate USB button.

The two devices must have distinct identities in Windows. Press **Esc** to keep
the previous binding. Holding the USB button triggers at most one confirmation
per press. Mira checks that the yellow action target is still fresh and visible
before clicking. Pause or profile setup blocks confirmation.

Mira uses Windows Raw Input, so both devices continue sending their normal keys
to the active application. If your button sends **Enter**, the active
application can still receive **Enter**. A programmable button can use **F13**
through **F24** to avoid common key bindings. Mira stores a hash of the device
identity and the selected key, along with the enable setting.

See [Windows Raw Input](https://learn.microsoft.com/windows/win32/inputdev/about-raw-input)
for device identification and event delivery details.
