# Mira for iPhone

Mira uses the iPhone's TrueDepth camera to send gaze measurements to the
Windows app. Build and install the companion on a TrueDepth iPhone running
iOS 17 or later.

<img src="../docs/images/mira-iphone.png" width="280" alt="Mira's iPhone companion ready to start tracking in the iOS Simulator">

## Build and install

On macOS, install Xcode and XcodeGen. From the repository root, run these
commands.

```sh
cd phone
xcodegen generate
open MyEyeTrackingPhone.xcodeproj
```

In Xcode, select the **MyEyeTrackingPhone** application target. Under
**Signing & Capabilities**, choose your own development team with automatic
signing. If your team requires a different bundle identifier, update
`PRODUCT_BUNDLE_IDENTIFIER` in `phone/project.yml` and regenerate the project.
Select your connected iPhone and run the app.

Follow the [Windows connection guide](../docs/phone-wifi.md) once the app is
installed. The first tracking session may request camera and local network
access.

## Connection and data

Wi-Fi is the default for new installs. It serves numeric gaze measurements on
TCP port `39472` over the private local network. The connection has no
authentication or encryption. Restrict it to a trusted network.

USB is an optional mode for an existing paired USB bridge on Windows. Set
`MIRA_IPHONE_USB_BRIDGE` to the path of its trusted Python adapter exporting
`open_stream()`. USB uses a loopback listener on port `39471`. Enabling RGB in
USB mode allows the bridge to request an in-memory face crop. Stop or
backgrounding closes the active listener and stops ARKit.

## Checks

From the repository root on macOS, the following checks run without signing,
an iPhone or camera access.

```sh
python3 scripts/check_phone_build.py
python3 scripts/check_phone_native.py
```

To run the Swift unit tests in the simulator, use the same scheme as CI.

```sh
cd phone
xcodegen generate
xcodebuild test -scheme MyEyeTrackingPhone \
  -destination 'platform=iOS Simulator,name=iPhone 17,OS=latest' \
  -parallel-testing-enabled NO CODE_SIGNING_ALLOWED=NO
```

Replace `iPhone 17` with the name of an installed simulator as needed.
