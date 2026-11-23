# Connect an iPhone over Wi-Fi

Mira on iPhone uses TrueDepth to send eye and head measurements to the Windows
app. First [build and install the iPhone companion](../phone/README.md) on a
TrueDepth iPhone running iOS 17 or later.

## Connect and set up

1. Put the iPhone and PC on the same trusted local network. The PC can use
   Ethernet. Guest networks may block connections between devices.
2. Open Mira on iPhone, select **Wi-Fi** in the top-right menu and tap
   **Start Tracking**. Grant camera and local network access if prompted.
3. On the PC, open **Overview → Get started** and select **Find iPhone**.
   Then select **Set up** and complete the gaze check.
4. In **Overview**, select **Connect iPhone** when you want to use gaze control.

Wi-Fi is selected on a fresh install. If the PC is set to USB, open
**Settings → iPhone**, select **Wi-Fi · home network** and choose
**Save connection**. If discovery fails, enter the private IPv4 address shown
on the phone in the same panel and select **Save connection**.

To repeat the gaze check, use **Profile → Refine setup · about a minute**.
**Details** shows the last accepted profile. A failed attempt preserves it.

## While tracking

Keep Mira in the foreground. After about 20 seconds without a touch, Mira dims
the display. Tap the dimmed screen to restore brightness. Locking the phone,
putting the app in the background or selecting **Stop** ends tracking.

Wi-Fi sends numeric eye and head measurements, tracking status and frame
timing. This TCP connection has no authentication or encryption. Use it only
on a local network you trust. A device on that network may be able to request
the stream while tracking is active.

## Transport details

The PC requests the latest numeric frame over TCP port `39472` with one request
outstanding at a time. The phone keeps the latest frame in memory. The PC
allows up to 400 ms for a complete response and rejects samples aged 200 ms or
more. A duplicate frame keeps its original timestamp. USB and Wi-Fi are
selected explicitly and use separate listeners.

The Windows status shows round-trip request time and a conservative estimate
of frame age based on the age reported by the phone plus the full round trip.
End-to-end gaze-to-highlight latency also includes capture, gaze processing,
smoothing and Windows rendering.

The Windows protocol tests use a local TCP fixture.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_phone_wifi
```
