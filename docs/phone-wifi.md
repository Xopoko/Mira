# Connect an iPhone over Wi-Fi

The [README](../README.md#try-gaze-tracking) walks through the first run.
This guide covers the connection itself. First
[build and install the iPhone companion](../phone/README.md).

## Connect

1. Put the iPhone and PC on the same trusted local network. The PC can use
   Ethernet. Guest networks may block connections between devices.
2. Open Mira on iPhone and tap **Start Tracking**. Grant camera and local
   network access if prompted. If you previously set the companion to USB,
   switch it back to **Wi-Fi** in the top-right menu first.
3. On the PC, select **Set up iPhone** in **Overview**, then select
   **Connect iPhone** in the setup window. If you already have a gaze
   profile, select **Connect iPhone** directly in **Overview**. Mira looks
   for the phone automatically over Wi-Fi.

## Enter the address manually

If Mira can’t find your iPhone, close the setup window if it’s open and select
**Connection settings** in **Overview**. In **Settings → iPhone**, select
**Wi‑Fi · home network**, enter the address shown on your phone, and select
**Save connection**. Return to **Overview** and try again.

## While tracking

Keep Mira in the foreground. After about 20 seconds without a tap, Mira dims
the display. Tap the dimmed screen to restore brightness. Locking the phone,
putting the app in the background or selecting **Stop** ends tracking. When
you return, tap **Start Tracking**. If the Wi-Fi port fails to open, Mira
retries up to 30 times, one second apart, before it shows the reason.

The phone sends numeric eye and head measurements, tracking status and frame
timing. This TCP connection has no authentication or encryption. Use it only
on a local network you trust. A device on that network can request the stream
while tracking is active.

## Transport details

Discovery probes the neighbouring addresses in the `/24` network of each private
IPv4 address on the PC. On a larger network, enter the phone’s address manually
if discovery misses it.

The PC requests the latest frame over TCP port `39472`, one request at a time.
The phone keeps only the latest frame in memory and reports its age with each
reply. The PC allows 400 ms for a complete response. It uses a frame only if
the reported age is at most 150 ms and that age plus the full request round
trip is under 200 ms. A repeated frame keeps its original timestamp. USB and
Wi-Fi are selected explicitly and use separate listeners.

Hover over the iPhone status in **Overview** to see the round trip and the
frame age. Gaze-to-highlight latency also includes capture, gaze processing,
smoothing and Windows rendering.

The Windows protocol tests use a local TCP fixture.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_phone_wifi
```
