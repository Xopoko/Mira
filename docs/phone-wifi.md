# Connect an iPhone over Wi-Fi

The [README](../README.md#try-gaze-tracking) walks through the first run.
This guide covers the connection itself. First
[build and install the iPhone companion](../phone/README.md).

## Connect

1. Put the iPhone and PC on the same trusted local network. The PC can use
   Ethernet. Guest networks may block connections between devices.
2. Open Mira on iPhone, select **Wi-Fi** in the top-right menu and tap
   **Start Tracking**. Grant camera and local network access if prompted.
3. On the PC, select **Find iPhone** in **Overview → Get started** or in
   **Settings → iPhone**.

**Find iPhone** probes the 254 neighbouring addresses of each private IPv4
address the PC has. If the phone is elsewhere on a larger network, or discovery
fails, enter the address shown on the phone in **Settings → iPhone** and
select **Save connection**. If the PC is set to USB, select **Wi-Fi · home
network** in the same panel first.

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
