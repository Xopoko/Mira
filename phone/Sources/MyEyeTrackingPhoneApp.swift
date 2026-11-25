import SwiftUI
import UIKit
import ARKit
import AVFoundation
import Network
import QuartzCore
import simd

@main
struct MyEyeTrackingPhoneApp: App {
    @StateObject private var tracker = EyeTracker()
    @Environment(\.scenePhase) private var phase
    @State private var resumePending = ProcessInfo.processInfo.arguments.contains("--resume-tracking")
    private func resumeIfRequested() {
        guard resumePending, UIApplication.shared.applicationState == .active,
              AVCaptureDevice.authorizationStatus(for: .video) == .authorized else { return }
        resumePending = false
        if !tracker.running { tracker.start() }
    }

    var body: some Scene {
        WindowGroup {
            PhoneView(tracker: tracker)
                .onAppear { tracker.enterForeground(); resumeIfRequested() }
                .onChange(of: phase) { _, value in
                    if value == .active {
                        tracker.enterForeground()
                        resumeIfRequested()
                    } else if value == .background {
                        tracker.leaveForeground()
                    } else {
                        tracker.revealScreen()
                    }
                }
        }
    }
}

enum PhoneTransport: String, CaseIterable, Identifiable, Hashable {
    case usb
    case wifi
    var id: String { rawValue }
    var title: String { self == .usb ? "USB" : "Wi-Fi" }
}

final class EyeTracker: NSObject, ObservableObject, ARSessionDelegate {
    // The connection choice survives relaunches, so iPhone and PC stay on the same transport.
    @Published var transport: PhoneTransport = EyeTracker.savedTransport {
        didSet {
            UserDefaults.standard.set(transport.rawValue, forKey: EyeTracker.transportKey)
            refreshWiFiEndpoint()
        }
    }
    private static let transportKey = "connection.transport"
    private static var savedTransport: PhoneTransport {
        PhoneTransport(rawValue: UserDefaults.standard.string(forKey: transportKey) ?? "") ?? .wifi
    }
    @Published var wifiEndpoint = ""
    @Published var running = false
    @Published var screenDimmed = false
    @Published var dimGeneration = 0
    @Published var tracked = false
    @Published var computerConnected = false
    @Published var depthReceived = false
    @Published var failure: String?
    @Published var status = "Ready"
    @Published var depthText = "Not checked yet"
    @Published var rateText = "Off"
    @Published var connectionText = "Not connected"
    let preview = PreviewModel()
    private let session = ARSession()
    private let server = USBServer()
    private let wifiServer = WiFiNumericServer()
    private let queue = DispatchQueue(label: "local.myeyetracking.arkit")
    private var sessionID = ""
    private var sequence = 0
    private var lastUI = 0.0
    private var lastFrame = 0.0
    private var depthFrames = 0
    private var trueDepthFormat = false
    private var acceptingFrames = false
    private var activeTransport: PhoneTransport = .usb
    private var priorIdleTimerDisabled: Bool?
    private var priorBrightness: CGFloat?
    private var startGeneration = 0
    // Live preview state; touched only on `queue`.
    private var previewEnabled = true
    private var lastPreview = 0.0
    private var smoothPreview: PreviewPose?

    var cameraBlocked: Bool {
        let access = AVCaptureDevice.authorizationStatus(for: .video)
        return access == .denied || access == .restricted
    }

    private func fail(_ message: String) {
        status = message
        failure = message
    }

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = queue
        server.onConnection = { [weak self] connected in
            DispatchQueue.main.async {
                guard let self = self, self.running, self.activeTransport == .usb else { return }
                self.computerConnected = connected
                self.connectionText = connected ? "Connected over USB" : "Waiting for the USB bridge…"
            }
        }
        // A failure that arrives after tracking stopped, for example while the app
        // leaves the foreground, must not leave a stale error on the screen.
        server.onError = { [weak self] in
            DispatchQueue.main.async {
                guard let self = self, self.running, self.activeTransport == .usb else { return }
                self.stop()
                self.fail("Couldn’t open the USB bridge")
            }
        }
        wifiServer.onEndpoint = { [weak self] endpoint in
            DispatchQueue.main.async { self?.wifiEndpoint = endpoint ?? "" }
        }
        wifiServer.onConnection = { [weak self] connected in
            DispatchQueue.main.async {
                guard let self = self, self.running, self.activeTransport == .wifi else { return }
                self.computerConnected = connected
                self.connectionText = connected ? "Connected over Wi-Fi" : "Waiting over Wi-Fi…"
            }
        }
        wifiServer.onError = { [weak self] reason in
            DispatchQueue.main.async {
                guard let self = self, self.running, self.activeTransport == .wifi else { return }
                self.stop()
                self.fail("Couldn’t open Wi-Fi port \(WiFiGazePolicy.port) (\(reason)). Tap Start Tracking to try again.")
            }
        }
        refreshWiFiEndpoint()
    }

    func enterForeground() {
        if priorIdleTimerDisabled == nil {
            priorIdleTimerDisabled = UIApplication.shared.isIdleTimerDisabled
        }
        UIApplication.shared.isIdleTimerDisabled = true
        revealScreen()
        refreshWiFiEndpoint()
    }

    func leaveForeground() {
        // Network teardown finishes asynchronously; keep running for a moment so the
        // listeners and connections close before iOS suspends the app.
        var task = UIBackgroundTaskIdentifier.invalid
        task = UIApplication.shared.beginBackgroundTask(withName: "Close connections") {
            UIApplication.shared.endBackgroundTask(task)
            task = .invalid
        }
        stop()
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            guard task != .invalid else { return }
            UIApplication.shared.endBackgroundTask(task)
            task = .invalid
        }
        if let previous = priorIdleTimerDisabled {
            UIApplication.shared.isIdleTimerDisabled = previous
            priorIdleTimerDisabled = nil
        }
    }

    private func refreshWiFiEndpoint() {
        wifiEndpoint = WiFiNumericServer.privateLANEndpoint ?? ""
    }

    func dimScreen() {
        guard running, !screenDimmed, UIApplication.shared.applicationState == .active else { return }
        priorBrightness = UIScreen.main.brightness
        UIScreen.main.brightness = min(UIScreen.main.brightness, 0.01)
        screenDimmed = true
        queue.async { self.previewEnabled = false }
    }

    func revealScreen() {
        if let previous = priorBrightness {
            UIScreen.main.brightness = previous
            priorBrightness = nil
        }
        screenDimmed = false
        dimGeneration += 1
        queue.async { self.previewEnabled = true }
    }

    func start() {
        guard !running else { return }
        refreshWiFiEndpoint()
        failure = nil
        startGeneration += 1
        let requestGeneration = startGeneration
        guard ARFaceTrackingConfiguration.isSupported else {
            fail("Face tracking isn’t supported on this iPhone")
            return
        }
        let requestedTransport = transport
        if requestedTransport == .wifi && wifiEndpoint.isEmpty {
            fail("Connect to a private Wi-Fi network")
            return
        }
        AVCaptureDevice.requestAccess(for: .video) { [weak self] allowed in
            DispatchQueue.main.async {
                guard let self = self, self.startGeneration == requestGeneration, !self.running else { return }
                guard allowed else { self.fail("Allow camera access in Settings"); return }
                guard UIApplication.shared.applicationState == .active else { return }
                let config = ARFaceTrackingConfiguration()
                let formats = ARFaceTrackingConfiguration.supportedVideoFormats.filter { $0.captureDeviceType == .builtInTrueDepthCamera }
                guard let format = formats.sorted(by: { $0.framesPerSecond > $1.framesPerSecond }).first else {
                    self.fail("The TrueDepth camera isn’t available")
                    return
                }
                config.videoFormat = format
                config.maximumNumberOfTrackedFaces = 1
                config.isLightEstimationEnabled = true
                self.queue.sync {
                    self.activeTransport = requestedTransport
                    self.trueDepthFormat = true
                    self.sessionID = UUID().uuidString
                    self.sequence = 0
                    self.depthFrames = 0
                    self.lastFrame = 0
                    self.smoothPreview = nil
                    self.acceptingFrames = true
                }
                self.computerConnected = false
                self.depthReceived = false
                self.depthText = "Waiting for depth data…"
                if requestedTransport == .usb {
                    self.server.start()
                    self.wifiServer.stop()
                    self.connectionText = "Waiting for the USB bridge…"
                } else {
                    self.server.stop()
                    self.wifiServer.start()
                    self.connectionText = "Waiting over Wi-Fi…"
                }
                self.running = true
                self.status = "Get comfortable in front of the camera"
                self.dimGeneration += 1
                self.session.run(config, options: [.resetTracking, .removeExistingAnchors])
            }
        }
    }

    func stop() {
        startGeneration += 1
        session.pause()
        queue.sync { acceptingFrames = false }
        server.stop()
        wifiServer.stop()
        running = false
        tracked = false
        computerConnected = false
        depthReceived = false
        preview.pose = nil
        status = "Tracking stopped"
        rateText = "Off"
        depthText = "Not checked yet"
        connectionText = "Connection closed"
        revealScreen()
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard acceptingFrames else { return }
        sequence += 1
        let now = CACurrentMediaTime()
        if frame.capturedDepthData != nil { depthFrames += 1 }
        let face = frame.anchors.compactMap { $0 as? ARFaceAnchor }.first
        let valid = face?.isTracked == true
        var packet: [String: Any] = ["version": 1, "session": sessionID, "seq": sequence,
                                    "timestamp": frame.timestamp, "tracked": valid,
                                    "trueDepth": trueDepthFormat]
        if let face = face, valid {
            let transform = simd_inverse(frame.camera.transform) * face.transform
            let left = transform * face.leftEyeTransform
            let right = transform * face.rightEyeTransform
            let look = transform * SIMD4<Float>(face.lookAtPoint, 1)
            packet["face"] = flatten(transform)
            packet["leftEye"] = flatten(left)
            packet["rightEye"] = flatten(right)
            packet["lookAt"] = [look.x, look.y, look.z]
            packet["blink"] = [face.blendShapes[.eyeBlinkLeft]?.floatValue ?? 0,
                               face.blendShapes[.eyeBlinkRight]?.floatValue ?? 0]
        }
        if activeTransport == .wifi {
            wifiServer.update(packet: packet, captured: frame.timestamp)
        } else {
            server.update(packet: packet, captured: frame.timestamp)
        }
        if previewEnabled && now-lastPreview >= 1.0/30.0 {
            lastPreview = now
            var next: PreviewPose?
            if let face = face, valid {
                let view = frame.camera.viewMatrix(for: .portrait) * face.transform
                func position(_ m: simd_float4x4) -> SIMD3<Float> { SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z) }
                let look = view * SIMD4<Float>(face.lookAtPoint, 1)
                next = PreviewPose.make(head: position(view),
                                        up: SIMD3(view.columns.1.x, view.columns.1.y, view.columns.1.z),
                                        leftEye: position(view * face.leftEyeTransform),
                                        rightEye: position(view * face.rightEyeTransform),
                                        lookAt: SIMD3(look.x, look.y, look.z))
            }
            if let next = next, let previous = smoothPreview {
                smoothPreview = previous.blended(toward: next, amount: 0.35)
            } else {
                smoothPreview = next
            }
            let pose = smoothPreview
            DispatchQueue.main.async { [weak self] in
                guard let self = self, self.running else { return }
                self.preview.pose = pose
            }
        }
        if now-lastUI > 0.2 {
            let fps = lastFrame > 0 ? 1/max(0.001, frame.timestamp-lastFrame) : 0
            let depthCount = depthFrames
            DispatchQueue.main.async { [weak self] in
                guard let self = self, self.running else { return }
                self.tracked = valid
                self.status = valid ? "Tracking head and eyes" : "Face out of view"
                self.rateText = EyeTracker.perSecondText(fps)
                self.depthReceived = depthCount > 0
                self.depthText = depthCount > 0 ? "Depth received" : "Waiting for depth data…"
            }
            lastUI = now
        }
        lastFrame = frame.timestamp
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async { [weak self] in self?.stop(); self?.fail("Tracking stopped unexpectedly. Start again.") }
    }
    func sessionWasInterrupted(_ session: ARSession) {
        DispatchQueue.main.async { [weak self] in self?.stop(); self?.fail("The camera is temporarily unavailable") }
    }
    static func perSecondText(_ fps: Double) -> String {
        "\(Int(fps.rounded())) per second"
    }
    private func flatten(_ m: simd_float4x4) -> [Float] {
        (0..<4).flatMap { c in (0..<4).map { r in m[c][r] } }
    }
}

// Only loopback is bound. usbmuxd forwards the exact phone's port through USB.
final class USBServer {
    private let port: UInt16 = 39471
    var onConnection: ((Bool) -> Void)?
    var onError: (() -> Void)?
    private let queue = DispatchQueue(label: "local.myeyetracking.usb")
    private var listener: NWListener?
    private final class Client {
        let connection: NWConnection
        var buffer = Data()
        init(_ connection: NWConnection) { self.connection = connection }
    }
    private var clients: [ObjectIdentifier: Client] = [:]
    private var latest: [String: Any]?
    private var captured = 0.0

    func start() {
        queue.async { [self] in
            self.close()
            do {
                let parameters = NWParameters.tcp
                parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: self.port)!)
                let listener = try NWListener(using: parameters)
                listener.stateUpdateHandler = { [weak self] state in
                    if case .failed = state { self?.onError?() }
                }
                listener.newConnectionHandler = { [weak self] connection in
                    guard let self = self else { connection.cancel(); return }
                    guard self.clients.count < 2 else { connection.cancel(); return }
                    let id = ObjectIdentifier(connection)
                    self.clients[id] = Client(connection)
                    connection.stateUpdateHandler = { [weak self, weak connection] state in
                        guard let self = self, let connection = connection,
                              self.clients[id]?.connection === connection else { return }
                        switch state {
                        case .ready: self.onConnection?(true); self.receive(connection)
                        case .failed, .cancelled:
                            self.clients.removeValue(forKey: id)
                            connection.cancel()
                            self.onConnection?(!self.clients.isEmpty)
                        default: break
                        }
                    }
                    connection.start(queue: self.queue)
                }
                self.listener = listener
                listener.start(queue: self.queue)
            } catch { self.onError?() }
        }
    }
    func update(packet: [String: Any], captured: Double) {
        queue.async { self.latest = packet; self.captured = captured }
    }
    func stop() { queue.async { self.close() } }
    private func close() {
        listener?.cancel(); listener = nil
        let previous = clients.values.map { $0.connection }
        clients.removeAll()
        for connection in previous { connection.cancel() }
        latest = nil
        onConnection?(false)
    }
    private func receive(_ connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64) { [weak self, weak connection] data, _, complete, error in
            guard let self = self, let connection = connection,
                  let client = self.clients[ObjectIdentifier(connection)] else { return }
            if let data = data { client.buffer.append(data) }
            if complete || error != nil || client.buffer.count > 64 {
                connection.cancel(); return
            }
            while let end = client.buffer.firstIndex(of: 10) {
                let request = Data(client.buffer.prefix(through: end))
                client.buffer.removeSubrange(...end)
                guard request == Data("GET\n".utf8) else { connection.cancel(); return }
                var packet = self.latest ?? ["version": 1, "tracked": false]
                packet["ageMs"] = max(0, (CACurrentMediaTime()-self.captured)*1000)
                guard JSONSerialization.isValidJSONObject(packet),
                      let data = try? JSONSerialization.data(withJSONObject: packet, options: [.sortedKeys]) else {
                    connection.cancel(); return
                }
                connection.send(content: data+Data([10]), completion: .contentProcessed { error in
                    if error != nil { connection.cancel() }
                })
            }
            self.receive(connection)
        }
    }
}
