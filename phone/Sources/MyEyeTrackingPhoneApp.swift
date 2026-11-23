import SwiftUI
import UIKit
import Darwin
import ARKit
import AVFoundation
import Network
import QuartzCore
import Vision
import CoreImage

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
    @Published var rgbEnabled = false
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
    let plan = PlanModel()
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
    private let rgbEncoder = RGBFaceEncoder()
    private var rgbBusy = false
    private var lastRGB = 0.0
    // Live plan state; touched only on `queue`.
    private var planEnabled = true
    private var lastPlan = 0.0
    private var smoothPlan: PlanPose?

    func configureRGB() { server.setRGBEnabled(rgbEnabled) }

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
        server.onError = { [weak self] in
            DispatchQueue.main.async { self?.stop(); self?.fail("Couldn’t open the USB bridge") }
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
        wifiServer.onError = { [weak self] in
            DispatchQueue.main.async { self?.stop(); self?.fail("Couldn’t open Wi-Fi port 39472") }
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
        stop()
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
        queue.async { self.planEnabled = false }
    }

    func revealScreen() {
        if let previous = priorBrightness {
            UIScreen.main.brightness = previous
            priorBrightness = nil
        }
        screenDimmed = false
        dimGeneration += 1
        queue.async { self.planEnabled = true }
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
                    self.smoothPlan = nil
                    self.acceptingFrames = true
                }
                self.computerConnected = false
                self.depthReceived = false
                self.depthText = "Waiting for depth data…"
                if requestedTransport == .usb {
                    self.server.start()
                    self.server.setRGBEnabled(self.rgbEnabled)
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
        plan.pose = nil
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
        let depth = frame.capturedDepthData != nil
        if depth { depthFrames += 1 }
        let face = frame.anchors.compactMap { $0 as? ARFaceAnchor }.first
        let valid = face?.isTracked == true
        var packet: [String: Any] = ["version": 1, "session": sessionID, "seq": sequence,
                                    "timestamp": frame.timestamp, "tracked": valid,
                                    "trueDepth": trueDepthFormat, "depthFrame": depth]
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
        if activeTransport == .usb && valid && !rgbBusy && now-lastRGB >= 1.0/15.0 && server.wantsRGB() {
            rgbBusy = true
            lastRGB = now
            let frameSession = sessionID
            let numeric = packet
            rgbEncoder.encode(frame.capturedImage) { [weak self] jpeg in
                guard let self = self else { return }
                self.queue.async {
                    self.rgbBusy = false
                    guard self.acceptingFrames, self.sessionID == frameSession,
                          self.server.wantsRGB(), let jpeg = jpeg else { return }
                    var result = numeric
                    result["rgb"] = ["format": "jpeg", "width": 448, "height": 448,
                                     "orientation": "portrait-unmirrored", "timestamp": frame.timestamp,
                                     "data": jpeg.base64EncodedString()]
                    self.server.updateRGB(packet: result, captured: frame.timestamp)
                }
            }
        }
        if planEnabled && now-lastPlan >= 1.0/30.0 {
            lastPlan = now
            var next: PlanPose?
            if let face = face, valid {
                let view = frame.camera.viewMatrix(for: .portrait) * face.transform
                func position(_ m: simd_float4x4) -> SIMD3<Float> { SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z) }
                let look = view * SIMD4<Float>(face.lookAtPoint, 1)
                next = PlanPose.make(head: position(view),
                                     forward: SIMD3(view.columns.2.x, view.columns.2.y, view.columns.2.z),
                                     leftEye: position(view * face.leftEyeTransform),
                                     rightEye: position(view * face.rightEyeTransform),
                                     lookAt: SIMD3(look.x, look.y, look.z))
            }
            if let next = next, let previous = smoothPlan {
                smoothPlan = previous.blended(toward: next, amount: 0.35)
            } else {
                smoothPlan = next
            }
            let pose = smoothPlan
            DispatchQueue.main.async { [weak self] in
                guard let self = self, self.running else { return }
                self.plan.pose = pose
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

// RGB processing is off the ARKit queue; at most one frame is in flight.
final class RGBFaceEncoder {
    private let queue = DispatchQueue(label: "local.myeyetracking.rgb", qos: .userInitiated)
    private let context = CIContext(options: [.cacheIntermediates: false])
    func encode(_ buffer: CVPixelBuffer, completion: @escaping (Data?) -> Void) {
        queue.async {
            autoreleasepool {
                // Fixed portrait, unmirrored sensor orientation. Screen calibration
                // binds this convention; it must not change with preview mirroring.
                var image = CIImage(cvPixelBuffer: buffer).oriented(.right)
                let scale = min(1.0, 720.0/max(image.extent.width, image.extent.height))
                image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
                let request = VNDetectFaceRectanglesRequest()
                do { try VNImageRequestHandler(ciImage: image, orientation: .up).perform([request]) }
                catch { completion(nil); return }
                guard let faces = request.results, faces.count == 1, let face = faces.first,
                      face.confidence >= 0.7 else { completion(nil); return }
                let box = face.boundingBox
                var rect = CGRect(x: box.minX*image.extent.width, y: box.minY*image.extent.height,
                                  width: box.width*image.extent.width, height: box.height*image.extent.height)
                rect = rect.insetBy(dx: -rect.width*0.10, dy: -rect.height*0.10).intersection(image.extent)
                guard rect.width >= 100, rect.height >= 100 else { completion(nil); return }
                var crop = image.cropped(to: rect)
                crop = crop.transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY))
                crop = crop.transformed(by: CGAffineTransform(scaleX: 448/rect.width, y: 448/rect.height))
                crop = crop.cropped(to: CGRect(x: 0, y: 0, width: 448, height: 448))
                let data = self.context.jpegRepresentation(of: crop, colorSpace: CGColorSpaceCreateDeviceRGB(),
                    options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.88])
                completion(data)
            }
        }
    }
}

// Pull protocol: GET keeps the numeric path; GETRGB opts into face crops.
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
        var rgbRequestedUntil = 0.0
        init(_ connection: NWConnection) { self.connection = connection }
    }
    // A numeric reader and an optional RGB reader have independent demand/buffers.
    private var clients: [ObjectIdentifier: Client] = [:]
    private var latest: [String: Any]?
    private var captured = 0.0
    private var rgbEnabled = false
    private var latestRGB: [String: Any]?
    private var rgbCaptured = 0.0

    private func hasRGBDemand() -> Bool {
        let now = CACurrentMediaTime()
        return clients.values.contains { now < $0.rgbRequestedUntil }
    }
    func setRGBEnabled(_ enabled: Bool) {
        queue.async {
            self.rgbEnabled = enabled
            self.latestRGB = nil
            for client in self.clients.values { client.rgbRequestedUntil = 0 }
        }
    }
    func wantsRGB() -> Bool { queue.sync { rgbEnabled && hasRGBDemand() } }
    func updateRGB(packet: [String: Any], captured: Double) {
        queue.async {
            guard self.rgbEnabled, self.hasRGBDemand() else { return }
            self.latestRGB = packet
            self.rgbCaptured = captured
        }
    }

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
                            if !self.hasRGBDemand() { self.latestRGB = nil }
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
        latest = nil; latestRGB = nil
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
                let wantsRGB = request == Data("GETRGB\n".utf8)
                guard wantsRGB || request == Data("GET\n".utf8) else { connection.cancel(); return }
                var packet: [String: Any]
                var captured = self.captured
                if wantsRGB {
                    client.rgbRequestedUntil = CACurrentMediaTime()+0.75
                    packet = self.rgbEnabled ? (self.latestRGB ?? ["version": 1, "tracked": false, "rgbStatus": "warming"]) : ["version": 1, "rgbStatus": "disabled"]
                    captured = self.rgbCaptured
                } else {
                    client.rgbRequestedUntil = 0
                    if !self.hasRGBDemand() { self.latestRGB = nil }
                    packet = self.latest ?? ["version": 1, "tracked": false]
                }
                packet["parallelGazeStreams"] = true
                packet["ageMs"] = max(0, (CACurrentMediaTime()-captured)*1000)
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
