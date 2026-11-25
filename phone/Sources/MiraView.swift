import SwiftUI
import UIKit
import simd

// Charcoal layers and one blue accent, the same values as the Windows app.
enum Palette {
    static let ground = Color(hex: 0x16181C)
    static let card = Color(hex: 0x1F2228)
    static let inset = Color(hex: 0x111317)
    static let line = Color(hex: 0x2E333B)
    static let outline = Color(hex: 0x5A626E)
    static let text = Color(hex: 0xE9EBEE)
    static let muted = Color(hex: 0xA4ABB6)
    static let accent = Color(hex: 0x3D8BFF)
    static let accentFill = Color(hex: 0x2F6FE4)
    static let waiting = Color(hex: 0xF5B544)
    static let failed = Color(hex: 0xFF6B5E)
    // The gaze light, gaze dot and focus frame of the Windows overlay.
    static let light = Color(hex: 0x78CAFF)
    static let gazeDot = Color(hex: 0xBFEFFF)
    static let focus = Color(hex: 0xF5CD69)
}

extension Color {
    init(hex: UInt32) {
        self.init(red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255)
    }
}

// What the screen says in one word and one sentence. Derived only from tracker state.
struct PhoneStatus: Equatable {
    enum Level: Equatable { case off, waiting, live, failed }
    let word: String
    let detail: String
    let level: Level

    static func resolve(running: Bool, connected: Bool, tracked: Bool, failure: String?, status: String) -> PhoneStatus {
        if !running {
            if let failure { return PhoneStatus(word: "Not tracking", detail: failure, level: .failed) }
            return PhoneStatus(word: "Ready", detail: "Stand iPhone under your screen with the front camera facing you, then tap Start Tracking.",
                               level: .off)
        }
        if !connected {
            return PhoneStatus(word: "Waiting for your PC", detail: "Open Mira on your PC and select Set up iPhone or Connect iPhone.",
                               level: .waiting)
        }
        if tracked {
            return PhoneStatus(word: "Tracking", detail: "Your PC is connected. Leave iPhone where it is.", level: .live)
        }
        return PhoneStatus(word: "Looking for you", detail: status, level: .waiting)
    }
}

// Where the person is and where they look, in meters on the plane of the iPhone camera.
// Left and right come from the eye transforms, so camera axis mirroring cannot flip it.
// On iPhone Air (2026-11-23) ARKit's left-eye transform sat on the person's right:
// moving left drew the head moving right, so the axis runs from the left-eye
// transform toward the right-eye transform. Up comes from the face's own up axis.
struct PreviewPose: Equatable {
    var depth: Double
    // x to the person's right, y up from the camera. Nil when they look away from the camera plane.
    var gaze: SIMD2<Double>?

    // The drawn monitor: an assumed screen standing just above the iPhone.
    static let screenHalfWidth = 0.30
    static let screenBottom = 0.02
    static let screenHeight = 0.34

    static func make(head: SIMD3<Float>, up: SIMD3<Float> = SIMD3(0, 1, 0), leftEye: SIMD3<Float>,
                     rightEye: SIMD3<Float>, lookAt: SIMD3<Float>) -> PreviewPose? {
        let eyes = rightEye - leftEye
        let horizontalIsX = abs(eyes.x) >= abs(eyes.y)
        let leftward: Float = (horizontalIsX ? eyes.x : eyes.y) >= 0 ? 1 : -1
        let upSign: Float = (horizontalIsX ? up.y : up.x) >= 0 ? 1 : -1
        func towardRight(_ v: SIMD3<Float>) -> Float { -leftward * (horizontalIsX ? v.x : v.y) }
        func upward(_ v: SIMD3<Float>) -> Float { upSign * (horizontalIsX ? v.y : v.x) }
        let depth = abs(head.z)
        guard depth > 0.1, depth < 3, simd_length(eyes) > 0.01 else { return nil }
        let toCamera: Float = head.z < 0 ? 1 : -1
        let center = (leftEye + rightEye) / 2
        let direction = lookAt - center
        var gaze: SIMD2<Double>?
        if direction.z * toCamera > 1e-4 {
            let hit = center + (-center.z / direction.z) * direction
            gaze = SIMD2(Double(towardRight(hit)), Double(upward(hit)))
        }
        return PreviewPose(depth: Double(depth), gaze: gaze)
    }

    func blended(toward next: PreviewPose, amount: Double) -> PreviewPose {
        var gaze = next.gaze
        if let a = self.gaze, let b = next.gaze { gaze = a + (b - a) * amount }
        return PreviewPose(depth: depth + (next.depth - depth) * amount, gaze: gaze)
    }

    // The gaze on the drawn monitor, as fractions from its top left corner.
    var screenPoint: (x: Double, y: Double, onScreen: Bool)? {
        guard let gaze else { return nil }
        let x = 0.5 + gaze.x / (2 * Self.screenHalfWidth)
        let y = 1 - (gaze.y - Self.screenBottom) / Self.screenHeight
        return (min(max(x, 0), 1), min(max(y, 0), 1), (0...1).contains(x) && (0...1).contains(y))
    }

    // Where the gaze lands, in words, for VoiceOver.
    var gazeDescription: String {
        guard let gaze else { return "away from the screen" }
        if gaze.y > Self.screenBottom + Self.screenHeight + 0.05 { return "above the screen" }
        if gaze.y < Self.screenBottom - 0.05 { return "below the screen" }
        if gaze.x < -Self.screenHalfWidth { return "past the left edge of the screen" }
        if gaze.x > Self.screenHalfWidth { return "past the right edge of the screen" }
        if gaze.x < -0.08 { return "at the left of the screen" }
        if gaze.x > 0.08 { return "at the right of the screen" }
        return "at the middle of the screen"
    }
}

final class PreviewModel: ObservableObject {
    @Published var pose: PreviewPose?
}

struct PhoneView: View {
    @ObservedObject var tracker: EyeTracker
    // A pushed page is being read. The screen must not dim under the reader.
    @State private var reading = false
    @Environment(\.openURL) private var openURL

    private var status: PhoneStatus {
        PhoneStatus.resolve(running: tracker.running, connected: tracker.computerConnected, tracked: tracker.tracked,
                            failure: tracker.failure, status: tracker.status)
    }

    private var connectionSummary: String {
        tracker.transport == .wifi && !tracker.wifiEndpoint.isEmpty
            ? "Wi-Fi · \(tracker.wifiEndpoint)" : tracker.transport.title
    }

    var body: some View {
        NavigationStack {
            List {
                Group {
                    Section {
                        VStack(alignment: .leading, spacing: 16) {
                            GazeScene(preview: tracker.preview, level: status.level)
                                .frame(height: 176)
                            VStack(alignment: .leading, spacing: 4) {
                                Text(status.word)
                                    .font(.title2.bold())
                                    .foregroundStyle(status.level == .failed ? Palette.failed : Palette.text)
                                Text(status.detail)
                                    .font(.subheadline)
                                    .foregroundStyle(Palette.muted)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            .accessibilityElement(children: .combine)
                        }
                        .padding(.vertical, 8)
                        .onTapGesture { tracker.revealScreen() }
                    }
                    Section {
                        StatusRow(level: tracker.tracked ? .live : (tracker.running ? .waiting : .off), name: "iPhone",
                                  value: tracker.running ? (tracker.tracked ? "Eyes and head tracked" : "Looking for your face") : "Camera off")
                            .onTapGesture { tracker.revealScreen() }
                        StatusRow(level: tracker.computerConnected ? .live : (tracker.running ? .waiting : .off),
                                  name: "Windows PC", value: tracker.connectionText)
                            .onTapGesture { tracker.revealScreen() }
                    }
                    Section {
                        NavigationLink {
                            ConnectionPage(tracker: tracker)
                                .onAppear { reading = true }
                                .onDisappear { reading = false }
                        } label: {
                            LabeledContent("Connection", value: connectionSummary)
                        }
                        NavigationLink("What Mira Sends") {
                            PrivacyPage()
                                .onAppear { reading = true }
                                .onDisappear { reading = false }
                        }
                    }
                }
                .miraRows()
            }
            .miraList()
            .navigationTitle("Mira")
            .safeAreaInset(edge: .bottom) {
                primaryAction
                    .padding(.horizontal, 16)
                    .padding(.top, 10)
                    .padding(.bottom, 6)
                    .background(Palette.ground)
                    .overlay(alignment: .top) { Rectangle().fill(Palette.line).frame(height: 1) }
            }
        }
        .tint(Palette.accent)
        .preferredColorScheme(.dark)
        .overlay {
            if tracker.screenDimmed {
                Button { tracker.revealScreen() } label: {
                    Color.black.ignoresSafeArea().contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Restore screen brightness")
            }
        }
        .onChange(of: reading) { _, _ in tracker.revealScreen() }
        .task(id: tracker.dimGeneration) {
            guard tracker.running, !reading else { return }
            do { try await Task.sleep(for: .seconds(20)) }
            catch { return }
            guard tracker.running, !reading else { return }
            tracker.dimScreen()
        }
    }

    @ViewBuilder private var primaryAction: some View {
        if !tracker.running && tracker.failure != nil && tracker.cameraBlocked {
            Button("Open Settings") {
                if let url = URL(string: UIApplication.openSettingsURLString) { openURL(url) }
            }
            .buttonStyle(ActionButtonStyle(filled: true))
        } else {
            Button(tracker.running ? "Stop" : "Start Tracking") {
                tracker.running ? tracker.stop() : tracker.start()
            }
            .buttonStyle(ActionButtonStyle(filled: !tracker.running))
        }
    }
}

private extension View {
    // The inset grouped list on Mira's charcoal ground.
    func miraList() -> some View {
        listStyle(.insetGrouped)
            .listSectionSpacing(.compact)
            .scrollContentBackground(.hidden)
            .background(Palette.ground)
            .foregroundStyle(Palette.text)
            .toolbarBackground(Palette.ground, for: .navigationBar)
    }

    func miraRows() -> some View {
        listRowBackground(Palette.card).listRowSeparatorTint(Palette.line)
    }
}

// How iPhone reaches the PC, and what the camera is doing. One tap from the main screen.
struct ConnectionPage: View {
    @ObservedObject var tracker: EyeTracker

    private var note: String {
        let locked = tracker.running ? "Stop tracking to change how iPhone connects. " : ""
        let trust = "Use a private network you trust. The numbers are sent unencrypted."
        if tracker.transport == .usb { return locked + "USB needs a bridge set up on your PC." }
        if tracker.wifiEndpoint.isEmpty { return locked + "Connect iPhone to a private Wi-Fi network. " + trust }
        return locked + "Mira on your PC finds this iPhone automatically. If it can’t, enter this address in Settings › iPhone. " + trust
    }

    var body: some View {
        List {
            Group {
                Section {
                    Picker("Connect over", selection: $tracker.transport) {
                        ForEach(PhoneTransport.allCases) { transport in
                            Text(transport.title).tag(transport)
                        }
                    }
                    .disabled(tracker.running)
                    if tracker.transport == .wifi && !tracker.wifiEndpoint.isEmpty {
                        LabeledContent("iPhone address") {
                            Text(tracker.wifiEndpoint).monospacedDigit().textSelection(.enabled)
                        }
                    }
                } footer: {
                    Text(note)
                }
                Section {
                    StatusRow(level: tracker.depthReceived ? .live : (tracker.running ? .waiting : .off),
                              name: "TrueDepth", value: tracker.depthText)
                    StatusRow(level: tracker.running ? .live : .off, name: "Camera updates", value: tracker.rateText)
                } header: {
                    Text("Camera")
                } footer: {
                    Text("Stand iPhone centered under your monitor with the front camera facing you. Don’t move it after calibration.")
                }
            }
            .miraRows()
        }
        .miraList()
        .navigationTitle("Connection")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct StatusRow: View {
    let level: PhoneStatus.Level
    let name: String
    let value: String
    @Environment(\.dynamicTypeSize) private var typeSize

    var body: some View {
        Group {
            if typeSize.isAccessibilitySize {
                // Very large text gets a line each instead of a squeezed, hyphenated column.
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 12) {
                        StatusDot(level: level)
                        Text(name)
                    }
                    Text(value).monospacedDigit().foregroundStyle(Palette.muted)
                }
            } else {
                HStack(spacing: 12) {
                    StatusDot(level: level)
                    Text(name)
                    Spacer(minLength: 12)
                    Text(value)
                        .monospacedDigit()
                        .foregroundStyle(Palette.muted)
                        .multilineTextAlignment(.trailing)
                }
            }
        }
        .accessibilityElement(children: .combine)
    }
}

// The word beside the dot always carries the state; the dot only repeats it.
struct StatusDot: View {
    let level: PhoneStatus.Level

    var body: some View {
        Group {
            switch level {
            case .live: Circle().fill(Palette.accent)
            case .waiting: Circle().fill(Palette.waiting)
            case .failed: Circle().fill(Palette.failed)
            case .off: Circle().strokeBorder(Palette.outline, lineWidth: 1.5)
            }
        }
        .frame(width: 9, height: 9)
        .accessibilityHidden(true)
    }
}

struct ActionButtonStyle: ButtonStyle {
    let filled: Bool

    func makeBody(configuration: Configuration) -> some View {
        let shape = RoundedRectangle(cornerRadius: 14, style: .continuous)
        configuration.label
            .font(.headline)
            .foregroundStyle(filled ? Color.white : Palette.text)
            .frame(maxWidth: .infinity, minHeight: 52)
            .background(filled ? Palette.accentFill : Palette.card, in: shape)
            .overlay { if !filled { shape.strokeBorder(Palette.line, lineWidth: 1) } }
            .opacity(configuration.isPressed ? 0.75 : 1)
            .contentShape(shape)
    }
}

// A monitor with the iPhone standing under it. As soon as iPhone sees the person's
// eyes, with or without a PC, the same light as the Windows overlay follows their gaze.
struct GazeScene: View {
    @ObservedObject var preview: PreviewModel
    let level: PhoneStatus.Level

    private var pose: PreviewPose? { preview.pose }

    // Window shapes inside the monitor, as fractions of it.
    private static let panes = [
        CGRect(x: 0.05, y: 0.09, width: 0.22, height: 0.82),
        CGRect(x: 0.31, y: 0.09, width: 0.64, height: 0.16),
        CGRect(x: 0.31, y: 0.32, width: 0.30, height: 0.32),
        CGRect(x: 0.65, y: 0.32, width: 0.30, height: 0.32),
        CGRect(x: 0.31, y: 0.71, width: 0.64, height: 0.20),
    ]

    var body: some View {
        Canvas { context, size in draw(&context, size) }
            .accessibilityElement()
            .accessibilityLabel(description)
    }

    private var description: String {
        if let pose {
            let centimeters = Int((pose.depth * 100).rounded())
            return "You are about \(centimeters) centimeters from iPhone, looking \(pose.gazeDescription)."
        }
        switch level {
        case .off, .failed: return "A monitor with iPhone standing under it. Tracking is off."
        case .waiting, .live: return "A monitor with iPhone standing under it. iPhone doesn’t see your eyes yet."
        }
    }

    private func draw(_ context: inout GraphicsContext, _ size: CGSize) {
        let phone = CGSize(width: 24, height: 44)
        let stand: CGFloat = 12, gap: CGFloat = 8
        let height = size.height - stand - gap - phone.height - 2
        let width = min(size.width - 2, height * 1.78)
        let monitor = CGRect(x: (size.width - width) / 2, y: 1, width: width, height: height)
        let glass = Path(roundedRect: monitor, cornerRadius: 8, style: .continuous)
        func pane(_ unit: CGRect) -> Path {
            Path(roundedRect: CGRect(x: monitor.minX + unit.minX * monitor.width, y: monitor.minY + unit.minY * monitor.height,
                                     width: unit.width * monitor.width, height: unit.height * monitor.height), cornerRadius: 3)
        }
        context.fill(glass, with: .color(Palette.inset))
        for unit in Self.panes { context.stroke(pane(unit), with: .color(Palette.line), lineWidth: 1) }

        if let gaze = pose?.screenPoint {
            let point = CGPoint(x: monitor.minX + gaze.x * monitor.width, y: monitor.minY + gaze.y * monitor.height)
            let radius = monitor.width * 0.36
            var lit = context
            lit.clip(to: glass)
            lit.opacity = gaze.onScreen ? 1 : 0.4
            lit.fill(Path(ellipseIn: CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2)),
                     with: .radialGradient(Gradient(colors: [Palette.light.opacity(0.28), Palette.light.opacity(0)]),
                                           center: point, startRadius: 0, endRadius: radius))
            let glow = GraphicsContext.Shading.radialGradient(Gradient(colors: [Palette.light, Palette.light.opacity(0)]),
                                                              center: point, startRadius: 0, endRadius: radius)
            for unit in Self.panes { lit.stroke(pane(unit), with: glow, lineWidth: 1.5) }
            if gaze.onScreen, let focused = Self.panes.first(where: { $0.contains(CGPoint(x: gaze.x, y: gaze.y)) }) {
                lit.stroke(pane(focused), with: .color(Palette.focus), lineWidth: 2)
            }
            lit.fill(Path(ellipseIn: CGRect(x: point.x - 3, y: point.y - 3, width: 6, height: 6)), with: .color(Palette.gazeDot))
        }
        context.stroke(glass, with: .color(Palette.outline), lineWidth: 2)

        let middle = size.width / 2, foot = monitor.maxY + stand
        var legs = Path()
        legs.move(to: CGPoint(x: middle, y: monitor.maxY + 1))
        legs.addLine(to: CGPoint(x: middle, y: foot))
        legs.move(to: CGPoint(x: middle - 30, y: foot))
        legs.addLine(to: CGPoint(x: middle + 30, y: foot))
        context.stroke(legs, with: .color(Palette.outline), style: StrokeStyle(lineWidth: 2, lineCap: .round))

        let body = CGRect(x: middle - phone.width / 2, y: size.height - phone.height - 1, width: phone.width, height: phone.height)
        let camera: Color = pose != nil ? Palette.accent : level == .waiting ? Palette.waiting : Palette.outline
        context.fill(Path(roundedRect: body, cornerRadius: 6, style: .continuous), with: .color(Palette.card))
        context.stroke(Path(roundedRect: body, cornerRadius: 6, style: .continuous), with: .color(Palette.text), lineWidth: 1.5)
        context.fill(Path(ellipseIn: CGRect(x: middle - 2.5, y: body.minY + 6, width: 5, height: 5)), with: .color(camera))
    }
}

struct PrivacyPage: View {
    var body: some View {
        List {
            Group {
                item("Numbers only", "Mira sends numeric head and eye measurements. No images leave your iPhone.")
                item("No cloud, no microphone", "No cloud services or microphone are used.")
                item("USB and Wi-Fi", "USB needs a bridge configured on Windows. Wi-Fi sends unencrypted gaze measurements over the local network. Use a network you trust.")
                item("Stopping", "Leaving the app, locking the screen, or tapping Stop turns off the camera and stops sending.")
            }
            .miraRows()
        }
        .miraList()
        .navigationTitle("What Mira Sends")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func item(_ title: String, _ text: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.headline).foregroundStyle(Palette.text)
            Text(text).font(.subheadline).foregroundStyle(Palette.muted).fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 4)
    }
}
