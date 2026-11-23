import SwiftUI
import UIKit
import simd

// Standby / Go: prompt-book paper, stage-black ink, one cue-lamp box.
enum Stage {
    static let paper = dynamic(light: 0xF3F3F0, dark: 0x111111)
    static let sheet = dynamic(light: 0xFAFAF8, dark: 0x1A1A1A)
    static let edge = dynamic(light: 0xD6D6D0, dark: 0x2E2E2E)
    static let ink = dynamic(light: 0x141414, dark: 0xF2F2EE)
    static let pencil = dynamic(light: 0x5C5C57, dark: 0xA3A39C)
    static let rule = dynamic(light: 0xDCDCD6, dark: 0x2A2A2A)
    static let box = dynamic(light: 0x141414, dark: 0x1F1F1F)
    static let boxEdge = dynamic(light: 0x141414, dark: 0x343434)
    static let boxText = Color(hex: 0xFFFFFF)
    static let boxMuted = Color(hex: 0xBDBDB7)
    static let lampOff = Color(hex: 0x3A3A3A)
    static let lampRing = Color(hex: 0x4A4A4A)
    static let amber = Color(hex: 0xFFB020)
    static let green = Color(hex: 0x2DBE60)
    static let hold = Color(hex: 0xFF6B5E)
    static let spike = dynamic(light: 0x3D6BFF, dark: 0x7D9BFF)
    static let spikeText = dynamic(light: 0x2C4FD1, dark: 0x9DB3FF)
    static let amberText = dynamic(light: 0x8A5300, dark: 0xFFB020)
    static let greenText = dynamic(light: 0x1A7A3D, dark: 0x4CD982)
    static let holdText = dynamic(light: 0xB3261E, dark: 0xFF8A80)

    private static func dynamic(light: UInt32, dark: UInt32) -> Color {
        Color(UIColor { $0.userInterfaceStyle == .dark ? UIColor(hex: dark) : UIColor(hex: light) })
    }
}

extension UIColor {
    convenience init(hex: UInt32) {
        self.init(red: CGFloat((hex >> 16) & 0xFF) / 255, green: CGFloat((hex >> 8) & 0xFF) / 255,
                  blue: CGFloat(hex & 0xFF) / 255, alpha: 1)
    }
}

extension Color {
    init(hex: UInt32) { self.init(UIColor(hex: hex)) }
}

// What the lamp box calls out. Derived only from tracker state.
struct StageCall: Equatable {
    enum Lamp: Equatable { case off, standby, go, hold }
    let word: String
    let detail: String
    let lamp: Lamp

    static func resolve(running: Bool, connected: Bool, tracked: Bool, failure: String?,
                        status: String, transport: PhoneTransport, endpoint: String) -> StageCall {
        if !running {
            if let failure { return StageCall(word: "Hold", detail: failure, lamp: .hold) }
            return StageCall(word: "Ready", detail: "Put iPhone under your screen, then tap Start.", lamp: .off)
        }
        if !connected {
            if transport == .wifi && !endpoint.isEmpty {
                return StageCall(word: "Standby", detail: "Waiting for your PC over Wi-Fi. iPhone address: \(endpoint).", lamp: .standby)
            }
            return StageCall(word: "Standby", detail: "Waiting for your PC over \(transport.title)…", lamp: .standby)
        }
        if tracked { return StageCall(word: "Go", detail: "Mira on your PC now follows your gaze.", lamp: .go) }
        return StageCall(word: "Standby", detail: status, lamp: .standby)
    }
}

// Top-down picture of the person in front of the screen, in meters and radians.
// Left and right come from the eye transforms, so camera axis mirroring cannot flip it.
// On iPhone Air (2026-11-23) ARKit's left-eye transform sat on the person's right:
// moving left drew the head moving right, so the axis runs from the left-eye
// transform toward the right-eye transform.
struct PlanPose: Equatable {
    var headX: Double
    var depth: Double
    var yaw: Double
    var gazeX: Double?

    static let resting = PlanPose(headX: 0, depth: 0.6, yaw: 0, gazeX: nil)
    static let screenHalfWidth = 0.30

    static func make(head: SIMD3<Float>, forward: SIMD3<Float>, leftEye: SIMD3<Float>,
                     rightEye: SIMD3<Float>, lookAt: SIMD3<Float>) -> PlanPose? {
        let eyes = rightEye - leftEye
        let horizontalIsX = abs(eyes.x) >= abs(eyes.y)
        let leftward: Float = (horizontalIsX ? eyes.x : eyes.y) >= 0 ? 1 : -1
        func towardRight(_ v: SIMD3<Float>) -> Float { -leftward * (horizontalIsX ? v.x : v.y) }
        let depth = abs(head.z)
        guard depth > 0.1, depth < 3, simd_length(eyes) > 0.01 else { return nil }
        let toCamera: Float = head.z < 0 ? 1 : -1
        let yaw = atan2(towardRight(forward), toCamera * forward.z)
        let center = (leftEye + rightEye) / 2
        let direction = lookAt - center
        var gaze: Double?
        if direction.z * toCamera > 1e-4 {
            let hit = center + (-center.z / direction.z) * direction
            gaze = Double(towardRight(hit))
        }
        return PlanPose(headX: Double(towardRight(head)), depth: Double(depth), yaw: Double(yaw), gazeX: gaze)
    }

    func blended(toward next: PlanPose, amount: Double) -> PlanPose {
        func mix(_ a: Double, _ b: Double) -> Double { a + (b - a) * amount }
        var gaze = next.gazeX
        if let a = gazeX, let b = next.gazeX { gaze = mix(a, b) }
        return PlanPose(headX: mix(headX, next.headX), depth: mix(depth, next.depth), yaw: mix(yaw, next.yaw), gazeX: gaze)
    }

    // Where the gaze lands, in words, for VoiceOver.
    var gazeDescription: String {
        guard let gaze = gazeX else { return "away from the screen" }
        if gaze < -Self.screenHalfWidth { return "past the left edge of the screen" }
        if gaze > Self.screenHalfWidth { return "past the right edge of the screen" }
        if gaze < -0.08 { return "at the left of the screen" }
        if gaze > 0.08 { return "at the right of the screen" }
        return "at the middle of the screen"
    }
}

final class PlanModel: ObservableObject {
    @Published var pose: PlanPose? { didSet { if let pose { lastLive = pose } } }
    private(set) var lastLive: PlanPose?
}

enum PlanMode { case idle, searching, live }

struct PhoneView: View {
    @ObservedObject var tracker: EyeTracker
    @State private var privacy = false
    @Environment(\.openURL) private var openURL
    @Environment(\.dynamicTypeSize) private var typeSize

    private var call: StageCall {
        StageCall.resolve(running: tracker.running, connected: tracker.computerConnected, tracked: tracker.tracked,
                          failure: tracker.failure, status: tracker.status, transport: tracker.transport,
                          endpoint: tracker.wifiEndpoint)
    }

    private var planMode: PlanMode {
        guard tracker.running else { return .idle }
        return tracker.tracked ? .live : .searching
    }

    var body: some View {
        ZStack {
            Stage.paper.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    header
                    StagePlan(plan: tracker.plan, mode: planMode)
                        .frame(height: typeSize.isAccessibilitySize ? 320 : 400)
                    LampBox(call: call)
                    statusRows
                    settings
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 20)
            }
            .safeAreaInset(edge: .bottom) {
                primaryAction
                    .buttonStyle(StagePrimaryButtonStyle())
                    .padding(.horizontal, 16)
                    .padding(.top, 10)
                    .padding(.bottom, 4)
                    .background(Stage.paper)
                    .overlay(alignment: .top) { StageRule() }
            }
            if tracker.screenDimmed {
                Button { tracker.revealScreen() } label: {
                    Color.black.ignoresSafeArea().contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Restore screen brightness")
                .zIndex(10)
            }
        }
        .simultaneousGesture(TapGesture().onEnded { tracker.revealScreen() })
        .onChange(of: privacy) { _, _ in tracker.revealScreen() }
        .task(id: tracker.dimGeneration) {
            guard tracker.running, !privacy else { return }
            do { try await Task.sleep(for: .seconds(20)) }
            catch { return }
            guard tracker.running, !privacy else { return }
            tracker.dimScreen()
        }
        .sheet(isPresented: $privacy) { PrivacySheet { privacy = false } }
    }

    @ViewBuilder private var primaryAction: some View {
        if !tracker.running && tracker.failure != nil && tracker.cameraBlocked {
            Button("Open Settings") {
                if let url = URL(string: UIApplication.openSettingsURLString) { openURL(url) }
            }
        } else {
            Button(tracker.running ? "Stop" : "Start Tracking") {
                tracker.running ? tracker.stop() : tracker.start()
            }
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("Mira").font(.largeTitle.bold()).foregroundStyle(Stage.ink)
            Spacer()
            Menu {
                Picker("Connection", selection: $tracker.transport) {
                    ForEach(PhoneTransport.allCases) { transport in
                        Text(transport.title).tag(transport)
                    }
                }
            } label: {
                HStack(spacing: 5) {
                    Text(tracker.transport.title)
                    Image(systemName: "chevron.down").font(.caption2.weight(.bold))
                }
                .font(.footnote.weight(.semibold))
                .foregroundStyle(Stage.ink)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .overlay(Capsule().strokeBorder(Stage.ink, lineWidth: 1.5))
                .contentShape(Capsule())
            }
            .disabled(tracker.running)
            .accessibilityLabel("Connection: \(tracker.transport.title)")
            .accessibilityHint(tracker.running ? "Stop tracking to change the connection." : "Choose USB or Wi-Fi.")
        }
        .padding(.horizontal, 6)
        .padding(.top, 6)
    }

    private var statusRows: some View {
        VStack(spacing: 0) {
            StatusRow(lamp: tracker.tracked ? .go : (tracker.running ? .standby : .off), name: "iPhone",
                      value: tracker.running ? (tracker.tracked ? "Head and eyes tracked" : "Looking for your face") : "Camera off")
            StageRule()
            StatusRow(lamp: tracker.computerConnected ? .go : (tracker.running ? .standby : .off),
                      name: "Windows PC", value: tracker.connectionText)
        }
        .padding(.horizontal, 6)
    }

    private var settings: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Connection").font(.headline).foregroundStyle(Stage.ink)
            Picker("Connection", selection: $tracker.transport) {
                ForEach(PhoneTransport.allCases) { transport in
                    Text(transport.title).tag(transport)
                }
            }
            .pickerStyle(.segmented)
            .disabled(tracker.running)
            if tracker.transport == .wifi {
                VStack(alignment: .leading, spacing: 4) {
                    if tracker.wifiEndpoint.isEmpty {
                        Text("Connect to a private Wi-Fi network").font(.callout).foregroundStyle(Stage.ink)
                    } else {
                        Text("Mira on your PC finds this iPhone automatically. If it can’t, enter this address in Settings › iPhone:")
                            .font(.callout).foregroundStyle(Stage.pencil)
                        Text(tracker.wifiEndpoint)
                            .font(.system(.title3, design: .monospaced).weight(.semibold))
                            .foregroundStyle(Stage.ink)
                            .textSelection(.enabled)
                    }
                    Text("Private local network only. Numeric data is sent unencrypted. RGB is off over Wi-Fi.")
                        .font(.footnote).foregroundStyle(Stage.pencil)
                }
            }
            StageRule()
            Text("Placement").font(.headline).foregroundStyle(Stage.ink)
            Text("Mount iPhone centered horizontally on your monitor. It can sit below the bottom edge. Point the front camera at your face. Don’t move iPhone after calibration.")
                .font(.callout).foregroundStyle(Stage.pencil)
            StageRule()
            Text("Camera").font(.headline).foregroundStyle(Stage.ink)
            StatusRow(lamp: tracker.depthReceived ? .go : (tracker.running ? .standby : .off),
                      name: "TrueDepth", value: tracker.depthText)
            StatusRow(lamp: tracker.running ? .go : .off, name: "Camera updates", value: tracker.rateText, numeric: true)
            StageRule()
            Toggle("RGB for Model Comparison (USB)", isOn: $tracker.rgbEnabled)
                .foregroundStyle(Stage.ink)
                .disabled(tracker.transport == .wifi)
                .onChange(of: tracker.rgbEnabled) { _, _ in tracker.configureRGB() }
            Text("Sent only when Windows asks: a cropped face image over USB. Nothing is saved.")
                .font(.footnote).foregroundStyle(Stage.pencil)
            StageRule()
            Button("What’s Sent to Your Computer") { privacy = true }
                .font(.callout.weight(.semibold))
                .foregroundStyle(Stage.spikeText)
                .frame(minHeight: 44)
        }
        .padding(18)
        .background(Stage.sheet, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(Stage.edge, lineWidth: 1))
        .padding(.top, 8)
    }
}

struct StageRule: View {
    var body: some View { Rectangle().fill(Stage.rule).frame(height: 1) }
}

struct LampBox: View {
    let call: StageCall
    @Environment(\.dynamicTypeSize) private var typeSize

    var body: some View {
        Group {
            if typeSize.isAccessibilitySize {
                VStack(alignment: .leading, spacing: 14) { lamps; words }
            } else {
                HStack(spacing: 14) {
                    lamps
                    words.padding(.leading, 4)
                    Spacer(minLength: 0)
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 20)
        .frame(maxWidth: .infinity, minHeight: 112, alignment: .leading)
        .background(Stage.box, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).strokeBorder(Stage.boxEdge, lineWidth: 1))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(call.word). \(call.detail)")
    }

    private var lamps: some View {
        HStack(spacing: 14) {
            BoxLamp(color: call.lamp == .standby ? Stage.amber : nil)
            BoxLamp(color: call.lamp == .go ? Stage.green : nil)
        }
    }

    private var words: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(call.word)
                .font(.largeTitle.bold())
                .foregroundStyle(call.lamp == .hold ? Stage.hold : Stage.boxText)
                .lineLimit(typeSize.isAccessibilitySize ? nil : 1)
                .minimumScaleFactor(typeSize.isAccessibilitySize ? 1 : 0.8)
            Text(call.detail)
                .font(.callout)
                .foregroundStyle(Stage.boxMuted)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

struct BoxLamp: View {
    let color: Color?

    var body: some View {
        Circle()
            .fill(color ?? Stage.lampOff)
            .overlay(Circle().strokeBorder(color == nil ? Stage.lampRing : .clear, lineWidth: 3))
            .frame(width: 50, height: 50)
            .accessibilityHidden(true)
    }
}

struct StatusRow: View {
    let lamp: StageCall.Lamp
    let name: String
    let value: String
    var numeric = false

    var body: some View {
        HStack(spacing: 12) {
            PaperLamp(state: lamp)
            Text(name).font(.body.weight(.semibold)).foregroundStyle(Stage.ink)
            Spacer(minLength: 12)
            Text(value)
                .font(numeric ? Font.callout.monospacedDigit() : Font.callout)
                .foregroundStyle(Stage.pencil)
                .multilineTextAlignment(.trailing)
        }
        .frame(minHeight: 44)
        .accessibilityElement(children: .combine)
    }
}

struct PaperLamp: View {
    let state: StageCall.Lamp

    var body: some View {
        Group {
            switch state {
            case .go: Circle().fill(Stage.green).overlay(Circle().strokeBorder(Stage.greenText, lineWidth: 1))
            case .standby: Circle().fill(Stage.amber).overlay(Circle().strokeBorder(Stage.amberText, lineWidth: 1))
            case .hold: Circle().fill(Stage.holdText)
            case .off: Circle().strokeBorder(Stage.pencil, lineWidth: 1.5)
            }
        }
        .frame(width: 11, height: 11)
        .accessibilityHidden(true)
    }
}

struct StagePrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .foregroundStyle(Stage.paper)
            .frame(maxWidth: .infinity, minHeight: 54)
            .background(Stage.ink.opacity(configuration.isPressed ? 0.78 : 1), in: Capsule())
            .contentShape(Capsule())
    }
}

struct StagePlan: View {
    @ObservedObject var plan: PlanModel
    let mode: PlanMode

    private var pose: PlanPose {
        switch mode {
        case .idle: return .resting
        case .searching: return plan.lastLive ?? .resting
        case .live: return plan.pose ?? plan.lastLive ?? .resting
        }
    }

    private var isLive: Bool { mode == .live && plan.pose != nil }

    var body: some View {
        Canvas { context, size in draw(&context, size) }
            .background(Stage.sheet)
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(Stage.edge, lineWidth: 1.5))
            .accessibilityElement()
            .accessibilityLabel(accessibilityText)
    }

    private var accessibilityText: String {
        guard isLive else {
            return mode == .idle ? "Plan: iPhone under the screen. Tracking is off." : "Plan: looking for your face."
        }
        let centimeters = Int((pose.depth * 100).rounded())
        return "Live plan: you are about \(centimeters) centimeters from iPhone, looking \(pose.gazeDescription)."
    }

    private func draw(_ context: inout GraphicsContext, _ size: CGSize) {
        let width = size.width, height = size.height
        let barLeft = width * 0.09, barRight = width * 0.91
        let barTop: CGFloat = 26, barHeight: CGFloat = 14
        let metersToPoints = (barRight - barLeft) / 2 / CGFloat(PlanPose.screenHalfWidth)
        let line = isLive ? Stage.ink : Stage.pencil
        let radius = min(width, height) * 0.12

        // Lay the plan out from the real label sizes, so large text never collides.
        let screenLabel = Text("Screen").font(.headline).foregroundColor(Stage.ink)
        let screenHeight = context.resolve(screenLabel).measure(in: CGSize(width: width - 24, height: 400)).height
        let screenTop = barTop + barHeight + 14
        let markY = screenTop + screenHeight + 28
        let iphoneLabel = Text("iPhone stays here").font(.footnote.weight(.medium)).foregroundColor(Stage.spikeText)
        let beside = width - (width / 2 + 26) - 12
        let iphoneSize = context.resolve(iphoneLabel).measure(in: CGSize(width: 1000, height: 200))
        let labelBeside = iphoneSize.width <= beside
        let belowHeight = context.resolve(iphoneLabel).measure(in: CGSize(width: width - 24, height: 400)).height
        let clearance = labelBeside ? markY + 30 : markY + 20 + belowHeight + 14

        context.fill(Path(roundedRect: CGRect(x: barLeft, y: barTop, width: barRight - barLeft, height: barHeight), cornerRadius: 3),
                     with: .color(Stage.ink))

        let youLabel = Text("You").font(.headline)
        let youHeight = context.resolve(youLabel).measure(in: CGSize(width: width, height: 200)).height
        let near = clearance + radius, far = max(near, height - radius - 10 - youHeight - 8)
        let depthShare = min(max((pose.depth - 0.30) / 0.60, 0), 1)
        let center = CGPoint(x: min(max(width / 2 + CGFloat(pose.headX) * metersToPoints, radius + 10), width - radius - 10),
                             y: near + CGFloat(depthShare) * (far - near))
        let forward = CGVector(dx: sin(pose.yaw), dy: -cos(pose.yaw))
        let side = CGVector(dx: -forward.dy, dy: forward.dx)
        func offset(_ along: CGFloat, _ across: CGFloat) -> CGPoint {
            CGPoint(x: center.x + forward.dx * along + side.dx * across, y: center.y + forward.dy * along + side.dy * across)
        }
        let leftEye = offset(radius * 0.42, -radius * 0.34)
        let rightEye = offset(radius * 0.42, radius * 0.34)

        // Head first, so the sightlines visibly start at the eyes.
        let outline = isLive ? StrokeStyle(lineWidth: 3, lineJoin: .round) : StrokeStyle(lineWidth: 3, lineJoin: .round, dash: [7, 6])
        func rotated(_ angle: Double) -> CGVector {
            CGVector(dx: forward.dx * cos(angle) - forward.dy * sin(angle), dy: forward.dx * sin(angle) + forward.dy * cos(angle))
        }
        let a = rotated(0.42), b = rotated(-0.42)
        var nose = Path()
        nose.move(to: CGPoint(x: center.x + a.dx * radius * 0.96, y: center.y + a.dy * radius * 0.96))
        nose.addLine(to: CGPoint(x: center.x + forward.dx * (radius + 16), y: center.y + forward.dy * (radius + 16)))
        nose.addLine(to: CGPoint(x: center.x + b.dx * radius * 0.96, y: center.y + b.dy * radius * 0.96))
        let head = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2))
        context.fill(nose, with: .color(Stage.sheet))
        context.stroke(nose, with: .color(line), style: outline)
        context.fill(head, with: .color(Stage.sheet))
        context.stroke(head, with: .color(line), style: outline)

        var ring: Path?
        if isLive, let gaze = pose.gazeX {
            let onScreen = abs(gaze) <= PlanPose.screenHalfWidth
            let x = min(max(width / 2 + CGFloat(gaze) * metersToPoints, barLeft + 8), barRight - 8)
            let target = CGPoint(x: x, y: barTop + barHeight)
            var sightlines = Path()
            sightlines.move(to: leftEye); sightlines.addLine(to: target)
            sightlines.move(to: rightEye); sightlines.addLine(to: target)
            let style = onScreen ? StrokeStyle(lineWidth: 2, lineCap: .round) : StrokeStyle(lineWidth: 2, lineCap: .round, dash: [5, 6])
            context.stroke(sightlines, with: .color(Stage.ink), style: style)
            if onScreen {
                ring = Path(roundedRect: CGRect(x: x - 26, y: barTop - 10, width: 52, height: barHeight + 20), cornerRadius: 8)
            } else {
                // Off the edge: an arrow beyond the end of the screen bar instead of a ring on it.
                let direction: CGFloat = gaze > 0 ? 1 : -1
                let base = gaze > 0 ? barRight + 7 : barLeft - 7
                let mid = barTop + barHeight / 2
                var arrow = Path()
                arrow.move(to: CGPoint(x: base, y: mid - 9))
                arrow.addLine(to: CGPoint(x: base + direction * 14, y: mid))
                arrow.addLine(to: CGPoint(x: base, y: mid + 9))
                arrow.closeSubpath()
                context.fill(arrow, with: .color(Stage.ink))
            }
        }
        for eye in [leftEye, rightEye] {
            context.fill(Path(ellipseIn: CGRect(x: eye.x - 4.5, y: eye.y - 4.5, width: 9, height: 9)), with: .color(line))
        }

        // Labels knock out the sightlines behind them, as on a printed plan.
        StagePlan.label(screenLabel, at: CGPoint(x: width / 2, y: screenTop), anchor: .top,
                        maxWidth: width - 24, in: &context)
        if let ring {
            // The ring sits on top of the label knockouts so it always reads whole.
            context.stroke(ring, with: .color(Stage.sheet), lineWidth: 7)
            context.stroke(ring, with: .color(Stage.ink), lineWidth: 3)
        }
        var cross = Path()
        cross.move(to: CGPoint(x: width / 2 - 13, y: markY))
        cross.addLine(to: CGPoint(x: width / 2 + 13, y: markY))
        cross.move(to: CGPoint(x: width / 2, y: markY - 13))
        cross.addLine(to: CGPoint(x: width / 2, y: markY + 13))
        context.stroke(cross, with: .color(Stage.spike), style: StrokeStyle(lineWidth: 6, lineCap: .square))
        if labelBeside {
            StagePlan.label(iphoneLabel, at: CGPoint(x: width / 2 + 26, y: markY), anchor: .leading, maxWidth: beside, in: &context)
        } else {
            StagePlan.label(iphoneLabel, at: CGPoint(x: width / 2, y: markY + 20), anchor: .top, maxWidth: width - 24, in: &context)
        }
        context.draw(youLabel.foregroundColor(line), at: CGPoint(x: center.x, y: center.y + radius + 10), anchor: .top)
    }

    private static func label(_ text: Text, at point: CGPoint, anchor: UnitPoint, maxWidth: CGFloat,
                              in context: inout GraphicsContext) {
        let resolved = context.resolve(text)
        let size = resolved.measure(in: CGSize(width: maxWidth, height: 400))
        let frame = CGRect(x: point.x - anchor.x * size.width, y: point.y - anchor.y * size.height,
                           width: size.width, height: size.height)
        context.fill(Path(roundedRect: frame.insetBy(dx: -5, dy: -2), cornerRadius: 4), with: .color(Stage.sheet))
        context.draw(resolved, in: frame)
    }
}

struct PrivacySheet: View {
    let done: () -> Void

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    item("Numbers by default", "Standard mode sends numeric head and eye data.")
                    item("Face images only on request", "RGB mode also sends a cropped image of your face, but only when Windows asks to compare models. Face images stay in memory on iPhone and on Windows. They’re never written to disk. RGB has its own switch, so you can turn it off separately.")
                    item("No cloud, no microphone", "No cloud services or microphone are used.")
                    item("USB and Wi-Fi", "USB needs a bridge configured on Windows. Wi-Fi sends unencrypted gaze measurements over the local network. Use a network you trust.")
                    item("Stopping", "Leaving the app, locking the screen, or tapping Stop turns off the camera and stops sending.")
                }
                .padding(24)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Stage.paper)
            .navigationTitle("What Mira Sends")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done", action: done).foregroundStyle(Stage.ink)
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private func item(_ title: String, _ text: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.headline).foregroundStyle(Stage.ink)
            Text(text).font(.body).foregroundStyle(Stage.pencil).fixedSize(horizontal: false, vertical: true)
        }
    }
}
