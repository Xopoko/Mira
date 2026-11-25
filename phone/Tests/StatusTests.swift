import XCTest
import simd
@testable import MyEyeTrackingPhone

final class StatusTests: XCTestCase {
    func testStatusFollowsConnectionTrackingAndFailure() {
        func status(running: Bool = true, connected: Bool = true, tracked: Bool = true, failure: String? = nil) -> PhoneStatus {
            PhoneStatus.resolve(running: running, connected: connected, tracked: tracked, failure: failure,
                                status: "Face out of view")
        }
        XCTAssertEqual(status(running: false, connected: false, tracked: false),
                       PhoneStatus(word: "Ready", detail: "Stand iPhone under your screen with the front camera facing you, then tap Start Tracking.", level: .off))
        XCTAssertEqual(status(running: false, failure: "Allow camera access in Settings"),
                       PhoneStatus(word: "Not tracking", detail: "Allow camera access in Settings", level: .failed))
        XCTAssertEqual(status(connected: false),
                       PhoneStatus(word: "Waiting for your PC", detail: "Open Mira on your PC and select Set up iPhone or Connect iPhone.", level: .waiting))
        XCTAssertEqual(status(tracked: false), PhoneStatus(word: "Looking for you", detail: "Face out of view", level: .waiting))
        XCTAssertEqual(status(), PhoneStatus(word: "Tracking", detail: "Your PC is connected. Leave iPhone where it is.", level: .live))
    }

    func testPreviewUsesThePersonsOwnRightAndUpUnderEitherCameraMirroring() {
        // 0.6 m away, looking 0.1 m to their right and 0.2 m above the camera.
        // Camera +x is the person's left before mirroring; ARKit's left-eye transform sits
        // on the person's right, as observed on the device.
        for mirror: Float in [1, -1] {
            func point(_ x: Float, _ y: Float, _ z: Float) -> SIMD3<Float> { SIMD3(x * mirror, y * mirror, z) }
            let pose = PreviewPose.make(head: point(0.10, 0, -0.6), up: point(0, 1, 0), leftEye: point(0.07, 0, -0.6),
                                        rightEye: point(0.13, 0, -0.6), lookAt: point(0, 0.1, -0.3))
            XCTAssertEqual(pose?.depth ?? 0, 0.6, accuracy: 1e-5)
            XCTAssertEqual(pose?.gaze?.x ?? 0, 0.10, accuracy: 1e-5)
            XCTAssertEqual(pose?.gaze?.y ?? 0, 0.20, accuracy: 1e-5)
            XCTAssertEqual(pose?.gazeDescription, "at the right of the screen")
            XCTAssertEqual(pose?.screenPoint?.x ?? 0, 0.5 + 0.10 / 0.60, accuracy: 1e-5)
            XCTAssertEqual(pose?.screenPoint?.y ?? 0, 1 - 0.18 / 0.34, accuracy: 1e-5)
            XCTAssertEqual(pose?.screenPoint?.onScreen, true)
        }
    }

    func testGazeOffTheDrawnMonitorIsNamedAndPinnedToItsEdge() {
        let away = PreviewPose.make(head: SIMD3(0, 0, -0.6), leftEye: SIMD3(-0.03, 0, -0.6),
                                    rightEye: SIMD3(0.03, 0, -0.6), lookAt: SIMD3(0, 0, -0.9))
        XCTAssertNil(away?.gaze)
        XCTAssertNil(away?.screenPoint)
        XCTAssertEqual(away?.gazeDescription, "away from the screen")

        let left = PreviewPose(depth: 0.6, gaze: SIMD2(-0.45, 0.2))
        XCTAssertEqual(left.gazeDescription, "past the left edge of the screen")
        XCTAssertEqual(left.screenPoint?.x, 0)
        XCTAssertEqual(left.screenPoint?.onScreen, false)
        XCTAssertEqual(PreviewPose(depth: 0.6, gaze: SIMD2(0, 0.6)).gazeDescription, "above the screen")
        XCTAssertEqual(PreviewPose(depth: 0.6, gaze: SIMD2(0, -0.2)).gazeDescription, "below the screen")
    }
}
