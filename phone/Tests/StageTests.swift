import XCTest
import simd
@testable import MyEyeTrackingPhone

final class StageTests: XCTestCase {
    func testLampFollowsConnectionTrackingAndFailure() {
        func call(running: Bool = true, connected: Bool = true, tracked: Bool = true, failure: String? = nil,
                  endpoint: String = "") -> StageCall {
            StageCall.resolve(running: running, connected: connected, tracked: tracked, failure: failure,
                              status: "Face out of view", transport: .wifi, endpoint: endpoint)
        }
        XCTAssertEqual(call(running: false, connected: false, tracked: false).lamp, .off)
        XCTAssertEqual(call(running: false, failure: "Allow camera access in Settings"),
                       StageCall(word: "Hold", detail: "Allow camera access in Settings", lamp: .hold))
        XCTAssertEqual(call(connected: false),
                       StageCall(word: "Standby", detail: "Waiting for your PC over Wi-Fi…", lamp: .standby))
        XCTAssertEqual(call(connected: false, endpoint: "192.168.1.24").detail,
                       "Waiting for your PC over Wi-Fi. iPhone address: 192.168.1.24.")
        XCTAssertEqual(call(tracked: false), StageCall(word: "Standby", detail: "Face out of view", lamp: .standby))
        XCTAssertEqual(call().lamp, .go)
    }

    func testPlanUsesThePersonsOwnLeftAndRightUnderEitherCameraMirroring() {
        // 0.6 m away, 0.1 m to the person's left, looking 0.1 m to their right on the screen.
        // Camera +x is the person's left before mirroring; ARKit's left-eye transform sits
        // on the person's right, as observed on the device.
        for mirror: Float in [1, -1] {
            func point(_ x: Float, _ z: Float) -> SIMD3<Float> { SIMD3(x * mirror, 0, z) }
            let pose = PlanPose.make(head: point(0.10, -0.6), forward: SIMD3(0, 0, 1),
                                     leftEye: point(0.07, -0.6), rightEye: point(0.13, -0.6), lookAt: point(0, -0.3))
            XCTAssertEqual(pose?.headX ?? 0, -0.10, accuracy: 1e-5)
            XCTAssertEqual(pose?.depth ?? 0, 0.6, accuracy: 1e-5)
            XCTAssertEqual(pose?.yaw ?? 1, 0, accuracy: 1e-5)
            XCTAssertEqual(pose?.gazeX ?? 0, 0.10, accuracy: 1e-5)
            XCTAssertEqual(pose?.gazeDescription, "at the right of the screen")
        }
    }

    func testTurningRightIsPositiveYawAndGazeBeyondTheScreenIsNamed() {
        let angle: Float = 0.3
        // Camera +x is the person's left here, so turning right moves the face axis toward -x.
        let away = PlanPose.make(head: SIMD3(0, 0, -0.6), forward: SIMD3(-sin(angle), 0, cos(angle)),
                                 leftEye: SIMD3(-0.03, 0, -0.6), rightEye: SIMD3(0.03, 0, -0.6), lookAt: SIMD3(0, 0, -0.9))
        XCTAssertEqual(away?.yaw ?? 0, Double(angle), accuracy: 1e-5)
        XCTAssertNil(away?.gazeX)
        XCTAssertEqual(away?.gazeDescription, "away from the screen")
        XCTAssertEqual(PlanPose(headX: 0, depth: 0.6, yaw: 0, gazeX: -0.45).gazeDescription, "past the left edge of the screen")
    }
}
