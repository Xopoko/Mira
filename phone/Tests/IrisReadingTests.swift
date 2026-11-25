import XCTest
import simd
@testable import MyEyeTrackingPhone

final class IrisReadingTests: XCTestCase {
    // An upright phone: the sensor image lies on its side and the front camera mirrors it.
    private let focal = 1100.0
    private let shift = SIMD2(720.0, 540.0)
    private let width = 1440, height = 1080
    private let camera = simd_float4x4(simd_quatf(angle: 0.3, axis: simd_normalize(SIMD3<Float>(0.2, 1, 0.1))))
        * simd_float4x4(columns: (SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, 1, 0), SIMD4(0.4, -0.2, 1.5, 1)))

    private func pixel(_ world: SIMD4<Float>) -> SIMD2<Double> {
        let point = simd_inverse(camera) * world
        return shift + SIMD2(-focal * Double(point.y / -point.z), -focal * Double(point.x / -point.z))
    }

    private func eye(_ x: Float) -> simd_float4x4 {
        simd_float4x4(columns: (SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, 1, 0), SIMD4(x, 0.028, 0.026, 1)))
    }

    func testFindsBothIrisesAndReportsHowCameraSpaceMapsToTheFrame() throws {
        var inCamera = simd_float4x4(simd_quatf(angle: 0.25, axis: simd_normalize(SIMD3<Float>(1, 0.6, 0.2))))
        inCamera.columns.3 = SIMD4(0.03, -0.02, -0.6, 1)
        let face = camera * inCamera
        // Both eyes look at a point beside the camera; each iris sits 10.5 mm from its eye centre.
        let target = camera * SIMD4<Float>(0.12, 0.05, 0, 1)
        var pixels = [UInt8](repeating: 185, count: width * height)
        var centres: [SIMD2<Double>] = []
        func drawEye(_ x: Float) -> simd_float4x4 {
            var transform = eye(x)
            let centre = (face * transform).columns.3
            let gaze = simd_normalize(target - centre)
            let iris = pixel(centre + Float(IrisReading.limbusDistance) * gaze)
            // ARKit's seed is off slightly in both frame axes.
            transform.columns.2 = simd_normalize(simd_inverse(face) * gaze + SIMD4(0.09, 0.05, 0, 0))
            let radius = IrisReading.limbusRadius * focal / Double(-(simd_inverse(camera) * centre).z)
            let displacement = iris - pixel(centre)
            let squared = simd_length_squared(displacement)
            let k = 1 / max(3.24 * radius * radius - squared, (9.0 / 16) * squared)
            centres.append(iris)
            for row in Int(iris.y - radius - 3)...Int(iris.y + radius + 3) {
                for column in Int(iris.x - radius - 3)...Int(iris.x + radius + 3) {
                    let offset = SIMD2(Double(column) + 0.5, Double(row) + 0.5) - iris
                    let projection = simd_dot(offset, displacement)
                    let distance = sqrt(simd_length_squared(offset) + k * projection * projection)
                    let inside = min(1, max(0, (radius - distance) / 1.5 + 0.5))
                    pixels[row * width + column] = UInt8(185 - 120 * inside)
                }
            }
            return transform
        }
        let eyes = (drawEye(0.032), drawEye(-0.032))
        let reading = try XCTUnwrap(pixels.withUnsafeBufferPointer { buffer in
            IrisReading.reading(image: LumaImage(base: buffer.baseAddress!, width: width, height: height, bytesPerRow: width),
                                camera: camera, face: face, eyes: eyes, pixel: pixel)
        })
        XCTAssertTrue(JSONSerialization.isValidJSONObject(reading))
        XCTAssertEqual(reading["image"] as? [Int], [width, height])
        let map = try XCTUnwrap(reading["camera"] as? [Double])
        for (value, expected) in zip(map, [0, -focal, -focal, 0, shift.x, shift.y]) {
            XCTAssertEqual(value, expected, accuracy: 0.05)
        }
        let profiles = try XCTUnwrap(reading["profile"] as? [String: Any])
        XCTAssertEqual(profiles["step"] as? Double, 0.05)
        for (side, centre) in zip(["left", "right"], centres) {
            let found = try XCTUnwrap(reading[side] as? [Double], side)
            XCTAssertEqual(found[0], centre.x, accuracy: 0.25, side)
            XCTAssertEqual(found[1], centre.y, accuracy: 0.25, side)
            XCTAssertEqual(found[2], IrisReading.limbusRadius * focal / 0.6, accuracy: 1.5, side)
            XCTAssertGreaterThan(found[3], 0.3, side)
            let profile = try XCTUnwrap(profiles[side] as? [[Double]], side)
            XCTAssertEqual(profile.map(\.count), [25, 25, 25, 25])
            XCTAssertTrue(profile.joined().allSatisfy { $0.isFinite && abs($0 * 10 - ($0 * 10).rounded()) < 1e-9 })
        }
    }

    func testFlatLumaReturnsMeasurementsWithLowScoresAndABrokenProjectionSendsNothing() throws {
        let flat = [UInt8](repeating: 150, count: width * height)
        var inCamera = matrix_identity_float4x4
        inCamera.columns.3 = SIMD4(0, 0, -0.6, 1)
        try flat.withUnsafeBufferPointer { buffer in
            let image = LumaImage(base: buffer.baseAddress!, width: width, height: height, bytesPerRow: width)
            let blank = try XCTUnwrap(IrisReading.reading(image: image, camera: camera, face: camera * inCamera,
                                                          eyes: (eye(0.032), eye(-0.032)), pixel: pixel))
            XCTAssertTrue(JSONSerialization.isValidJSONObject(blank))
            let profiles = try XCTUnwrap(blank["profile"] as? [String: Any])
            for side in ["left", "right"] {
                let fit = try XCTUnwrap(blank[side] as? [Double])
                XCTAssertLessThan(fit[3], 0.3)
                XCTAssertEqual(profiles[side] as? [[Double]],
                               Array(repeating: Array(repeating: 150, count: 25), count: 4))
            }
            XCTAssertNil(IrisReading.reading(image: image, camera: camera, face: camera * inCamera,
                                             eyes: (eye(0.032), eye(-0.032))) { _ in SIMD2(.nan, .nan) })
        }
    }

    func testOffFrameEyeHasNeitherFitNorProfiles() throws {
        let pixels = [UInt8](repeating: 150, count: width * height)
        var inCamera = matrix_identity_float4x4
        inCamera.columns.3 = SIMD4(0, 0, -0.6, 1)
        let reading = try XCTUnwrap(pixels.withUnsafeBufferPointer { buffer in
            let image = LumaImage(base: buffer.baseAddress!, width: width, height: height, bytesPerRow: width)
            return IrisReading.reading(image: image, camera: camera, face: camera * inCamera,
                                       eyes: (eye(0.35), eye(-0.032)), pixel: pixel)
        })
        let profiles = try XCTUnwrap(reading["profile"] as? [String: Any])
        XCTAssertTrue(reading["left"] is NSNull)
        XCTAssertTrue(profiles["left"] is NSNull)
        XCTAssertNotNil(reading["right"] as? [Double])
        XCTAssertNotNil(profiles["right"] as? [[Double]])
        XCTAssertTrue(JSONSerialization.isValidJSONObject(reading))
    }
}
