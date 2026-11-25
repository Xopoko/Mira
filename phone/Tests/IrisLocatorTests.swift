import XCTest
import simd
@testable import MyEyeTrackingPhone

final class IrisLocatorTests: XCTestCase {
    // Median and 95th-percentile error in pixels at a limbus radius of 12, on
    // rendered eyes. Up and down is the weaker axis: lids hide that part of the limbus.
    func testSoftOpenEyes() { check(.open, across: (0.10, 0.30), up: (0.25, 0.50)) }
    func testLidsLashesGlintsAndMonitorReflection() { check(.realistic, across: (0.25, 0.70), up: (0.25, 0.90)) }
    func testSeenFromBelow() { check(.below, across: (0.25, 0.70), up: (0.25, 0.90)) }
    func testHardScenes() {
        for scenario in [EyeSweep.corner, .dim, .darkSkin, .darkerSclera, .sideLight, .light] {
            check(scenario, across: (0.40, 1.0), up: (0.60, 1.7))
        }
    }

    func testCleanLateralProfilesAreCenteredOnTheLimbus() throws {
        for radius in [9.0, 12, 16] {
            for index in 0..<4 {
                let eye = EyeSweep.eye(.disc, radius: radius, index: index)
                let fit = try XCTUnwrap(eye.locate())
                XCTAssertGreaterThan(fit.score, 0.5)
                XCTAssertEqual(fit.profiles.count, 4)
                for profile in fit.profiles {
                    XCTAssertEqual(profile.count, 25)
                    XCTAssertTrue(profile.allSatisfy { $0.isFinite })
                }
                for profile in fit.profiles.prefix(2) {
                    let peak = (2...22).max { i, j in
                        profile[i + 2] - profile[i - 2] < profile[j + 2] - profile[j - 2]
                    }!
                    XCTAssertLessThanOrEqual(abs(peak - 12), 1)
                }
            }
        }
    }

    func testGuidanceWindowDoesNotPullTheFitOffTheLimbus() throws {
        let eye = EyeSweep.eye(.disc, radius: 12, index: 1)
        let pixels = eye.render()
        for direction in [eye.across, -eye.across, eye.up, -eye.up] {
            let fit = try XCTUnwrap(eye.locate(pixels: pixels, predicted: eye.center + 6 * direction))
            XCTAssertEqual(fit.x, eye.center.x, accuracy: 0.10)
            XCTAssertEqual(fit.y, eye.center.y, accuracy: 0.10)
        }
    }

    func testClosedEyesPlainSkinAndNoiseStillReturnLowScoringProfiles() throws {
        for scenario in [EyeSweep.closed, .skin, .noise] {
            for radius in [9.0, 12, 16] {
                for index in 0..<8 {
                    let fit = try XCTUnwrap(EyeSweep.eye(scenario, radius: radius, index: index).locate())
                    XCTAssertLessThan(fit.score, 0.3, "\(scenario) R=\(radius) sample=\(index)")
                }
            }
        }
    }

    func testInvalidInputsAndPartialWindowsReturnNil() {
        let eye = EyeSweep.eye(.open, radius: 12, index: 0)
        let pixels = eye.render()
        XCTAssertNil(eye.locate(pixels: pixels, predicted: SIMD2(10, eye.eyeball.y)))
        XCTAssertNil(eye.locate(pixels: pixels, predicted: SIMD2(eye.eyeball.x, Double(eye.side) - 10)))
        XCTAssertNil(eye.locate(pixels: pixels, predicted: SIMD2(.nan, 50)))
        XCTAssertNil(eye.locate(pixels: pixels, up: eye.across))
        for radius in [4.99, 80.01, .nan, .infinity] {
            XCTAssertNil(eye.locate(pixels: pixels, expectedRadius: radius))
        }
    }

    private func check(_ scenario: EyeSweep, across: (median: Double, p95: Double), up: (median: Double, p95: Double),
                       file: StaticString = #filePath, line: UInt = #line) {
        for radius in [9.0, 12, 16] {
            let result = EyeSweep.measure(scenario, radius: radius, count: 64)
            XCTAssertGreaterThanOrEqual(result.rate, 0.97, result.description, file: file, line: line)
            if radius == 12 {
                XCTAssertLessThanOrEqual(result.across.median, across.median, result.description, file: file, line: line)
                XCTAssertLessThanOrEqual(result.across.p95, across.p95, result.description, file: file, line: line)
                XCTAssertLessThanOrEqual(result.up.median, up.median, result.description, file: file, line: line)
                XCTAssertLessThanOrEqual(result.up.p95, up.p95, result.description, file: file, line: line)
            }
        }
    }
}

private struct SplitMix64 {
    var state: UInt64

    mutating func unit() -> Double {
        state &+= 0x9E3779B97F4A7C15
        var z = (state ^ (state >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return Double((z ^ (z >> 31)) >> 11) * 0x1p-53
    }
}

private struct SyntheticEye {
    typealias Point = SIMD2<Double>

    let radius: Double
    let roll: Double
    let eyeball: Point
    let guessOffset: Point
    let expectedRadius: Double
    let displacement: Point
    let skin: Double
    let sclera: Double
    let iris: Double
    let pupil: Double
    let pupilFraction: Double
    let lidTop: Double
    let lidBottom: Double
    let halfWidth: Double
    let lashWidth: Double
    let glint: Point
    let reflectionRadius: Double
    let monitorBand: Point
    let lightSlope: Point
    let sideLight: Double
    let exposure: Double
    let blackLevel: Double
    let blur: Double
    let noise: Double
    let seed: UInt64
    let closed: Bool
    let disc: Bool
    let blank: Bool

    var side: Int { 2 * Int(ceil(3.3 * max(radius, expectedRadius) + 13)) }
    var stride: Int { side + 9 }
    var across: Point { Point(cos(roll), sin(roll)) }
    var up: Point { Point(across.y, -across.x) }
    var center: Point { eyeball + displacement.x * across + displacement.y * up }
    var shapeK: Double {
        let d2 = simd_length_squared(displacement)
        return 1 / max(3.24 * radius * radius - d2, 9.0 / 16 * d2)
    }

    @inline(__always)
    func lids(at x: Double) -> (Double, Double) {
        let u = x / halfWidth
        let opening = max(0, 1 - u * u)
        let slant = 0.042 * x
        return (lidTop * opening + slant, -lidBottom * opening + slant)
    }

    func render() -> [UInt8] {
        let side = side, stride = stride
        var samples = [Double](repeating: skin, count: side * side)
        let across = across, up = up
        let k = shapeK
        let a = 1 + k * displacement.x * displacement.x
        let b = 2 * k * displacement.x * displacement.y
        let c = 1 + k * displacement.y * displacement.y
        let reflectionSquared = reflectionRadius * reflectionRadius
        let renderTop = max(0, lidTop) + lashWidth + 0.042 * halfWidth + 2
        let renderBottom = min(0, -lidBottom) - 0.042 * halfWidth - 2
        for row in 0..<side {
            let rowY = Double(row) + 0.125 - eyeball.y
            for column in 0..<side {
                let columnX = Double(column) + 0.125 - eyeball.x
                let midX = across.x * (columnX + 0.375) + across.y * (rowY + 0.375)
                let midY = up.x * (columnX + 0.375) + up.y * (rowY + 0.375)
                let illumination = lightSlope.x * midX + lightSlope.y * midY
                let sideGain = 1 - abs(sideLight) * min(1, max(0, 0.5 * (1 + (sideLight < 0 ? -midX : midX) / halfWidth)))
                let insideRenderBounds = disc ? abs(midX - displacement.x) < radius + 2
                    && abs(midY - displacement.y) < radius + 2
                    : abs(midX) <= halfWidth + 1 && midY >= renderBottom && midY <= renderTop
                var value = skin
                if !blank, insideRenderBounds {
                    var sum = 0.0
                    for sy in 0..<4 {
                        let py = rowY + Double(sy) * 0.25
                        var x = across.x * columnX + across.y * py
                        var y = up.x * columnX + up.y * py
                        for _ in 0..<4 {
                            let (top, bottom) = lids(at: x)
                            var luma = skin
                            if disc || (!closed && abs(x) < halfWidth && y < top && y > bottom) {
                                let dx = x - displacement.x, dy = y - displacement.y
                                let radial = sqrt((a * dx + b * dy) * dx + c * dy * dy)
                                let limbus = min(1, max(0, (radial - radius) / 1.5 + 0.5))
                                let pupilEdge = min(1, max(0, radial - pupilFraction * radius + 0.5))
                                let texture = disc ? 0 : 5 * sin(1.2 * dx + 0.4 * dy) + 3 * sin(1.6 * dy - 0.7 * dx)
                                luma = pupil + (iris + texture - pupil) * pupilEdge
                                luma += (sclera - luma) * limbus
                                if reflectionSquared > 0, radial < radius {
                                    let glintDelta = Point(dx, dy) - glint
                                    let bandDelta = Point(dx, dy) - monitorBand
                                    if simd_length_squared(glintDelta) < reflectionSquared {
                                        luma = 230
                                    } else if abs(bandDelta.x) < 0.45 * radius, abs(bandDelta.y) < 0.125 * radius {
                                        luma += 0.48 * (sclera - luma)
                                    }
                                }
                            } else if abs(x) < halfWidth, y >= top,
                                      y < top + lashWidth * (0.76 + 0.24 * cos(1.8 * x)) {
                                luma = 0.29 * skin
                            }
                            sum += luma
                            x += across.x * 0.25
                            y += up.x * 0.25
                        }
                    }
                    value = sum / 16
                }
                samples[row * side + column] = exposure * (value + illumination) * sideGain + blackLevel
            }
        }

        // Blur the whole rendered frame before adding independent sensor noise.
        let halfKernel = Int(ceil(3 * blur))
        let kernel = (-halfKernel...halfKernel).map { exp(-0.5 * pow(Double($0) / blur, 2)) }
        let weight = kernel.reduce(0, +)
        var horizontal = samples
        for row in 0..<side {
            for column in 0..<side {
                var value = 0.0
                for tap in -halfKernel...halfKernel {
                    value += kernel[tap + halfKernel] * samples[row * side + min(side - 1, max(0, column + tap))]
                }
                horizontal[row * side + column] = value / weight
            }
        }
        var pixels = [UInt8](repeating: 0, count: side * stride)
        var random = SplitMix64(state: seed)
        for row in 0..<side {
            for column in 0..<side {
                var value = 0.0
                for tap in -halfKernel...halfKernel {
                    value += kernel[tap + halfKernel] * horizontal[min(side - 1, max(0, row + tap)) * side + column]
                }
                let jitter = noise > 0 ? noise * sqrt(-2 * log(max(0x1p-53, random.unit())))
                    * cos(2 * Double.pi * random.unit()) : 0
                pixels[row * stride + column] = UInt8(min(255, max(0, (value / weight + jitter).rounded())))
            }
        }
        return pixels
    }

    func locate(pixels: [UInt8]? = nil, predicted: Point? = nil, expectedRadius: Double? = nil,
                up: Point? = nil) -> IrisFit? {
        (pixels ?? render()).withUnsafeBufferPointer {
            IrisLocator.locate(in: LumaImage(base: $0.baseAddress!, width: side, height: side, bytesPerRow: stride),
                               eyeball: eyeball, expectedRadius: expectedRadius ?? self.expectedRadius,
                               across: across, up: up ?? self.up, near: predicted ?? center + guessOffset)
        }
    }
}

private enum EyeSweep: String, CaseIterable {
    case open, realistic, below, corner, dim, darkerSclera, darkSkin, sideLight, light, disc, closed, skin, noise

    struct Distribution {
        let values: [Double]
        var median: Double { percentile(0.5) }
        var p95: Double { percentile(0.95) }

        private func percentile(_ fraction: Double) -> Double {
            values.isEmpty ? .infinity : values[Int(ceil(Double(values.count) * fraction)) - 1]
        }
    }

    struct Result: CustomStringConvertible {
        let name: String
        let across: Distribution
        let up: Distribution
        let count: Int
        let medianScore: Double
        var rate: Double { Double(across.values.count) / Double(count) }
        var description: String {
            String(format: "%@ %d/%d across %.3f/%.3f up %.3f/%.3f score %.3f",
                   name, across.values.count, count, across.median, across.p95, up.median, up.p95, medianScore)
        }
    }

    static func measure(_ scenario: Self, radius: Double, count: Int) -> Result {
        var horizontal: [Double] = [], vertical: [Double] = [], scores: [Double] = []
        for index in 0..<count {
            let sample = eye(scenario, radius: radius, index: index)
            guard let fit = sample.locate(), fit.score >= 0.30 else { continue }
            let difference = SIMD2(fit.x, fit.y) - sample.center
            horizontal.append(abs(simd_dot(difference, sample.across)))
            vertical.append(abs(simd_dot(difference, sample.up)))
            scores.append(fit.score)
        }
        return Result(name: "\(scenario.rawValue) R=\(radius)", across: Distribution(values: horizontal.sorted()),
                      up: Distribution(values: vertical.sorted()), count: count,
                      medianScore: Distribution(values: scores.sorted()).median)
    }

    static func eye(_ scenario: Self, radius: Double, index: Int) -> SyntheticEye {
        var random = SplitMix64(state: 0x68BAE11A3 &+ UInt64(index) &* 0x9E3779B97F4A7C15)
        let disc = scenario == .disc
        let clean = scenario == .open || disc
        let closed = scenario == .closed
        let blank = scenario == .skin || scenario == .noise
        let corner = scenario == .corner
        let below = scenario == .below
        let direction = random.unit() < 0.5 ? -1.0 : 1.0
        let across = corner ? direction * (1.10 + 0.13 * random.unit())
            : (2 * random.unit() - 1) * (below ? 1 : (clean ? 0.48 : 0.72))
        let verticalSample = random.unit()
        let vertical = below ? 0.30 + 0.65 * verticalSample
            : (2 * verticalSample - 1) * (corner ? 0.19 : (clean ? 0.18 : 0.34))
        let x = radius * across, y = radius * vertical
        let halfWidth = radius * (corner ? 1.77 + 0.12 * random.unit() : 2.05 + 0.18 * random.unit())
        let upperSample = random.unit(), lowerSample = random.unit()
        let d2 = x * x + y * y
        let chord = radius / sqrt(1 + y * y / max(3.24 * radius * radius - d2, 9.0 / 16 * d2))
        let lidCurve = 1 - pow(x / halfWidth, 2)
        let upperFraction = clean ? 1.10 + 0.20 * upperSample : 0.49 + 0.37 * upperSample
        let lowerFraction = clean ? 1.15 + 0.20 * lowerSample
            : (corner && index % 8 == 0 ? 0.85 : 1.02) + 0.27 * lowerSample
        let top = (y + chord * upperFraction - 0.042 * x) / lidCurve
        let lower = (-y + chord * lowerFraction + 0.042 * x) / lidCurve
        let expected = radius * (0.92 + 0.16 * random.unit())
        let side = Double(Int(ceil(3.3 * max(radius, expected) + 13)))
        let eyeball = SIMD2(side + random.unit() - 0.5, side + random.unit() - 0.5)
        let guessAngle = 2 * Double.pi * random.unit()
        let guessFraction = index % 9 == 0 ? 0.5 : 0.1 + 0.2 * random.unit()
        let guess = radius * guessFraction * SIMD2(cos(guessAngle), sin(guessAngle))
        let angle = [0, 65 * Double.pi / 180, Double.pi / 2, Double.pi][index % 4]
        let glint = radius * SIMD2(0.9 * (random.unit() - 0.5), 0.9 * (random.unit() - 0.5))
        let band = radius * SIMD2(1.25 * (random.unit() - 0.5), 1.3 * (random.unit() - 0.5))
        let lightSlope = SIMD2(0.45 * (2 * random.unit() - 1), 0.6 * (2 * random.unit() - 1))
        let iris = 60 + 20 * random.unit()
        let skin: Double, sclera: Double, irisLevel: Double, pupil: Double
        switch scenario {
        case .disc: (skin, sclera, irisLevel, pupil) = (185, 185, 65, 65)
        case .dim: (skin, sclera, irisLevel, pupil) = (170, 200, iris, 29)
        case .darkerSclera: (skin, sclera, irisLevel, pupil) = (178, 145, 60, 22)
        case .darkSkin: (skin, sclera, irisLevel, pupil) = (65, 170, 30, 14)
        case .light: (skin, sclera, irisLevel, pupil) = (154, 218, 112, 30)
        case .noise: (skin, sclera, irisLevel, pupil) = (10 + 40 * random.unit(), 150, 60, 22)
        default: (skin, sclera, irisLevel, pupil) = (150, 200, iris, 29)
        }
        return SyntheticEye(radius: radius, roll: angle, eyeball: eyeball, guessOffset: guess,
                            expectedRadius: expected, displacement: disc ? .zero : SIMD2(x, y),
                            skin: skin, sclera: sclera, iris: irisLevel, pupil: pupil,
                            pupilFraction: 0.30 + 0.25 * random.unit(),
                            lidTop: closed ? 0.07 * radius : top, lidBottom: closed ? 0 : lower,
                            halfWidth: halfWidth, lashWidth: clean ? 0 : (closed ? 1.35 : 0.85),
                            glint: glint, reflectionRadius: clean || closed ? 0 : radius * (0.08 + 0.06 * random.unit()),
                            monitorBand: band, lightSlope: disc || blank ? .zero : lightSlope,
                            sideLight: scenario == .sideLight ? 0.5 * direction : 0,
                            exposure: scenario == .dim ? 0.30 : 1, blackLevel: scenario == .dim ? 12 : 0,
                            blur: 1 + random.unit(),
                            noise: disc || scenario == .skin ? 0 : 2 + 4 * random.unit(),
                            seed: random.state, closed: closed, disc: disc, blank: blank)
    }
}
