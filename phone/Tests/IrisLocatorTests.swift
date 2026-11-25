import XCTest
import simd
@testable import MyEyeTrackingPhone

final class IrisLocatorTests: XCTestCase {
    func testOpenEyesAtFourScales() {
        for radius in [9.0, 12, 16, 22] {
            let result = EyeSweep.measure(.open, radius: radius, count: 40)
            check(result, rate: 0.97, across: (0.08, 0.15), up: (0.08, 0.18))
            XCTAssertGreaterThan(result.medianScore, 0.75)
            XCTAssertLessThan(result.radius.p95, 0.30)
        }
    }

    func testLidsLashesReflectionsAndNoise() {
        check(EyeSweep.measure(.realistic, radius: 12, count: 160),
              rate: 0.93, across: (0.15, 0.50), up: (0.35, 0.65))
    }

    func testSeenFromBelowAtFourScales() {
        for radius in [9.0, 12, 16, 22] {
            let result = EyeSweep.measure(.realistic, radius: radius, count: 80, view: .below)
            check(result, rate: 0.94, across: (0.15, 0.50), up: (0.35, 0.65))
            XCTAssertLessThan(result.worstCenterError, 0.25 * radius, result.description)
        }
        for scenario in [EyeSweep.dim, .darkerSclera, .sideLight, .lidShadow] {
            check(EyeSweep.measure(scenario, radius: 12, count: 48, view: .below),
                  rate: 0.90, across: (0.18, 0.75), up: (0.35, 0.85))
        }
        for (index, across) in [-0.94, 0.94].enumerated() {
            let eye = EyeSweep.eye(.realistic, radius: 12, index: index, view: .below, pose: SIMD2(across, 0.99))
            let fit = eye.locate()
            XCTAssertGreaterThanOrEqual(fit?.score ?? 0, 0.30)
            if let fit {
                XCTAssertLessThan(simd_length(SIMD2(fit.x, fit.y) - eye.center), 1.15)
            }
        }
        check(EyeSweep.measure(.realistic, radius: 12, count: 48, view: .above),
              rate: 0.90, across: (0.18, 0.75), up: (0.35, 2.15))
    }

    func testOneSidedLimbusBenefitsFromTheEllipse() {
        let result = EyeSweep.measure(.corner, radius: 12, count: 64)
        check(result, rate: 0.86, across: (0.18, 1.20), up: (0.18, 0.55))
        XCTAssertGreaterThanOrEqual(result.circleAcross.count, 48)
        XCTAssertLessThan(result.pairedAcross.median, 0.75 * result.circleAcross.median)
    }

    func testPlainDiscOnUniformBackground() {
        for radius in [9.0, 10.8, 12, 16, 22] {
            let result = EyeSweep.measure(.disc, radius: radius, count: 32)
            check(result, rate: 1, across: (0.055, 0.08), up: (0.055, 0.08))
            XCTAssertLessThan(result.worstCenterError, 0.10, result.description)
            XCTAssertGreaterThan(result.medianScore, 0.85, result.description)
        }
        let disc = EyeSweep.eye(.disc, radius: 10.8, index: 7)
        let pixels = disc.render()
        for (gain, offset) in [(0.3, 5.0), (0.7, 105.0)] {
            let shifted = pixels.map { UInt8((gain * Double($0) + offset).rounded()) }
            guard let fit = disc.locate(pixels: shifted) else {
                XCTFail("Disc lost after an affine exposure change")
                continue
            }
            XCTAssertGreaterThan(fit.score, 0.85)
            XCTAssertLessThan(simd_length(SIMD2(fit.x, fit.y) - disc.center), 0.10)
        }
        for (index, pose) in [SIMD2(1.29, 0), SIMD2(-1.29, 0), SIMD2(0, 1.29), SIMD2(0, -1.29)].enumerated() {
            let eye = EyeSweep.eye(.disc, radius: 12, index: index, pose: pose)
            guard let fit = eye.locate() else {
                XCTFail("Ellipse lost at the edge of the search window")
                continue
            }
            XCTAssertLessThan(simd_length(SIMD2(fit.x, fit.y) - eye.center), 0.25)
        }
    }

    func testExposureSkinToneAndShadows() {
        check(EyeSweep.measure(.light, radius: 12, count: 48),
              rate: 0.90, across: (0.18, 1.10), up: (0.25, 0.85))
        for radius in [9.0, 12, 16, 22] {
            for scenario in EyeSweep.lighting {
                let acrossP95 = radius == 9 ? 0.90 : (radius == 12 && scenario == .sideLight ? 0.70 : 0.50)
                let acrossMedian = radius == 9 ? 0.18 : 0.15
                check(EyeSweep.measure(scenario, radius: radius, count: 64),
                      rate: radius == 9 ? 0.80 : 0.89, across: (acrossMedian, acrossP95), up: (0.35, 0.85))
            }
        }
    }

    func testApproximateEyeballAndRadius() {
        check(EyeSweep.measure(.wrongInputs, radius: 12, count: 64),
              rate: 0.90, across: (0.18, 0.55), up: (0.32, 0.65))
    }

    func testHorizontalVerticalAndObliqueRoll() {
        for roll in [0.0, Double.pi / 2, 0.63] {
            check(EyeSweep.measure(.realistic, radius: 12, count: 64, roll: roll),
                  rate: 0.90, across: (0.15, 0.80), up: (0.30, 0.75))
        }
    }

    func testRejectsClosedEyesAndUniformLuma() {
        for index in 0..<24 {
            let eye = EyeSweep.eye(.closed, radius: 12, index: index)
            XCTAssertLessThan(eye.locate()?.score ?? 0, 0.30)
        }
        let eye = EyeSweep.eye(.open, radius: 12, index: 0)
        let flat = [UInt8](repeating: 150, count: eye.side * eye.stride)
        XCTAssertLessThan(eye.locate(pixels: flat)?.score ?? 0, 0.30)
        for index in 0..<12 {
            let noise = EyeSweep.eye(.noiseOnly, radius: 12, index: index)
            XCTAssertLessThan(noise.locate()?.score ?? 0, 0.30)
        }
    }

    func testRejectsPartialWindowAndUnsupportedRadius() {
        let eye = EyeSweep.eye(.open, radius: 12, index: 0)
        let pixels = eye.render()
        XCTAssertNil(eye.locate(pixels: pixels, eyeball: SIMD2(10, eye.eyeball.y)))
        XCTAssertNil(eye.locate(pixels: pixels, eyeball: SIMD2(eye.eyeball.x, Double(eye.side) - 10)))
        let nearTop = EyeSweep.eye(.disc, radius: 12, index: 0, origin: SIMD2(eye.eyeball.x, 36))
        XCTAssertNil(nearTop.locate())
        for radius in [4.99, 80.01, .nan, .infinity] {
            XCTAssertNil(eye.locate(pixels: pixels, expectedRadius: radius))
        }
    }

    private func check(_ result: EyeSweep.Result, rate: Double,
                       across: (Double, Double), up: (Double, Double),
                       file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertGreaterThanOrEqual(result.rate, rate, result.description, file: file, line: line)
        XCTAssertLessThanOrEqual(result.across.median, across.0, result.description, file: file, line: line)
        XCTAssertLessThanOrEqual(result.across.p95, across.1, result.description, file: file, line: line)
        XCTAssertLessThanOrEqual(result.up.median, up.0, result.description, file: file, line: line)
        XCTAssertLessThanOrEqual(result.up.p95, up.1, result.description, file: file, line: line)
    }
}

private struct SplitMix64 {
    var state: UInt64

    mutating func unit() -> Double {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
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
    let lidTop: Double
    let lidBottom: Double
    let halfWidth: Double
    let lashWidth: Double
    let reflections: (Point, Point)
    let reflectionRadius: Double
    let lightSlope: Point
    let sideLight: Double
    let lidShadow: Double
    let exposure: Double
    let blackLevel: Double
    let noise: Double
    let seed: UInt64
    let closed: Bool
    let disc: Bool
    let noiseOnly: Bool
    let eyebrow: Bool

    var side: Int { 2 * Int(ceil(3.3 * max(radius, expectedRadius) + 13)) }
    var stride: Int { side + 9 }
    var across: Point { roll == Double.pi / 2 ? Point(0, 1) : Point(cos(roll), sin(roll)) }
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
        let browHeight = max(0, lidTop) + 0.36 * radius
        let renderTop = (eyebrow ? browHeight + 0.17 * radius : max(0, lidTop) + lashWidth)
            + 0.042 * halfWidth + 2
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
                guard !noiseOnly, insideRenderBounds else {
                    samples[row * side + column] = exposure * (skin + illumination) * sideGain + blackLevel
                    continue
                }
                var sum = 0.0
                for sy in 0..<4 {
                    let py = rowY + Double(sy) * 0.25
                    var x = across.x * columnX + across.y * py
                    var y = up.x * columnX + up.y * py
                    for _ in 0..<4 {
                        let (top, bottom) = lids(at: x)
                        var value = skin
                        if disc {
                            let dx = x - displacement.x, dy = y - displacement.y
                            let radial = sqrt((a * dx + b * dy) * dx + c * dy * dy)
                            let edge = min(1, max(0, (radial - radius) / 1.5 + 0.5))
                            value = iris + (skin - iris) * edge
                        } else if abs(x) < halfWidth {
                            if !closed && y < top && y > bottom {
                                let dx = x - displacement.x, dy = y - displacement.y
                                let radial = sqrt((a * dx + b * dy) * dx + c * dy * dy)
                                let limbus = min(1, max(0, (radial - radius) / 1.5 + 0.5))
                                let pupilEdge = min(1, max(0, (radial - 0.35 * radius) + 0.5))
                                value = pupil + (iris - pupil) * pupilEdge + (sclera - iris) * limbus
                                if reflectionSquared > 0 {
                                    let p1x = dx - reflections.0.x, p1y = dy - reflections.0.y
                                    let p2x = dx - reflections.1.x, p2y = dy - reflections.1.y
                                    if p1x * p1x + p1y * p1y < reflectionSquared
                                        || p2x * p2x + p2y * p2y < reflectionSquared {
                                        value = 255
                                    }
                                }
                                if lidShadow > 0 {
                                    let band = bottom + 0.70 * (top - bottom)
                                    let fade = min(1, max(0, (y - band) / 1.6 + 0.5))
                                    value *= 1 - lidShadow * fade * fade * (3 - 2 * fade)
                                }
                            } else if y >= top && y < top + lashWidth * (1 - 0.45 * abs(x) / halfWidth) {
                                value = 44
                            } else if eyebrow && abs(x) < 0.84 * halfWidth {
                                let u = x / halfWidth
                                let arch = browHeight - 0.32 * radius * u * u + 0.07 * x
                                let hair = min(1, max(0, 0.14 * radius + 0.5 - abs(y - arch)))
                                value -= hair * 0.78 * (skin - pupil)
                            }
                        }
                        sum += value
                        x += across.x * 0.25
                        y += up.x * 0.25
                    }
                }
                samples[row * side + column] = exposure * (sum / 16 + illumination) * sideGain + blackLevel
            }
        }

        // The separable blur has a standard deviation of 0.7 pixel.
        var horizontal = samples
        for row in 0..<side {
            for column in 1..<(side - 1) {
                let i = row * side + column
                horizontal[i] = 0.245 * (samples[i - 1] + samples[i + 1]) + 0.51 * samples[i]
            }
        }
        var pixels = [UInt8](repeating: 0, count: side * stride)
        var random = SplitMix64(state: seed)
        for row in 1..<(side - 1) {
            for column in 1..<(side - 1) {
                let i = row * side + column
                let blur = 0.245 * (horizontal[i - side] + horizontal[i + side]) + 0.51 * horizontal[i]
                let jitter = noise > 0 ? noise * sqrt(-2 * log(max(0x1p-53, random.unit())))
                    * cos(2 * Double.pi * random.unit()) : 0
                pixels[row * stride + column] = UInt8(min(255, max(0, (blur + jitter).rounded())))
            }
        }
        return pixels
    }

    func locate(pixels: [UInt8]? = nil, eyeball: Point? = nil, expectedRadius: Double? = nil) -> IrisFit? {
        (pixels ?? render()).withUnsafeBufferPointer {
            IrisLocator.locate(in: LumaImage(base: $0.baseAddress!, width: side, height: side, bytesPerRow: stride),
                               eyeball: eyeball ?? self.eyeball + guessOffset,
                               expectedRadius: expectedRadius ?? self.expectedRadius, across: across, up: up)
        }
    }

    /// Give the circle baseline exact limbus points from the clean inner arc.
    func circleAcrossError() -> Double? {
        let side = displacement.x > 0 ? -1.0 : 1.0
        func edge(on direction: Point) -> Double {
            let projection = simd_dot(direction, displacement)
            return radius / sqrt(1 + shapeK * projection * projection)
        }
        var points: [Point] = []
        for index in -8...8 {
            let angle = Double(index) * Double.pi * 7 / 180
            let direction = Point(side * cos(angle), sin(angle))
            let outside = Point(-direction.x, direction.y)
            let outer = displacement + outside * (edge(on: outside) + 2)
            let (outerTop, outerBottom) = lids(at: outer.x)
            if abs(outer.x) < halfWidth, outer.y < outerTop, outer.y > outerBottom { return nil }
            let point = direction * edge(on: direction)
            let position = displacement + point
            let (top, bottom) = lids(at: position.x)
            if position.y < top - 1, position.y > bottom + 1 { points.append(point) }
        }
        guard points.count >= 5 else { return nil }
        let mean = points.reduce(Point.zero, +) / Double(points.count)
        var xx = 0.0, xy = 0.0, yy = 0.0
        var rhs = Point.zero
        for point in points {
            let p = point - mean
            xx += p.x * p.x
            xy += p.x * p.y
            yy += p.y * p.y
            rhs += p * (0.5 * simd_length_squared(point))
        }
        let det = xx * yy - xy * xy
        return abs((yy * rhs.x - xy * rhs.y) / det)
    }
}

private enum EyeSweep: String {
    case open, realistic, corner, light, wrongInputs, closed, noiseOnly
    case disc, dim, darkerSclera, darkSkin, bright, sideLight, lidShadow

    static let lighting: [Self] = [.dim, .darkerSclera, .darkSkin, .bright, .sideLight, .lidShadow]

    enum View: Double {
        case level = 0, below = 1, above = -1
    }

    struct Distribution {
        let values: [Double]
        var count: Int { values.count }
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
        let radius: Distribution
        let circleAcross: Distribution
        let pairedAcross: Distribution
        let count: Int
        let medianScore: Double
        let worstCenterError: Double
        var rate: Double { Double(across.count) / Double(count) }
        var description: String {
            String(format: "%@ %d/%d across %.3f/%.3f up %.3f/%.3f circle %.3f score %.3f",
                   name, across.count, count, across.median, across.p95, up.median, up.p95,
                   circleAcross.median, medianScore)
        }
    }

    static func measure(_ scenario: Self, radius: Double, count: Int, roll: Double? = nil,
                        view: View = .level) -> Result {
        var horizontal: [Double] = [], vertical: [Double] = [], scores: [Double] = [], circles: [Double] = []
        var radii: [Double] = [], paired: [Double] = [], worstCenterError = 0.0
        for index in 0..<count {
            let sample = eye(scenario, radius: radius, index: index, roll: roll, view: view)
            guard let fit = sample.locate(), fit.score >= 0.30 else { continue }
            let difference = SIMD2(fit.x, fit.y) - sample.center
            worstCenterError = max(worstCenterError, simd_length(difference))
            let acrossError = abs(simd_dot(difference, sample.across))
            horizontal.append(acrossError)
            vertical.append(abs(simd_dot(difference, sample.up)))
            radii.append(abs(fit.radius - sample.radius))
            scores.append(fit.score)
            if scenario == .corner, let error = sample.circleAcrossError() {
                circles.append(error)
                paired.append(acrossError)
            }
        }
        return Result(name: "\(scenario.rawValue) \(view) R=\(radius) roll=\(roll?.description ?? "mixed")",
                      across: Distribution(values: horizontal.sorted()), up: Distribution(values: vertical.sorted()),
                      radius: Distribution(values: radii.sorted()),
                      circleAcross: Distribution(values: circles.sorted()),
                      pairedAcross: Distribution(values: paired.sorted()), count: count,
                      medianScore: Distribution(values: scores.sorted()).median, worstCenterError: worstCenterError)
    }

    static func eye(_ scenario: Self, radius: Double, index: Int, roll: Double? = nil,
                    view: View = .level, origin: SIMD2<Double>? = nil, pose: SIMD2<Double>? = nil) -> SyntheticEye {
        var random = SplitMix64(state: 0x68BAE11A3 &+ UInt64(index) &* 0x9E3779B97F4A7C15)
        let disc = scenario == .disc
        let noiseOnly = scenario == .noiseOnly
        let clean = scenario == .open || disc
        let closed = scenario == .closed
        let corner = scenario == .corner
        let direction = random.unit() < 0.5 ? -1.0 : 1.0
        let across = corner ? direction * (1.04 + 0.15 * random.unit())
            : (2 * random.unit() - 1) * (view == .level ? (clean ? 0.48 : 0.78) : 1)
        let verticalSample = random.unit()
        let vertical = view == .level ? (2 * verticalSample - 1) * (corner ? 0.19 : (clean ? 0.18 : 0.34))
            : view.rawValue * (0.30 + 0.70 * verticalSample)
        let x = radius * (pose?.x ?? across), y = radius * (pose?.y ?? vertical)
        let halfWidth = radius * (corner ? 1.98 + 0.12 * random.unit() : 2.01 + 0.18 * random.unit())
        let lidSlope = 0.042
        let upperSample = random.unit(), lowerSample = random.unit()
        let upperCover = view == .level ? (clean ? 0.06 + 0.06 * upperSample : 0.15 + 0.20 * upperSample)
            : 0.05 + 0.25 * upperSample
        let d2 = x * x + y * y
        let chord = radius / sqrt(1 + y * y / max(3.24 * radius * radius - d2, 9.0 / 16 * d2))
        let lidCurve = 1 - pow(x / halfWidth, 2)
        let top = (y + (view == .level ? radius : chord) * (1 - 2 * upperCover) - x * lidSlope) / lidCurve
        let lower = view == .level ? radius * (0.82 + 0.34 * lowerSample)
            : (-y + chord * (1 - 0.40 * lowerSample) + x * lidSlope) / lidCurve
        let expected = scenario == .wrongInputs ? radius * (index % 2 == 0 ? 0.88 : 1.12) : radius
        let side = Double(Int(ceil(3.3 * max(radius, expected) + 13)))
        let eyeball = origin ?? SIMD2(side + random.unit() - 0.5, side + random.unit() - 0.5)
        let guessAngle = 2 * Double.pi * random.unit()
        let guess = scenario == .wrongInputs ? 2 * SIMD2(cos(guessAngle), sin(guessAngle)) : .zero
        let angle = roll ?? [0, Double.pi / 2, 0.63, -0.47][index % 4]
        let secondAngle = (2 * random.unit() - 1) * 0.55
        let edgeSide = random.unit() < 0.5 ? -1.0 : 1.0
        let highlight = SIMD2(edgeSide * cos(secondAngle), sin(secondAngle))
        let displacement = disc && pose == nil ? SIMD2<Double>.zero : SIMD2(x, y)
        let projection = simd_dot(highlight, displacement)
        let edge = radius / sqrt(1 + (projection * projection)
                                 / (3.24 * radius * radius - simd_length_squared(displacement)))
        let iris = scenario == .light ? 105 + 10 * random.unit() : 60 + 20 * random.unit()
        let reflectionRadius = clean || closed ? 0 : radius * (0.15 + 0.10 * random.unit())
        let lightSlope = SIMD2(0.45 * (2 * random.unit() - 1), 0.6 * (2 * random.unit() - 1))
        let skin: Double, sclera: Double, irisLevel: Double, pupil: Double
        switch scenario {
        case .disc: (skin, sclera, irisLevel, pupil) = (185, 185, 65, 65)
        case .dim: (skin, sclera, irisLevel, pupil) = (170, 200, iris, 29)
        case .darkerSclera: (skin, sclera, irisLevel, pupil) = (175, 150, 60, 22)
        case .darkSkin: (skin, sclera, irisLevel, pupil) = (70, 170, 30, 14)
        case .bright: (skin, sclera, irisLevel, pupil) = (220, 235, 125, 65)
        default: (skin, sclera, irisLevel, pupil) = (150, 200, iris, 29)
        }
        return SyntheticEye(radius: radius, roll: angle, eyeball: eyeball, guessOffset: guess,
                            expectedRadius: expected, displacement: displacement,
                            skin: skin, sclera: sclera, iris: irisLevel, pupil: pupil,
                            lidTop: closed ? 0.07 * radius : top, lidBottom: closed ? 0 : lower,
                            halfWidth: halfWidth, lashWidth: clean ? 0 : (closed ? 1.35 : 0.85),
                            reflections: (radius * SIMD2(-0.20, 0.19), edge * highlight),
                            reflectionRadius: reflectionRadius,
                            lightSlope: disc || noiseOnly ? .zero : lightSlope,
                            sideLight: scenario == .sideLight ? 0.35 * direction : 0,
                            lidShadow: scenario == .lidShadow ? 0.25 : 0,
                            exposure: scenario == .dim ? 0.34 : 1, blackLevel: scenario == .dim ? 2 : 0,
                            noise: clean ? 0 : (scenario == .dim ? 2.5 : 4),
                            seed: random.state, closed: closed, disc: disc, noiseOnly: noiseOnly,
                            eyebrow: view != .level)
    }
}
