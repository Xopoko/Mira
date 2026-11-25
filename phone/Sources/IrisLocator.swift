import Foundation
import simd

/// Borrowed 8-bit luma. Pixel centres in frame coordinates lie at half-integers.
struct LumaImage {
    let base: UnsafePointer<UInt8>
    let width: Int
    let height: Int
    let bytesPerRow: Int

    @inline(__always)
    fileprivate func sample(_ p: SIMD2<Double>) -> Double {
        let x = Int(p.x), y = Int(p.y)
        let fx = p.x - Double(x), fy = p.y - Double(y)
        let row = base + y * bytesPerRow + x
        let a = Double(row[0]), b = Double(row[1])
        let c = Double(row[bytesPerRow]), d = Double(row[bytesPerRow + 1])
        let top = a + fx * (b - a), bottom = c + fx * (d - c)
        return top + fy * (bottom - top)
    }

    fileprivate func contains(_ p: SIMD2<Double>) -> Bool {
        p.x >= 0 && p.y >= 0 && p.x < Double(width - 1) && p.y < Double(height - 1)
    }
}

struct IrisFit: Equatable {
    // Centre and longer semi-axis are in frame pixels.
    var x: Double
    var y: Double
    var radius: Double
    var score: Double
    var profiles: [[Double]]
}

enum IrisLocator {
    static let sectors = 4
    static let samples = 25
    static let step = 0.05

    private typealias Point = SIMD2<Double>
    private static let rotations = (-5...5).map { index in
        let angle = Double(index) * (.pi * 7 / 180)
        return Point(cos(angle), sin(angle))
    }

    /// The projected circle on a sphere, with its minor axis along the
    /// displacement from the eyeball. The axis ratio has a floor of 0.6.
    private struct Shape {
        let displacement: Point
        let k: Double

        init(_ displacement: Point, depthSquared: Double) {
            self.displacement = displacement
            let distanceSquared = simd_length_squared(displacement)
            k = 1 / max(depthSquared - distanceSquared, (9.0 / 16) * distanceSquared)
        }

        @inline(__always)
        func scale(_ direction: Point) -> Double {
            let projection = simd_dot(displacement, direction)
            return 1 / sqrt(1 + k * projection * projection)
        }

        func slope(along direction: Point, toward tangent: Point) -> Double {
            let projection = simd_dot(displacement, direction)
            return k * projection * simd_dot(displacement, tangent) / (1 + k * projection * projection)
        }
    }

    private struct Fit {
        var center: Point
        var radius: Double
    }

    private struct Edge {
        let offset: Double
        let strength: Double
    }

    private struct Photometry {
        let dark: Double
        let span: Double
        let noise: Double
    }

    private struct Search {
        let image: LumaImage
        let eye: Point
        let prediction: Point
        let across: Point
        let up: Point
        let radius: Double
        let extent: Double
        let rays: [Point]
        let needsBoundsCheck: Bool

        var depthSquared: Double { 3.24 * radius * radius }
        var minRadius: Double { 0.88 * radius }
        var maxRadius: Double { 1.12 * radius }

        func locate() -> IrisFit? {
            let light = photometry()
            guard var fit = coarse(light) else { return nil }
            for _ in 0..<3 {
                guard let profiles = profiles(at: fit) else { return nil }
                fit = recenter(fit, profiles: profiles)
            }
            guard let profiles = profiles(at: fit) else { return nil }
            return IrisFit(x: fit.center.x + 0.5, y: fit.center.y + 0.5, radius: fit.radius,
                           score: score(profiles, noise: light.noise), profiles: profiles)
        }

        private func coarse(_ light: Photometry) -> Fit? {
            // About one pixel at camera scale. Larger irises also have wider
            // contrast flanks, so sample their window at a coarser resolution.
            let n = min(13, Int(ceil(extent)))
            let spacing = extent / Double(n)
            let flank = max(1.65, 0.17 * radius)
            let radii = SIMD4(minRadius, 0.96 * radius, 1.04 * radius, maxRadius)
            let cap = SIMD4(repeating: 0.8 * light.span)
            let dark = SIMD4(repeating: light.dark)
            let one = SIMD4<Double>(repeating: 1)
            let inverseSpan = 1 / light.span
            var best = Fit(center: prediction, radius: radius)
            var bestMerit = 0.0
            for row in -n...n {
                let start = prediction + up * (Double(row) * spacing)
                for column in -n...n {
                    let center = start + across * (Double(column) * spacing)
                    let shape = Shape(center - eye, depthSquared: depthSquared)
                    var left = SIMD4<Double>.zero
                    var right = SIMD4<Double>.zero
                    var lower = SIMD4<Double>.zero
                    for sector in 0..<3 {
                        var support = SIMD4<Double>.zero
                        for index in stride(from: 1, through: 9, by: 2) {
                            let direction = rays[11 * sector + index]
                            let distances = radii * shape.scale(direction)
                            let a = center - flank * direction
                            let b = center + flank * direction
                            if needsBoundsCheck, !image.contains(a + distances.x * direction)
                                || !image.contains(b + distances.w * direction) { return nil }
                            let first = distances.x * direction, second = distances.y * direction
                            let third = distances.z * direction, fourth = distances.w * direction
                            let inside = SIMD4(image.sample(a + first), image.sample(a + second),
                                               image.sample(a + third), image.sample(a + fourth))
                            let outside = SIMD4(image.sample(b + first), image.sample(b + second),
                                                image.sample(b + third), image.sample(b + fourth))
                            let relative = (outside - dark) * (2 * inverseSpan)
                            let bright = simd_clamp(relative - one, .zero, one)
                            let difference = simd_clamp(outside - inside, -0.4 * cap, cap)
                            support += bright * difference
                        }
                        switch sector {
                        case 0: left = support
                        case 1: right = support
                        default: lower = support
                        }
                    }
                    for r in 0..<4 {
                        // Give the weaker side less influence when a corner
                        // leaves only one lateral arc visible.
                        let merit = max(left[r], right[r]) + 0.50 * min(left[r], right[r]) + 0.55 * lower[r]
                        if merit > bestMerit {
                            best = Fit(center: center, radius: radii[r])
                            bestMerit = merit
                        }
                    }
                }
            }
            return best
        }

        private func photometry() -> Photometry {
            // Use percentiles for the local luma range and the median adjacent
            // difference for the noise floor. Both resist isolated glints.
            withUnsafeTemporaryAllocation(of: Int.self, capacity: 512) { bins in
                bins.initialize(repeating: 0)
                var levels = 0, differences = 0
                for row in -5...5 {
                    for column in -5...5 {
                        let p = prediction + (0.3 * radius) * (Double(column) * across + Double(row) * up)
                        if !needsBoundsCheck || image.contains(p) {
                            let level = image.sample(p)
                            bins[Int(level)] += 1
                            levels += 1
                            let adjacent = p + Point(1, 0)
                            if !needsBoundsCheck || image.contains(adjacent) {
                                bins[256 + Int(abs(image.sample(adjacent) - level))] += 1
                                differences += 1
                            }
                        }
                    }
                }
                func percentile(start: Int, rank: Int) -> Double {
                    var cumulative = 0
                    for value in 0..<256 {
                        cumulative += bins[start + value]
                        if cumulative >= rank { return Double(value) }
                    }
                    return 0
                }
                let dark = percentile(start: 0, rank: (levels + 9) / 10)
                let bright = percentile(start: 0, rank: (9 * levels + 9) / 10)
                let noise = percentile(start: 256, rank: (differences + 1) / 2)
                return Photometry(dark: dark, span: max(1, bright - dark), noise: noise)
            }
        }

        private func profiles(at fit: Fit) -> [[Double]]? {
            let shape = Shape(fit.center - eye, depthSquared: depthSquared)
            let spacing = step * fit.radius
            let reach = 12 * spacing
            var result = [[Double]]()
            result.reserveCapacity(sectors)
            for sector in 0..<sectors {
                var profile = [Double]()
                profile.reserveCapacity(samples)
                let offsets = (0..<11).map { index in
                    fit.center + (fit.radius * shape.scale(rays[sector * 11 + index])) * rays[sector * 11 + index]
                }
                if needsBoundsCheck {
                    for index in 0..<11 {
                        let delta = reach * rays[sector * 11 + index]
                        if !image.contains(offsets[index] - delta) || !image.contains(offsets[index] + delta) {
                            return nil
                        }
                    }
                }
                for j in 0..<samples {
                    let offset = Double(j - 12) * spacing
                    var sum = 0.0
                    var low = Double.infinity, secondLow = Double.infinity
                    var high = -Double.infinity, secondHigh = -Double.infinity
                    for index in 0..<11 {
                        let value = image.sample(offsets[index] + offset * rays[sector * 11 + index])
                        sum += value
                        secondLow = min(secondLow, max(low, value))
                        secondHigh = max(secondHigh, min(high, value))
                        low = min(low, value)
                        high = max(high, value)
                    }
                    profile.append((sum - low - secondLow - high - secondHigh) / 7)
                }
                result.append(profile)
            }
            return result
        }

        private func edges(in p: [Double], spacing: Double) -> [Edge] {
            // The difference of two two-sample averages is a lightly smoothed derivative.
            func rise(_ i: Int) -> Double {
                (p[i + 2] + p[i + 1] - p[i - 1] - p[i - 2]) * 0.5
            }
            var result = [Edge]()
            for i in 6...18 {
                let left = rise(i - 1), peak = rise(i), right = rise(i + 1)
                guard peak > 0, peak >= left, peak > right else { continue }
                let outside = (p[i + 5] + p[i + 4]) * 0.5
                let inside = (p[i - 5] + p[i - 4]) * 0.5
                let contrast = (outside - inside) / max(1, outside + inside)
                guard contrast > 0.045 else { continue }
                let fraction = 0.5 * (left - right) / (left + right - 2 * peak)
                result.append(Edge(offset: (Double(i - 12) + fraction) * spacing, strength: contrast))
            }
            return result
        }

        private func lowerOffset(_ edge: Edge, profiles: [[Double]], spacing: Double) -> Double {
            let p = profiles[2]
            let peak = 12 + Int((edge.offset / spacing).rounded())
            let crest = p[(peak + 1)...min(24, peak + 8)].max()!
            // A dark lid just outside the lower limbus pulls the derivative
            // peak inwards. Estimate the opposing step from the outer flank.
            let sclera = 0.5 * (profiles[0][14...23].max()! + profiles[1][14...23].max()!)
            let inside = p[max(0, peak - 7)...(peak - 3)].min()!
            let contrast = sclera - inside
            guard contrast > 0, crest > inside else { return edge.offset }
            let difference = sclera - (p[22] + p[23] + p[24]) / 3
            let drop = min(contrast, max(-contrast, difference))
            guard drop > 0 || -drop > 0.15 * contrast else { return edge.offset }
            let slope = (p[peak + 1] - p[peak - 1]) / (2 * spacing)
            guard slope > 0 else { return edge.offset }
            let blur = min(2.5, contrast / (sqrt(2 * .pi) * slope))
            let fall = ((peak + 2)...23).min { p[$0 + 1] - p[$0 - 1] < p[$1 + 1] - p[$1 - 1] }!
            let separation = Double(fall - peak) * spacing
            let distance = max(0.6 * blur, max(0, separation * separation - 3 * blur * blur).squareRoot())
            let z = distance / blur
            // First-order shift of the peak for two Gaussian-blurred steps.
            return edge.offset + drop / contrast * distance * exp(-0.5 * z * z)
        }

        private func recenter(_ fit: Fit, profiles: [[Double]]) -> Fit {
            let spacing = step * fit.radius
            let left = edges(in: profiles[0], spacing: spacing)
            let right = edges(in: profiles[1], spacing: spacing)
            let lower = edges(in: profiles[2], spacing: spacing)
            let upper = edges(in: profiles[3], spacing: spacing)
            let shape = Shape(fit.center - eye, depthSquared: depthSquared)
            let acrossScale = shape.scale(across)
            var x = 0.0, y = 0.0
            var newRadius = fit.radius
            var bestPair = 0.0
            let horizontalPosition = simd_dot(fit.center - eye, across)
            let outward = horizontalPosition < 0 ? 0 : 1
            let weakOuter = abs(horizontalPosition) > 0.65 * radius
                && profiles[outward][15...22].max()! < 0.82 * profiles[1 - outward][15...22].max()!
            let bestLeft = weakOuter && outward == 0 ? nil : left.max { $0.strength < $1.strength }
            let bestRight = weakOuter && outward == 1 ? nil : right.max { $0.strength < $1.strength }
            for a in left {
                for b in right where !weakOuter {
                    let r = fit.radius + 0.5 * (a.offset + b.offset) / acrossScale
                    let support = min(a.strength, b.strength)
                    if r >= minRadius, r <= maxRadius, support > bestPair {
                        bestPair = support
                        x = 0.5 * (b.offset - a.offset)
                        newRadius = r
                    }
                }
            }
            if bestPair == 0 {
                if let a = bestLeft, a.strength > (bestRight?.strength ?? 0) {
                    x = -a.offset
                } else if let b = bestRight {
                    x = b.offset
                }
            }
            if let a = lower.max(by: { $0.strength < $1.strength }) {
                let verticalScale = shape.scale(up)
                let radiusDelta = (newRadius - fit.radius) * verticalScale
                let offset = lowerOffset(a, profiles: profiles, spacing: spacing)
                let scaleX = (Shape(fit.center - eye + 0.25 * across, depthSquared: depthSquared).scale(up) -
                              Shape(fit.center - eye - 0.25 * across, depthSquared: depthSquared).scale(up)) * 2
                let scaleY = (Shape(fit.center - eye + 0.25 * up, depthSquared: depthSquared).scale(up) -
                              Shape(fit.center - eye - 0.25 * up, depthSquared: depthSquared).scale(up)) * 2
                let slope = shape.slope(along: up, toward: across)
                y = (radiusDelta - offset + (fit.radius * scaleX - slope) * x) / max(0.65, 1 - fit.radius * scaleY)
                if let b = upper.max(by: { $0.strength < $1.strength }),
                   b.strength >= 0.85 * a.strength,
                   abs(offset + b.offset - 2 * radiusDelta) <= 2 * spacing {
                    y = 0.5 * (b.offset - offset) - slope * x
                }
            }
            x -= shape.slope(along: across, toward: up) * y
            let relative = fit.center - prediction
            let horizontal = min(extent, max(-extent, simd_dot(relative, across) + x))
            let vertical = min(extent, max(-extent, simd_dot(relative, up) + y))
            return Fit(center: prediction + horizontal * across + vertical * up, radius: newRadius)
        }

        /// Sum the nonnegative lateral Michelson contrasts between the flanks
        /// of the limbus, discounting a local median adjacent-pixel difference
        /// from each rise. The gain lets one unobscured side support an eye in
        /// the corner. The result is clipped to one.
        private func score(_ profiles: [[Double]], noise: Double) -> Double {
            var sum = 0.0
            for k in 0..<2 {
                let p = profiles[k]
                let inside = p[6] + p[7] + p[8]
                let outside = p[16] + p[17] + p[18]
                sum += max(0, (outside - inside - 3 * noise) / max(1, outside + inside))
            }
            return min(1, 1.45 * sum)
        }
    }

    static func locate(in image: LumaImage, eyeball: SIMD2<Double>, expectedRadius: Double,
                       across: SIMD2<Double>, up: SIMD2<Double>, near predicted: SIMD2<Double>) -> IrisFit? {
        guard (5...80).contains(expectedRadius), image.width > 1, image.height > 1,
              image.bytesPerRow >= image.width, image.bytesPerRow <= Int.max / image.height,
              eyeball.x.isFinite, eyeball.y.isFinite, predicted.x.isFinite, predicted.y.isFinite else { return nil }
        let acrossLength = simd_length(across), upLength = simd_length(up)
        guard acrossLength.isFinite, acrossLength > 0, upLength.isFinite, upLength > 0 else { return nil }
        let horizontal = across / acrossLength
        let perpendicular = Point(-horizontal.y, horizontal.x)
        let orientation = simd_dot(up / upLength, perpendicular)
        guard abs(orientation) > 0.1 else { return nil }
        let vertical = orientation < 0 ? -perpendicular : perpendicular
        let extent = 0.7 * expectedRadius + 1
        let origin = Point(repeating: 0.5)
        let prediction = predicted - origin
        guard image.contains(prediction), simd_length_squared(prediction - eyeball).isFinite else { return nil }
        let reach = max(1.12 * expectedRadius + max(1.65, 0.17 * expectedRadius), 1.12 * 1.6 * expectedRadius)
        let limit = extent * (simd_abs(horizontal) + simd_abs(vertical)) + Point(repeating: reach)
        let needsBoundsCheck = !image.contains(prediction - limit) || !image.contains(prediction + limit)
        let rays = [-horizontal, horizontal, -vertical, vertical].flatMap { axis in
            let tangent = Point(-axis.y, axis.x)
            return rotations.map { axis * $0.x + tangent * $0.y }
        }
        return Search(image: image, eye: eyeball - origin, prediction: prediction, across: horizontal, up: vertical,
                      radius: expectedRadius, extent: extent, rays: rays, needsBoundsCheck: needsBoundsCheck).locate()
    }
}
