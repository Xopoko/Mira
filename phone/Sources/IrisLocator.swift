import Foundation
import simd

/// Borrowed 8-bit luma. The top-left corner is (0, 0), and pixel centres are at half-integer coordinates.
struct LumaImage {
    let base: UnsafePointer<UInt8>
    let width: Int
    let height: Int
    let bytesPerRow: Int
}

struct IrisFit: Equatable {
    // Centre and longer semi-axis are in image pixels.
    var x: Double
    var y: Double
    var radius: Double
    // Weighted edge support in 0...1.
    var score: Double
}

enum IrisLocator {
    private typealias Point = SIMD2<Double>

    private struct Ray {
        let direction: Point
        let weight: Double
    }

    private struct Edge {
        let offset: Point
        let weight: Double
    }

    private struct Seed {
        let center: Point
        let radius: Double
        let score: Double
    }

    private struct Contrast {
        let high: Double
        let span: Double
        let noise: Double
        let brightTilt: Double

        var scale: Double { 1 / span }
        var floor: Double { max(0.045 * span, 1.5 * noise) }
        var highlight: Double { high + max(0.055 * span, 2.1 * noise) }
    }

    private static let angles = (0...16).map { i -> Ray in
        let angle = Double(i - 8) * (Double.pi * 7 / 180)
        let weight = max(0.30, 1 - (angle >= 0 ? 1.05 : 0.62) * angle * angle)
        return Ray(direction: Point(cos(angle), sin(angle)), weight: weight)
    }

    private static let radiusFactors = SIMD4(0.88, 0.96, 1.05, 1.13)
    private static let arcWeight = 2 * angles.reduce(0.0) { $0 + $1.weight }

    /// Ellipse metric I + k ddᵀ, with the minor axis parallel to the displacement d.
    private struct Shape {
        let displacement: Point
        let k: Double
        let halfDerivative: Double

        init(_ displacement: Point, _ depthSquared: Double) {
            self.displacement = displacement
            let distanceSquared = simd_length_squared(displacement)
            if depthSquared - distanceSquared >= 0.36 * depthSquared {
                k = 1 / (depthSquared - distanceSquared)
                halfDerivative = k * k
            } else {
                k = (16.0 / 9) / distanceSquared
                halfDerivative = -k / distanceSquared
            }
        }

        func scale(_ direction: Point) -> Double {
            let projection = simd_dot(displacement, direction)
            return 1 / sqrt(1 + k * projection * projection)
        }

        func normal(at offset: Point) -> Point {
            simd_normalize(offset + (k * simd_dot(offset, displacement)) * displacement)
        }

        func residual(at offset: Point, radius: Double) -> (Double, SIMD3<Double>) {
            let projection = simd_dot(offset, displacement)
            let length = sqrt(simd_length_squared(offset) + k * projection * projection)
            let gradient = (-offset + (k * projection) * (offset - displacement)
                            + (halfDerivative * projection * projection) * displacement) / length
            return (length - radius, SIMD3(gradient.x, gradient.y, -1))
        }
    }

    private struct Crop {
        let pixels: UnsafeBufferPointer<UInt8>
        let stride: Int
        let height: Int
        let source: UnsafePointer<UInt8>
        let sourceStride: Int

        @inline(__always)
        func nearest(_ p: Point) -> Double {
            Double(pixels[Int(p.y + 0.5) * stride + Int(p.x + 0.5)])
        }

        @inline(__always)
        func raw(_ p: Point) -> Double {
            Double(source[Int(p.y + 0.5) * sourceStride + Int(p.x + 0.5)])
        }

        @inline(__always)
        func rawClamped(_ p: Point) -> Double {
            raw(Point(min(Double(stride - 1), max(0, p.x)), min(Double(height - 1), max(0, p.y))))
        }

        @inline(__always)
        func sample(_ p: Point) -> Double {
            let x = Int(p.x), y = Int(p.y)
            let fx = p.x - Double(x), fy = p.y - Double(y)
            let i = y * stride + x
            let a = Double(pixels[i]), b = Double(pixels[i + 1])
            let c = Double(pixels[i + stride]), d = Double(pixels[i + stride + 1])
            let top = a + fx * (b - a), bottom = c + fx * (d - c)
            return top + fy * (bottom - top)
        }
    }

    private struct Search {
        let crop: Crop
        let eye: Point
        let across: Point
        let up: Point
        let radius: Double
        let extent: Double

        var depthSquared: Double { 3.24 * radius * radius }
        var minRadius: Double { 0.86 * radius }
        var maxRadius: Double { 1.16 * radius }

        func locate() -> IrisFit? {
            guard let contrast = contrast() else { return nil }
            let (seed, backup, lower, upper) = coarseFit(contrast)
            guard seed.score > 0.35 else { return nil }
            return withUnsafeTemporaryAllocation(of: Edge.self, capacity: 2 * IrisLocator.angles.count) { edges in
                edges.initialize(repeating: Edge(offset: .zero, weight: 0))
                var fit = refine(seed, contrast, into: edges)
                if (fit?.score ?? 0) < 0.50 || abs((fit?.radius ?? radius) - radius) > 0.09 * radius,
                   backup.score > 0.35,
                   simd_length(seed.center - backup.center) + abs(seed.radius - backup.radius) > 0.5,
                   let second = refine(backup, contrast, into: edges), second.score > (fit?.score ?? 0) {
                    fit = second
                }
                // A weak fit can follow the lid into another row of the wider search window.
                if (fit?.score ?? 0) < 0.54 {
                    for alternate in [lower, upper] where alternate.score > 0.68 * seed.score {
                        if let next = refine(alternate, contrast, into: edges),
                           next.score > (fit?.score ?? 0) + 0.025 {
                            fit = next
                        }
                    }
                }
                return (fit?.score ?? 0) >= 0.30 ? fit : nil
            }
        }

        private func refine(_ seed: Seed, _ contrast: Contrast,
                            into edges: UnsafeMutableBufferPointer<Edge>) -> IrisFit? {
            var center = seed.center
            var fittedRadius = seed.radius
            var score = 0.0
            for pass in 0..<4 {
                let count = trace(center: center, radius: fittedRadius, contrast: contrast,
                                  reach: pass == 0 ? max(3.0, 0.15 * radius) : 2.2, into: edges)
                let observations = UnsafeBufferPointer(start: edges.baseAddress, count: count)
                guard count >= 6 else { return nil }
                let startingCenter = center
                for iteration in 0..<3 {
                    let cutoff = pass == 0 && iteration == 0 ? 3.3 : (pass < 2 ? 1.45 : 0.90)
                    guard let step = correction(center: center, radius: fittedRadius, origin: startingCenter,
                                                edges: observations, cutoff: cutoff) else { break }
                    let delta = center + Point(step.x, step.y) - eye
                    center = eye + across * min(extent, max(-extent, simd_dot(delta, across)))
                        + up * min(extent, max(-extent, simd_dot(delta, up)))
                    fittedRadius = min(maxRadius, max(minRadius, fittedRadius + step.z))
                }
                if pass == 3 {
                    score = support(center: center, radius: fittedRadius, origin: startingCenter, edges: observations)
                }
            }
            return IrisFit(x: center.x, y: center.y, radius: fittedRadius, score: score)
        }

        private func contrast() -> Contrast? {
            withUnsafeTemporaryAllocation(of: Int.self, capacity: 1024) { bins in
                bins.initialize(repeating: 0)
                for row in -6...6 {
                    let start = eye + up * (Double(row) * 0.20 * radius)
                    for column in -11...11 {
                        let point = start + across * (Double(column) * 0.18 * radius)
                        let raw = crop.raw(point)
                        let luma = Int(crop.nearest(point))
                        bins[luma] += 1
                        if column != 0 {
                            bins[(column < 0 ? 512 : 768) + luma] += 1
                        }
                        bins[256 + Int(abs(raw - crop.raw(point + Point(1, 0))))] += 1
                        bins[256 + Int(abs(raw - crop.raw(point + Point(0, 1))))] += 1
                    }
                }
                func quantile(_ from: Int, _ rank: Int) -> Double {
                    var total = 0
                    for value in 0...255 {
                        total += bins[from + value]
                        if total >= rank { return Double(value) }
                    }
                    return 0
                }
                let low = quantile(0, 30)
                let high = quantile(0, 276)
                let span = high - low
                // The median adjacent difference estimates sensor noise without counting isolated edges.
                let noise = quantile(256, 299) / 0.954
                guard span > 4.0 * noise else { return nil }
                let midDifference = quantile(768, 100) - quantile(512, 100)
                let highDifference = quantile(768, 132) - quantile(512, 132)
                // Shift the bright reference only if a broad range of tones tilts the same way.
                let tilt = midDifference * highDifference > 0
                    ? min(abs(midDifference), abs(highDifference), 0.30 * span) * (highDifference > 0 ? 0.5 : -0.5) : 0
                return Contrast(high: high, span: span, noise: noise, brightTilt: tilt)
            }
        }

        private func coarseFit(_ contrast: Contrast) -> (Seed, Seed, Seed, Seed) {
            let step = max(1.45, 0.15 * radius)
            let n = Int(ceil(extent / step))
            return withUnsafeTemporaryAllocation(of: Seed.self, capacity: 2 * n + 1) { rows in
                coarseFit(contrast, gridRadius: n, rows: rows)
            }
        }

        private func coarseFit(_ contrast: Contrast, gridRadius n: Int,
                               rows: UnsafeMutableBufferPointer<Seed>) -> (Seed, Seed, Seed, Seed) {
            let baseline = max(1.6, 0.10 * radius)
            let scale = contrast.scale, floor = contrast.floor, bright = contrast.highlight
            let empty = Seed(center: eye, radius: radius, score: 0)
            var best = empty, nominal = empty
            var bestRow = 0
            let spacing = extent / Double(n)
            for j in -n...n {
                let vertical = up * (Double(j) * spacing)
                var row = empty
                for i in -n...n {
                    let delta = across * (Double(i) * spacing) + vertical
                    let center = eye + delta
                    let shape = Shape(delta, depthSquared)
                    var sums = SIMD4<Double>.zero
                    for index in stride(from: 0, to: IrisLocator.angles.count, by: 3) {
                        let ray = IrisLocator.angles[index]
                        for side in [-1.0, 1.0] {
                            let direction = across * (side * ray.direction.x) + up * ray.direction.y
                            let distance = radius * shape.scale(direction)
                            let normal = shape.normal(at: direction)
                            let tangent = baseline * Point(-normal.y, normal.x)
                            let radii = IrisLocator.radiusFactors * distance
                            let inner = radii - baseline
                            let outer = radii + baseline
                            for r in 0..<4 {
                                let inside = crop.nearest(center + direction * inner[r])
                                let outside = crop.nearest(center + direction * outer[r])
                                let strength = min(0.8, max(0, (outside - inside - floor) * scale))
                                guard strength > 0 else { continue }
                                let excess = max(0, (outside - bright) * scale)
                                let edge = center + direction * radii[r]
                                let tangential = abs(crop.nearest(edge + tangent) - crop.nearest(edge - tangent)) * scale
                                let merit = max(0, strength - 0.75 * tangential)
                                sums[r] += ray.weight * merit / (1 + 6 * excess)
                            }
                        }
                    }
                    for r in 0..<4 {
                        let factor = IrisLocator.radiusFactors[r]
                        let score = sums[r] * (1 - 8 * (factor - 1) * (factor - 1))
                        if score > row.score { row = Seed(center: center, radius: radius * factor, score: score) }
                        if (r == 1 || r == 2) && score > nominal.score {
                            nominal = Seed(center: center, radius: radius * factor, score: score)
                        }
                    }
                }
                rows.initializeElement(at: j + n, to: row)
                if row.score > best.score {
                    best = row
                    bestRow = j + n
                }
            }
            var lower = empty, upper = empty
            for j in rows.indices {
                if j < bestRow, rows[j].score > lower.score { lower = rows[j] }
                if j > bestRow, rows[j].score > upper.score { upper = rows[j] }
            }
            return (best, nominal, lower, upper)
        }

        private func trace(center: Point, radius: Double, contrast: Contrast, reach: Double,
                           into edges: UnsafeMutableBufferPointer<Edge>) -> Int {
            let shape = Shape(center - eye, depthSquared)
            let spacing = 0.55
            let flank = max(2.0, min(3.0, radius * 0.16))
            let count = Int(ceil(reach / spacing))
            let scale = contrast.scale, threshold = contrast.floor, bright = contrast.highlight
            let glintReach = 0.43 * radius + 1.3
            var written = 0
            for ray in IrisLocator.angles {
                for side in [-1.0, 1.0] {
                    let direction = across * (side * ray.direction.x) + up * ray.direction.y
                    let distance = radius * shape.scale(direction)
                    let normal = shape.normal(at: direction)
                    let tangent = Point(-normal.y, normal.x)
                    let obliquity = simd_dot(direction, tangent)
                    let radialToNormal = 1 / simd_dot(direction, normal)
                    let predicted = center + distance * direction
                    let stride = direction * spacing
                    let start = predicted - Double(count + 2) * stride
                    var lower = crop.sample(start + stride)
                    var middle = crop.sample(start + 2 * stride)
                    var left = middle - crop.sample(start)
                    var peak = 0.0
                    var position = 0.0
                    var edgeWeight = 0.0
                    var previous = 0.0
                    for index in -count...(count + 1) {
                        let upper = crop.sample(predicted + Double(index + 1) * stride)
                        let right = upper - lower
                        if index > -count, left > threshold, left >= right, left >= previous {
                            let curve = previous + right - 2 * left
                            let fraction = curve < 0 ? 0.5 * (previous - right) / curve : 0
                            let candidate = (Double(index - 1) + fraction) * spacing
                            let point = predicted + candidate * direction
                            let inside = crop.sample(point - flank * direction)
                            let outside = crop.sample(point + flank * direction)
                            let strength = min(1, max(0, (outside - inside - threshold) * scale))
                            let tangential = 0.5 * (crop.sample(point + tangent) - crop.sample(point - tangent))
                            let normalGradient = (left / (2 * spacing) - obliquity * tangential) * radialToNormal
                            let alignment = max(0, 1 - 1.5 * abs(tangential) / max(threshold, normalGradient))
                            let horizontal = simd_dot(point - eye, across) / self.radius
                            let localBright = bright + contrast.brightTilt * min(1, max(-1, horizontal))
                            let glare = glint(at: point, outward: direction, tangent: tangent, flank: flank,
                                              reach: glintReach, scale: scale, bright: localBright)
                            let weight = alignment * alignment * glare / (1 + 0.045 * candidate * candidate)
                            let merit = strength * strength * weight
                            if merit > peak {
                                position = candidate
                                peak = merit
                                edgeWeight = alignment * glare * min(1, strength / 0.47)
                            }
                        }
                        previous = left
                        left = right
                        lower = middle
                        middle = upper
                    }
                    guard distance + position > 0 else { continue }
                    let weight = ray.weight * edgeWeight
                    if weight > 0.07 {
                        edges[written] = Edge(offset: (distance + position) * direction, weight: weight)
                        written += 1
                    }
                }
            }
            return written
        }

        private func glint(at point: Point, outward: Point, tangent: Point, flank: Double,
                           reach: Double, scale: Double, bright: Double) -> Double {
            let outer = point + (flank + 0.5) * outward
            let inner = point + 1.25 * outward
            let raw = max(crop.raw(point), crop.raw(inner), crop.raw(outer),
                          crop.raw(inner + tangent), crop.raw(inner - tangent))
            guard raw > bright else { return 1 }
            // A broad bright surface has neighbours at the same level. An isolated glint does not.
            let lateral = (0.52 * reach + 0.7) * tangent
            let lateralLevel = max(crop.rawClamped(outer + lateral), crop.rawClamped(outer - lateral))
            let radialLevel = crop.rawClamped(point + reach * outward)
            let excess = (raw - bright) * scale
            let isolation = max(0, (raw - max(radialLevel, lateralLevel)) * scale)
            return max(0, 1 - min(15 * excess, 13 * isolation))
        }

        private func correction(center: Point, radius: Double, origin: Point,
                                edges: UnsafeBufferPointer<Edge>, cutoff: Double) -> SIMD3<Double>? {
            let shape = Shape(center - eye, depthSquared)
            let shift = origin - center
            var xx = 0.035, xy = 0.0, xz = 0.0
            var yy = 0.035, yz = 0.0, zz = 0.055
            var rhs = SIMD3<Double>.zero
            var sides = Point.zero
            for edge in edges {
                let (error, gradient) = shape.residual(at: edge.offset + shift, radius: radius)
                let robust = max(0, 1 - (error / cutoff) * (error / cutoff))
                let weighted = (edge.weight * robust * robust) * gradient
                xx += weighted.x * gradient.x
                xy += weighted.x * gradient.y
                xz += weighted.x * gradient.z
                yy += weighted.y * gradient.y
                yz += weighted.y * gradient.z
                zz += weighted.z * gradient.z
                rhs -= error * weighted
                sides[simd_dot(edge.offset + shift, across) < 0 ? 0 : 1] -= weighted.z
            }
            // Radius and horizontal position are coupled when most of one arc is hidden.
            let balance = min(sides.x, sides.y) / max(0.01, max(sides.x, sides.y))
            let sideways = abs(simd_dot(center - eye, across)) / self.radius
            let occlusion = min(1, max(0.15, 3 * (sideways - 0.72)))
            let prior = 0.70 * occlusion * (1 - balance) * (1 - balance)
            zz += prior
            rhs.z += prior * (self.radius - radius)
            let determinant = xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz)
                + xz * (xy * yz - yy * xz)
            guard determinant > 0.001 else { return nil }
            let x = simd_dot(rhs, SIMD3(yy * zz - yz * yz, xz * yz - xy * zz, xy * yz - xz * yy))
            let y = simd_dot(rhs, SIMD3(xz * yz - xy * zz, xx * zz - xz * xz, xy * xz - xx * yz))
            let z = simd_dot(rhs, SIMD3(xy * yz - xz * yy, xy * xz - xx * yz, xx * yy - xy * xy))
            let step = SIMD3(x, y, z) / determinant
            return step / max(1, simd_length(step) / 1.5)
        }

        private func support(center: Point, radius: Double, origin: Point,
                             edges: UnsafeBufferPointer<Edge>) -> Double {
            let shape = Shape(center - eye, depthSquared)
            var support = 0.0
            for edge in edges {
                let error = shape.residual(at: edge.offset + (origin - center), radius: radius).0
                support += edge.weight * max(0, 1 - abs(error) / 1.15)
            }
            let sideways = abs(simd_dot(center - eye, across)) / self.radius
            let visibility = max(0.59, 1 - max(0, sideways - 0.65))
            return min(1, 1.25 * support / (IrisLocator.arcWeight * visibility))
        }
    }

    /// The eye axes are in image coordinates and the radius is the longer limbus semi-axis.
    static func locate(in image: LumaImage, eyeball: SIMD2<Double>, expectedRadius: Double,
                       across: SIMD2<Double>, up: SIMD2<Double>) -> IrisFit? {
        guard (5...80).contains(expectedRadius), image.width > 0, image.height > 0,
              image.bytesPerRow >= image.width, image.bytesPerRow <= Int.max / image.height,
              eyeball.x.isFinite, eyeball.y.isFinite else { return nil }
        let acrossLength = simd_length(across)
        guard acrossLength.isFinite, acrossLength > 0 else { return nil }
        let horizontal = across / acrossLength
        let perpendicular = Point(-horizontal.y, horizontal.x)
        let orientation = simd_dot(up, perpendicular)
        guard orientation.isFinite, abs(orientation) > 0.1 else { return nil }
        let vertical = orientation < 0 ? -perpendicular : perpendicular
        let extent = 1.3 * expectedRadius + 2
        let limit = extent * (simd_abs(horizontal) + simd_abs(vertical))
            + Point(repeating: 1.34 * expectedRadius + 8)
        let first = eyeball - limit, last = eyeball + limit
        guard first.x >= 1, first.y >= 1, last.x < Double(image.width - 1),
              last.y < Double(image.height - 1) else { return nil }
        let x = Int(first.x), y = Int(first.y)
        let width = Int(last.x) + 1 - x, height = Int(last.y) + 1 - y
        let origin = Point(Double(x) + 0.5, Double(y) + 0.5)

        // A separable binomial blur keeps noise out of the radial derivative.
        return withUnsafeTemporaryAllocation(of: UInt16.self, capacity: width * (height + 2)) { rows in
            for row in 0..<(height + 2) {
                let source = image.base + (y + row - 1) * image.bytesPerRow + x - 1
                for column in 0..<width {
                    rows.initializeElement(at: row * width + column,
                                           to: UInt16(source[column]) + 2 * UInt16(source[column + 1])
                                               + UInt16(source[column + 2]))
                }
            }
            return withUnsafeTemporaryAllocation(of: UInt8.self, capacity: width * height) { pixels in
                for index in 0..<pixels.count {
                    let sum = rows[index] + 2 * rows[index + width] + rows[index + 2 * width]
                    pixels.initializeElement(at: index, to: UInt8((sum + 8) / 16))
                }
                let crop = Crop(pixels: UnsafeBufferPointer(pixels), stride: width, height: height,
                                source: image.base + y * image.bytesPerRow + x, sourceStride: image.bytesPerRow)
                var fit = Search(crop: crop, eye: eyeball - origin, across: horizontal,
                                 up: vertical, radius: expectedRadius, extent: extent).locate()
                fit?.x += origin.x
                fit?.y += origin.y
                return fit
            }
        }
    }
}
