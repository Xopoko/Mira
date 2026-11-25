import ARKit
import CoreVideo
import simd

/// Where both irises are in the camera frame, measured on the device. The
/// frame's brightness plane is read here and never leaves the phone. Only the
/// numbers returned below are added to a reply.
enum IrisReading {
    /// Half the visible width of an iris, in metres. It varies little between people.
    static let limbusRadius = 0.0059

    static func measure(frame: ARFrame, face: ARFaceAnchor) -> [String: Any]? {
        let buffer = frame.capturedImage
        let format = CVPixelBufferGetPixelFormatType(buffer)
        guard format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ||
              format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange else { return nil }
        let width = CVPixelBufferGetWidthOfPlane(buffer, 0), height = CVPixelBufferGetHeightOfPlane(buffer, 0)
        let size = CGSize(width: width, height: height)
        // ARKit projects into a view. Undoing the display transform gives the
        // pixel of the captured frame, front-camera mirroring included.
        let toFrame = frame.displayTransform(for: .landscapeRight, viewportSize: size).inverted()
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return nil }
        let image = LumaImage(base: UnsafePointer(base.assumingMemoryBound(to: UInt8.self)), width: width,
                              height: height, bytesPerRow: CVPixelBufferGetBytesPerRowOfPlane(buffer, 0))
        return reading(image: image, camera: frame.camera.transform, face: face.transform,
                       eyes: (face.leftEyeTransform, face.rightEyeTransform)) { world in
            let point = frame.camera.projectPoint(SIMD3(world.x, world.y, world.z), orientation: .landscapeRight,
                                                  viewportSize: size)
            let unit = CGPoint(x: point.x/size.width, y: point.y/size.height).applying(toFrame)
            return SIMD2(Double(unit.x)*Double(width), Double(unit.y)*Double(height))
        }
    }

    /// `pixel` maps a point in world space to continuous frame pixels.
    static func reading(image: LumaImage, camera: simd_float4x4, face: simd_float4x4,
                        eyes: (simd_float4x4, simd_float4x4),
                        pixel: (SIMD4<Float>) -> SIMD2<Double>) -> [String: Any]? {
        // The PC needs the same mapping to turn an iris pixel back into a ray:
        // pixel = shift + map * (x, y) / -z for a point in camera space.
        let shift = pixel(camera * SIMD4(0, 0, -1, 1))
        let alongX = (pixel(camera * SIMD4(0.1, 0, -1, 1)) - shift) * 10
        let alongY = (pixel(camera * SIMD4(0, 0.1, -1, 1)) - shift) * 10
        let scale = (alongX.x*alongY.y - alongX.y*alongY.x).magnitude.squareRoot()
        let map = [alongX.x, alongY.x, alongX.y, alongY.y, shift.x, shift.y]
        guard map.allSatisfy({ $0.isFinite }), scale > 1 else { return nil }

        func eye(_ transform: simd_float4x4) -> Any {
            let centre = (face * transform).columns.3
            let depth = Double(-(simd_inverse(camera) * centre).z)
            let eyeball = pixel(centre)
            let across = pixel(centre + 0.01 * face.columns.0) - eyeball
            let up = pixel(centre + 0.01 * face.columns.1) - eyeball
            guard depth > 0.05, simd_length(across) > 0, simd_length(up) > 0,
                  let fit = IrisLocator.locate(in: image, eyeball: eyeball, expectedRadius: limbusRadius*scale/depth,
                                               across: simd_normalize(across), up: simd_normalize(up)) else {
                return NSNull()
            }
            let numbers = [fit.x, fit.y, fit.radius, fit.score]
            return numbers.allSatisfy({ $0.isFinite }) ? numbers.map { ($0*1000).rounded()/1000 } : NSNull()
        }
        return ["image": [image.width, image.height], "camera": map, "left": eye(eyes.0), "right": eye(eyes.1)]
    }
}
