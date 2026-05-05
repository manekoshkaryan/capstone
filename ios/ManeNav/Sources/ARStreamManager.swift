import ARKit
import CoreVideo
import Foundation
import UIKit

protocol ARStreamDelegate: AnyObject {
    func didReceiveFrame(jpeg: Data, timestamp: Double)
    func didReceiveDepth(depth: Data, width: Int, height: Int, timestamp: Double)
}

final class ARStreamManager: NSObject, ARSessionDelegate {
    weak var delegate: ARStreamDelegate?

    let session = ARSession()
    private var frameCount = 0
    private let videoFPS = 15
    private let depthFPS = 10
    private let jpegQuality: CGFloat = 0.6

    override init() {
        super.init()
        session.delegate = self
    }

    func start() {
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            print("[ARStream] LiDAR not available")
            return
        }
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = .sceneDepth
        config.videoFormat = ARWorldTrackingConfiguration.supportedVideoFormats
            .first(where: { $0.framesPerSecond == 30 }) ?? ARWorldTrackingConfiguration.supportedVideoFormats[0]
        session.run(config)
        print("[ARStream] Started — video=\(videoFPS)fps depth=\(depthFPS)fps")
    }

    func stop() { session.pause() }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        frameCount += 1
        let ts = frame.timestamp

        // Video frame — rotate 90° CW so portrait phone → portrait image on server
        let videoEvery = max(1, 30 / videoFPS)
        if frameCount % videoEvery == 0 {
            let pixelBuffer = frame.capturedImage
            if let jpeg = pixelBufferToJPEG(pixelBuffer, quality: jpegQuality) {
                let delegate = self.delegate
                DispatchQueue.global(qos: .userInitiated).async {
                    delegate?.didReceiveFrame(jpeg: jpeg, timestamp: ts)
                }
            }
        }

        // Depth frame — rotate 90° CW to match video orientation
        let depthEvery = max(1, 30 / depthFPS)
        if frameCount % depthEvery == 0, let sceneDepth = frame.sceneDepth {
            let depthMap = sceneDepth.depthMap
            let srcW = CVPixelBufferGetWidth(depthMap)
            let srcH = CVPixelBufferGetHeight(depthMap)

            CVPixelBufferLockBaseAddress(depthMap, .readOnly)
            guard let base = CVPixelBufferGetBaseAddress(depthMap) else {
                CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
                return
            }
            let floats = base.bindMemory(to: Float32.self, capacity: srcW * srcH)

            // Rotate 90° CW: dst[x][srcH-1-y] = src[y][x]
            // After rotation: new width = srcH, new height = srcW
            let dstW = srcH
            let dstH = srcW
            var buf = [UInt16](repeating: 0, count: dstW * dstH)
            for y in 0 ..< srcH {
                for x in 0 ..< srcW {
                    let srcIdx = y * srcW + x
                    let dstIdx = x * dstW + (srcH - 1 - y)
                    buf[dstIdx] = UInt16(min(max(floats[srcIdx] * 1000.0, 0), 65535))
                }
            }
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)

            let depthData = buf.withUnsafeBytes { Data($0) }
            let delegate = self.delegate
            DispatchQueue.global(qos: .userInitiated).async {
                delegate?.didReceiveDepth(depth: depthData, width: dstW, height: dstH, timestamp: ts)
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[ARStream] Error: \(error)")
    }

    // MARK: - Private

    private func pixelBufferToJPEG(_ buffer: CVPixelBuffer, quality: CGFloat) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        // .right = 90° CW — ARKit landscape buffer → portrait display
        let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
        return uiImage.jpegData(compressionQuality: quality)
    }
}
