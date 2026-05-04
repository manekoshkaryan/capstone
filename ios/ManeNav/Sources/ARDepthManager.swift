import ARKit
import Accelerate
import Foundation

protocol ARDepthDelegate: AnyObject {
    func didReceiveDepth(_ data: Data, width: Int, height: Int, timestamp: Double)
}

final class ARDepthManager: NSObject, ARSessionDelegate {
    weak var delegate: ARDepthDelegate?

    private let session = ARSession()
    private var frameCount = 0
    private let depthFPS = 10  // send depth at 10fps max

    override init() {
        super.init()
        session.delegate = self
    }

    func start() {
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            print("[ARDepth] LiDAR not available on this device")
            return
        }
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = .sceneDepth
        session.run(config)
        print("[ARDepth] ARKit session started with sceneDepth")
    }

    func stop() {
        session.pause()
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        frameCount += 1
        let targetEvery = max(1, 60 / depthFPS)
        guard frameCount % targetEvery == 0 else { return }
        guard let sceneDepth = frame.sceneDepth else { return }

        let depthMap = sceneDepth.depthMap
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let timestamp = frame.timestamp  // copy scalar — don't retain frame

        // Copy pixel data synchronously then release immediately
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        guard let baseAddr = CVPixelBufferGetBaseAddress(depthMap) else {
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
            return
        }
        let floatPtr = baseAddr.bindMemory(to: Float32.self, capacity: width * height)
        var uint16Buf = [UInt16](repeating: 0, count: width * height)
        for i in 0 ..< width * height {
            let mm = floatPtr[i] * 1000.0
            uint16Buf[i] = UInt16(min(max(mm, 0), 65535))
        }
        CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
        // frame and depthMap no longer referenced past this point

        let rawData = uint16Buf.withUnsafeBytes { Data($0) }
        let delegate = self.delegate
        DispatchQueue.global(qos: .userInitiated).async {
            delegate?.didReceiveDepth(rawData, width: width, height: height, timestamp: timestamp)
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[ARDepth] Session error: \(error)")
    }
}
