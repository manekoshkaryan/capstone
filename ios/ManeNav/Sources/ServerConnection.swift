import Foundation

/// Single WebSocket connection to the Python server.
/// Sends both JPEG video frames and LiDAR depth frames over one connection.
/// Message format — binary frames with 1-byte type prefix:
///   0x01 + 8-byte f64 ts + JPEG bytes              (phone → server: video)
///   0x02 + 8-byte f64 ts + u16 w + u16 h + uint16[] (phone → server: depth)
///   0x03 + JPEG bytes                               (server → phone: annotated frame)
final class ServerConnection: NSObject, URLSessionWebSocketDelegate {
    private var task: URLSessionWebSocketTask?
    private var urlSession: URLSession!
    private let url: URL
    private(set) var isConnected = false
    private var reconnectTimer: Timer?
    var onAnnotatedFrame: ((Data) -> Void)?

    init(host: String, port: Int) {
        self.url = URL(string: "ws://\(host):\(port)/stream")!
        super.init()
        let cfg = URLSessionConfiguration.default
        self.urlSession = URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
    }

    func connect() {
        task?.cancel()
        let t = urlSession.webSocketTask(with: url)
        task = t
        t.resume()
        print("[ServerConn] Connecting to \(url)")
        receiveLoop(t)
    }

    private func receiveLoop(_ t: URLSessionWebSocketTask) {
        t.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let msg):
                if case .data(let data) = msg, !data.isEmpty, data[0] == 0x03 {
                    let jpeg = data.dropFirst()
                    self.onAnnotatedFrame?(jpeg)
                }
                self.receiveLoop(t)
            case .failure:
                break  // connection closed, delegate handles reconnect
            }
        }
    }

    func sendVideo(_ jpeg: Data, timestamp: Double) {
        guard isConnected else { return }
        var msg = Data(capacity: 1 + 8 + jpeg.count)
        msg.append(0x01)
        var ts = timestamp
        msg.append(contentsOf: withUnsafeBytes(of: &ts) { Array($0) })
        msg.append(jpeg)
        send(msg)
    }

    func sendDepth(_ depth: Data, width: Int, height: Int, timestamp: Double) {
        guard isConnected else { return }
        var msg = Data(capacity: 1 + 8 + 2 + 2 + depth.count)
        msg.append(0x02)
        var ts = timestamp; var w = UInt16(width); var h = UInt16(height)
        msg.append(contentsOf: withUnsafeBytes(of: &ts) { Array($0) })
        msg.append(contentsOf: withUnsafeBytes(of: &w) { Array($0) })
        msg.append(contentsOf: withUnsafeBytes(of: &h) { Array($0) })
        msg.append(depth)
        send(msg)
    }

    private func send(_ data: Data) {
        task?.send(.data(data)) { [weak self] error in
            if let error = error {
                print("[ServerConn] Send error: \(error)")
                self?.isConnected = false
                self?.scheduleReconnect()
            }
        }
    }

    private func scheduleReconnect() {
        reconnectTimer?.invalidate()
        reconnectTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: false) { [weak self] _ in
            self?.connect()
        }
    }

    // MARK: URLSessionWebSocketDelegate

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol protocol: String?) {
        print("[ServerConn] Connected")
        isConnected = true
        reconnectTimer?.invalidate()
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        print("[ServerConn] Closed")
        isConnected = false
        scheduleReconnect()
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        if error != nil {
            isConnected = false
            scheduleReconnect()
        }
    }
}
