import Foundation

final class DepthWebSocketSender: NSObject, URLSessionWebSocketDelegate {
    private var task: URLSessionWebSocketTask?
    private var session: URLSession!
    private var serverURL: URL
    private(set) var isConnected = false
    private var connected: Bool {
        get { isConnected }
        set { isConnected = newValue }
    }
    private var reconnectTimer: Timer?

    init(host: String, port: Int) {
        // ws:// — plain WebSocket (server is HTTP on same port as HTTPS,
        // but depth endpoint runs on separate plain HTTP port 8444)
        self.serverURL = URL(string: "ws://\(host):\(port)/depth")!
        super.init()
        let cfg = URLSessionConfiguration.default
        self.session = URLSession(configuration: cfg, delegate: self, delegateQueue: .main)
    }

    func connect() {
        task?.cancel()
        task = session.webSocketTask(with: serverURL)
        task?.resume()
        print("[DepthWS] Connecting to \(serverURL)")
    }

    func send(_ depthData: Data, width: Int, height: Int, timestamp: Double) {
        guard connected, let task = task else { return }

        // 12-byte header: timestamp(f64) + width(u16) + height(u16)
        var header = Data(capacity: 12)
        var ts = timestamp
        var w = UInt16(width)
        var h = UInt16(height)
        header.append(contentsOf: withUnsafeBytes(of: &ts) { Array($0) })
        header.append(contentsOf: withUnsafeBytes(of: &w) { Array($0) })
        header.append(contentsOf: withUnsafeBytes(of: &h) { Array($0) })

        let payload = header + depthData
        task.send(.data(payload)) { error in
            if let error = error {
                print("[DepthWS] Send error: \(error)")
                self.connected = false
                self.scheduleReconnect()
            }
        }
    }

    private func scheduleReconnect() {
        reconnectTimer?.invalidate()
        reconnectTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: false) { _ in
            self.connect()
        }
    }

    // MARK: URLSessionWebSocketDelegate

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol protocol: String?) {
        print("[DepthWS] Connected")
        connected = true
        reconnectTimer?.invalidate()
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        print("[DepthWS] Closed: \(closeCode)")
        connected = false
        scheduleReconnect()
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        if let error = error {
            print("[DepthWS] Task error: \(error)")
            connected = false
            scheduleReconnect()
        }
    }
}
