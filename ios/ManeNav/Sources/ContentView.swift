import SwiftUI
import WebKit

// MARK: - WebView with self-signed cert support

struct PhoneWebView: UIViewRepresentable {
    let url: URL

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []
        let wv = WKWebView(frame: .zero, configuration: config)
        wv.navigationDelegate = context.coordinator
        wv.load(URLRequest(url: url))
        return wv
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    class Coordinator: NSObject, WKNavigationDelegate {
        // Accept self-signed cert from local server
        func webView(_ webView: WKWebView,
                     didReceive challenge: URLAuthenticationChallenge,
                     completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
            if challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
               let trust = challenge.protectionSpace.serverTrust {
                completionHandler(.useCredential, URLCredential(trust: trust))
            } else {
                completionHandler(.performDefaultHandling, nil)
            }
        }
    }
}

// MARK: - Main view

struct ContentView: View {
    // Edit these to match your server
    private let serverHost = "10.10.150.159"
    private let webrtcPort = 8443
    private let depthPort  = 8444

    @StateObject private var coordinator = AppCoordinator()

    var body: some View {
        ZStack(alignment: .topTrailing) {
            PhoneWebView(url: URL(string: "https://\(serverHost):\(webrtcPort)/?device=iphone")!)
                .ignoresSafeArea()

            // Status badge
            VStack(alignment: .trailing, spacing: 4) {
                StatusBadge(label: "LiDAR", active: coordinator.lidarActive, color: .green)
                StatusBadge(label: "Depth WS", active: coordinator.wsConnected, color: .blue)
            }
            .padding(12)
        }
        .onAppear {
            coordinator.start(host: serverHost, depthPort: depthPort)
        }
    }
}

struct StatusBadge: View {
    let label: String
    let active: Bool
    let color: Color

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(active ? color : Color.gray)
                .frame(width: 8, height: 8)
            Text(label)
                .font(.caption2.monospaced())
                .foregroundColor(.white)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.black.opacity(0.6))
        .cornerRadius(8)
    }
}

// MARK: - Coordinator

@MainActor
final class AppCoordinator: NSObject, ObservableObject, ARDepthDelegate {
    @Published var lidarActive = false
    @Published var wsConnected = false

    private var depthManager: ARDepthManager?
    private var wsSender: DepthWebSocketSender?

    func start(host: String, depthPort: Int) {
        let dm = ARDepthManager()
        dm.delegate = self
        dm.start()
        depthManager = dm
        lidarActive = true

        let ws = DepthWebSocketSender(host: host, port: depthPort)
        ws.connect()
        wsSender = ws

        // Poll WS connection state
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in
                self.wsConnected = ws.isConnected
            }
        }
    }

    nonisolated func didReceiveDepth(_ data: Data, width: Int, height: Int, timestamp: Double) {
        Task { @MainActor [weak self] in
            self?.wsSender?.send(data, width: width, height: height, timestamp: timestamp)
        }
    }
}
