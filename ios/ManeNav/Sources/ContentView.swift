import SwiftUI
import ARKit
import SceneKit

// MARK: - ARKit camera preview

struct ARCameraView: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView()
        view.session = session
        view.scene = SCNScene()
        view.autoenablesDefaultLighting = false
        view.automaticallyUpdatesLighting = false
        view.rendersContinuously = false
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

// MARK: - Main view

struct ContentView: View {
    private let serverHost = "192.168.11.105"
    private let serverPort = 8444

    @StateObject private var coordinator = AppCoordinator()

    var body: some View {
        ZStack {
            // Full-screen camera feed
            if let session = coordinator.arSession {
                ARCameraView(session: session)
                    .ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
            }

            // Detection overlay from server
            if let img = coordinator.annotatedFrame {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFill()
                    .ignoresSafeArea()
                    .opacity(0.55)
                    .allowsHitTesting(false)
            }

            // Subtle dark gradient at top and bottom for controls
            VStack {
                LinearGradient(
                    colors: [.black.opacity(0.6), .clear],
                    startPoint: .top, endPoint: .bottom
                )
                .frame(height: 120)
                .ignoresSafeArea(edges: .top)

                Spacer()

                LinearGradient(
                    colors: [.clear, .black.opacity(0.7)],
                    startPoint: .top, endPoint: .bottom
                )
                .frame(height: 180)
                .ignoresSafeArea(edges: .bottom)
            }

            // Top status bar
            VStack {
                HStack(spacing: 12) {
                    Dot(active: coordinator.arActive, color: .green, label: "ARKit")
                    Dot(active: coordinator.lidarActive, color: .orange, label: "LiDAR")
                    Dot(active: coordinator.serverConnected, color: .blue, label: "Server")
                    Spacer()
                    Text(coordinator.statsText)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.white.opacity(0.6))
                }
                .padding(.horizontal, 20)
                .padding(.top, 56)

                Spacer()

                // Bottom controls
                HStack(spacing: 32) {
                    // Mute button
                    CircleButton(
                        icon: coordinator.muted ? "speaker.slash.fill" : "speaker.wave.2.fill",
                        color: coordinator.muted ? .red : .white,
                        action: { coordinator.muted.toggle() }
                    )

                    // Start / Stop
                    Button(action: { coordinator.toggleStreaming() }) {
                        ZStack {
                            Circle()
                                .fill(coordinator.streaming ? Color.red : Color.white)
                                .frame(width: 72, height: 72)
                            Image(systemName: coordinator.streaming ? "stop.fill" : "play.fill")
                                .font(.system(size: 26, weight: .bold))
                                .foregroundColor(coordinator.streaming ? .white : .black)
                        }
                        .shadow(color: .black.opacity(0.4), radius: 10)
                    }
                    .scaleEffect(coordinator.streaming ? 1.05 : 1.0)
                    .animation(.easeInOut(duration: 0.15), value: coordinator.streaming)

                    // Placeholder for symmetry / future button
                    CircleButton(icon: "info.circle", color: .white.opacity(0.6), action: {})
                }
                .padding(.bottom, 48)
            }
        }
        .onAppear {
            coordinator.start(host: serverHost, port: serverPort)
        }
    }
}

// MARK: - Sub-views

struct Dot: View {
    let active: Bool
    let color: Color
    let label: String

    var body: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(active ? color : Color.gray.opacity(0.4))
                .frame(width: 7, height: 7)
            Text(label)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundColor(active ? .white : .gray)
        }
    }
}

struct CircleButton: View {
    let icon: String
    let color: Color
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(.white.opacity(0.12))
                    .frame(width: 52, height: 52)
                Image(systemName: icon)
                    .font(.system(size: 20, weight: .medium))
                    .foregroundColor(color)
            }
        }
    }
}

// MARK: - Coordinator

@MainActor
final class AppCoordinator: NSObject, ObservableObject, ARStreamDelegate {
    @Published var arActive = false
    @Published var lidarActive = false
    @Published var serverConnected = false
    @Published var streaming = false
    @Published var muted = false
    @Published var statsText = ""
    @Published var arSession: ARSession?
    @Published var annotatedFrame: UIImage? = nil

    private var arManager: ARStreamManager?
    private var connection: ServerConnection?
    private var framesSent = 0
    private var depthSent = 0

    func start(host: String, port: Int) {
        UIApplication.shared.isIdleTimerDisabled = true  // keep screen on, ARKit alive

        let conn = ServerConnection(host: host, port: port)
        conn.connect()
        conn.onAnnotatedFrame = { [weak self] jpeg in
            Task { @MainActor [weak self] in
                self?.annotatedFrame = UIImage(data: jpeg)
            }
        }
        connection = conn

        let ar = ARStreamManager()
        ar.delegate = self
        ar.start()
        arManager = ar
        arSession = ar.session
        arActive = true
        streaming = true

        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.serverConnected = conn.isConnected
                self.statsText = "↑\(self.framesSent)f \(self.depthSent)d"
            }
        }
    }

    func toggleStreaming() {
        streaming.toggle()
        if streaming {
            arManager?.start()
        } else {
            arManager?.stop()
        }
    }

    nonisolated func didReceiveFrame(jpeg: Data, timestamp: Double) {
        Task { @MainActor [weak self] in
            guard let self, self.streaming else { return }
            self.connection?.sendVideo(jpeg, timestamp: timestamp)
            self.framesSent += 1
        }
    }

    nonisolated func didReceiveDepth(depth: Data, width: Int, height: Int, timestamp: Double) {
        Task { @MainActor [weak self] in
            guard let self, self.streaming else { return }
            self.connection?.sendDepth(depth, width: width, height: height, timestamp: timestamp)
            self.lidarActive = true
            self.depthSent += 1
        }
    }
}
