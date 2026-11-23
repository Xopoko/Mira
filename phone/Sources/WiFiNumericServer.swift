import Darwin
import Foundation
import Network
import QuartzCore

enum WiFiGazePolicy {
    static let port: UInt16 = 39472

    static func isPrivateLANIPv4(_ address: String) -> Bool {
        let strings = address.split(separator: ".", omittingEmptySubsequences: false)
        let parts = strings.compactMap { UInt8($0) }
        guard strings.count == 4, parts.count == 4,
              zip(strings, parts).allSatisfy({ String($0.0) == String($0.1) }) else { return false }
        return parts[0] == 10 ||
            (parts[0] == 172 && (16...31).contains(parts[1])) ||
            (parts[0] == 192 && parts[1] == 168)
    }

    static func accepts(_ request: Data) -> Bool {
        request == Data("GET\n".utf8)
    }
}

// The phone listener binds its private Wi-Fi address and accepts local paths only.
final class WiFiNumericServer {
    var onEndpoint: ((String?) -> Void)?
    var onConnection: ((Bool) -> Void)?
    var onError: (() -> Void)?
    private let queue = DispatchQueue(label: "local.myeyetracking.wifi")
    private var listener: NWListener?
    private final class Client {
        let connection: NWConnection
        var buffer = Data()
        var sending = false
        init(_ connection: NWConnection) { self.connection = connection }
    }
    private var clients: [ObjectIdentifier: Client] = [:]
    private var latest: [String: Any]?
    private var captured = 0.0

    static var privateLANEndpoint: String? {
        var head: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&head) == 0, let first = head else { return nil }
        defer { freeifaddrs(head) }
        var item: UnsafeMutablePointer<ifaddrs>? = first
        while let current = item {
            let value = current.pointee
            item = value.ifa_next
            guard String(cString: value.ifa_name) == "en0",
                  let address = value.ifa_addr,
                  address.pointee.sa_family == sa_family_t(AF_INET) else { continue }
            let ipv4 = UnsafeRawPointer(address).assumingMemoryBound(to: sockaddr_in.self).pointee.sin_addr
            guard let text = inet_ntoa(ipv4) else { continue }
            let addressText = String(cString: text)
            if WiFiGazePolicy.isPrivateLANIPv4(addressText) { return addressText }
        }
        return nil
    }

    func start() {
        start(address: Self.privateLANEndpoint, port: NWEndpoint.Port(rawValue: WiFiGazePolicy.port),
              interfaceType: .wifi)
    }

    #if os(macOS)
    func startLoopback(onListening: @escaping (UInt16) -> Void) {
        start(address: "127.0.0.1", port: .any, interfaceType: .loopback, onListening: onListening)
    }
    #endif

    private func start(address: @autoclosure @escaping () -> String?, port: NWEndpoint.Port?,
                       interfaceType: NWInterface.InterfaceType,
                       onListening: ((UInt16) -> Void)? = nil) {
        queue.async { [self] in
            self.close()
            guard let address = address(), let port else {
                self.onEndpoint?(nil)
                self.onError?()
                return
            }
            self.onEndpoint?(address)
            do {
                let tcp = NWProtocolTCP.Options()
                tcp.noDelay = true
                let parameters = NWParameters(tls: nil, tcp: tcp)
                parameters.requiredInterfaceType = interfaceType
                parameters.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(address), port: port)
                parameters.acceptLocalOnly = true
                let listener = try NWListener(using: parameters)
                listener.stateUpdateHandler = { [weak self, weak listener] state in
                    guard let self = self, let listener = listener, self.listener === listener else { return }
                    switch state {
                    case .ready:
                        if let boundPort = listener.port { onListening?(boundPort.rawValue) }
                    case .failed:
                        self.onError?()
                    default: break
                    }
                }
                listener.newConnectionHandler = { [weak self, weak listener] connection in
                    guard let self = self, let listener = listener, self.listener === listener else { connection.cancel(); return }
                    guard self.clients.count < 2 else { connection.cancel(); return }
                    let id = ObjectIdentifier(connection)
                    self.clients[id] = Client(connection)
                    connection.stateUpdateHandler = { [weak self, weak connection] state in
                        guard let self = self, let connection = connection,
                              self.clients[id]?.connection === connection else { return }
                        switch state {
                        case .ready:
                            self.onConnection?(true)
                            self.receive(connection)
                        case .failed, .cancelled:
                            self.clients.removeValue(forKey: id)
                            connection.cancel()
                            self.onConnection?(!self.clients.isEmpty)
                        default: break
                        }
                    }
                    connection.start(queue: self.queue)
                }
                self.listener = listener
                listener.start(queue: self.queue)
            } catch { self.onError?() }
        }
    }

    func update(packet: [String: Any], captured: Double) {
        queue.async {
            guard self.listener != nil else { return }
            self.latest = packet
            self.captured = captured
        }
    }

    func stop() { queue.sync { close() } }

    private func close() {
        listener?.cancel()
        listener = nil
        let previous = clients.values.map { $0.connection }
        clients.removeAll()
        for connection in previous { connection.cancel() }
        latest = nil
        onConnection?(false)
    }

    private func receive(_ connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64) { [weak self, weak connection] data, _, complete, error in
            guard let self = self, let connection = connection,
                  let client = self.clients[ObjectIdentifier(connection)] else { return }
            if let data = data { client.buffer.append(data) }
            if complete || error != nil || client.buffer.count > 64 {
                connection.cancel()
                return
            }
            guard let end = client.buffer.firstIndex(of: 10) else {
                self.receive(connection)
                return
            }
            let request = Data(client.buffer.prefix(through: end))
            client.buffer.removeSubrange(...end)
            // One request and one response may be in flight per client. Closing
            // on pipelining prevents an old request backlog from amplifying age.
            guard WiFiGazePolicy.accepts(request), client.buffer.isEmpty, !client.sending else {
                connection.cancel()
                return
            }
            client.sending = true
            var packet = self.latest ?? ["version": 1, "tracked": false]
            packet["ageMs"] = max(0, (CACurrentMediaTime()-self.captured)*1000)
            guard JSONSerialization.isValidJSONObject(packet),
                  let body = try? JSONSerialization.data(withJSONObject: packet, options: [.sortedKeys]),
                  body.count <= 16_384 else {
                connection.cancel()
                return
            }
            connection.send(content: body+Data([10]), completion: .contentProcessed { sendError in
                self.queue.async {
                    guard self.clients[ObjectIdentifier(connection)] === client else { return }
                    if sendError != nil {
                        connection.cancel()
                        return
                    }
                    client.sending = false
                    self.receive(connection)
                }
            })
        }
    }
}
