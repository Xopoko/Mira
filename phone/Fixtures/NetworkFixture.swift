import Darwin
import Foundation
import QuartzCore

@main
struct NetworkFixture {
    static func main() {
        do {
            try verifyPolicy()
            try verifyServer()
            try verifyRetry()
            print("Phone Wi-Fi loopback fixture passed.")
        } catch {
            let reason = (error as? FixtureError)?.reason ?? "fixture_assertion"
            print("Phone Wi-Fi loopback fixture failed (\(reason)).")
            exit(1)
        }
    }

    static func verifyPolicy() throws {
        let accepted = ["10.0.0.1", "172.16.0.1", "172.31.255.254", "192.168.1.88"]
        let rejected = ["172.15.255.255", "172.32.0.1", "192.167.1.1", "127.0.0.1",
                        "169.254.2.3", "8.8.8.8", "::1", "192.168.1", "10.0.0.1.2",
                        "10..0.1", "", "010.0.0.1"]
        guard accepted.allSatisfy(WiFiGazePolicy.isPrivateLANIPv4),
              rejected.allSatisfy({ !WiFiGazePolicy.isPrivateLANIPv4($0) }),
              WiFiGazePolicy.accepts(Data("GET\n".utf8)),
              !WiFiGazePolicy.accepts(Data("GETRGB\n".utf8)),
              !WiFiGazePolicy.accepts(Data("GET\nGET\n".utf8)) else {
            throw FixtureError.failed("policy")
        }
    }

    static func verifyServer() throws {
        let state = FixtureState()
        let server = WiFiNumericServer()
        server.onConnection = { state.setConnected($0) }
        server.onError = { _ in state.setFailed() }
        server.startLoopback { state.setPort($0) }
        defer { server.stop() }

        guard waitUntil(timeout: 2, condition: { state.port != nil || state.failed }),
              !state.failed, let port = state.port else {
            throw FixtureError.failed("listener_start")
        }

        let first = try connect(port: port)
        guard waitUntil(timeout: 2, condition: { state.connected && !state.failed }) else {
            Darwin.close(first)
            throw FixtureError.failed("client_connection")
        }

        let captured = CACurrentMediaTime()
        server.update(packet: packet(seq: 11), captured: captured)
        server.update(packet: packet(seq: 12), captured: captured)
        sendAll(first, Data("G".utf8))
        Thread.sleep(forTimeInterval: 0.025)
        sendAll(first, Data("ET\n".utf8))
        let firstReply = try readJSONLine(first)
        guard firstReply["seq"] as? Int == 12,
              let age1 = firstReply["ageMs"] as? Double else { throw FixtureError.failed("fragmented_get_latest") }

        Thread.sleep(forTimeInterval: 0.10)
        sendAll(first, Data("GET\n".utf8))
        let secondReply = try readJSONLine(first)
        guard secondReply["seq"] as? Int == 12,
              let age2 = secondReply["ageMs"] as? Double,
              age2 > age1 + 35 else { throw FixtureError.failed("stale_age_growth") }

        let rgb = try connect(port: port)
        sendAll(rgb, Data("GETRGB\n".utf8))
        guard reachesEOF(rgb) else { Darwin.close(rgb); throw FixtureError.failed("getrgb_rejection") }
        Darwin.close(rgb)

        let pipelined = try connect(port: port)
        sendAll(pipelined, Data("GET\nGET\n".utf8))
        guard reachesEOF(pipelined) else { Darwin.close(pipelined); throw FixtureError.failed("pipelined_get_rejection") }
        Darwin.close(pipelined)

        Darwin.close(first)
        guard waitUntil(timeout: 2, condition: { !state.connected }) else { throw FixtureError.failed("disconnect") }

        let activeAtStop = try connect(port: port)
        guard waitUntil(timeout: 2, condition: { state.connected }) else {
            Darwin.close(activeAtStop)
            throw FixtureError.failed("stop_connect")
        }
        server.stop()
        guard reachesEOF(activeAtStop), waitUntil(timeout: 1, condition: { !state.connected }) else {
            Darwin.close(activeAtStop)
            throw FixtureError.failed("stop_close")
        }
        Darwin.close(activeAtStop)
        guard !state.failed else { throw FixtureError.failed("listener_error") }

        // The connection closed by stop() still holds the port, as after backgrounding.
        server.startLoopback(port: port) { state.setPort($0) }
        guard waitUntil(timeout: 2, condition: { state.starts == 2 || state.failed }),
              !state.failed, state.port == port else {
            throw FixtureError.failed("listener_restart")
        }
        let afterRestart = try connect(port: port)
        defer { Darwin.close(afterRestart) }
        sendAll(afterRestart, Data("GET\n".utf8))
        guard try readJSONLine(afterRestart)["version"] as? Int == 1 else {
            throw FixtureError.failed("restart_get")
        }
    }

    // A busy port is retried quietly; only a port that stays busy is reported, with its reason.
    static func verifyRetry() throws {
        let blocker = try occupyPort()
        let state = FixtureState()
        let server = WiFiNumericServer()
        server.retryDelay = 0.1
        server.onError = { _ in state.setFailed() }
        server.startLoopback(port: blocker.port) { state.setPort($0) }
        defer { server.stop() }
        Thread.sleep(forTimeInterval: 0.35)
        let (openedWhileBusy, reportedWhileBusy) = (state.starts != 0, state.failed)
        Darwin.close(blocker.fd)
        guard !openedWhileBusy else { throw FixtureError.failed("busy_port_opened") }
        guard !reportedWhileBusy else { throw FixtureError.failed("busy_port_reported_early") }
        guard waitUntil(timeout: 3, condition: { state.starts == 1 || state.failed }), !state.failed else {
            throw FixtureError.failed("busy_port_retry")
        }
        let client = try connect(port: blocker.port)
        defer { Darwin.close(client) }
        sendAll(client, Data("GET\n".utf8))
        guard try readJSONLine(client)["version"] as? Int == 1 else { throw FixtureError.failed("retry_get") }

        let busy = try occupyPort()
        defer { Darwin.close(busy.fd) }
        let reason = FixtureState()
        let failing = WiFiNumericServer()
        failing.retryDelay = 0.05
        failing.retryLimit = 2
        failing.onError = { reason.setReason($0) }
        failing.startLoopback(port: busy.port) { _ in reason.setPort(busy.port) }
        defer { failing.stop() }
        guard waitUntil(timeout: 2, condition: { reason.reason != nil }), reason.starts == 0,
              reason.reason == String(cString: strerror(EADDRINUSE)) else {
            throw FixtureError.failed("retry_limit_reason")
        }
    }

    static func occupyPort() throws -> (fd: Int32, port: UInt16) {
        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw FixtureError.failed("socket_create") }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { pointer -> Bool in
                Darwin.bind(fd, pointer, length) == 0 && Darwin.listen(fd, 1) == 0 &&
                    getsockname(fd, pointer, &length) == 0
            }
        }
        guard bound else { Darwin.close(fd); throw FixtureError.failed("socket_occupy") }
        return (fd, UInt16(bigEndian: address.sin_port))
    }

    static func packet(seq: Int) -> [String: Any] {
        ["version": 1, "session": "loopback-fixture", "seq": seq,
         "timestamp": CACurrentMediaTime(), "tracked": true, "trueDepth": true,
         "depthFrame": true, "ageMs": 0]
    }

    static func connect(port: UInt16) throws -> Int32 {
        let deadline = Date().addingTimeInterval(2)
        while Date() < deadline {
            let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
            guard fd >= 0 else { throw FixtureError.failed("socket_create") }
            var timeout = timeval(tv_sec: 2, tv_usec: 0)
            _ = withUnsafePointer(to: &timeout) {
                setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, $0, socklen_t(MemoryLayout<timeval>.size))
            }
            var address = sockaddr_in()
            address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
            address.sin_family = sa_family_t(AF_INET)
            address.sin_port = port.bigEndian
            address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
            let result = withUnsafePointer(to: &address) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            if result == 0 { return fd }
            Darwin.close(fd)
            Thread.sleep(forTimeInterval: 0.025)
        }
        throw FixtureError.failed("socket_connect")
    }

    static func sendAll(_ fd: Int32, _ data: Data) {
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            var sent = 0
            while sent < raw.count {
                let count = Darwin.send(fd, base.advanced(by: sent), raw.count-sent, 0)
                if count <= 0 { return }
                sent += count
            }
        }
    }

    static func readJSONLine(_ fd: Int32) throws -> [String: Any] {
        var bytes = Data()
        while bytes.count < 16_384 {
            var byte: UInt8 = 0
            let count = Darwin.recv(fd, &byte, 1, 0)
            guard count == 1 else { throw FixtureError.failed("response_read") }
            if byte == 10 {
                guard let json = try? JSONSerialization.jsonObject(with: bytes),
                      let object = json as? [String: Any] else { throw FixtureError.failed("response_json") }
                return object
            }
            bytes.append(byte)
        }
        throw FixtureError.failed("response_size")
    }

    static func reachesEOF(_ fd: Int32) -> Bool {
        var byte: UInt8 = 0
        return Darwin.recv(fd, &byte, 1, 0) == 0
    }

    static func waitUntil(timeout: TimeInterval, condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            Thread.sleep(forTimeInterval: 0.01)
        }
        return condition()
    }
}

final class FixtureState {
    private let lock = NSLock()
    private var isConnected = false
    private var didFail = false
    private var listeningPort: UInt16?
    private var listenCount = 0
    private var failureReason: String?
    var connected: Bool { lock.lock(); defer { lock.unlock() }; return isConnected }
    var reason: String? { lock.lock(); defer { lock.unlock() }; return failureReason }
    func setReason(_ value: String) { lock.lock(); failureReason = value; lock.unlock() }
    var failed: Bool { lock.lock(); defer { lock.unlock() }; return didFail }
    var port: UInt16? { lock.lock(); defer { lock.unlock() }; return listeningPort }
    var starts: Int { lock.lock(); defer { lock.unlock() }; return listenCount }
    func setConnected(_ value: Bool) { lock.lock(); isConnected = value; lock.unlock() }
    func setFailed() { lock.lock(); didFail = true; lock.unlock() }
    func setPort(_ value: UInt16) { lock.lock(); listeningPort = value; listenCount += 1; lock.unlock() }
}

enum FixtureError: Error {
    case failed(String)
    var reason: String {
        switch self { case .failed(let value): return value }
    }
}
