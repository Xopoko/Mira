import XCTest
@testable import MyEyeTrackingPhone

final class WiFiGazePolicyTests: XCTestCase {
    func testClassifiesRFC1918IPv4Addresses() {
        for address in ["10.0.0.8", "172.16.0.2", "172.31.255.254", "192.168.1.88"] {
            XCTAssertTrue(WiFiGazePolicy.isPrivateLANIPv4(address), address)
        }
        for address in ["172.15.255.255", "172.32.0.1", "192.167.1.2", "127.0.0.1",
                        "169.254.1.2", "8.8.8.8", "::1", "192.168.1", "192.168.x.1.1", "192..168.1.1", "192.168.001.1"] {
            XCTAssertFalse(WiFiGazePolicy.isPrivateLANIPv4(address), address)
        }
    }

    func testPolicyAcceptsExactlyOneCompleteGETCommand() {
        XCTAssertTrue(WiFiGazePolicy.accepts(Data("GET\n".utf8)))
        for request in ["PING\n", "POST\n", "GET", "GET\nGET\n"] {
            XCTAssertFalse(WiFiGazePolicy.accepts(Data(request.utf8)), request)
        }
        XCTAssertEqual(WiFiGazePolicy.port, 39472)
    }
}
