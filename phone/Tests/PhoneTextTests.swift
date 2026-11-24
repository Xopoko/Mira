import XCTest
@testable import MyEyeTrackingPhone

final class PhoneTextTests: XCTestCase {
    func testUpdateRateShowsRoundedCountPerSecond() {
        XCTAssertEqual(EyeTracker.perSecondText(0), "0 per second")
        XCTAssertEqual(EyeTracker.perSecondText(1.2), "1 per second")
        XCTAssertEqual(EyeTracker.perSecondText(59.6), "60 per second")
    }

    func testBundleDeclaresAppIconAsPrimaryIconName() {
        let icons = Bundle.main.infoDictionary?["CFBundleIcons"] as? [String: Any]
        let primary = icons?["CFBundlePrimaryIcon"] as? [String: Any]
        XCTAssertEqual(primary?["CFBundleIconName"] as? String, "AppIcon")
    }

    func testBundleDeclaresEnglishRegionAndMiraNameWithNonCyrillicPermissionText() {
        let info = Bundle.main.infoDictionary ?? [:]
        XCTAssertEqual(info["CFBundleDevelopmentRegion"] as? String, "en")
        XCTAssertEqual(info["CFBundleDisplayName"] as? String, "Mira")
        for key in ["NSCameraUsageDescription", "NSLocalNetworkUsageDescription"] {
            let text = info[key] as? String ?? ""
            XCTAssertFalse(text.isEmpty, key)
            XCTAssertNil(text.range(of: "\\p{Cyrillic}", options: .regularExpression), key)
        }
    }
}
