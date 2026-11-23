"""Exercise the iPhone Wi-Fi transport on macOS using a loopback socket."""

from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.platform != "darwin":
        raise SystemExit("Run this check on macOS with Xcode installed.")

    try:
        with tempfile.TemporaryDirectory(prefix="mira-phone-loopback-") as temporary:
            executable = Path(temporary) / "network-fixture"
            subprocess.run(
                [
                    "xcrun", "swiftc", "-parse-as-library",
                    str(ROOT / "phone" / "Sources" / "WiFiNumericServer.swift"),
                    str(ROOT / "phone" / "Fixtures" / "NetworkFixture.swift"),
                    "-o", str(executable),
                ],
                check=True,
            )
            subprocess.run([str(executable)], check=True, timeout=15)
    except FileNotFoundError as error:
        raise SystemExit(f"Missing required file or tool: {error.filename}") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
    except subprocess.TimeoutExpired as error:
        raise SystemExit("Phone Wi-Fi loopback fixture timed out.") from error


if __name__ == "__main__":
    main()
