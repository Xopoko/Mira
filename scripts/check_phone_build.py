"""Build the iPhone app on macOS without signing or installing it."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.platform != "darwin":
        raise SystemExit("Run this check on macOS with Xcode and XcodeGen installed.")

    try:
        with tempfile.TemporaryDirectory(prefix="mira-phone-build-") as temporary:
            project = Path(temporary)
            for name in ("Info.plist", "project.yml"):
                shutil.copy2(ROOT / "phone" / name, project / name)
            for name in ("Sources", "Tests"):
                shutil.copytree(ROOT / "phone" / name, project / name)
            subprocess.run(["xcodegen", "generate"], cwd=project, check=True)
            subprocess.run(
                [
                    "xcodebuild", "-quiet", "-project", "MyEyeTrackingPhone.xcodeproj",
                    "-scheme", "MyEyeTrackingPhone", "-destination", "generic/platform=iOS",
                    "-derivedDataPath", str(project / "build"), "CODE_SIGNING_ALLOWED=NO",
                ],
                cwd=project,
                check=True,
            )
    except FileNotFoundError as error:
        raise SystemExit(f"Missing required file or tool: {error.filename}") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
    print("Unsigned iOS build passed.")


if __name__ == "__main__":
    main()
