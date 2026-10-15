"""Fetch the pinned Tectonic release into tools/ (project-local, gitignored).

Nothing is installed system-wide: no PATH, registry or user-directory changes.
Remove tools/ to uninstall.  Usage: python scripts/get_tectonic.py
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

VERSION = "0.17.0"
ASSET = f"tectonic-{VERSION}-x86_64-pc-windows-msvc.zip"
URL = f"https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40{VERSION}/{ASSET}"
# SHA-256 published for this asset on the official GitHub release page (release asset digest).
SHA256 = "f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f"

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def main() -> int:
    TOOLS.mkdir(exist_ok=True)
    exe = TOOLS / "tectonic.exe"
    if exe.exists():
        print(f"already present: {exe}")
        return 0
    zpath = TOOLS / ASSET
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, zpath)
    got = hashlib.sha256(zpath.read_bytes()).hexdigest()
    if got != SHA256:
        zpath.unlink()
        print(f"CHECKSUM MISMATCH: expected {SHA256}, got {got}", file=sys.stderr)
        return 1
    print(f"sha256 verified: {got}")
    with zipfile.ZipFile(zpath) as z:
        z.extract("tectonic.exe", TOOLS)
    zpath.unlink()
    print(f"extracted {exe}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
