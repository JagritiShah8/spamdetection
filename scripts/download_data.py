"""Dataset download script for SMS Spam Collection.

Downloads and extracts the SMSSpamCollection dataset from the UCI Machine
Learning Repository. Fails explicitly if the download cannot be completed.
"""

from __future__ import annotations

import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
TARGET_DIR = Path("data")
TARGET_FILE = TARGET_DIR / "SMSSpamCollection"


def download_and_extract(url: str = UCI_URL, target_dir: Path = TARGET_DIR) -> Path:
    """Download the SMS spam collection zip and extract SMSSpamCollection.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "SMSSpamCollection"

    if target_file.exists() and target_file.stat().st_size > 0:
        print(f"Dataset already exists at {target_file} ({target_file.stat().st_size} bytes).")
        return target_file

    print(f"Downloading SMS Spam Collection from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download dataset from {url}: {exc}. "
            "Please check network connection or download manually into data/SMSSpamCollection."
        ) from exc

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            file_names = zf.namelist()
            # Look for SMSSpamCollection file
            target_name = None
            for name in file_names:
                if name.endswith("SMSSpamCollection") or name == "SMSSpamCollection":
                    target_name = name
                    break
            if not target_name:
                raise RuntimeError(
                    f"SMSSpamCollection not found inside zip archive. Found files: {file_names}"
                )

            extracted_data = zf.read(target_name)
            target_file.write_bytes(extracted_data)
            print(f"Successfully extracted {target_file} ({len(extracted_data)} bytes).")
            return target_file
    except Exception as exc:
        raise RuntimeError(f"Failed to extract zip contents: {exc}") from exc


if __name__ == "__main__":
    try:
        path = download_and_extract()
        print(f"Dataset ready at {path}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
