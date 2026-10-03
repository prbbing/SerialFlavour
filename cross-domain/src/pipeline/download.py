"""Verified downloads; each domain supplies its own sources."""

import hashlib
from pathlib import Path
import urllib.request


def download_verified(url, destination, expected_md5, max_bytes=5_000_000_000):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def md5_file(path):
        digest = hashlib.md5()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    if destination.exists():
        if destination.stat().st_size <= max_bytes and md5_file(destination) == expected_md5:
            return destination
        raise ValueError(f"existing download fails checksum/size validation: {destination}")
    temporary = destination.with_name(destination.name + ".part")
    digest = hashlib.md5()
    received = 0
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > max_bytes:
            raise ValueError("download exceeds the configured single-file limit")
        while chunk := response.read(1024 * 1024):
            received += len(chunk)
            if received > max_bytes:
                raise ValueError("download exceeds the configured single-file limit")
            digest.update(chunk)
            stream.write(chunk)
        if declared and received != int(declared):
            raise ValueError("download ended before Content-Length bytes were received")
    if digest.hexdigest() != expected_md5:
        raise ValueError(f"download checksum mismatch: {url}")
    temporary.replace(destination)
    print(f"downloaded {destination.name}: {received:,} bytes", flush=True)
    return destination


def run(context):
    return context.module("data").download(context)
