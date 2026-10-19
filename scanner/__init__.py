"""AI Scanner: check any file for signs it was made by AI."""

from .pipeline import scan_bytes, scan_path, scan_text

__all__ = ["scan_bytes", "scan_path", "scan_text"]
