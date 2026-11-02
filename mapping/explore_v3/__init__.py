"""Frozen v3 exploration algorithms; shared unchanged helpers resolve in mapping.

Copied algorithms originate from commit 599499a; the LiDAR worker also has an
optional FAST-LIO recording hook. The manifest records both source and runtime hashes.
"""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parent.parent))
