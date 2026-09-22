#!/usr/bin/env python
"""Internal isolated eval worker. Not a user-facing CLI."""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401

from src.eval import worker_main

if __name__ == "__main__":
    worker_main(sys.argv)
