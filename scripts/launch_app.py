#!/usr/bin/env python3
"""Launch the Pitch Mechanics Analyzer desktop application."""

import sys

from src.desktop.app import PitchAnalyzerApp


def main():
    app = PitchAnalyzerApp(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
