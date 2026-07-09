#!/usr/bin/env python3
"""
NFluid — Interactive 2D CFD Solver with CAD Frontend

Launcher script for the cfdeditor package.
Usage: python NFluid.py
"""

import sys
import os

# Add the project root to the Python path so cfdeditor package can be found
sys.path.insert(0, os.path.dirname(__file__))

# Import and run the application
from cfdeditor.main import run_app

if __name__ == "__main__":
    run_app()