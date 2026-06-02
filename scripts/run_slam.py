#!/usr/bin/env python3
"""Wrapper to run latest dev SNN SLAM with correct pathing."""
import sys
import os
import runpy

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

# Run the live SLAM system orchestrator
runpy.run_path(os.path.join(ROOT, 'src/snn_live_slam.py'), run_name='__main__')
