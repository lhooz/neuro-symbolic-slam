#!/usr/bin/env python3
"""Wrapper to run stable1 SNN SLAM with correct pathing."""
import sys, os
ROOT = '/Users/lhooz/.openclaw/workspace'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src/stable1'))

# Patch __file__ for the script we're about to exec
import types
script_path = os.path.join(ROOT, 'src/stable1/snn_slam_system.py')

# Read and patch the script to fix project_root
with open(script_path) as f:
    src = f.read()

# The script computes project_root from __file__, which doesn't exist in exec()
# Patch it to use the known path
src = src.replace(
    "project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))",
    f"project_root = '{ROOT}'"
)

exec(compile(src, script_path, 'exec'), {'__name__': '__main__', '__file__': script_path})
