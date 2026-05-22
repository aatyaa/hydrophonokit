"""HydroPhonoKit Entry Point"""
import os
import sys

# Force UTF-8 encoding globally to prevent Windows 'charmap' errors with VASP files
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from hydrophonokit.cli import main
main()
