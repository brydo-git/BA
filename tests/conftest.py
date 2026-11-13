import sys
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required

# Make the ba/ package root importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
