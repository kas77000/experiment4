import os
import sys

# The standalone folder is its own root: `python -m tier5.fit` is always run
# from inside it, so tests must import the same way.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
