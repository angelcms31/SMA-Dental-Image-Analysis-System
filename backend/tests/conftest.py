import os
import sys

# make `import sma_algorithms` work regardless of the directory pytest is run from
BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
