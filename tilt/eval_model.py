"""Board planarity / size of COLMAP text models of the gopro10 corrected images.

    python tilt/eval_model.py HULLS.npz MODEL_TXT [MODEL_TXT ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tilt.real import load_model, board_ids, planarity  # noqa: E402

hulls = np.load(sys.argv[1], allow_pickle=True)["hulls"].item()
for d in sys.argv[2:]:
    m = load_model(d)
    b = board_ids(m, hulls)
    print(d, json.dumps(dict(images=len(m["names"]), points=len(m["X"]), obs=len(m["ci"]),
                             K=[round(v, 2) for v in m["K"]], **planarity(m["X"][b]))))
