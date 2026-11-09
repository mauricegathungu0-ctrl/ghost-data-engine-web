"""Run engine_v3 (with its default blends) on every *_MISSING.csv in tests/data/<dataset>/ and write
<name>_repaired.csv next to it. The engine reads only the MISSING file; the answer key is not opened here.
Then, from the dataset folder:  python score_all.py     (your scoring script, unchanged except for its name).
Usage (from tests/):  python make_repaired.py [temperature|humidity ...]   [--no-blend]"""
import contextlib, glob, io, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v3"))
import engine_v3 as E

args = [a for a in sys.argv[1:] if not a.startswith("--")]
kw = {"knn_blend": None, "row_blend": None} if "--no-blend" in sys.argv else {}
for ds in (args or ["temperature", "humidity"]):
    folder = os.path.join(HERE, "data", ds)
    for src in sorted(glob.glob(os.path.join(folder, "*_MISSING.csv"))):
        out = src.replace("_MISSING.csv", "_repaired.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            E.process_and_fix_file(src, out, **kw)
        print("wrote", os.path.relpath(out, HERE))
