# data_wrapper.py
"""Compatibility alias: the CSV pipeline now lives in engine_v3.py.

Every earlier import (`from data_wrapper import process_and_fix_file`, load_csv_with_gaps,
export_reconstructed_csv) keeps working. See engine_v3.py for the post-fill blend options.
"""
from engine_v3 import (  # noqa: F401
    DEFAULT_KNN_BLEND,
    DEFAULT_ROW_BLEND,
    apply_blends,
    export_reconstructed_csv,
    load_csv_with_gaps,
    process_and_fix_file,
    repair_grid,
)

if __name__ == "__main__":
    import runpy
    runpy.run_module("engine_v3", run_name="__main__")
