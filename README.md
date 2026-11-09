# Ghost Data Engine (web app)

Streamlit interface for the Ghost Data Engine 3.3 (pure-Python gap filler for gridded data).

- `app.py` - the web app (tabs: repair a CSV, 2D demo, space-time 3D, benchmarks, about)
- `app_core.py` - Streamlit-free helpers around the engine (grid mode and table mode)
- `.streamlit/config.toml` - dark theme
- engine modules: `scanner.py`, `tension.py`, `reconstructor.py`, `engine_v3.py`, `gde_spatiotemporal.py`, `generator.py`, ...

Repair modes: **Table** (header and/or text columns: each numeric column is repaired on its own, text columns
are kept) and **Grid** (the whole file is one numeric matrix; post-fill blend available).

Run locally: `pip install -r requirements.txt` then `streamlit run app.py` (needs Streamlit 1.50 or newer)
Tests: `python -m unittest test_app_core test_engine test_gde_spatiotemporal test_calibrate_confidence test_blend`
