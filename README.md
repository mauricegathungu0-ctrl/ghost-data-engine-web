# Ghost Data Engine (web app)

Streamlit interface for the Ghost Data Engine 3.3 (pure-Python gap filler for gridded data).

- `app.py` - the web app (tabs: repair a CSV, 2D demo, space-time 3D, benchmarks, about)
- `app_core.py` - Streamlit-free helpers around the engine
- engine modules: `scanner.py`, `tension.py`, `reconstructor.py`, `engine_v3.py`, `gde_spatiotemporal.py`, `generator.py`, ...

Run locally: `pip install -r requirements.txt` then `streamlit run app.py`
Tests: `python -m unittest test_engine test_gde_spatiotemporal test_calibrate_confidence test_blend`
