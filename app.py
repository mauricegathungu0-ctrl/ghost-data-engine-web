"""Ghost Data Engine - web app (Streamlit).

Run locally:   streamlit run app.py
The engine itself is pure Python; this file only provides the interface.
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import app_core as core
from generator import SHAPE_BUILDERS

st.set_page_config(page_title="Ghost Data Engine", page_icon="👻", layout="wide")

CMAP = plt.get_cmap("viridis").copy()
CMAP.set_bad("#d9d9d9")


def heat(ax, m, title, vmin=None, vmax=None):
    im = ax.imshow(m, cmap=CMAP, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def show_summary(s):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Voids found", s["total_voids"])
    c2.metric("Cells reconstructed", s["cells_reconstructed"])
    c3.metric("Cells not filled", s["cells_stalled"])
    c4.metric("Mean confidence (heuristic)", s["mean_confidence"])


# ---------------------------------------------------------------------------
st.title("👻 Ghost Data Engine")
st.caption("Auditable gap filling for gridded data. Pure-Python engine; every filled cell has an audit row. "
           "Filled values are estimates, not recovered truth.")

tab_csv, tab_2d, tab_3d, tab_bench, tab_about = st.tabs(
    ["Repair my CSV", "2D demo with ground truth", "Space-time (3D)", "Benchmarks & calibration", "About"])

# ---------------------------------------------------------------------------
# Tab 1: repair an uploaded CSV
# ---------------------------------------------------------------------------
with tab_csv:
    st.subheader("Repair a CSV matrix")
    st.write("Upload a numeric CSV grid. Empty cells and the words `nan`, `null`, `missing`, `bad`, `gap`, "
             "`-999` are treated as gaps. Recorded cells are never changed.")
    left, right = st.columns([2, 1])
    with right:
        conn = st.radio("Void connectivity", [4, 8], horizontal=True, key="csv_conn")
        decay = st.select_slider("Inward decay", options=[1.0, 0.95, 0.85], value=1.0, key="csv_decay")
        blend = st.checkbox("Post-fill blend (recommended)", value=True, key="csv_blend")
        header = st.checkbox("First row is a header (skip it)", value=False)
        delim = st.selectbox("Delimiter", [",", ";", "\t"], format_func=lambda d: {",": "comma", ";": "semicolon", "\t": "tab"}[d])
    with left:
        up = st.file_uploader("CSV file", type=["csv", "txt"])
        use_sample = st.checkbox("Use the built-in sample instead", value=(up is None))

    data = None
    if up is not None and not use_sample:
        data = up.getvalue()
    elif use_sample:
        data = core.sample_csv_text().encode()

    if data is not None:
        if header:
            data = data.split(b"\n", 1)[1] if b"\n" in data else b""
        try:
            grid, w, h = core.load_grid_from_bytes(data, delim)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the file: {exc}")
            grid = None
        if grid is not None:
            if w * h > core.MAX_CELLS_2D:
                st.error(f"This grid is {w}x{h} = {w*h} cells. The hosted app is limited to "
                         f"{core.MAX_CELLS_2D} cells. Use the command line version for larger files.")
            elif not grid:
                st.error("No numeric values found. Check the delimiter and the header option.")
            else:
                res = core.repair_matrix(grid, w, h, conn, decay, blend)
                if res["report"] is None:
                    st.success("No missing cells were found, so nothing was changed.")
                else:
                    show_summary(res["summary"])
                    vmin, vmax = np.nanmin(res["original"]), np.nanmax(res["original"])
                    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
                    heat(axs[0], res["original"], "Original (grey = missing)", vmin, vmax)
                    heat(axs[1], res["repaired"], "Repaired", vmin, vmax)
                    st.pyplot(fig)
                    plt.close(fig)
                    st.download_button("Download repaired CSV", core.matrix_to_csv(res["repaired"]),
                                       file_name="gde_repaired.csv", mime="text/csv")
                    if res["summary"]["cells_stalled"]:
                        st.warning(f"{res['summary']['cells_stalled']} cell(s) had no path to any recorded "
                                   "value and stay as NaN.")
                    with st.expander("Audit trail (one row per reconstructed cell)"):
                        df = pd.DataFrame(core.audit_rows(res["report"], w, h))
                        st.dataframe(df, use_container_width=True)
                        st.download_button("Download audit trail (CSV)", df.to_csv(index=False),
                                           file_name="gde_audit.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Tab 2: 2D demo with known truth
# ---------------------------------------------------------------------------
with tab_2d:
    st.subheader("Hide cells from a known field, repair them, and score the result")
    c1, c2, c3 = st.columns(3)
    size = c1.slider("Grid size", 15, 70, 40)
    seed = c1.number_input("Seed", 0, 9999, 1)
    nv = c2.slider("Number of voids", 1, 12, 5)
    vs = c2.slider("Max void size", 2, 12, 6)
    field = c3.selectbox("True field", ["waves", "plane", "bump"])
    shapes = c3.multiselect("Void shapes", list(SHAPE_BUILDERS), default=["blob", "rectangle", "l_shape"])
    d_conn = c1.radio("Connectivity", [4, 8], horizontal=True, key="d2_conn")
    d_blend = c2.checkbox("Post-fill blend", value=True, key="d2_blend")
    if not shapes:
        st.info("Pick at least one void shape.")
    else:
        r = core.run_demo_2d(size, size, int(seed), nv, vs, shapes, field, d_conn, 1.0, d_blend)
        if r["report"] is None:
            st.info("No voids were generated; try another seed.")
        else:
            show_summary(r["summary"])
            m1, m2 = st.columns(2)
            m1.metric("Mean absolute error on hidden cells", f"{r['mae']:.3f}")
            m2.metric("Field range (max - min)", f"{r['value_range']:.1f}")
            masked = np.where(r["hidden"], np.nan, r["truth"])
            vmin, vmax = r["truth"].min(), r["truth"].max()
            fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
            heat(axs[0], r["truth"], "Truth", vmin, vmax)
            heat(axs[1], masked, "Given to engine", vmin, vmax)
            heat(axs[2], r["repaired"], "Repaired", vmin, vmax)
            im = axs[3].imshow(np.where(r["filled_mask"], np.abs(r["repaired"] - r["truth"]), np.nan),
                               cmap="magma", interpolation="nearest")
            axs[3].set_title("Absolute error (filled cells)", fontsize=10)
            axs[3].set_xticks([]); axs[3].set_yticks([])
            fig.colorbar(im, ax=axs[3], fraction=0.046)
            st.pyplot(fig)
            plt.close(fig)
            st.caption("The engine only sees the second panel. The truth is used afterwards to score.")

# ---------------------------------------------------------------------------
# Tab 3: 3D space-time
# ---------------------------------------------------------------------------
with tab_3d:
    st.subheader("Space-time reconstruction (drifting cloud masks)")
    c1, c2, c3 = st.columns(3)
    sz = c1.slider("Grid size", 8, 30, 16, key="s3")
    fr = c1.slider("Frames", 2, 12, 6)
    sd = c2.number_input("Seed", 0, 9999, 3, key="seed3")
    nc = c2.slider("Clouds", 1, 5, 3)
    mf = c3.slider("Max hidden share", 0.1, 0.6, 0.4)
    tdec = c3.slider("Temporal decay", 0.05, 1.0, 0.5)
    mw_on = st.checkbox("Cap propagation depth (decline cells deeper than N waves)")
    mw = st.slider("Max waves", 1, 10, 4) if mw_on else None
    if sz * sz * fr > core.MAX_CELLS_3D:
        st.error(f"{sz*sz*fr} cells exceeds the hosted limit of {core.MAX_CELLS_3D}.")
    else:
        r3 = core.run_demo_3d(sz, fr, int(sd), nc, mf, tdec, 4, 0.9, mw)
        s = r3["report"]["summary"]
        a, b, c, d = st.columns(4)
        a.metric("Hidden cells", s["missing_cells"])
        b.metric("Reconstructed", s["cells_reconstructed"])
        c.metric("Left unresolved", s["cells_unresolved"])
        d.metric("MAE / 95th pct error", f"{r3['mae']:.2f} / {r3['p95']:.2f}" if r3["mae"] is not None else "n/a")
        t_idx = st.slider("Frame to display", 0, fr - 1, 0)
        vmin, vmax = r3["truth"].min(), r3["truth"].max()
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
        heat(axs[0], r3["truth"][t_idx], f"Truth (t={t_idx})", vmin, vmax)
        heat(axs[1], r3["observed"][t_idx], "Given to engine", vmin, vmax)
        heat(axs[2], r3["predicted"][t_idx], "Reconstructed", vmin, vmax)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Confidence scores are heuristic diagnostics, not probabilities. See the calibration tab.")

# ---------------------------------------------------------------------------
# Tab 4: benchmarks and calibration
# ---------------------------------------------------------------------------
with tab_bench:
    st.subheader("Synthetic benchmark (2D)")
    st.write("Ghost engine vs a global-mean baseline and an iterative neighbour-average baseline, "
             "on random hidden cells of a smooth field.")
    bc1, bc2 = st.columns(2)
    bsize = bc1.slider("Grid size", 15, 50, 30, key="bs")
    bfrac = bc2.slider("Hidden fraction", 0.05, 0.5, 0.2, key="bf")
    if st.button("Run benchmark"):
        from benchmark import run_benchmark
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            out = run_benchmark(width=bsize, height=bsize, missing_fraction=bfrac, seed=7)
        rows = [{"method": k, **v} for k, v in out["methods"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("Lower MAE/RMSE is better. One synthetic field and one seed: indicative only.")

    st.divider()
    st.subheader("Does the confidence score track real error?")
    k1, k2 = st.columns(2)
    nseeds = k1.slider("Seeds", 3, 20, 8)
    csize = k2.slider("Grid size", 10, 24, 14, key="cs")
    if st.button("Run calibration"):
        res = core.run_calibration(nseeds, csize, 6)
        rho = res["spearman_confidence_vs_error"]
        x1, x2, x3 = st.columns(3)
        x1.metric("Cells scored", res["reconstructed_cells"])
        x2.metric("Overall 95th pct error", f"{res['overall_p95_error']:.2f}" if res["overall_p95_error"] is not None else "n/a")
        x3.metric("Spearman rho (conf vs error)", f"{rho:.3f}" if rho is not None else "n/a")
        bdf = pd.DataFrame([{"confidence": f"{b['low']:.1f}-{b['high']:.1f}", "cells": b["cells"],
                             "mean error": b["mean_error"], "P95 error": b["p95_error"]}
                            for b in res["buckets"]])
        st.dataframe(bdf, use_container_width=True)
        st.caption("A negative rho means higher confidence goes with lower error. Buckets with under 30 cells are unreliable.")

# ---------------------------------------------------------------------------
# Tab 5: about
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("""
**Ghost Data Engine** finds voids in a 2D grid (connected missing regions), measures the tension of the
recorded cells around each void, and fills it with inward propagation waves, optional smoothing, and
optional post-fill blends. A 3D version does the same across time frames.

**What it is not:** it does not recover the true hidden values. It produces auditable estimates, and the
confidence score is a heuristic. Use the benchmark and calibration tabs to see how large errors really are.

Author: Maurice Gathungu. Engine version 3.3.
""")
