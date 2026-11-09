"""Ghost Data Engine - web app (Streamlit).

Run locally:   streamlit run app.py
The engine itself is pure Python; this file only provides the interface.
"""
import contextlib
import hashlib
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

# ---- dark plots to match the dark theme ------------------------------------
BG, FG = "#0e1117", "#fafafa"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "axes.edgecolor": "#3a3f4b"})
CMAP = plt.get_cmap("viridis").copy()
CMAP.set_bad("#3a3f4b")
FILL_CSS = "background-color: rgba(46, 160, 67, 0.28);"


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


MISS_CSS = "background-color: rgba(255, 75, 75, 0.30);"


def preview(df, filled):
    """Repaired Output Preview table; reconstructed cells are tinted green, leftovers show None."""
    styled = (df.style.apply(lambda _d: filled.map(lambda b: FILL_CSS if b else ""), axis=None)
              .format(na_rep="None"))
    st.dataframe(styled, width="stretch")


def scan_preview(df, missing):
    """Scanned Data table: every value as read, missing cells shown as None and tinted red."""
    styled = (df.style.apply(lambda _d: missing.map(lambda b: MISS_CSS if b else ""), axis=None)
              .format(na_rep="None"))
    st.dataframe(styled, width="stretch")


def _check_size(cells, label):
    limit = core.MAX_TABLE_CELLS if label == "table" else core.MAX_CELLS_2D
    if cells > limit:
        raise ValueError(f"This {label} has {cells} cells. The hosted app is limited to {limit} cells; "
                         "use the command line version for larger files.")


@st.cache_data(show_spinner="Scanning file...", max_entries=16)
def scan_table(data, delim):
    df, numeric = core.read_table(data, delim)
    _check_size(len(df) * max(1, len(numeric)), "table")
    if not numeric:
        raise ValueError("No numeric columns were found to repair.")
    missing = pd.DataFrame(False, index=df.index, columns=df.columns)
    missing[numeric] = df[numeric].isna()
    return df, missing


@st.cache_data(show_spinner="Scanning file...", max_entries=16)
def scan_grid(data, delim, header):
    if header:
        data = data.split(b"\n", 1)[1] if b"\n" in data else b""
    grid, w, h = core.load_grid_from_bytes(data, delim)
    _check_size(w * h, "grid")
    if not grid:
        raise ValueError("No numeric values found. Check the delimiter and the header option.")
    df = pd.DataFrame(core.grid_to_matrix(grid, w, h), columns=[f"col_{i + 1}" for i in range(w)])
    return df, df.isna()


# ---- cached engine runs (Streamlit re-runs the whole script on every click) ----
@st.cache_data(show_spinner="Reconstructing...", max_entries=16)
def run_table(data, delim, decay):
    df, numeric = core.read_table(data, delim)
    if len(df) * max(1, len(numeric)) > core.MAX_TABLE_CELLS:
        raise ValueError(f"This table has {len(df)} rows x {len(numeric)} numeric columns. The hosted app "
                         f"is limited to {core.MAX_TABLE_CELLS} cells; use the command line for larger files.")
    if not numeric:
        raise ValueError("No numeric columns were found to repair.")
    res = core.repair_table(df, numeric, decay)
    res["numeric_cols"] = numeric
    return res


@st.cache_data(show_spinner="Reconstructing...", max_entries=16)
def run_grid(data, delim, header, conn, decay, blend):
    if header:
        data = data.split(b"\n", 1)[1] if b"\n" in data else b""
    grid, w, h = core.load_grid_from_bytes(data, delim)
    if w * h > core.MAX_CELLS_2D:
        raise ValueError(f"This grid is {w}x{h} = {w * h} cells. The hosted app is limited to "
                         f"{core.MAX_CELLS_2D} cells. Use the command line version for larger files.")
    if not grid:
        raise ValueError("No numeric values found. Check the delimiter and the header option.")
    res = core.repair_matrix(grid, w, h, conn, decay, blend)
    res["w"], res["h"] = w, h
    if res["report"] is not None:
        res["audit"] = core.audit_rows(res["report"], w, h)
    return res


@st.cache_data(show_spinner="Running 2D demo...", max_entries=8)
def demo_2d(*args):
    return core.run_demo_2d(*args)


@st.cache_data(show_spinner="Running space-time demo...", max_entries=8)
def demo_3d(*args):
    return core.run_demo_3d(*args)


@st.cache_data(show_spinner="Running calibration...", max_entries=4)
def calibration(*args):
    return core.run_calibration(*args)


@st.cache_data(show_spinner="Running benchmark...", max_entries=4)
def synthetic_benchmark(size, frac):
    from benchmark import run_benchmark
    with contextlib.redirect_stdout(io.StringIO()):
        return run_benchmark(width=size, height=size, missing_fraction=frac, seed=7)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Architecture Note")
st.sidebar.info(
    "**Open Core / Public Demo**\n\n"
    "This public demo runs the real Ghost Data Engine directly: pure-Python, tension-weighted "
    "multi-wave reconstruction with optional post-fill blends. Every filled cell has an audit row. "
    "File size limits keep the hosted app responsive; use the command line for larger files."
)

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
    st.subheader("Repair a CSV")
    st.write("Upload a CSV. Empty cells and the words `nan`, `null`, `missing`, `bad`, `gap`, `-999` are "
             "treated as gaps. Recorded cells and text columns are never changed.")
    left, right = st.columns([2, 1])
    with left:
        up = st.file_uploader("CSV file", type=["csv", "txt"])
        use_sample = st.checkbox("Use the built-in sample", value=False, key="csv_sample",
                                 help="Used only while no file is uploaded.")
    with right:
        delim = st.selectbox("Delimiter", [",", ";", "\t"],
                             format_func=lambda d: {",": "comma", ";": "semicolon", "\t": "tab"}[d])

    data, source = None, None
    if up is not None:
        data, source = up.getvalue(), up.name
    elif use_sample:
        source = "sample"

    if source is None:
        st.info("Upload a CSV, or tick **Use the built-in sample**. The scanned data appears here, "
                "then press **Run reconstruction**.")
    else:
        auto_table = core.looks_like_table(data or core.sample_table_text().encode(), delim)
        mode = right.radio(
            "Data layout",
            ["Table (columns are separate variables)", "Grid (one numeric matrix)"],
            index=0 if auto_table else 1, key=f"mode_{source}_{auto_table}",
            help="Table: each numeric column is repaired on its own, text columns are kept. "
                 "Grid: the whole file is one spatial matrix (rows x columns).")
        is_table = mode.startswith("Table")
        if data is None:
            data = (core.sample_table_text() if is_table else core.sample_csv_text()).encode()
        decay = right.select_slider("Inward decay", options=[1.0, 0.95, 0.85], value=1.0, key="csv_decay")
        if is_table:
            conn, blend, header = 4, False, True
            right.caption("Post-fill blend is off in table mode (it was tuned on 2D grids).")
        else:
            conn = right.radio("Void connectivity", [4, 8], horizontal=True, key="csv_conn")
            blend = right.checkbox("Post-fill blend (recommended)", value=True, key="csv_blend")
            header = right.checkbox("First row is a header (skip it)", value=False)

        # ---- step 1: scan -------------------------------------------------
        try:
            scanned, missing = scan_table(data, delim) if is_table else scan_grid(data, delim, header)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the file: {exc}")
            scanned = None
        if scanned is not None:
            n_missing = int(missing.values.sum())
            n_cells = int(len(scanned) * len(scanned.columns))
            st.subheader("📋 Scanned Data")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows", len(scanned))
            m2.metric("Columns", len(scanned.columns))
            m3.metric("Missing cells", n_missing)
            m4.metric("Missing share", f"{100 * n_missing / max(1, n_cells):.1f}%")
            scan_preview(scanned, missing)
            st.caption("Red cells are missing and shown as None. Everything else is exactly as uploaded.")

            if n_missing == 0:
                st.success("No missing cells were found, so there is nothing to repair.")
            else:
                # ---- step 2: reconstruct ------------------------------------
                sig = hashlib.sha1(repr((data, delim, is_table, decay, conn, blend, header)).encode()).hexdigest()
                if st.button("🚀 Run reconstruction", type="primary"):
                    st.session_state["csv_sig"] = sig

                if st.session_state.get("csv_sig") == sig:
                    try:
                        res = (run_table(data, delim, decay) if is_table
                               else run_grid(data, delim, header, conn, decay, blend))
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not process the file: {exc}")
                        res = None
                    if res is not None:
                        if is_table:
                            repaired, filled = res["repaired"], res["filled"]
                            summary, audit = res["summary"], res["audit"]
                        else:
                            summary, audit = res["summary"], res.get("audit")
                            repaired = pd.DataFrame(res["repaired"], columns=scanned.columns)
                            filled = pd.DataFrame(res["filled_mask"], columns=scanned.columns)

                        st.success("Reconstruction Complete!")
                        show_summary(summary)
                        st.subheader("✨ Repaired Output Preview")
                        preview(repaired, filled)
                        st.caption("Green cells were reconstructed; everything else is exactly as uploaded.")
                        if is_table and res["skipped"]:
                            st.warning("No recorded value at all in: " + ", ".join(res["skipped"]) + ". Left as None.")
                        elif summary["cells_stalled"]:
                            st.warning(f"{summary['cells_stalled']} cell(s) had no path to any recorded "
                                       "value and stay as None.")
                        csv_out = repaired.to_csv(index=False) if is_table else core.matrix_to_csv(res["repaired"])
                        st.download_button("Download repaired CSV", csv_out, file_name="gde_repaired.csv",
                                           mime="text/csv")
                        if not is_table:
                            with st.expander("Heatmaps"):
                                vmin, vmax = np.nanmin(res["original"]), np.nanmax(res["original"])
                                fig, axs = plt.subplots(1, 2, figsize=(9, 4))
                                heat(axs[0], res["original"], "Original (grey = missing)", vmin, vmax)
                                heat(axs[1], res["repaired"], "Repaired", vmin, vmax)
                                st.pyplot(fig)
                                plt.close(fig)
                        with st.expander("Audit trail (one row per reconstructed cell)"):
                            df = pd.DataFrame(audit)
                            st.dataframe(df, width="stretch")
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
    if st.button("Run 2D demo", type="primary"):
        st.session_state["d2_ran"] = True
    if not shapes:
        st.info("Pick at least one void shape.")
    elif st.session_state.get("d2_ran"):
        r = demo_2d(size, size, int(seed), nv, vs, shapes, field, d_conn, 1.0, d_blend)
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
    else:
        st.info("Set the options, then press **Run 2D demo**.")

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
    if st.button("Run space-time demo", type="primary"):
        st.session_state["d3_ran"] = True
    if sz * sz * fr > core.MAX_CELLS_3D:
        st.error(f"{sz * sz * fr} cells exceeds the hosted limit of {core.MAX_CELLS_3D}.")
    elif st.session_state.get("d3_ran"):
        r3 = demo_3d(sz, fr, int(sd), nc, mf, tdec, 4, 0.9, mw)
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
    else:
        st.info("Set the options, then press **Run space-time demo**.")

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
        out = synthetic_benchmark(bsize, bfrac)
        rows = [{"method": k, **v} for k, v in out["methods"].items()]
        st.dataframe(pd.DataFrame(rows), width="stretch")
        st.caption("Lower MAE/RMSE is better. One synthetic field and one seed: indicative only.")

    st.divider()
    st.subheader("Does the confidence score track real error?")
    k1, k2 = st.columns(2)
    nseeds = k1.slider("Seeds", 3, 20, 8)
    csize = k2.slider("Grid size", 10, 24, 14, key="cs")
    if st.button("Run calibration"):
        res = calibration(nseeds, csize, 6)
        rho = res["spearman_confidence_vs_error"]
        x1, x2, x3 = st.columns(3)
        x1.metric("Cells scored", res["reconstructed_cells"])
        x2.metric("Overall 95th pct error", f"{res['overall_p95_error']:.2f}" if res["overall_p95_error"] is not None else "n/a")
        x3.metric("Spearman rho (conf vs error)", f"{rho:.3f}" if rho is not None else "n/a")
        bdf = pd.DataFrame([{"confidence": f"{b['low']:.1f}-{b['high']:.1f}", "cells": b["cells"],
                             "mean error": b["mean_error"], "P95 error": b["p95_error"]}
                            for b in res["buckets"]])
        st.dataframe(bdf, width="stretch")
        st.caption("A negative rho means higher confidence goes with lower error. Buckets with under 30 cells are unreliable.")

# ---------------------------------------------------------------------------
# Tab 5: about
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("""
**Ghost Data Engine** finds voids in a 2D grid (connected missing regions), measures the tension of the
recorded cells around each void, and fills it with inward propagation waves, optional smoothing, and
optional post-fill blends. A 3D version does the same across time frames.

**Table mode** repairs each numeric column of a CSV on its own (gaps are filled from the recorded values
above and below) and leaves text columns untouched. **Grid mode** treats the whole file as one spatial matrix.

**What it is not:** it does not recover the true hidden values. It produces auditable estimates, and the
confidence score is a heuristic. Use the benchmark and calibration tabs to see how large errors really are.

Author: Maurice Gathungu. Engine version 3.3.
""")
