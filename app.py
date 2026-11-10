"""Ghost Data Engine - web app (Streamlit).

Run locally:   streamlit run app.py
The engine itself is pure Python; this file only provides the interface.
"""
import contextlib
import hashlib
import io
import os

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
    if df.attrs.get("skipped_lines"):
        st.warning(f"{df.attrs['skipped_lines']} malformed line(s) in the file were skipped while reading it.")
    _check_size(len(df) * max(1, len(numeric)), "table")
    if not numeric:
        raise ValueError("No numeric columns were found to repair.")
    missing = pd.DataFrame(False, index=df.index, columns=df.columns)
    missing[numeric] = df[numeric].isna()
    return df, missing


@st.cache_data(show_spinner="Scanning file...", max_entries=16)
def scan_grid(data, delim, header, labelled=False):
    cols = rows = None
    if labelled:
        _corner, cols, rows, data = core.split_labelled(data, delim)
    elif header:
        data = data.split(b"\n", 1)[1] if b"\n" in data else b""
    grid, w, h = core.load_grid_from_bytes(data, delim)
    _check_size(w * h, "grid")
    if not grid:
        raise ValueError("No numeric values found. Check the delimiter and the header option.")
    df = pd.DataFrame(core.grid_to_matrix(grid, w, h),
                      columns=cols if cols else [f"col_{i + 1}" for i in range(w)],
                      index=rows if rows else None)
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
def run_grid(data, delim, header, conn, decay, blend, labelled=False):
    if labelled:
        data = core.split_labelled(data, delim)[3]
    elif header:
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


@st.cache_data(show_spinner="Hiding cells, repairing and scoring...", max_entries=8)
def live_test(data, delim, scatter, blocks, bmin, bmax, seed, custom, blend):
    truth = core.load_complete_with_labels(data, delim)[0]
    mask = core.make_hide_mask(truth.shape, scatter, blocks, bmin, bmax, seed, custom)
    return truth, mask, core.accuracy_test(truth, mask, blend=blend)


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
    "Uploads up to 500 MB are accepted; a cell cap per request keeps the hosted app responsive, so use the command line for very large grids."
)

# ---------------------------------------------------------------------------
st.title("👻 Ghost Data Engine")
st.caption("Auditable gap filling for gridded data. Pure-Python engine; every filled cell has an audit row. "
           "Filled values are estimates, not recovered truth.")

tab_csv, tab_live, tab_2d, tab_3d, tab_bench, tab_about = st.tabs(
    ["Repair my CSV", "🎯 Live accuracy test", "2D demo with ground truth", "Space-time (3D)",
     "Benchmarks & calibration", "About"])

# ---------------------------------------------------------------------------
# Tab 1: repair an uploaded CSV
# ---------------------------------------------------------------------------
with tab_csv:
    st.subheader("Repair a CSV")
    st.write("Upload a CSV. Empty cells and the words `nan`, `null`, `missing`, `bad`, `gap`, `-999` are "
             "treated as gaps. Recorded cells and text columns are never changed.")
    left, right = st.columns([2, 1])
    with left:
        up = st.file_uploader("CSV file", type=["csv", "txt"],
                              help="Uploads up to 500 MB are accepted. Files with more cells than the processing cap are scanned but not repaired here; use the command line for those.")
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
        auto_labelled = bool(data) and core.detect_labelled(data, delim)
        auto_table = (not auto_labelled) and core.looks_like_table(data or core.sample_table_text().encode(), delim)
        mode = right.radio(
            "Data layout",
            ["Table (columns are separate variables)", "Grid (one numeric matrix)"],
            index=0 if auto_table else 1, key=f"mode_{source}_{auto_table}_{auto_labelled}",
            help="Table: each numeric column is repaired on its own, text columns are kept. "
                 "Grid: the whole file is one spatial matrix (rows x columns).")
        is_table = mode.startswith("Table")
        if data is None:
            data = (core.sample_table_text() if is_table else core.sample_csv_text()).encode()
        decay = right.select_slider("Inward decay", options=[1.0, 0.95, 0.85], value=1.0, key="csv_decay")
        if is_table:
            conn, blend, header, labelled = 4, False, True, False
            if auto_labelled:
                right.warning("This file looks like a Year x Day grid. Table mode repairs each column "
                              "separately, which is much less accurate here. Choose **Grid** instead.")
            right.caption("Post-fill blend is off in table mode (it was tuned on 2D grids).")
        else:
            conn = right.radio("Void connectivity", [4, 8], horizontal=True, key="csv_conn")
            blend = right.checkbox("Post-fill blend (recommended)", value=True, key="csv_blend")
            labelled = right.checkbox("Keep headings and row labels (first row and first column)",
                                      value=auto_labelled, key=f"lab_{source}_{auto_labelled}",
                                      help="For files such as Year x Day: the first row holds column headings and "
                                           "the first column holds labels. They are shown and kept, never repaired.")
            header = False if labelled else right.checkbox("First row is a header (skip it)", value=False)

        # ---- step 1: scan -------------------------------------------------
        try:
            scanned, missing = scan_table(data, delim) if is_table else scan_grid(data, delim, header, labelled)
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
                sig = hashlib.sha1(repr((data, delim, is_table, decay, conn, blend, header, labelled)).encode()).hexdigest()
                if st.button("🚀 Run reconstruction", type="primary"):
                    st.session_state["csv_sig"] = sig

                if st.session_state.get("csv_sig") == sig:
                    try:
                        res = (run_table(data, delim, decay) if is_table
                               else run_grid(data, delim, header, conn, decay, blend, labelled))
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not process the file: {exc}")
                        res = None
                    if res is not None:
                        if is_table:
                            repaired, filled = res["repaired"], res["filled"]
                            summary, audit = res["summary"], res["audit"]
                        else:
                            summary, audit = res["summary"], res.get("audit")
                            repaired = pd.DataFrame(res["repaired"], columns=scanned.columns, index=scanned.index)
                            filled = pd.DataFrame(res["filled_mask"], columns=scanned.columns, index=scanned.index)

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
                        if is_table:
                            csv_out = repaired.to_csv(index=False)
                        elif labelled:
                            csv_out = repaired.round(4).to_csv(index_label=core.split_labelled(data, delim)[0])
                        else:
                            csv_out = core.matrix_to_csv(res["repaired"])
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
# Tab: live accuracy test (hide known cells yourself, repair, score)
# ---------------------------------------------------------------------------
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data",
                           "machakos_temperature_complete.csv")


def _labelled(m, rows=None, cols=None):
    return pd.DataFrame(m, index=rows or [f"Row {i + 1}" for i in range(m.shape[0])],
                        columns=cols or [str(j + 1) for j in range(m.shape[1])])


with tab_live:
    st.subheader("Hide real data yourself, let the engine repair it, and check the answer")
    st.write("Start from a **complete** dataset where every true value is known. You choose what to hide, "
             "the engine sees only what is left, and the repaired values are scored against the truth "
             "and against simple methods.")
    src = st.radio("Data", ["Built-in: Machakos daily temperature (35 years x 365 days, NASA POWER)",
                            "Upload my own complete CSV (all numbers; optional heading row and label column; no blanks)"],
                   key="live_src")
    live_data = None
    if src.startswith("Built-in"):
        if os.path.exists(SAMPLE_PATH):
            with open(SAMPLE_PATH, "rb") as fh:
                live_data = fh.read()
        else:
            st.error("The built-in file sample_data/machakos_temperature_complete.csv is missing.")
    else:
        up = st.file_uploader("Complete CSV", type=["csv", "txt"], key="live_up")
        if up is not None:
            live_data = up.getvalue()
    if live_data is not None:
        try:
            truth_m, row_lab, col_lab = core.load_complete_with_labels(live_data)
            if truth_m.size > core.MAX_CELLS_2D:
                st.error(f"{truth_m.size} cells is above the hosted limit of {core.MAX_CELLS_2D}.")
                truth_m = None
        except ValueError as e:
            st.error(str(e))
            truth_m = None
        if truth_m is not None:
            H, W = truth_m.shape
            st.caption(f"{H} rows x {W} columns = {truth_m.size} cells.")
            c1, c2, c3 = st.columns(3)
            scatter_pct = c1.slider("Scattered missing cells (%)", 0, 30, 5, key="lv_sc")
            nblocks = c2.slider("Long outages (blocks)", 0, 20, 6, key="lv_nb")
            lo, hi = c3.slider("Outage length (columns)", 1, max(2, min(W - 1, 200)),
                               (min(30, max(1, W // 4)), min(90, max(2, W // 2))), key="lv_len")
            seed_l = c1.number_input("Random seed", 0, 9999, 0, key="lv_seed")
            lv_blend = c2.checkbox("Post-fill blend", value=True, key="lv_blend")
            use_custom = c3.checkbox("Also hide one block I choose", key="lv_cb")
            custom = None
            if use_custom:
                d1, d2, d3, d4 = st.columns(4)
                r0 = d1.number_input("Row from", 1, H, min(10, H), key="lv_r0")
                r1 = d2.number_input("Row to", 1, H, min(12, H), key="lv_r1")
                k0 = d3.number_input("Column from", 1, W, min(100, W), key="lv_c0")
                k1 = d4.number_input("Column to", 1, W, min(140, W), key="lv_c1")
                custom = (int(min(r0, r1)), int(max(r0, r1)), int(min(k0, k1)), int(max(k0, k1)))

            sig = hashlib.sha1(repr((hashlib.sha1(live_data).hexdigest(), scatter_pct, nblocks, lo, hi,
                                     int(seed_l), custom, lv_blend)).encode()).hexdigest()
            if st.button("1️⃣ Hide cells", type="primary", key="lv_hide"):
                st.session_state["lv_hidden"] = sig
                st.session_state.pop("lv_done", None)
            if st.session_state.get("lv_hidden") != sig:
                st.info("Set what to hide, then press **1️⃣ Hide cells**. Changing any setting starts over.")
            else:
                truth_m, mask_m, res = live_test(live_data, ",", scatter_pct / 100.0, nblocks, lo, hi,
                                                  int(seed_l), custom, lv_blend)
                n_hid = int(mask_m.sum())
                st.markdown("### 📋 Scanned Data (what the engine is given)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Cells hidden", n_hid)
                m2.metric("Share hidden", f"{100 * n_hid / mask_m.size:.1f}%")
                m3.metric("Cells recorded", int(mask_m.size - n_hid))
                shown = _labelled(np.where(mask_m, np.nan, truth_m), row_lab, col_lab)
                miss = _labelled(mask_m.astype(bool), row_lab, col_lab)
                scan_preview(shown, miss)
                st.caption("Red cells show None: they are hidden from the engine.")
                if st.button("2️⃣ Repair the hidden cells", type="primary", key="lv_go"):
                    st.session_state["lv_done"] = sig
                if st.session_state.get("lv_done") == sig:
                    st.success("Reconstruction Complete!")
                    sc = pd.DataFrame(res["scores"])
                    g = sc.iloc[0]
                    best_other = sc.iloc[1:]["Mean abs. error"].min()
                    k1_, k2_, k3_ = st.columns(3)
                    k1_.metric("Hidden cells filled", f"{res['n_filled']} of {res['n_hidden']}")
                    k2_.metric("Engine mean abs. error", f"{g['Mean abs. error']:.3f}")
                    k3_.metric("Best simple method", f"{best_other:.3f}")
                    st.markdown("### ✨ Repaired Output Preview")
                    preview(_labelled(res["repaired"], row_lab, col_lab), _labelled(mask_m.astype(bool), row_lab, col_lab))
                    st.caption("Green cells were filled by the engine. Recorded cells are unchanged.")
                    st.markdown("### 🎯 Accuracy against the true values")
                    st.dataframe(sc.style.format({"Mean abs. error": "{:.3f}", "RMSE": "{:.3f}",
                                                  "Seconds": "{:.2f}"}), width="stretch", hide_index=True)
                    if g["Mean abs. error"] < best_other:
                        st.caption("On this run the engine's error is lower than the best simple method. "
                                   "Results vary with what you hide: short, scattered gaps are easy for "
                                   "linear interpolation, long outages are where the engine helps most.")
                    else:
                        st.caption("On this run the engine did not beat the best simple method. That is "
                                   "expected for short or scattered gaps; try longer outages.")
                    err = np.where(mask_m, np.abs(res["repaired"] - truth_m), np.nan)
                    vmin, vmax = float(truth_m.min()), float(truth_m.max())
                    fig, axs = plt.subplots(4, 1, figsize=(12, 9))
                    heat(axs[0], truth_m, "Truth (answer key)", vmin, vmax)
                    heat(axs[1], np.where(mask_m, np.nan, truth_m), "Given to the engine", vmin, vmax)
                    heat(axs[2], res["repaired"], "Repaired", vmin, vmax)
                    im = axs[3].imshow(err, cmap="magma", interpolation="nearest", aspect="auto")
                    axs[3].set_title("Absolute error on hidden cells", fontsize=10)
                    axs[3].set_xticks([]); axs[3].set_yticks([])
                    fig.colorbar(im, ax=axs[3], fraction=0.02)
                    for a_ in axs[:3]:
                        a_.set_aspect("auto")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.download_button("Download repaired CSV", core.matrix_to_csv(res["repaired"]),
                                       "repaired.csv", "text/csv", key="lv_dl")
                    with st.expander("Show the answer key (true values)"):
                        st.dataframe(_labelled(truth_m, row_lab, col_lab), width="stretch")
                else:
                    st.info("Now press **2️⃣ Repair the hidden cells**.")

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
