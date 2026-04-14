"""
========================================================
  Automated EDA Report Generator
  Compatible with: Databricks, Jupyter, local Python
  Supports: CSV, XLSX input  |  HTML output
  Author:   Data Science Toolkit
  Version:  2.0
========================================================

USAGE — DATABRICKS:
    1. Upload this script or paste into a notebook cell
    2. Set FILE_PATH and OUTPUT_PATH below
    3. Run: %run /path/to/eda_report  or paste & execute

USAGE — LOCAL:
    pip install pandas numpy scipy matplotlib seaborn openpyxl jinja2
    python eda_report.py

DATABRICKS DEPENDENCIES (already available in DBR 10+):
    pandas, numpy, scipy, matplotlib, seaborn, openpyxl
"""

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION — Edit these before running
# ─────────────────────────────────────────────────────────────
FILE_PATH   = "your_dataset.csv"          # CSV or XLSX path (DBFS: /dbfs/path/file.csv)
OUTPUT_PATH = "/tmp/eda_report.html"      # Output HTML path  (DBFS: /dbfs/tmp/eda_report.html)
SAMPLE_ROWS = 5                           # Rows shown in data preview
MAX_UNIQUE_CAT = 30                       # Max unique values to plot for categoricals
CORRELATION_THRESHOLD = 0.7              # Highlight correlations above this
# ─────────────────────────────────────────────────────────────

import os
import sys
import warnings
import base64
import io
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # Non-interactive backend (safe for Databricks)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Seaborn / Matplotlib style ─────────────────────────────
PALETTE   = ["#00B4D8", "#0D1B2A", "#F4A261", "#10B981", "#EF4444",
             "#7C3AED", "#F97316", "#14B8A6", "#EC4899", "#84CC16"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
    "axes.edgecolor":   "#E2E8F0",
    "axes.labelcolor":  "#1E293B",
    "xtick.color":      "#64748B",
    "ytick.color":      "#64748B",
    "grid.color":       "#E2E8F0",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.titlecolor":  "#0D1B2A",
})


# ════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════

def fig_to_b64(fig, dpi=130):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def human_size(n_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def safe_skew(series):
    try:
        return round(float(series.skew()), 4)
    except Exception:
        return None


def safe_kurt(series):
    try:
        return round(float(series.kurt()), 4)
    except Exception:
        return None


def shapiro_result(series):
    try:
        clean = series.dropna()
        n = min(len(clean), 5000)
        _, p = stats.shapiro(clean.sample(n, random_state=42) if len(clean) > n else clean)
        return ("✅ Normal" if p > 0.05 else "⚠ Non-normal"), round(p, 5)
    except Exception:
        return "N/A", None


# ════════════════════════════════════════════════════════════
#  DATA LOADER
# ════════════════════════════════════════════════════════════

def load_data(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .xlsx")
    print(f"[✓] Loaded {len(df):,} rows × {len(df.columns)} columns from '{path}'")
    return df


# ════════════════════════════════════════════════════════════
#  EDA COMPUTATIONS
# ════════════════════════════════════════════════════════════

def compute_overview(df, path):
    try:
        fsize = human_size(os.path.getsize(path))
    except Exception:
        fsize = "N/A"

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols  = df.select_dtypes(include=["datetime64"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    dup_rows      = int(df.duplicated().sum())
    total_missing = int(df.isnull().sum().sum())
    total_cells   = df.shape[0] * df.shape[1]
    miss_pct      = round(100 * total_missing / total_cells, 2) if total_cells else 0

    return {
        "file_path":      path,
        "file_size":      fsize,
        "rows":           len(df),
        "columns":        len(df.columns),
        "numeric_cols":   len(num_cols),
        "categorical_cols": len(cat_cols),
        "datetime_cols":  len(dt_cols),
        "bool_cols":      len(bool_cols),
        "duplicate_rows": dup_rows,
        "total_missing":  total_missing,
        "missing_pct":    miss_pct,
        "memory_mb":      round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "num_col_names":  num_cols,
        "cat_col_names":  cat_cols,
        "dt_col_names":   dt_cols,
    }


def compute_missing(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    dtype_map = df.dtypes.astype(str)
    unique_cnt = df.nunique()
    result = []
    for col in df.columns:
        result.append({
            "column":  col,
            "dtype":   dtype_map[col],
            "missing": int(missing[col]),
            "missing_pct": float(missing_pct[col]),
            "unique":  int(unique_cnt[col]),
        })
    return sorted(result, key=lambda x: x["missing_pct"], reverse=True)


def compute_numeric_stats(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    stats_list = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        norm_label, norm_p = shapiro_result(df[col])
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        stats_list.append({
            "column":   col,
            "count":    int(s.count()),
            "mean":     round(float(s.mean()), 4),
            "median":   round(float(s.median()), 4),
            "std":      round(float(s.std()), 4),
            "min":      round(float(s.min()), 4),
            "max":      round(float(s.max()), 4),
            "q25":      round(float(q1), 4),
            "q75":      round(float(q3), 4),
            "skewness": safe_skew(s),
            "kurtosis": safe_kurt(s),
            "outliers": outliers,
            "normality": norm_label,
            "norm_p":   norm_p,
        })
    return stats_list


def compute_cat_stats(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    stats_list = []
    for col in cat_cols:
        s = df[col].dropna()
        vc = s.value_counts()
        stats_list.append({
            "column":     col,
            "count":      int(s.count()),
            "missing":    int(df[col].isnull().sum()),
            "unique":     int(s.nunique()),
            "top_value":  str(vc.index[0]) if len(vc) > 0 else "N/A",
            "top_freq":   int(vc.iloc[0]) if len(vc) > 0 else 0,
            "top_pct":    round(100 * vc.iloc[0] / len(s), 2) if len(vc) > 0 else 0,
            "value_counts": vc.head(MAX_UNIQUE_CAT).to_dict(),
        })
    return stats_list


# ════════════════════════════════════════════════════════════
#  CHART GENERATORS
# ════════════════════════════════════════════════════════════

def chart_missing_heatmap(df):
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(8, len(missing_pct) * 0.5), 4))
    bars = ax.bar(missing_pct.index, missing_pct.values,
                  color=[PALETTE[0] if v < 20 else PALETTE[4] for v in missing_pct.values],
                  edgecolor="white", linewidth=0.5, width=0.6)
    ax.axhline(20, color=PALETTE[4], linestyle="--", linewidth=1, alpha=0.7, label="20% threshold")
    for bar, val in zip(bars, missing_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color="#1E293B")
    ax.set_title("Missing Values by Column (%)", pad=12)
    ax.set_ylabel("Missing (%)")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, min(100, missing_pct.max() * 1.25))
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_dtype_pie(overview):
    sizes = [overview["numeric_cols"], overview["categorical_cols"],
             overview["datetime_cols"], overview["bool_cols"]]
    labels = ["Numeric", "Categorical", "Datetime", "Boolean"]
    sizes_f = [(l, s) for l, s in zip(labels, sizes) if s > 0]
    if not sizes_f:
        return None
    labels_f, sizes_f2 = zip(*sizes_f)
    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        sizes_f2, labels=labels_f, autopct="%1.1f%%",
        colors=PALETTE[:len(sizes_f2)], startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.78, labeldistance=1.12
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Column Type Distribution", pad=12)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_numeric_histograms(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        return None
    ncols = min(3, len(num_cols))
    nrows = (len(num_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
    axes = np.array(axes).flatten() if len(num_cols) > 1 else [axes]
    for i, col in enumerate(num_cols):
        ax = axes[i]
        data = df[col].dropna()
        if len(data) == 0:
            ax.set_visible(False)
            continue
        ax.hist(data, bins=30, color=PALETTE[i % len(PALETTE)],
                edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.axvline(data.mean(), color="#EF4444", linestyle="--", linewidth=1.2, label=f"Mean: {data.mean():.2f}")
        ax.axvline(data.median(), color="#F4A261", linestyle=":", linewidth=1.2, label=f"Median: {data.median():.2f}")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.legend(fontsize=7, framealpha=0.7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numeric Column Distributions", fontsize=13, fontweight="bold", y=1.01, color="#0D1B2A")
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_boxplots(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        return None
    ncols = min(3, len(num_cols))
    nrows = (len(num_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.0))
    axes = np.array(axes).flatten() if len(num_cols) > 1 else [axes]
    for i, col in enumerate(num_cols):
        ax = axes[i]
        data = df[col].dropna()
        if len(data) == 0:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                        widths=0.5, showfliers=True,
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor=PALETTE[4], alpha=0.5))
        bp["boxes"][0].set_facecolor(PALETTE[i % len(PALETTE)])
        bp["boxes"][0].set_alpha(0.7)
        bp["medians"][0].set_color("#EF4444")
        bp["medians"][0].set_linewidth(2)
        ax.set_title(col, fontsize=10)
        ax.set_xticks([])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Box Plots — Outlier Detection", fontsize=13, fontweight="bold", y=1.01, color="#0D1B2A")
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_correlation(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        return None
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(7, len(num_cols) * 0.85), max(6, len(num_cols) * 0.75)))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=0.5,
                linecolor="#E2E8F0", ax=ax, annot_kws={"size": 8},
                cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation Matrix (Pearson)", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_categorical_bars(cat_stats):
    charts = {}
    for cs in cat_stats:
        if not cs["value_counts"] or cs["unique"] > MAX_UNIQUE_CAT:
            continue
        vc = cs["value_counts"]
        labels = [str(k)[:30] for k in vc.keys()]
        values = list(vc.values())
        top_n = min(15, len(labels))
        labels, values = labels[:top_n], values[:top_n]
        fig, ax = plt.subplots(figsize=(max(6, top_n * 0.55), 4))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
        bars = ax.bar(labels, values, color=colors, edgecolor="white",
                      linewidth=0.5, width=0.65)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:,}", ha="center", va="bottom", fontsize=8, color="#1E293B")
        ax.set_title(f"{cs['column']} — Value Counts (top {top_n})", fontsize=10)
        ax.set_ylabel("Count")
        plt.xticks(rotation=40, ha="right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        charts[cs["column"]] = fig_to_b64(fig)
    return charts


def chart_scatter_matrix(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        return None
    cols = num_cols[:min(6, len(num_cols))]
    sample = df[cols].dropna().sample(min(2000, len(df)), random_state=42)
    fig = plt.figure(figsize=(max(8, len(cols) * 2.2), max(7, len(cols) * 2.0)))
    axes = pd.plotting.scatter_matrix(sample, alpha=0.35, figsize=(max(8, len(cols) * 2.2),
                                       max(7, len(cols) * 2.0)),
                                       diagonal="kde", color=PALETTE[0],
                                       hist_kwds={"bins": 20, "color": PALETTE[0], "edgecolor": "white"})
    for ax in axes.flatten():
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)
        ax.tick_params(labelsize=7)
    plt.suptitle("Scatter Matrix (numeric columns)", fontsize=12,
                 fontweight="bold", color="#0D1B2A", y=1.01)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_high_corr(df, threshold=CORRELATION_THRESHOLD):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        return []
    corr = df[num_cols].corr().abs()
    pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            val = corr.iloc[i, j]
            if val >= threshold:
                pairs.append({
                    "col1": num_cols[i],
                    "col2": num_cols[j],
                    "corr": round(float(corr.iloc[i, j]), 4),
                    "sign": "positive" if df[num_cols].corr().iloc[i, j] > 0 else "negative"
                })
    return sorted(pairs, key=lambda x: x["corr"], reverse=True)


# ════════════════════════════════════════════════════════════
#  HTML REPORT BUILDER
# ════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>EDA Report — {filename}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {{
    --navy:    #0D1B2A;
    --teal:    #00B4D8;
    --gold:    #F4A261;
    --green:   #10B981;
    --red:     #EF4444;
    --orange:  #F97316;
    --purple:  #7C3AED;
    --bg:      #F1F5F9;
    --surface: #FFFFFF;
    --border:  #E2E8F0;
    --text:    #1E293B;
    --muted:   #64748B;
    --radius:  12px;
  }}

  * {{ margin:0; padding:0; box-sizing:border-box; }}

  body {{
    font-family: 'Space Grotesk', sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.6;
  }}

  /* ── HEADER ── */
  .header {{
    background: var(--navy);
    color: white;
    padding: 36px 48px 28px;
    position: relative;
    overflow: hidden;
  }}
  .header::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: rgba(0,180,216,.12);
  }}
  .header::after {{
    content: '';
    position: absolute;
    bottom: -40px; right: 120px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(244,162,97,.08);
  }}
  .header-badge {{
    display: inline-block;
    background: rgba(0,180,216,.2);
    color: var(--teal);
    border: 1px solid rgba(0,180,216,.4);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: 12px;
  }}
  .header h1 {{
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 6px;
    letter-spacing: -.02em;
  }}
  .header h1 span {{ color: var(--teal); }}
  .header .meta {{
    color: rgba(255,255,255,.55);
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 8px;
  }}

  /* ── NAV ── */
  .nav {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0 48px;
    display: flex;
    gap: 0;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 1px 8px rgba(0,0,0,.06);
    overflow-x: auto;
  }}
  .nav a {{
    padding: 14px 20px;
    text-decoration: none;
    color: var(--muted);
    font-size: 13px;
    font-weight: 500;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
    transition: all .15s;
  }}
  .nav a:hover {{
    color: var(--teal);
    border-bottom-color: var(--teal);
  }}

  /* ── LAYOUT ── */
  .container {{ max-width: 1280px; margin: 0 auto; padding: 32px 48px 64px; }}
  .section {{
    background: var(--surface);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,.04);
  }}
  .section-title {{
    font-size: 17px;
    font-weight: 700;
    color: var(--navy);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .section-title .icon {{
    width: 32px; height: 32px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
  }}

  /* ── STAT CARDS ── */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px;
    margin-bottom: 4px;
  }}
  .kpi {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
    transition: box-shadow .15s;
  }}
  .kpi:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,.08); }}
  .kpi-value {{
    font-size: 26px;
    font-weight: 700;
    color: var(--navy);
    line-height: 1.1;
  }}
  .kpi-label {{
    font-size: 11px;
    color: var(--muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-top: 4px;
  }}
  .kpi.warn .kpi-value {{ color: var(--red); }}
  .kpi.info .kpi-value {{ color: var(--teal); }}
  .kpi.ok   .kpi-value {{ color: var(--green); }}
  .kpi.gold .kpi-value {{ color: var(--gold); }}

  /* ── TABLES ── */
  .tbl-wrap {{ overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  thead tr {{
    background: var(--navy);
    color: white;
  }}
  thead th {{
    padding: 11px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: .04em;
    white-space: nowrap;
  }}
  tbody tr:nth-child(even) {{ background: #F8FAFC; }}
  tbody tr:hover {{ background: #EFF6FF; }}
  tbody td {{
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text);
  }}
  tbody td.label {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: var(--navy);
    font-size: 13px;
  }}

  /* ── BADGES ── */
  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: .04em;
  }}
  .badge-red    {{ background:#FEF2F2; color:#B91C1C; border:1px solid #FECACA; }}
  .badge-green  {{ background:#F0FDF4; color:#15803D; border:1px solid #BBF7D0; }}
  .badge-orange {{ background:#FFF7ED; color:#C2410C; border:1px solid #FED7AA; }}
  .badge-blue   {{ background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE; }}
  .badge-gray   {{ background:#F8FAFC; color:#475569; border:1px solid #CBD5E1; }}
  .badge-purple {{ background:#F5F3FF; color:#6D28D9; border:1px solid #DDD6FE; }}

  /* ── CHART GRIDS ── */
  .chart-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 20px;
  }}
  .chart-card {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }}
  .chart-card h4 {{
    font-size: 13px;
    font-weight: 600;
    color: var(--navy);
    margin-bottom: 12px;
  }}
  .chart-card img {{ width: 100%; border-radius: 6px; }}
  .chart-full {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    margin-top: 16px;
  }}
  .chart-full img {{ width: 100%; border-radius: 6px; }}

  /* ── MISSING BAR ── */
  .miss-bar-bg {{
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
    width: 120px;
    display: inline-block;
    vertical-align: middle;
  }}
  .miss-bar-fill {{
    height: 100%;
    border-radius: 4px;
    background: var(--teal);
  }}
  .miss-bar-fill.high {{ background: var(--red); }}

  /* ── SKEW CHIP ── */
  .skew-chip {{
    display: inline-block;
    padding: 1px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
  }}
  .skew-right {{ background:#FFF7ED; color:#C2410C; }}
  .skew-left  {{ background:#EFF6FF; color:#1D4ED8; }}
  .skew-sym   {{ background:#F0FDF4; color:#15803D; }}

  /* ── CORR PAIRS ── */
  .corr-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 12px;
  }}
  .corr-card {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
  }}
  .corr-val {{
    font-size: 22px;
    font-weight: 700;
    color: var(--navy);
  }}
  .corr-val.high {{ color: var(--red); }}
  .corr-names {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
  }}

  /* ── PREVIEW TABLE ── */
  .preview-wrap {{ overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); }}
  .preview-wrap table thead tr {{ background: #1B4F72; }}

  /* ── FOOTER ── */
  .footer {{
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    padding: 32px 0 16px;
    border-top: 1px solid var(--border);
    margin-top: 32px;
  }}
  .footer strong {{ color: var(--teal); }}

  /* ── INSIGHTS BOX ── */
  .insight-box {{
    background: linear-gradient(135deg,#EFF6FF,#F0FDF4);
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 20px;
  }}
  .insight-box h4 {{
    font-size: 13px;
    font-weight: 700;
    color: #1D4ED8;
    margin-bottom: 8px;
  }}
  .insight-box ul {{
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }}
  .insight-box li::before {{ content: "›  "; color: var(--teal); font-weight: 700; }}
  .insight-box li {{ font-size: 13px; color: var(--text); }}

  /* ── ACCORDION ── */
  details {{ margin-bottom: 10px; }}
  summary {{
    cursor: pointer;
    padding: 10px 14px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    color: var(--navy);
    list-style: none;
    display: flex;
    justify-content: space-between;
    align-items: center;
    user-select: none;
  }}
  summary::-webkit-details-marker {{ display: none; }}
  summary::after {{ content: "▾"; color: var(--muted); font-size: 14px; }}
  details[open] summary::after {{ content: "▴"; }}
  details[open] summary {{ border-radius: 8px 8px 0 0; }}
  .detail-body {{
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 16px;
    background: var(--surface);
  }}

  /* ── PROGRESS RINGS ── */
  .ring-wrap {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-top: 8px;
  }}
  .ring-item {{
    text-align: center;
    min-width: 80px;
  }}
  .ring-label {{
    font-size: 10px;
    color: var(--muted);
    font-weight: 500;
    margin-top: 4px;
    max-width: 90px;
    word-break: break-word;
  }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-badge">Automated EDA Report</div>
  <h1>Dataset Analysis <span>Report</span></h1>
  <div class="meta">
    File: {filename} &nbsp;|&nbsp;
    Generated: {generated_at} &nbsp;|&nbsp;
    {rows:,} rows &times; {columns} columns &nbsp;|&nbsp;
    {memory_mb} MB in memory
  </div>
</div>

<!-- NAV -->
<nav class="nav">
  <a href="#overview">📊 Overview</a>
  <a href="#data-types">🔢 Data Types</a>
  <a href="#missing">❓ Missing Data</a>
  <a href="#numeric">📈 Numeric</a>
  <a href="#categorical">🏷 Categorical</a>
  <a href="#correlation">🔗 Correlation</a>
  <a href="#distributions">📉 Distributions</a>
  <a href="#preview">👁 Data Preview</a>
  <a href="#insights">💡 Insights</a>
</nav>

<!-- BODY -->
<div class="container">

<!-- ①  OVERVIEW -->
<div class="section" id="overview">
  <div class="section-title">
    <span class="icon" style="background:#EFF6FF">📊</span>
    Dataset Overview
  </div>
  <div class="kpi-grid">
    <div class="kpi info">
      <div class="kpi-value">{rows:,}</div>
      <div class="kpi-label">Total Rows</div>
    </div>
    <div class="kpi info">
      <div class="kpi-value">{columns}</div>
      <div class="kpi-label">Total Columns</div>
    </div>
    <div class="kpi ok">
      <div class="kpi-value">{numeric_cols}</div>
      <div class="kpi-label">Numeric</div>
    </div>
    <div class="kpi gold">
      <div class="kpi-value">{categorical_cols}</div>
      <div class="kpi-label">Categorical</div>
    </div>
    <div class="kpi {missing_cls}">
      <div class="kpi-value">{missing_pct}%</div>
      <div class="kpi-label">Missing Data</div>
    </div>
    <div class="kpi {dup_cls}">
      <div class="kpi-value">{duplicate_rows:,}</div>
      <div class="kpi-label">Duplicate Rows</div>
    </div>
    <div class="kpi">
      <div class="kpi-value">{total_missing:,}</div>
      <div class="kpi-label">Missing Cells</div>
    </div>
    <div class="kpi">
      <div class="kpi-value">{memory_mb} MB</div>
      <div class="kpi-label">Memory Usage</div>
    </div>
    <div class="kpi">
      <div class="kpi-value">{file_size}</div>
      <div class="kpi-label">File Size</div>
    </div>
  </div>
</div>

<!-- ② DATA TYPES -->
<div class="section" id="data-types">
  <div class="section-title">
    <span class="icon" style="background:#F5F3FF">🔢</span>
    Column Data Types
  </div>
  <div class="chart-grid">
    {dtype_pie_html}
  </div>
  <div class="tbl-wrap" style="margin-top:20px">
    <table>
      <thead><tr><th>#</th><th>Column Name</th><th>Data Type</th><th>Non-Null</th><th>Unique Values</th><th>Missing</th><th>Missing %</th></tr></thead>
      <tbody>{col_rows}</tbody>
    </table>
  </div>
</div>

<!-- ③ MISSING DATA -->
<div class="section" id="missing">
  <div class="section-title">
    <span class="icon" style="background:#FEF2F2">❓</span>
    Missing Data Analysis
  </div>
  {missing_chart_html}
  <div class="tbl-wrap" style="margin-top:20px">
    <table>
      <thead><tr><th>Column</th><th>Missing Count</th><th>Missing %</th><th>Visual</th><th>Severity</th></tr></thead>
      <tbody>{missing_rows}</tbody>
    </table>
  </div>
</div>

<!-- ④ NUMERIC STATS -->
<div class="section" id="numeric">
  <div class="section-title">
    <span class="icon" style="background:#F0FDF4">📈</span>
    Numeric Column Statistics
  </div>
  {numeric_html}
</div>

<!-- ⑤ CATEGORICAL -->
<div class="section" id="categorical">
  <div class="section-title">
    <span class="icon" style="background:#FFFBEB">🏷</span>
    Categorical Column Analysis
  </div>
  {cat_html}
</div>

<!-- ⑥ CORRELATION -->
<div class="section" id="correlation">
  <div class="section-title">
    <span class="icon" style="background:#FFF7ED">🔗</span>
    Correlation Analysis
  </div>
  {corr_html}
</div>

<!-- ⑦ DISTRIBUTIONS -->
<div class="section" id="distributions">
  <div class="section-title">
    <span class="icon" style="background:#EFF6FF">📉</span>
    Distributions &amp; Outliers
  </div>
  {dist_html}
</div>

<!-- ⑧ DATA PREVIEW -->
<div class="section" id="preview">
  <div class="section-title">
    <span class="icon" style="background:#F8FAFC">👁</span>
    Data Preview (first {sample_rows} rows)
  </div>
  <div class="preview-wrap">
    {preview_table}
  </div>
</div>

<!-- ⑨ INSIGHTS -->
<div class="section" id="insights">
  <div class="section-title">
    <span class="icon" style="background:#F0FDF4">💡</span>
    Automated Insights &amp; Recommendations
  </div>
  {insights_html}
</div>

</div><!-- /container -->

<div class="footer container">
  Generated by <strong>EDA Report Generator v2.0</strong> &nbsp;·&nbsp;
  {generated_at} &nbsp;·&nbsp;
  Built for Databricks &amp; Python environments
</div>

<script>
// Smooth scroll
document.querySelectorAll('.nav a').forEach(a => {{
  a.addEventListener('click', e => {{
    e.preventDefault();
    const target = document.querySelector(a.getAttribute('href'));
    if (target) target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
  }});
}});
// Active nav highlight
const sections = document.querySelectorAll('.section');
const navLinks = document.querySelectorAll('.nav a');
window.addEventListener('scroll', () => {{
  let current = '';
  sections.forEach(s => {{
    if (window.scrollY >= s.offsetTop - 80) current = s.id;
  }});
  navLinks.forEach(a => {{
    a.style.color = a.getAttribute('href') === '#' + current ? 'var(--teal)' : '';
    a.style.borderBottomColor = a.getAttribute('href') === '#' + current ? 'var(--teal)' : 'transparent';
  }});
}});
</script>
</body>
</html>
"""


def skew_chip(skew_val):
    if skew_val is None:
        return '<span class="badge badge-gray">N/A</span>'
    if skew_val > 1:
        return f'<span class="skew-chip skew-right">Right-skewed ({skew_val:+.2f})</span>'
    elif skew_val < -1:
        return f'<span class="skew-chip skew-left">Left-skewed ({skew_val:+.2f})</span>'
    else:
        return f'<span class="skew-chip skew-sym">Symmetric ({skew_val:+.2f})</span>'


def miss_bar_html(pct):
    cls = "high" if pct > 20 else ""
    return (f'<span class="miss-bar-bg"><span class="miss-bar-fill {cls}" '
            f'style="width:{min(pct,100):.1f}%"></span></span>')


def severity_badge(pct):
    if pct == 0:
        return '<span class="badge badge-green">None</span>'
    elif pct < 5:
        return '<span class="badge badge-blue">Low</span>'
    elif pct < 20:
        return '<span class="badge badge-orange">Moderate</span>'
    else:
        return '<span class="badge badge-red">High</span>'


def norm_badge(label):
    if "Normal" in label:
        return f'<span class="badge badge-green">{label}</span>'
    elif label == "N/A":
        return f'<span class="badge badge-gray">{label}</span>'
    else:
        return f'<span class="badge badge-orange">{label}</span>'


def build_html_report(df, overview, missing_info, num_stats, cat_stats,
                      charts, path):

    filename = os.path.basename(path)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Column rows ──
    col_rows = ""
    for i, mi in enumerate(missing_info):
        dtype_badge_map = {
            "int64": "badge-blue", "int32": "badge-blue", "float64": "badge-blue",
            "float32": "badge-blue", "object": "badge-purple", "bool": "badge-orange",
            "datetime64[ns]": "badge-green", "category": "badge-gray",
        }
        bc = dtype_badge_map.get(mi["dtype"], "badge-gray")
        non_null = overview["rows"] - mi["missing"]
        col_rows += (
            f"<tr><td>{i+1}</td><td class='label'>{mi['column']}</td>"
            f"<td><span class='badge {bc}'>{mi['dtype']}</span></td>"
            f"<td>{non_null:,}</td><td>{mi['unique']:,}</td>"
            f"<td>{mi['missing']:,}</td>"
            f"<td>{mi['missing_pct']:.2f}%</td></tr>"
        )

    # ── Missing chart ──
    missing_chart_html = ""
    if charts.get("missing"):
        missing_chart_html = (
            f'<div class="chart-full"><img src="{charts["missing"]}" '
            f'alt="Missing values chart"/></div>'
        )
    else:
        missing_chart_html = '<p style="color:var(--green);font-weight:600">✅ No missing values detected in this dataset.</p>'

    # ── Missing rows ──
    missing_rows = ""
    for mi in missing_info:
        missing_rows += (
            f"<tr><td class='label'>{mi['column']}</td>"
            f"<td>{mi['missing']:,}</td>"
            f"<td>{mi['missing_pct']:.2f}%</td>"
            f"<td>{miss_bar_html(mi['missing_pct'])}</td>"
            f"<td>{severity_badge(mi['missing_pct'])}</td></tr>"
        )
    if not missing_rows:
        missing_rows = "<tr><td colspan='5' style='text-align:center;color:var(--green)'>✅ No missing values</td></tr>"

    # ── Numeric stats ──
    if num_stats:
        num_rows = ""
        for ns in num_stats:
            num_rows += (
                f"<tr>"
                f"<td class='label'>{ns['column']}</td>"
                f"<td>{ns['count']:,}</td>"
                f"<td>{ns['mean']:,.4f}</td>"
                f"<td>{ns['median']:,.4f}</td>"
                f"<td>{ns['std']:,.4f}</td>"
                f"<td>{ns['min']:,.4f}</td>"
                f"<td>{ns['max']:,.4f}</td>"
                f"<td>{ns['q25']:,.4f}</td>"
                f"<td>{ns['q75']:,.4f}</td>"
                f"<td>{skew_chip(ns['skewness'])}</td>"
                f"<td>{'<span class=\"badge badge-red\">' + str(ns['outliers']) + '</span>' if ns['outliers'] > 0 else '<span class=\"badge badge-green\">0</span>'}</td>"
                f"<td>{norm_badge(ns['normality'])}</td>"
                f"</tr>"
            )
        numeric_html = (
            '<div class="tbl-wrap">'
            '<table><thead><tr>'
            '<th>Column</th><th>Count</th><th>Mean</th><th>Median</th>'
            '<th>Std Dev</th><th>Min</th><th>Max</th>'
            '<th>Q25</th><th>Q75</th><th>Skewness</th>'
            '<th>Outliers (IQR)</th><th>Normality</th>'
            '</tr></thead><tbody>'
            + num_rows + '</tbody></table></div>'
        )
    else:
        numeric_html = '<p style="color:var(--muted)">No numeric columns found.</p>'

    # ── Categorical ──
    if cat_stats:
        cat_parts = []
        for cs in cat_stats:
            bar_img = charts.get("cat_bars", {}).get(cs["column"], "")
            img_html = f'<div class="chart-full"><img src="{bar_img}" alt="{cs["column"]} distribution"/></div>' if bar_img else ""
            top_vc_html = ""
            if cs["value_counts"]:
                top_vc_html = "<table style='width:100%;font-size:12px'><thead><tr><th>Value</th><th>Count</th><th>%</th></tr></thead><tbody>"
                total = sum(cs["value_counts"].values())
                for k, v in list(cs["value_counts"].items())[:10]:
                    pct = 100 * v / total if total else 0
                    top_vc_html += f"<tr><td class='label'>{k}</td><td>{v:,}</td><td>{pct:.1f}%</td></tr>"
                top_vc_html += "</tbody></table>"

            unique_badge = (
                '<span class="badge badge-green">Low cardinality</span>' if cs["unique"] <= 10
                else '<span class="badge badge-orange">Medium cardinality</span>' if cs["unique"] <= 50
                else '<span class="badge badge-red">High cardinality</span>'
            )
            cat_parts.append(
                f'<details><summary>{cs["column"]} &nbsp; '
                f'<span class="badge badge-gray">{cs["unique"]} unique</span> &nbsp;'
                f'{unique_badge}</summary>'
                f'<div class="detail-body">'
                f'<div class="kpi-grid" style="margin-bottom:14px">'
                f'<div class="kpi"><div class="kpi-value">{cs["unique"]}</div><div class="kpi-label">Unique Values</div></div>'
                f'<div class="kpi"><div class="kpi-value">{cs["count"]:,}</div><div class="kpi-label">Non-null Count</div></div>'
                f'<div class="kpi warn"><div class="kpi-value">{cs["missing"]}</div><div class="kpi-label">Missing</div></div>'
                f'<div class="kpi info"><div class="kpi-value">{cs["top_pct"]}%</div><div class="kpi-label">Top Value Freq</div></div>'
                f'</div>'
                f'<p style="margin-bottom:12px;font-size:13px">Top value: <strong>{cs["top_value"]}</strong> ({cs["top_freq"]:,} occurrences)</p>'
                f'<div class="chart-grid">'
                f'<div>{top_vc_html}</div>'
                f'<div>{img_html}</div>'
                f'</div>'
                f'</div></details>'
            )
        cat_html = "\n".join(cat_parts)
    else:
        cat_html = '<p style="color:var(--muted)">No categorical columns found.</p>'

    # ── Correlation ──
    corr_pairs = charts.get("high_corr", [])
    corr_img   = charts.get("correlation", "")
    corr_chart_html = f'<div class="chart-full"><img src="{corr_img}" alt="Correlation matrix"/></div>' if corr_img else ""

    high_corr_html = ""
    if corr_pairs:
        cards = ""
        for p in corr_pairs:
            cls = "high" if p["corr"] >= 0.9 else ""
            sign_badge = '<span class="badge badge-red">Positive</span>' if p["sign"] == "positive" else '<span class="badge badge-blue">Negative</span>'
            cards += (
                f'<div class="corr-card">'
                f'<div class="corr-val {cls}">{p["corr"]:.4f}</div>'
                f'<div class="corr-names">{p["col1"]} ↔ {p["col2"]}</div>'
                f'<div style="margin-top:6px">{sign_badge} '
                f'{"<span class=\\'badge badge-red\\'>Very High ≥0.9</span>" if p["corr"] >= 0.9 else "<span class=\\'badge badge-orange\\'>High</span>"}'
                f'</div></div>'
            )
        high_corr_html = (
            f'<h4 style="margin:20px 0 12px;font-size:14px;color:var(--navy)">⚠ High Correlation Pairs (≥{CORRELATION_THRESHOLD})</h4>'
            f'<div class="corr-grid">{cards}</div>'
        )

    corr_html = corr_chart_html + high_corr_html if (corr_img or high_corr_html) else '<p style="color:var(--muted)">Fewer than 2 numeric columns — correlation not available.</p>'

    # ── Distributions ──
    hist_img  = charts.get("histograms", "")
    box_img   = charts.get("boxplots", "")
    scat_img  = charts.get("scatter", "")
    dist_html = ""
    if hist_img:
        dist_html += f'<div class="chart-full"><img src="{hist_img}" alt="Histograms"/></div>'
    if box_img:
        dist_html += f'<div class="chart-full" style="margin-top:16px"><img src="{box_img}" alt="Box plots"/></div>'
    if scat_img:
        dist_html += f'<div class="chart-full" style="margin-top:16px"><img src="{scat_img}" alt="Scatter matrix"/></div>'
    if not dist_html:
        dist_html = '<p style="color:var(--muted)">No numeric columns available for distribution plots.</p>'

    # ── Dtype pie ──
    dtype_pie_img = charts.get("dtype_pie", "")
    dtype_pie_html = (
        f'<div class="chart-card"><img src="{dtype_pie_img}" alt="Data types pie"/></div>'
        if dtype_pie_img else ""
    )

    # ── Data preview ──
    preview_table = df.head(SAMPLE_ROWS).to_html(
        classes="", border=0, index=True, na_rep="<em style='color:#94A3B8'>NaN</em>"
    )

    # ── Automated insights ──
    insights = []
    if overview["duplicate_rows"] > 0:
        insights.append(f"🔴 <strong>{overview['duplicate_rows']:,} duplicate rows</strong> detected — consider deduplication before modelling.")
    if overview["missing_pct"] > 20:
        insights.append(f"🔴 Overall missing data is <strong>{overview['missing_pct']}%</strong> — significant imputation or column removal strategy required.")
    elif overview["missing_pct"] > 5:
        insights.append(f"🟡 <strong>{overview['missing_pct']}% missing data</strong> detected — review imputation strategy per column.")
    else:
        insights.append(f"🟢 Missing data is low at <strong>{overview['missing_pct']}%</strong> — dataset is largely complete.")

    high_miss_cols = [m["column"] for m in missing_info if m["missing_pct"] > 50]
    if high_miss_cols:
        insights.append(f"🔴 Columns with >50% missing: <strong>{', '.join(high_miss_cols)}</strong> — consider dropping or advanced imputation.")

    skewed_cols = [n["column"] for n in num_stats if n["skewness"] and abs(n["skewness"]) > 1]
    if skewed_cols:
        insights.append(f"📊 Highly skewed numeric columns: <strong>{', '.join(skewed_cols)}</strong> — consider log/sqrt transformation.")

    outlier_cols = [(n["column"], n["outliers"]) for n in num_stats if n["outliers"] > 0]
    if outlier_cols:
        col_str = ", ".join(f"{c} ({o:,})" for c, o in outlier_cols[:5])
        insights.append(f"⚠ Outliers detected (IQR method): <strong>{col_str}</strong> — investigate before modelling.")

    if corr_pairs:
        top_pair = corr_pairs[0]
        insights.append(f"🔗 Highest correlation: <strong>{top_pair['col1']} ↔ {top_pair['col2']}</strong> ({top_pair['corr']:.4f}) — potential multicollinearity risk.")

    high_card = [c for c in cat_stats if c["unique"] > 50]
    if high_card:
        insights.append(f"🏷 High cardinality categoricals: <strong>{', '.join(c['column'] for c in high_card)}</strong> — consider target encoding or embeddings.")

    non_normal = [n["column"] for n in num_stats if "Non-normal" in n["normality"]]
    if non_normal:
        insights.append(f"📉 Non-normal distributions: <strong>{', '.join(non_normal[:5])}</strong> — use non-parametric tests or transform features.")

    if overview["categorical_cols"] > 0:
        insights.append(f"🏷 {overview['categorical_cols']} categorical column(s) present — one-hot or ordinal encoding needed for ML models.")

    insights.append(f"✅ Dataset shape: <strong>{overview['rows']:,} rows × {overview['columns']} columns</strong> ({overview['memory_mb']} MB) — ready for next EDA steps.")

    insights_html = (
        '<div class="insight-box"><h4>🤖 Automated Observations</h4><ul>'
        + "".join(f"<li>{i}</li>" for i in insights)
        + "</ul></div>"
    )

    # ── Assemble ──
    html = HTML_TEMPLATE.format(
        filename        = filename,
        generated_at    = generated_at,
        rows            = overview["rows"],
        columns         = overview["columns"],
        memory_mb       = overview["memory_mb"],
        numeric_cols    = overview["numeric_cols"],
        categorical_cols= overview["categorical_cols"],
        missing_pct     = overview["missing_pct"],
        missing_cls     = "warn" if overview["missing_pct"] > 10 else ("ok" if overview["missing_pct"] == 0 else ""),
        duplicate_rows  = overview["duplicate_rows"],
        dup_cls         = "warn" if overview["duplicate_rows"] > 0 else "ok",
        total_missing   = overview["total_missing"],
        file_size       = overview["file_size"],
        col_rows        = col_rows,
        dtype_pie_html  = dtype_pie_html,
        missing_chart_html = missing_chart_html,
        missing_rows    = missing_rows,
        numeric_html    = numeric_html,
        cat_html        = cat_html,
        corr_html       = corr_html,
        dist_html       = dist_html,
        sample_rows     = SAMPLE_ROWS,
        preview_table   = preview_table,
        insights_html   = insights_html,
    )
    return html


# ════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ════════════════════════════════════════════════════════════

def run_eda(file_path=FILE_PATH, output_path=OUTPUT_PATH):
    print("=" * 60)
    print("  EDA Report Generator — Starting")
    print("=" * 60)

    # Load
    df = load_data(file_path)

    # Compute
    print("[→] Computing overview...")
    overview = compute_overview(df, file_path)

    print("[→] Analysing missing values...")
    missing_info = compute_missing(df)

    print("[→] Computing numeric statistics...")
    num_stats = compute_numeric_stats(df)

    print("[→] Computing categorical statistics...")
    cat_stats = compute_cat_stats(df)

    # Charts
    print("[→] Generating charts (this may take a moment)...")
    charts = {}

    charts["dtype_pie"]   = chart_dtype_pie(overview)
    charts["missing"]     = chart_missing_heatmap(df)
    charts["histograms"]  = chart_numeric_histograms(df)
    charts["boxplots"]    = chart_boxplots(df)
    charts["correlation"] = chart_correlation(df)
    charts["scatter"]     = chart_scatter_matrix(df)
    charts["cat_bars"]    = chart_categorical_bars(cat_stats)
    charts["high_corr"]   = chart_high_corr(df)

    # Build HTML
    print("[→] Building HTML report...")
    html = build_html_report(df, overview, missing_info, num_stats,
                             cat_stats, charts, file_path)

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n{'='*60}")
    print(f"  ✅ Report saved → {output_path}")
    print(f"  Rows:    {overview['rows']:,}")
    print(f"  Columns: {overview['columns']}")
    print(f"  Missing: {overview['missing_pct']}%")
    print(f"  Dupes:   {overview['duplicate_rows']:,}")
    print(f"{'='*60}\n")

    # Databricks display support
    try:
        from IPython.display import display, HTML as iHTML
        display(iHTML(f'<a href="{output_path}" target="_blank">📄 Open EDA Report</a>'))
    except Exception:
        pass

    return output_path


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Allow CLI override: python eda_report.py mydata.csv /tmp/out.html
    if len(sys.argv) >= 2:
        FILE_PATH = sys.argv[1]
    if len(sys.argv) >= 3:
        OUTPUT_PATH = sys.argv[2]
    run_eda(FILE_PATH, OUTPUT_PATH)
