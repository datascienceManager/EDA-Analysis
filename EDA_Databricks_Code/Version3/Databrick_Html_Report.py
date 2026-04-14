"""
════════════════════════════════════════════════════════════════
  Enhanced Automated EDA Report Generator  v3.0
  ─────────────────────────────────────────────
  Compatible with : Databricks (DBR 10+), Jupyter, local Python
  Input formats   : CSV, XLSX
  Output          : Single self-contained HTML file
  ────────────────────────────────────────────────────────────
  NEW IN v3.0
  ───────────
  ✦ Sankey chart          – categorical flow / co-occurrence
  ✦ Bubble chart          – 3-variable numeric relationship
  ✦ geom_line             – time-series & ordered trend lines
  ✦ geom_point            – scatter with optional color group
  ✦ geom_bar              – grouped & stacked bar variants
  ✦ facet_grid            – grid of mini-plots (row × col split)
  ✦ facet_wrap            – wrapped panels by category
  ✦ Violin plots          – distribution shape + box overlay
  ✦ QQ plots              – normality visual check
  ✦ Pair-correlation dots – correlation dot-matrix
  ✦ Missing heatmap       – row-level missingness pattern
  ────────────────────────────────────────────────────────────
  DATABRICKS USAGE
  ────────────────
  Option A – paste into a notebook cell, then call:
      run_eda("/dbfs/FileStore/your_data.csv",
              "/dbfs/tmp/eda_v3_report.html")

  Option B – upload file and %run it:
      %run /Workspace/Shared/eda_report_v3
      run_eda("/dbfs/FileStore/your_data.csv")

  Option C – CLI:
      python eda_report_v3.py mydata.csv /tmp/report.html
  ────────────────────────────────────────────────────────────
  DEPENDENCIES (pre-installed on DBR 10+)
      pip install pandas numpy scipy matplotlib seaborn openpyxl
════════════════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
FILE_PATH            = "your_dataset.csv"
OUTPUT_PATH          = "/tmp/eda_report_v3.html"
SAMPLE_ROWS          = 8
MAX_UNIQUE_CAT       = 25
CORRELATION_THRESHOLD = 0.65
FACET_MAX_PANELS     = 12        # max panels in facet plots
SANKEY_TOP_N         = 8        # top N values per node
BUBBLE_SAMPLE        = 500      # rows sampled for bubble chart
# ─────────────────────────────────────────────────────────────

import os, sys, io, base64, warnings, json, math
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────
PAL = ["#378ADD","#10B981","#F4A261","#EF4444","#7C3AED",
       "#F97316","#14B8A6","#EC4899","#84CC16","#06B6D4",
       "#8B5CF6","#F59E0B","#3B82F6","#22D3EE","#A3E635"]

sns.set_theme(style="whitegrid", palette=PAL)
plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"#FAFAFA",
    "axes.edgecolor":"#E2E8F0","axes.labelcolor":"#1E293B",
    "xtick.color":"#64748B","ytick.color":"#64748B",
    "grid.color":"#E2E8F0","grid.linewidth":0.5,
    "font.family":"DejaVu Sans","axes.titlesize":11,
    "axes.titleweight":"bold","axes.titlecolor":"#0D1B2A",
    "axes.spines.top":False,"axes.spines.right":False,
})


# ════════════════════════════════════════════════════════════
#  UTILITY
# ════════════════════════════════════════════════════════════

def fig_b64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"

def img_tag(b64, alt="chart", style="width:100%"):
    return f'<img src="{b64}" alt="{alt}" style="{style};border-radius:6px"/>'

def human(n):
    for u in ["B","KB","MB","GB"]:
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"

def num_vals(df, col):
    return pd.to_numeric(df[col], errors="coerce").dropna().values

def safe_skew(s):
    try: return round(float(s.skew()), 4)
    except: return None

def safe_kurt(s):
    try: return round(float(s.kurt()), 4)
    except: return None

def shapiro(series):
    try:
        clean = series.dropna()
        n = min(len(clean), 5000)
        samp = clean.sample(n, random_state=42) if len(clean) > n else clean
        _, p = stats.shapiro(samp)
        return ("✅ Normal" if p > 0.05 else "⚠ Non-normal"), round(p, 5)
    except: return "N/A", None

def color_cycle(n):
    return [PAL[i % len(PAL)] for i in range(n)]


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
        raise ValueError(f"Unsupported format: {ext}")
    print(f"[✓] Loaded {len(df):,} rows × {len(df.columns)} cols from '{path}'")
    return df


# ════════════════════════════════════════════════════════════
#  STATISTICS
# ════════════════════════════════════════════════════════════

def overview(df, path):
    try: fsize = human(os.path.getsize(path))
    except: fsize = "N/A"
    num = df.select_dtypes(include=np.number).columns.tolist()
    cat = df.select_dtypes(include=["object","category"]).columns.tolist()
    dt  = df.select_dtypes(include=["datetime64"]).columns.tolist()
    miss = int(df.isnull().sum().sum())
    cells = df.shape[0] * df.shape[1]
    return dict(
        file_path=path, file_size=fsize, rows=len(df), columns=len(df.columns),
        num_cols=num, cat_cols=cat, dt_cols=dt,
        n_num=len(num), n_cat=len(cat), n_dt=len(dt),
        dups=int(df.duplicated().sum()),
        total_miss=miss, miss_pct=round(100*miss/cells,2) if cells else 0,
        mem_mb=round(df.memory_usage(deep=True).sum()/1e6, 2),
    )

def missing_info(df):
    miss = df.isnull().sum()
    pct  = (miss / len(df) * 100).round(2)
    uniq = df.nunique()
    dtypes = df.dtypes.astype(str)
    return sorted([
        dict(col=c, dtype=dtypes[c], missing=int(miss[c]),
             miss_pct=float(pct[c]), unique=int(uniq[c]))
        for c in df.columns
    ], key=lambda x: x["miss_pct"], reverse=True)

def num_stats(df):
    rows = []
    for col in df.select_dtypes(include=np.number).columns:
        s = df[col].dropna()
        if not len(s): continue
        q1,q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3-q1
        outliers = int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum())
        nl, np_ = shapiro(df[col])
        rows.append(dict(
            col=col, count=int(s.count()), mean=round(float(s.mean()),4),
            median=round(float(s.median()),4), std=round(float(s.std()),4),
            min=round(float(s.min()),4), max=round(float(s.max()),4),
            q1=round(float(q1),4), q3=round(float(q3),4),
            skew=safe_skew(s), kurt=safe_kurt(s),
            outliers=outliers, norm_label=nl, norm_p=np_,
        ))
    return rows

def cat_stats(df, max_unique=MAX_UNIQUE_CAT):
    rows = []
    for col in df.select_dtypes(include=["object","category"]).columns:
        s = df[col].dropna()
        vc = s.value_counts()
        rows.append(dict(
            col=col, count=int(s.count()),
            missing=int(df[col].isnull().sum()),
            unique=int(s.nunique()),
            top=str(vc.index[0]) if len(vc) else "N/A",
            top_freq=int(vc.iloc[0]) if len(vc) else 0,
            top_pct=round(100*vc.iloc[0]/len(s),2) if len(vc) else 0,
            vc=vc.head(max_unique).to_dict(),
        ))
    return rows


# ════════════════════════════════════════════════════════════
#  ──────────────── CHART GENERATORS ────────────────────────
# ════════════════════════════════════════════════════════════


# ── 1. Data-type pie ───────────────────────────────────────
def chart_dtype_pie(ov):
    data = [(l,v) for l,v in [("Numeric",ov["n_num"]),("Categorical",ov["n_cat"]),("Datetime",ov["n_dt"])] if v>0]
    if not data: return None
    labels, sizes = zip(*data)
    fig, ax = plt.subplots(figsize=(5,4))
    wedges,_,at = ax.pie(sizes, labels=labels, autopct="%1.1f%%",
        colors=PAL[:len(data)], startangle=140,
        wedgeprops={"edgecolor":"white","linewidth":2},
        pctdistance=0.78, labeldistance=1.12)
    for t in at: t.set_fontsize(9); t.set_color("white"); t.set_fontweight("bold")
    ax.set_title("Column Type Distribution", pad=10)
    plt.tight_layout()
    return fig_b64(fig)


# ── 2. Missing heatmap bar ─────────────────────────────────
def chart_missing_bar(df):
    mp = (df.isnull().mean()*100).sort_values(ascending=False)
    mp = mp[mp>0]
    if mp.empty: return None
    fig, ax = plt.subplots(figsize=(max(8,len(mp)*.55), 4))
    colors = [PAL[3] if v>20 else PAL[0] for v in mp.values]
    ax.bar(mp.index, mp.values, color=colors, edgecolor="white", linewidth=.5, width=.65)
    ax.axhline(20, color=PAL[3], linestyle="--", linewidth=1, alpha=.7, label="20% threshold")
    for i,(x,v) in enumerate(zip(mp.index, mp.values)):
        ax.text(i, v+.4, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color="#1E293B")
    ax.set_title("Missing Values by Column (%)", pad=10)
    ax.set_ylabel("Missing %")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, min(100, mp.max()*1.25))
    plt.tight_layout()
    return fig_b64(fig)


# ── 3. Missing row-level heatmap ───────────────────────────
def chart_missing_heatmap(df, max_rows=200, max_cols=30):
    miss = df.isnull()
    cols_with_miss = miss.columns[miss.any()].tolist()[:max_cols]
    if not cols_with_miss: return None
    sample = miss[cols_with_miss].head(max_rows).astype(int)
    fig, ax = plt.subplots(figsize=(min(14, len(cols_with_miss)*.55+2), min(6, len(sample)*.04+2)))
    cmap = mcolors.ListedColormap(["#E2E8F0","#EF4444"])
    ax.imshow(sample.values, cmap=cmap, aspect="auto", interpolation="none")
    ax.set_xticks(range(len(cols_with_miss)))
    ax.set_xticklabels(cols_with_miss, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_title(f"Row-level Missing Pattern (first {len(sample)} rows · red=missing)", pad=10)
    red_patch = mpatches.Patch(color="#EF4444", label="Missing")
    gray_patch = mpatches.Patch(color="#E2E8F0", label="Present")
    ax.legend(handles=[gray_patch, red_patch], loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig_b64(fig)


# ── 4. Histograms ──────────────────────────────────────────
def chart_histograms(df):
    num = df.select_dtypes(include=np.number).columns.tolist()
    if not num: return None
    nc = min(3, len(num)); nr = math.ceil(len(num)/nc)
    fig, axes = plt.subplots(nr, nc, figsize=(nc*4.5, nr*3.2))
    axes = np.array(axes).flatten() if len(num)>1 else [axes]
    for i, col in enumerate(num):
        ax = axes[i]; d = df[col].dropna()
        if not len(d): ax.set_visible(False); continue
        ax.hist(d, bins=30, color=PAL[i%len(PAL)], edgecolor="white", linewidth=.4, alpha=.85)
        ax.axvline(d.mean(), color="#EF4444", lw=1.5, ls="--", label=f"μ={d.mean():.2f}")
        ax.axvline(d.median(), color="#F4A261", lw=1.5, ls=":", label=f"med={d.median():.2f}")
        ax.set_title(col, fontsize=10); ax.legend(fontsize=7)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle("Numeric Distributions", fontsize=13, fontweight="bold", y=1.01, color="#0D1B2A")
    plt.tight_layout(); return fig_b64(fig)


# ── 5. Box plots ───────────────────────────────────────────
def chart_boxplots(df):
    num = df.select_dtypes(include=np.number).columns.tolist()
    if not num: return None
    nc = min(3, len(num)); nr = math.ceil(len(num)/nc)
    fig, axes = plt.subplots(nr, nc, figsize=(nc*4, nr*3))
    axes = np.array(axes).flatten() if len(num)>1 else [axes]
    for i, col in enumerate(num):
        ax = axes[i]; d = df[col].dropna()
        if not len(d): ax.set_visible(False); continue
        bp = ax.boxplot(d, patch_artist=True, vert=True, widths=.5, showfliers=True,
            flierprops=dict(marker="o",markersize=3,markerfacecolor=PAL[3],alpha=.5))
        bp["boxes"][0].set_facecolor(PAL[i%len(PAL)]); bp["boxes"][0].set_alpha(.7)
        bp["medians"][0].set_color("#EF4444"); bp["medians"][0].set_linewidth(2)
        ax.set_title(col, fontsize=10); ax.set_xticks([])
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle("Box Plots — Outlier Detection", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(); return fig_b64(fig)


# ── 6. Violin plots ────────────────────────────────────────
def chart_violins(df):
    num = df.select_dtypes(include=np.number).columns.tolist()
    if not num: return None
    nc = min(3, len(num)); nr = math.ceil(len(num)/nc)
    fig, axes = plt.subplots(nr, nc, figsize=(nc*4, nr*3.2))
    axes = np.array(axes).flatten() if len(num)>1 else [axes]
    for i, col in enumerate(num):
        ax = axes[i]; d = df[col].dropna()
        if len(d) < 5: ax.set_visible(False); continue
        parts = ax.violinplot(d, showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(PAL[i%len(PAL)]); pc.set_alpha(.7)
        parts["cmedians"].set_color("#EF4444"); parts["cmedians"].set_linewidth(2)
        parts["cmeans"].set_color("#F4A261"); parts["cmeans"].set_linewidth(2)
        ax.set_title(col, fontsize=10); ax.set_xticks([])
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle("Violin Plots — Distribution Shape", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(); return fig_b64(fig)


# ── 7. QQ plots ────────────────────────────────────────────
def chart_qq(df):
    num = df.select_dtypes(include=np.number).columns.tolist()
    if not num: return None
    nc = min(3, len(num)); nr = math.ceil(len(num)/nc)
    fig, axes = plt.subplots(nr, nc, figsize=(nc*4, nr*3.2))
    axes = np.array(axes).flatten() if len(num)>1 else [axes]
    for i, col in enumerate(num):
        ax = axes[i]; d = df[col].dropna()
        if len(d) < 5: ax.set_visible(False); continue
        samp = d.sample(min(1000,len(d)), random_state=42) if len(d)>1000 else d
        (osm, osr), (slope, intercept, r) = stats.probplot(samp, dist="norm")
        ax.plot(osm, osr, "o", color=PAL[i%len(PAL)], markersize=3, alpha=.6)
        ax.plot(osm, slope*np.array(osm)+intercept, color="#EF4444", lw=1.5, ls="--")
        ax.set_title(f"{col} (r={r:.3f})", fontsize=10)
        ax.set_xlabel("Theoretical Quantiles", fontsize=8)
        ax.set_ylabel("Sample Quantiles", fontsize=8)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle("QQ Plots — Normality Check", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(); return fig_b64(fig)


# ── 8. Correlation heatmap ─────────────────────────────────
def chart_corr_heatmap(df):
    num = df.select_dtypes(include=np.number).columns.tolist()
    if len(num) < 2: return None
    corr = df[num].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(7,len(num)*.85), max(6,len(num)*.75)))
    cmap = sns.diverging_palette(220,10,as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
        annot=True, fmt=".2f", square=True, linewidths=.5,
        linecolor="#E2E8F0", ax=ax, annot_kws={"size":8},
        cbar_kws={"shrink":.7})
    ax.set_title("Correlation Matrix (Pearson)", pad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout(); return fig_b64(fig)


# ── 9. Correlation dot matrix ──────────────────────────────
def chart_corr_dots(df):
    num = df.select_dtypes(include=np.number).columns.tolist()
    if len(num) < 2: return None
    corr = df[num].corr()
    n = len(num)
    fig, ax = plt.subplots(figsize=(max(5, n*.8), max(4, n*.8)))
    for i, c1 in enumerate(num):
        for j, c2 in enumerate(num):
            v = corr.loc[c1,c2]
            size = abs(v)*500
            color = PAL[0] if v>=0 else PAL[3]
            ax.scatter(j, n-1-i, s=size, color=color, alpha=.7)
            if n<=10: ax.text(j, n-1-i, f"{v:.2f}", ha="center", va="center",
                              fontsize=7, color="white", fontweight="bold")
    ax.set_xticks(range(n)); ax.set_xticklabels(num, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(reversed(num), fontsize=9)
    ax.set_xlim(-.6, n-.4); ax.set_ylim(-.6, n-.4)
    blue_p = mpatches.Patch(color=PAL[0], label="Positive")
    red_p  = mpatches.Patch(color=PAL[3], label="Negative")
    ax.legend(handles=[blue_p,red_p], fontsize=8, loc="upper right")
    ax.set_title("Correlation Dot Matrix (size = |r|)", pad=10)
    plt.tight_layout(); return fig_b64(fig)


# ── 10. Categorical bar charts ─────────────────────────────
def chart_cat_bars(cat_stats_list):
    charts = {}
    for cs in cat_stats_list:
        if not cs["vc"] or cs["unique"] > MAX_UNIQUE_CAT: continue
        vc = cs["vc"]; labels=[str(k)[:25] for k in vc.keys()]; vals=list(vc.values())
        top = min(15, len(labels)); labels,vals = labels[:top],vals[:top]
        fig, ax = plt.subplots(figsize=(max(6,top*.55),4))
        colors = color_cycle(len(labels))
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=.5, width=.65)
        for bar,val in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.2,
                    f"{val:,}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{cs['col']} — Value Counts", fontsize=10)
        ax.set_ylabel("Count")
        plt.xticks(rotation=40, ha="right", fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        charts[cs["col"]] = fig_b64(fig)
    return charts


# ── 11. SCATTER (geom_point) ───────────────────────────────
def chart_geom_point(df):
    """Scatter plots for all numeric pairs (up to 6 combos), optionally coloured by a cat column."""
    num = df.select_dtypes(include=np.number).columns.tolist()
    cat = df.select_dtypes(include=["object","category"]).columns.tolist()
    if len(num) < 2: return None

    pairs = list(combinations(num[:6], 2))[:6]
    color_col = next((c for c in cat if df[c].nunique()<=8), None)
    nc = min(3, len(pairs)); nr = math.ceil(len(pairs)/nc)
    fig, axes = plt.subplots(nr, nc, figsize=(nc*4.5, nr*3.8))
    axes = np.array(axes).flatten() if len(pairs)>1 else [axes]

    legend_done = False
    for i, (x,y) in enumerate(pairs):
        ax = axes[i]
        sample = df[[x,y]+(([color_col]) if color_col else [])].dropna()
        sample = sample.sample(min(1000,len(sample)), random_state=42) if len(sample)>1000 else sample
        if color_col:
            groups = sample[color_col].unique()
            for j,grp in enumerate(groups):
                mask = sample[color_col]==grp
                ax.scatter(sample.loc[mask,x], sample.loc[mask,y],
                           color=PAL[j%len(PAL)], alpha=.6, s=20, label=str(grp)[:15])
            if not legend_done and len(groups)<=8:
                ax.legend(fontsize=7, title=color_col[:15], title_fontsize=7)
                legend_done = True
        else:
            ax.scatter(sample[x], sample[y], color=PAL[i%len(PAL)], alpha=.5, s=20)
        # trend line
        try:
            xv,yv = sample[x].values, sample[y].values
            z = np.polyfit(xv,yv,1); p = np.poly1d(z)
            xs = np.linspace(xv.min(),xv.max(),50)
            ax.plot(xs, p(xs), color="#EF4444", lw=1.2, ls="--", alpha=.8)
        except: pass
        r,pv = stats.pearsonr(sample[x].values, sample[y].values) if len(sample)>2 else (0,1)
        ax.set_xlabel(x, fontsize=9); ax.set_ylabel(y, fontsize=9)
        ax.set_title(f"{x} vs {y}  (r={r:.3f})", fontsize=9)

    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle("geom_point — Scatter Plots with Trend Lines", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(); return fig_b64(fig)


# ── 12. LINE chart (geom_line) ─────────────────────────────
def chart_geom_line(df):
    """Line charts: if datetime col exists → time series; else → sorted index trends."""
    num = df.select_dtypes(include=np.number).columns.tolist()
    dt  = df.select_dtypes(include=["datetime64"]).columns.tolist()
    if not num: return None

    if dt:
        time_col = dt[0]
        plot_cols = num[:5]
        nc = min(2, len(plot_cols)); nr = math.ceil(len(plot_cols)/nc)
        fig, axes = plt.subplots(nr, nc, figsize=(nc*5.5, nr*3.2))
        axes = np.array(axes).flatten() if len(plot_cols)>1 else [axes]
        for i, col in enumerate(plot_cols):
            ax = axes[i]
            sub = df[[time_col,col]].dropna().sort_values(time_col)
            # rolling mean
            rm = sub[col].rolling(max(1,len(sub)//30)).mean()
            ax.plot(sub[time_col], sub[col], color=PAL[i%len(PAL)], alpha=.4, lw=.8)
            ax.plot(sub[time_col], rm, color=PAL[i%len(PAL)], lw=2, label="Rolling mean")
            ax.fill_between(sub[time_col], sub[col], alpha=.12, color=PAL[i%len(PAL)])
            ax.set_title(col, fontsize=10); ax.set_xlabel("")
            ax.legend(fontsize=8)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
        for j in range(i+1, len(axes)): axes[j].set_visible(False)
        fig.suptitle(f"geom_line — Time Series (x = {time_col})", fontsize=13, fontweight="bold", y=1.01)
    else:
        # index-based trend
        plot_cols = num[:6]
        nc = min(3, len(plot_cols)); nr = math.ceil(len(plot_cols)/nc)
        fig, axes = plt.subplots(nr, nc, figsize=(nc*4.5, nr*3))
        axes = np.array(axes).flatten() if len(plot_cols)>1 else [axes]
        for i, col in enumerate(plot_cols):
            ax = axes[i]
            d = df[col].dropna().reset_index(drop=True)
            ax.plot(d.index, d.values, color=PAL[i%len(PAL)], lw=1, alpha=.7)
            # LOESS-style rolling
            w = max(1, len(d)//20)
            ax.plot(d.index, d.rolling(w).mean(), color="#EF4444", lw=1.5, label=f"Rolling {w}")
            ax.fill_between(d.index, d.values, alpha=.1, color=PAL[i%len(PAL)])
            ax.set_title(col, fontsize=10); ax.legend(fontsize=7)
        for j in range(i+1, len(axes)): axes[j].set_visible(False)
        fig.suptitle("geom_line — Trend Lines (by row index)", fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout(); return fig_b64(fig)


# ── 13. BAR charts (geom_bar) ──────────────────────────────
def chart_geom_bar(df):
    """Grouped & stacked bar charts for categorical × numeric relationships."""
    cat = df.select_dtypes(include=["object","category"]).columns.tolist()
    num = df.select_dtypes(include=np.number).columns.tolist()
    if not cat or not num: return None

    cat_col = min(cat, key=lambda c: df[c].nunique()) if cat else None
    if df[cat_col].nunique() > 20: return None
    num_col = num[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Grouped bar – mean per category
    ax1 = axes[0]
    grp = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(15)
    colors = color_cycle(len(grp))
    bars = ax1.bar(grp.index.astype(str), grp.values, color=colors, edgecolor="white", width=.7)
    for bar,val in zip(bars,grp.values):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_title(f"Mean {num_col} by {cat_col}", fontsize=10)
    ax1.set_xlabel(cat_col, fontsize=9); ax1.set_ylabel(f"Mean {num_col}", fontsize=9)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40, ha="right", fontsize=8)

    # Stacked bar – if second numeric col exists
    ax2 = axes[1]
    if len(num) >= 2:
        num_col2 = num[1]
        grp2 = df.groupby(cat_col)[[num_col,num_col2]].mean().head(12)
        x = np.arange(len(grp2))
        ax2.bar(x, grp2[num_col], label=num_col, color=PAL[0], edgecolor="white", width=.6, alpha=.85)
        ax2.bar(x, grp2[num_col2], bottom=grp2[num_col], label=num_col2,
                color=PAL[1], edgecolor="white", width=.6, alpha=.85)
        ax2.set_xticks(x); ax2.set_xticklabels(grp2.index.astype(str), rotation=40, ha="right", fontsize=8)
        ax2.set_title(f"Stacked: {num_col} + {num_col2} by {cat_col}", fontsize=10)
        ax2.legend(fontsize=8)
        ax2.set_xlabel(cat_col, fontsize=9)
    else:
        # Count bar
        vc = df[cat_col].value_counts().head(15)
        ax2.barh(vc.index.astype(str), vc.values, color=PAL[2], edgecolor="white", height=.7)
        ax2.set_title(f"Count per {cat_col}", fontsize=10)
        ax2.set_xlabel("Count", fontsize=9)

    fig.suptitle("geom_bar — Grouped & Stacked Bar Charts", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(); return fig_b64(fig)


# ── 14. BUBBLE chart ──────────────────────────────────────
def chart_bubble(df):
    """Bubble chart: x=num1, y=num2, size=num3, color=cat (optional)."""
    num = df.select_dtypes(include=np.number).columns.tolist()
    cat = df.select_dtypes(include=["object","category"]).columns.tolist()
    if len(num) < 3: return None

    x_col, y_col, s_col = num[0], num[1], num[2]
    color_col = next((c for c in cat if 2<=df[c].nunique()<=8), None)

    use_cols = [x_col,y_col,s_col] + ([color_col] if color_col else [])
    sample = df[use_cols].dropna().sample(min(BUBBLE_SAMPLE,len(df)), random_state=42)

    # normalise bubble size
    sv = pd.to_numeric(sample[s_col], errors="coerce").fillna(0)
    sv_norm = (sv - sv.min()) / (sv.max()-sv.min()+1e-9)
    sizes = (sv_norm*600 + 20).clip(lower=20)

    fig, ax = plt.subplots(figsize=(8,5.5))
    if color_col:
        grps = sample[color_col].unique()
        for j,grp in enumerate(grps):
            mask = sample[color_col]==grp
            ax.scatter(sample.loc[mask,x_col], sample.loc[mask,y_col],
                       s=sizes[mask], color=PAL[j%len(PAL)], alpha=.6,
                       edgecolors="white", linewidths=.5, label=str(grp)[:20])
        ax.legend(title=color_col[:20], fontsize=8, title_fontsize=8, loc="upper left")
    else:
        sc = ax.scatter(sample[x_col], sample[y_col], s=sizes,
                        c=sizes, cmap="coolwarm", alpha=.65,
                        edgecolors="white", linewidths=.5)
        plt.colorbar(sc, ax=ax, label=f"{s_col} (encoded as colour)", shrink=.7)

    ax.set_xlabel(x_col, fontsize=10); ax.set_ylabel(y_col, fontsize=10)
    ax.set_title(f"Bubble Chart — x:{x_col}  y:{y_col}  size:{s_col}", fontsize=11, pad=10)

    # Size legend
    for sz,label in [(20,"min"),(300,"mid"),(620,"max")]:
        ax.scatter([],[], s=sz, color="#94A3B8", alpha=.5, label=f"{s_col}~{label}", edgecolors="white")
    ax.legend(fontsize=7, loc="lower right", ncol=2)

    plt.tight_layout(); return fig_b64(fig)


# ── 15. SANKEY chart ──────────────────────────────────────
def chart_sankey(df):
    """
    Draws a Sankey-style flow diagram using stacked bar + flow bands.
    Uses the top-2 categorical columns as source → target nodes.
    Pure matplotlib (no plotly / no external lib required).
    """
    cat = df.select_dtypes(include=["object","category"]).columns.tolist()
    # pick two categoricals with manageable cardinality
    eligible = [c for c in cat if 2<=df[c].nunique()<=SANKEY_TOP_N*2]
    if len(eligible) < 2: return None

    src_col, tgt_col = eligible[0], eligible[1]
    ct = pd.crosstab(df[src_col], df[tgt_col])
    # keep top N rows/cols
    ct = ct.loc[ct.sum(axis=1).nlargest(SANKEY_TOP_N).index,
                ct.sum(axis=0).nlargest(SANKEY_TOP_N).index]

    src_labels = ct.index.astype(str).tolist()
    tgt_labels = ct.columns.astype(str).tolist()
    src_totals = ct.sum(axis=1).values
    tgt_totals = ct.sum(axis=0).values
    grand = max(src_totals.sum(), 1)

    fig, ax = plt.subplots(figsize=(11, max(5, max(len(src_labels),len(tgt_labels))*.6)))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

    # ─ Draw source bars (left side) ─
    src_y = {}; cur = 1.0
    for i,lbl in enumerate(src_labels):
        h = src_totals[i]/grand*.88
        y0 = cur - h
        color = PAL[i%len(PAL)]
        ax.add_patch(mpatches.FancyBboxPatch((0.02,y0),0.14,h,
            boxstyle="round,pad=0.005", fc=color, ec="white", lw=.8, alpha=.85))
        ax.text(0.17, y0+h/2, f"{lbl[:18]}\n({src_totals[i]:,})",
                va="center", ha="left", fontsize=8, color="#1E293B", fontweight="500")
        src_y[lbl] = (y0, y0+h, color)
        cur = y0 - 0.012

    # ─ Draw target bars (right side) ─
    tgt_y = {}; cur = 1.0
    for j,lbl in enumerate(tgt_labels):
        h = tgt_totals[j]/grand*.88
        y0 = cur - h
        color = PAL[(j+len(src_labels))%len(PAL)]
        ax.add_patch(mpatches.FancyBboxPatch((0.84,y0),0.14,h,
            boxstyle="round,pad=0.005", fc=color, ec="white", lw=.8, alpha=.85))
        ax.text(0.83, y0+h/2, f"{lbl[:18]}\n({tgt_totals[j]:,})",
                va="center", ha="right", fontsize=8, color="#1E293B", fontweight="500")
        tgt_y[lbl] = (y0, y0+h, color)
        cur = y0 - 0.012

    # ─ Draw flow bands ─
    src_offsets  = {l: src_y[l][0]  for l in src_labels}
    tgt_offsets  = {l: tgt_y[l][0]  for l in tgt_labels}
    for i,src in enumerate(src_labels):
        for j,tgt in enumerate(tgt_labels):
            v = ct.loc[src,tgt] if src in ct.index and tgt in ct.columns else 0
            if v==0: continue
            fh = v/grand*.88
            y_src = src_offsets[src]
            y_tgt = tgt_offsets[tgt]
            # cubic bezier path
            from matplotlib.path import Path
            import matplotlib.patches as patches
            verts = [
                (0.16, y_src+fh), (0.5, y_src+fh),
                (0.5, y_tgt+fh), (0.84, y_tgt+fh),
                (0.84, y_tgt),   (0.5, y_tgt),
                (0.5, y_src),    (0.16, y_src),
                (0.16, y_src+fh)
            ]
            codes = [Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,
                     Path.LINETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.CLOSEPOLY]
            path = Path(verts, codes)
            patch = patches.PathPatch(path,
                fc=src_y[src][2], ec="none", alpha=.22)
            ax.add_patch(patch)
            src_offsets[src] += fh
            tgt_offsets[tgt] += fh

    ax.text(0.09,1.01,src_col[:25],ha="center",va="bottom",fontsize=9,
            fontweight="bold",color="#0D1B2A",transform=ax.transAxes)
    ax.text(0.91,1.01,tgt_col[:25],ha="center",va="bottom",fontsize=9,
            fontweight="bold",color="#0D1B2A",transform=ax.transAxes)
    ax.set_title(f"Sankey Flow: {src_col} → {tgt_col}", fontsize=12,
                 fontweight="bold", pad=14)
    plt.tight_layout(); return fig_b64(fig)


# ── 16. FACET WRAP ─────────────────────────────────────────
def chart_facet_wrap(df):
    """
    facet_wrap: one numeric histogram per category level.
    Uses best numeric + best categorical pair.
    """
    cat = df.select_dtypes(include=["object","category"]).columns.tolist()
    num = df.select_dtypes(include=np.number).columns.tolist()
    if not cat or not num: return None

    num_col = num[0]
    cat_col = next((c for c in cat if 2<=df[c].nunique()<=FACET_MAX_PANELS), None)
    if cat_col is None: return None

    levels = df[cat_col].dropna().unique()[:FACET_MAX_PANELS]
    nc = min(4, len(levels)); nr = math.ceil(len(levels)/nc)
    fig, axes = plt.subplots(nr, nc, figsize=(nc*3.5, nr*2.8), sharex=True)
    axes = np.array(axes).flatten() if len(levels)>1 else [axes]

    all_vals = df[num_col].dropna()
    bins = np.histogram_bin_edges(all_vals, bins=20)
    overall_mean = all_vals.mean()

    for i,lvl in enumerate(levels):
        ax = axes[i]
        d = df.loc[df[cat_col]==lvl, num_col].dropna()
        ax.hist(d, bins=bins, color=PAL[i%len(PAL)], edgecolor="white",
                linewidth=.4, alpha=.85)
        ax.axvline(d.mean() if len(d) else overall_mean,
                   color="#EF4444", lw=1.2, ls="--")
        ax.set_title(f"{str(lvl)[:20]}\n(n={len(d):,})", fontsize=9)
        ax.set_xlabel(num_col, fontsize=8)

    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle(f"facet_wrap({cat_col}) — Distribution of {num_col}",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(); return fig_b64(fig)


# ── 17. FACET GRID ─────────────────────────────────────────
def chart_facet_grid(df):
    """
    facet_grid: rows = cat1 levels, cols = cat2 levels,
    each panel = scatter of num1 vs num2.
    Falls back to one cat if only one available.
    """
    cat = df.select_dtypes(include=["object","category"]).columns.tolist()
    num = df.select_dtypes(include=np.number).columns.tolist()
    if len(num) < 2: return None

    x_col,y_col = num[0], num[1]
    row_col = next((c for c in cat if 2<=df[c].nunique()<=4), None)
    col_col = next((c for c in cat if c!=row_col and 2<=df[c].nunique()<=4), None)

    if row_col is None: return None

    row_levels = df[row_col].dropna().unique()[:4]

    if col_col:
        col_levels = df[col_col].dropna().unique()[:4]
    else:
        # single-cat version: use row_col for rows only, one column
        col_levels = ["All"]

    nr, nc = len(row_levels), len(col_levels)
    fig, axes = plt.subplots(nr, nc, figsize=(nc*3.8+.5, nr*3.2+.5), squeeze=False)

    for ri,rlvl in enumerate(row_levels):
        for ci,clvl in enumerate(col_levels):
            ax = axes[ri][ci]
            if col_col:
                mask = (df[row_col]==rlvl) & (df[col_col]==clvl)
            else:
                mask = df[row_col]==rlvl
            sub = df.loc[mask, [x_col,y_col]].dropna()
            sub = sub.sample(min(300,len(sub)), random_state=42) if len(sub)>300 else sub
            color = PAL[(ri*nc+ci)%len(PAL)]
            ax.scatter(sub[x_col], sub[y_col], color=color, alpha=.55, s=18)
            if len(sub)>2:
                try:
                    z = np.polyfit(sub[x_col],sub[y_col],1); p=np.poly1d(z)
                    xs=np.linspace(sub[x_col].min(),sub[x_col].max(),40)
                    ax.plot(xs,p(xs),color="#EF4444",lw=1.2,ls="--")
                except: pass
            lbl = f"{str(rlvl)[:12]}" + (f"\n{str(clvl)[:12]}" if col_col else "")
            ax.set_title(lbl, fontsize=8)
            if ri==nr-1: ax.set_xlabel(x_col, fontsize=8)
            if ci==0: ax.set_ylabel(y_col, fontsize=8)

    row_title = f"rows={row_col}" + (f"  cols={col_col}" if col_col else "")
    fig.suptitle(f"facet_grid({row_title}) — {x_col} vs {y_col}",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(); return fig_b64(fig)


# ── 18. Scatter matrix ─────────────────────────────────────
def chart_scatter_matrix(df):
    num = df.select_dtypes(include=np.number).columns.tolist()
    if len(num) < 2: return None
    cols = num[:min(6,len(num))]
    avail = df[cols].dropna()
    sample = avail.sample(min(1500,len(avail)), random_state=42)
    fig = plt.figure(figsize=(max(8,len(cols)*2), max(7,len(cols)*1.9)))
    pd.plotting.scatter_matrix(sample, alpha=.35, figsize=(max(8,len(cols)*2),max(7,len(cols)*1.9)),
        diagonal="kde", color=PAL[0],
        hist_kwds={"bins":18,"color":PAL[0],"edgecolor":"white"})
    plt.suptitle("Scatter Matrix (numeric cols)", fontsize=12, fontweight="bold",
                 color="#0D1B2A", y=1.01)
    plt.tight_layout(); return fig_b64(fig)


# ════════════════════════════════════════════════════════════
#  HTML HELPERS
# ════════════════════════════════════════════════════════════

def skew_chip(v):
    if v is None: return badge("N/A","gray")
    if v>1:  return badge(f"Right-skewed ({v:+.2f})","orange")
    if v<-1: return badge(f"Left-skewed ({v:+.2f})","blue")
    return badge(f"Symmetric ({v:+.2f})","green")

def miss_bar(pct):
    cls = "#EF4444" if pct>20 else "#378ADD"
    return (f'<div style="height:7px;background:#E2E8F0;border-radius:4px;'
            f'width:110px;display:inline-block;vertical-align:middle">'
            f'<div style="height:100%;width:{min(pct,100):.1f}%;'
            f'background:{cls};border-radius:4px"></div></div>')

def severity(pct):
    if pct==0:    return badge("None","green")
    if pct<5:     return badge("Low","blue")
    if pct<20:    return badge("Moderate","orange")
    return badge("High","red")

def norm_badge(lbl):
    if "Normal" in lbl: return badge(lbl,"green")
    if lbl=="N/A":      return badge(lbl,"gray")
    return badge(lbl,"orange")

def badge(text, typ="gray"):
    styles = {
        "gray":   "background:#F1F5F9;color:#475569",
        "blue":   "background:#EFF6FF;color:#1D4ED8",
        "green":  "background:#F0FDF4;color:#15803D",
        "red":    "background:#FEF2F2;color:#B91C1C",
        "orange": "background:#FFF7ED;color:#C2410C",
        "purple": "background:#F5F3FF;color:#6D28D9",
    }
    s = styles.get(typ, styles["gray"])
    return (f'<span style="{s};padding:2px 9px;border-radius:12px;'
            f'font-size:11px;font-weight:600">{text}</span>')

def kpi(val, label, color="#0D1B2A"):
    return f'''<div style="background:#F8FAFC;border:0.5px solid #E2E8F0;
border-radius:10px;padding:14px 16px;text-align:center">
<div style="font-size:24px;font-weight:700;color:{color};line-height:1.1">{val}</div>
<div style="font-size:10px;color:#64748B;font-weight:500;text-transform:uppercase;
letter-spacing:.06em;margin-top:4px">{label}</div></div>'''

def section(id_, icon_bg, icon, title, content):
    return f'''<div class="section" id="{id_}">
<div class="section-title">
  <span class="icon-box" style="background:{icon_bg}">{icon}</span>{title}
</div>
{content}</div>'''

def accordion(title, badges_html, body_html):
    return f'''<details style="margin-bottom:8px">
<summary style="cursor:pointer;padding:10px 14px;background:#F8FAFC;
border:0.5px solid #E2E8F0;border-radius:8px;font-weight:600;font-size:13px;
color:#0D1B2A;list-style:none;display:flex;justify-content:space-between;align-items:center">
<span style="font-family:monospace">{title}</span>
<span style="display:flex;gap:6px;align-items:center">{badges_html}</span>
</summary>
<div style="border:0.5px solid #E2E8F0;border-top:none;border-radius:0 0 8px 8px;
padding:14px 16px;background:#fff">{body_html}</div></details>'''

def chart_card(b64_or_none, title="", extra=""):
    if not b64_or_none: return ""
    return f'''<div class="chart-full" style="margin-bottom:18px">
{"<h4 style='font-size:13px;font-weight:600;color:#0D1B2A;margin-bottom:10px'>"+title+"</h4>" if title else ""}
<img src="{b64_or_none}" alt="{title}" style="width:100%;border-radius:8px"/>
{extra}</div>'''

TBL_WRAP_OPEN  = '<div style="overflow-x:auto;border-radius:8px;border:0.5px solid #E2E8F0">'
TBL_WRAP_CLOSE = '</div>'
TH = 'style="padding:9px 12px;background:#0D1B2A;color:#e2e8f0;text-align:left;font-size:11px;font-weight:600;white-space:nowrap"'
TD = 'style="padding:7px 12px;border-bottom:0.5px solid #E2E8F0;font-family:monospace;font-size:12px;color:#1E293B"'
TDL= 'style="padding:7px 12px;border-bottom:0.5px solid #E2E8F0;font-weight:600;color:#0D1B2A;font-size:13px"'


# ════════════════════════════════════════════════════════════
#  FULL HTML ASSEMBLER
# ════════════════════════════════════════════════════════════

def build_report(df, ov, miss, nstats, cstats, charts, path):
    filename = os.path.basename(path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = ov["rows"]

    # ── dtype-to-badge map ──
    def dtype_badge(t):
        m={"int64":"blue","int32":"blue","float64":"blue","float32":"blue",
           "object":"purple","bool":"orange","datetime64[ns]":"green","category":"gray"}
        return badge(t, m.get(t,"gray"))

    # ── Sections ──────────────────────────────────────────

    # 1. Overview KPIs
    mp = ov["miss_pct"]
    sec1 = f'''<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px">
{kpi(f"{n:,}","Total rows","#378ADD")}
{kpi(ov["columns"],"Total columns","#378ADD")}
{kpi(ov["n_num"],"Numeric","#10B981")}
{kpi(ov["n_cat"],"Categorical","#7C3AED")}
{kpi(ov["n_dt"],"Datetime","#F4A261")}
{kpi(f'{mp}%',"Missing data","#EF4444" if mp>10 else "#F97316" if mp>0 else "#10B981")}
{kpi(f'{ov["dups"]:,}',"Duplicate rows","#EF4444" if ov["dups"]>0 else "#10B981")}
{kpi(f'{ov["total_miss"]:,}',"Missing cells","#64748B")}
{kpi(f'{ov["mem_mb"]} MB',"Memory","#64748B")}
{kpi(ov["file_size"],"File size","#64748B")}
</div>'''

    # 2. Column table
    col_rows = "".join(
        f'<tr><td {TD}>{i+1}</td><td {TDL}>{m["col"]}</td>'
        f'<td {TD}>{dtype_badge(m["dtype"])}</td>'
        f'<td {TD}>{(n-m["missing"]):,}</td>'
        f'<td {TD}>{m["missing"]:,}</td>'
        f'<td {TD}>{m["miss_pct"]:.2f}% {miss_bar(m["miss_pct"])}</td>'
        f'<td {TD}>{m["unique"]:,}</td></tr>'
        for i,m in enumerate(miss)
    )
    dtype_pie_img = chart_card(charts.get("dtype_pie"), "Column type distribution")
    sec2 = dtype_pie_img + TBL_WRAP_OPEN + f'''<table style="width:100%;border-collapse:collapse;font-size:13px">
<thead><tr><th {TH}>#</th><th {TH}>Column</th><th {TH}>Type</th><th {TH}>Non-null</th>
<th {TH}>Missing</th><th {TH}>Missing %</th><th {TH}>Unique</th></tr></thead>
<tbody>{col_rows}</tbody></table>''' + TBL_WRAP_CLOSE

    # 3. Missing analysis
    miss_bar_chart = chart_card(charts.get("miss_bar"), "Missing values by column")
    miss_heatmap   = chart_card(charts.get("miss_heatmap"), "Row-level missing pattern")
    miss_rows = "".join(
        f'<tr><td {TDL}>{m["col"]}</td><td {TD}>{m["missing"]:,}</td>'
        f'<td {TD}>{m["miss_pct"]:.2f}%</td><td {TD}>{severity(m["miss_pct"])}</td>'
        f'<td {TD}>{miss_bar(m["miss_pct"])}</td></tr>'
        for m in miss
    ) or f'<tr><td colspan="5" style="text-align:center;color:#15803D;padding:12px">✅ No missing values</td></tr>'
    sec3 = (miss_bar_chart + miss_heatmap +
            TBL_WRAP_OPEN + f'''<table style="width:100%;border-collapse:collapse;font-size:13px">
<thead><tr><th {TH}>Column</th><th {TH}>Missing count</th><th {TH}>Missing %</th>
<th {TH}>Severity</th><th {TH}>Visual</th></tr></thead>
<tbody>{miss_rows}</tbody></table>''' + TBL_WRAP_CLOSE)

    # 4. Numeric stats
    if nstats:
        num_rows = "".join(
            f'<tr><td {TDL}>{s["col"]}</td><td {TD}>{s["count"]:,}</td>'
            f'<td {TD}>{s["mean"]:,.4f}</td><td {TD}>{s["median"]:,.4f}</td>'
            f'<td {TD}>{s["std"]:,.4f}</td><td {TD}>{s["min"]:,.4f}</td>'
            f'<td {TD}>{s["max"]:,.4f}</td><td {TD}>{s["q1"]:,.4f}</td>'
            f'<td {TD}>{s["q3"]:,.4f}</td><td {TD}>{skew_chip(s["skew"])}</td>'
            f'<td {TD}>{s["kurt"] if s["kurt"] is not None else "N/A"}</td>'
            f'<td {TD}>{"<span style=color:#B91C1C;font-weight:700>"+str(s["outliers"])+"</span>" if s["outliers"]>0 else "<span style=color:#15803D>0</span>"}</td>'
            f'<td {TD}>{norm_badge(s["norm_label"])}</td></tr>'
            for s in nstats
        )
        sec4 = TBL_WRAP_OPEN + f'''<table style="width:100%;border-collapse:collapse;font-size:12px">
<thead><tr><th {TH}>Column</th><th {TH}>Count</th><th {TH}>Mean</th><th {TH}>Median</th>
<th {TH}>Std</th><th {TH}>Min</th><th {TH}>Max</th><th {TH}>Q25</th><th {TH}>Q75</th>
<th {TH}>Skewness</th><th {TH}>Kurtosis</th><th {TH}>Outliers(IQR)</th>
<th {TH}>Normality</th></tr></thead>
<tbody>{num_rows}</tbody></table>''' + TBL_WRAP_CLOSE
    else:
        sec4 = '<p style="color:#64748B">No numeric columns found.</p>'

    # 5. Categorical
    cat_bars = charts.get("cat_bars", {})
    if cstats:
        parts = []
        for cs in cstats:
            img = cat_bars.get(cs["col"],"")
            img_html = f'<img src="{img}" style="width:100%;border-radius:6px;margin-top:10px"/>' if img else ""
            vc_rows = "".join(
                f'<tr><td {TDL}>{k}</td><td {TD}>{v:,}</td>'
                f'<td {TD}>{100*v/cs["count"]:.1f}%</td></tr>'
                for k,v in list(cs["vc"].items())[:10]
            )
            vc_table = (TBL_WRAP_OPEN + f'<table style="width:100%;border-collapse:collapse;font-size:12px">'
                        f'<thead><tr><th {TH}>Value</th><th {TH}>Count</th><th {TH}>%</th></tr></thead>'
                        f'<tbody>{vc_rows}</tbody></table>' + TBL_WRAP_CLOSE)
            u = cs["unique"]
            ub = (badge("Low cardinality","green") if u<=10
                  else badge("Medium cardinality","orange") if u<=50
                  else badge("High cardinality","red"))
            kpi_row = f'''<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px">
{kpi(u,"Unique","#7C3AED")}{kpi(f'{cs["count"]:,}',"Non-null","#0D1B2A")}
{kpi(cs["missing"],"Missing","#EF4444")}{kpi(f'{cs["top_pct"]}%',"Top freq","#F4A261")}</div>'''
            parts.append(accordion(cs["col"], f'{badge(str(u)+" unique","gray")} {ub}',
                                   kpi_row + f'<p style="font-size:12px;color:#64748B;margin-bottom:10px">Top: <strong>{cs["top"]}</strong> ({cs["top_freq"]:,}×)</p>'
                                   + vc_table + img_html))
        sec5 = "\n".join(parts)
    else:
        sec5 = '<p style="color:#64748B">No categorical columns found.</p>'

    # 6. Correlation
    corr_hm  = chart_card(charts.get("corr_heatmap"), "Correlation matrix (Pearson)")
    corr_dot = chart_card(charts.get("corr_dots"), "Correlation dot matrix (size = |r|)")
    high_pairs = charts.get("high_corr", [])
    pairs_html = ""
    if high_pairs:
        pairs_html = ('<h4 style="font-size:13px;font-weight:600;color:#0D1B2A;margin:20px 0 10px">'
                      f'⚠ High-correlation pairs (≥{CORRELATION_THRESHOLD})</h4>'
                      '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px">')
        for p in high_pairs:
            col = "#B91C1C" if p["abs"]>=.9 else "#C2410C"
            pairs_html += (f'<div style="background:#F8FAFC;border:0.5px solid #E2E8F0;'
                           f'border-radius:8px;padding:12px 14px">'
                           f'<div style="font-size:20px;font-weight:700;color:{col}">{p["v"]:.4f}</div>'
                           f'<div style="font-family:monospace;font-size:11px;color:#64748B;margin:4px 0">'
                           f'{p["c1"]} ↔ {p["c2"]}</div>'
                           f'<div style="margin-top:5px">'
                           f'{badge("Positive","red") if p["v"]>0 else badge("Negative","blue")} '
                           f'{badge("Very high ≥0.9","red") if p["abs"]>=.9 else badge("High","orange")}'
                           f'</div></div>')
        pairs_html += "</div>"
    if not corr_hm and not corr_dot:
        sec6 = '<p style="color:#64748B">Fewer than 2 numeric columns — correlation not available.</p>'
    else:
        sec6 = corr_hm + corr_dot + pairs_html

    # 7. Distribution charts (histograms, box, violin, qq)
    sec7 = (chart_card(charts.get("histograms"), "Histograms with mean & median lines") +
            chart_card(charts.get("boxplots"), "Box plots — outlier detection") +
            chart_card(charts.get("violins"), "Violin plots — distribution shape") +
            chart_card(charts.get("qq"), "QQ plots — normality check"))
    if not sec7.strip():
        sec7 = '<p style="color:#64748B">No numeric columns for distribution plots.</p>'

    # 8. Advanced charts
    geom_note = lambda t: f'<p style="font-size:11px;color:#94A3B8;margin-bottom:8px;font-style:italic">{t}</p>'
    sec8 = (
        chart_card(charts.get("geom_point"),
                   "geom_point — Scatter plots with trend lines",
                   geom_note("Each panel = one numeric pair. Red dashed = linear trend. Coloured by first categorical column (≤8 levels).")) +
        chart_card(charts.get("geom_line"),
                   "geom_line — Trend lines",
                   geom_note("If a datetime column exists: time series + rolling mean. Otherwise: row-index trend.")) +
        chart_card(charts.get("geom_bar"),
                   "geom_bar — Grouped & stacked bar charts",
                   geom_note("Left: mean of numeric by category. Right: stacked means of two numeric columns (or count if only one numeric).")) +
        chart_card(charts.get("bubble"),
                   "Bubble chart — 3-variable relationship",
                   geom_note("x = 1st numeric, y = 2nd numeric, bubble size = 3rd numeric, colour = categorical (if ≤8 levels).")) +
        chart_card(charts.get("sankey"),
                   "Sankey flow diagram",
                   geom_note("Flow between top-2 categorical columns. Band width ∝ co-occurrence count.")) +
        chart_card(charts.get("facet_wrap"),
                   "facet_wrap — Distribution per category level",
                   geom_note("One histogram panel per level of the first suitable categorical column.")) +
        chart_card(charts.get("facet_grid"),
                   "facet_grid — Scatter grid (row × col)",
                   geom_note("Grid of scatter plots split by one or two categorical columns.")) +
        chart_card(charts.get("scatter_matrix"),
                   "Scatter matrix")
    )
    if not sec8.strip():
        sec8 = '<p style="color:#64748B">Insufficient columns for advanced charts.</p>'

    # 9. Data preview
    preview = df.head(SAMPLE_ROWS).to_html(classes="", border=0, index=True,
                                            na_rep='<em style="color:#94A3B8">NaN</em>')
    sec9 = (TBL_WRAP_OPEN + preview + TBL_WRAP_CLOSE)

    # 10. Insights
    insights = []
    if ov["dups"]>0: insights.append(("🔴",f'<strong>{ov["dups"]:,} duplicate rows</strong> — deduplicate before modelling.'))
    if mp>20: insights.append(("🔴",f'Overall missing <strong>{mp}%</strong> — major imputation or removal required.'))
    elif mp>5: insights.append(("🟡",f'<strong>{mp}% missing</strong> — review imputation strategy.'))
    else: insights.append(("🟢",f'Missing data is low at <strong>{mp}%</strong> — dataset largely complete.'))
    hm=[m["col"] for m in miss if m["miss_pct"]>50]
    if hm: insights.append(("🔴",f'Cols >50% missing: <strong>{", ".join(hm)}</strong> — consider dropping.'))
    sk=[s["col"] for s in nstats if s["skew"] and abs(s["skew"])>1]
    if sk: insights.append(("📊",f'Highly skewed: <strong>{", ".join(sk)}</strong> — log/sqrt transform advised.'))
    ot=[(s["col"],s["outliers"]) for s in nstats if s["outliers"]>0]
    if ot: insights.append(("⚠",f'Outliers (IQR): <strong>{", ".join(f"{c}({o:,})" for c,o in ot[:5])}</strong>'))
    hp=charts.get("high_corr",[])
    if hp: insights.append(("🔗",f'Highest correlation: <strong>{hp[0]["c1"]} ↔ {hp[0]["c2"]}</strong> ({hp[0]["v"]:.4f})'))
    hc=[c for c in cstats if c["unique"]>50]
    if hc: insights.append(("🏷",f'High cardinality: <strong>{", ".join(c["col"] for c in hc)}</strong> — target encode.'))
    nn=[s["col"] for s in nstats if "Non-normal" in s["norm_label"]]
    if nn: insights.append(("📉",f'Non-normal: <strong>{", ".join(nn[:5])}</strong> — use non-parametric tests.'))
    insights.append(("✅",f'Shape: <strong>{n:,} rows × {ov["columns"]} cols</strong> · {ov["n_num"]} numeric · {ov["n_cat"]} categorical'))

    sec10 = ('<div style="background:linear-gradient(135deg,#EFF6FF,#F0FDF4);'
             'border:0.5px solid #BFDBFE;border-radius:10px;padding:16px 20px">'
             '<h4 style="font-size:13px;font-weight:700;color:#1D4ED8;margin-bottom:10px">🤖 Automated Observations</h4>'
             '<ul style="list-style:none;display:flex;flex-direction:column;gap:5px">'
             + "".join(f'<li style="font-size:13px;color:#1E293B"><span style="margin-right:8px">{ic}</span>{txt}</li>'
                       for ic,txt in insights)
             + '</ul></div>')

    # ── CSS & skeleton ──
    HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>EDA Report v3 — {filename}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#F1F5F9;color:#1E293B;font-size:14px;line-height:1.6}}
.header{{background:#0D1B2A;color:#fff;padding:32px 48px 24px;position:relative;overflow:hidden}}
.header::before{{content:'';position:absolute;top:-60px;right:-60px;width:260px;height:260px;border-radius:50%;background:rgba(55,138,221,.1)}}
.header::after{{content:'';position:absolute;bottom:-40px;right:140px;width:160px;height:160px;border-radius:50%;background:rgba(244,162,97,.07)}}
.hbadge{{display:inline-block;background:rgba(55,138,221,.2);color:#60B4F8;border:1px solid rgba(55,138,221,.3);border-radius:20px;padding:2px 12px;font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px}}
.header h1{{font-size:26px;font-weight:700;letter-spacing:-.02em;margin-bottom:4px}}
.header h1 span{{color:#60B4F8}}
.meta{{color:rgba(255,255,255,.45);font-size:11px;font-family:monospace;margin-top:6px}}
nav{{background:#fff;border-bottom:1px solid #E2E8F0;padding:0 48px;display:flex;position:sticky;top:0;z-index:100;box-shadow:0 1px 6px rgba(0,0,0,.05);overflow-x:auto}}
nav a{{padding:12px 16px;text-decoration:none;color:#64748B;font-size:12px;font-weight:600;border-bottom:2px solid transparent;white-space:nowrap;transition:all .15s}}
nav a:hover{{color:#378ADD;border-bottom-color:#378ADD}}
.container{{max-width:1320px;margin:0 auto;padding:28px 48px 64px}}
.section{{background:#fff;border-radius:12px;border:0.5px solid #E2E8F0;padding:24px 28px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.04)}}
.section-title{{font-size:16px;font-weight:700;color:#0D1B2A;margin-bottom:18px;display:flex;align-items:center;gap:8px}}
.icon-box{{width:28px;height:28px;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
thead tr{{background:#0D1B2A}}
tbody tr:nth-child(even){{background:#F8FAFC}}
tbody tr:hover{{background:#EFF6FF}}
.chart-full{{background:#F8FAFC;border:0.5px solid #E2E8F0;border-radius:10px;padding:14px 16px;overflow-x:auto}}
details summary::-webkit-details-marker{{display:none}}
.footer{{text-align:center;color:#94A3B8;font-size:11px;padding:24px 0 12px;border-top:0.5px solid #E2E8F0;margin-top:24px}}
@media(max-width:640px){{.container{{padding:16px 16px 40px}}nav{{padding:0 12px}}.header{{padding:20px 16px 16px}}}}
</style>
</head>
<body>
<div class="header">
  <div class="hbadge">Automated EDA Report v3.0</div>
  <h1>Dataset Analysis <span>Report</span></h1>
  <div class="meta">File: {filename} &nbsp;|&nbsp; Generated: {now} &nbsp;|&nbsp;
  {n:,} rows × {ov["columns"]} cols &nbsp;|&nbsp; {ov["mem_mb"]} MB</div>
</div>
<nav>
  <a href="#s1">📊 Overview</a>
  <a href="#s2">🔢 Columns</a>
  <a href="#s3">❓ Missing</a>
  <a href="#s4">📈 Numeric</a>
  <a href="#s5">🏷 Categorical</a>
  <a href="#s6">🔗 Correlation</a>
  <a href="#s7">📉 Distributions</a>
  <a href="#s8">🚀 Advanced Charts</a>
  <a href="#s9">👁 Preview</a>
  <a href="#s10">💡 Insights</a>
</nav>
<div class="container">
{section("s1","#EFF6FF","📊","Dataset Overview",sec1)}
{section("s2","#F5F3FF","🔢","Column Summary",sec2)}
{section("s3","#FEF2F2","❓","Missing Data Analysis",sec3)}
{section("s4","#F0FDF4","📈","Numeric Statistics",sec4)}
{section("s5","#FFFBEB","🏷","Categorical Analysis",sec5)}
{section("s6","#FFF7ED","🔗","Correlation Analysis",sec6)}
{section("s7","#EFF6FF","📉","Distributions & Outliers",sec7)}
{section("s8","#F0FDF4","🚀","Advanced Charts — geom · Sankey · Bubble · Facets",sec8)}
{section("s9","#F8FAFC","👁",f"Data Preview (first {SAMPLE_ROWS} rows)",sec9)}
{section("s10","#F0FDF4","💡","Automated Insights & Recommendations",sec10)}
</div>
<div class="footer container">
  Generated by <strong style="color:#378ADD">EDA Report Generator v3.0</strong>
  &nbsp;·&nbsp; {now} &nbsp;·&nbsp; {filename}
</div>
<script>
document.querySelectorAll('nav a').forEach(a=>{{
  a.addEventListener('click',e=>{{
    e.preventDefault();
    const t=document.querySelector(a.getAttribute('href'));
    if(t) t.scrollIntoView({{behavior:'smooth',block:'start'}});
  }});
}});
const secs=document.querySelectorAll('.section');
const navAs=document.querySelectorAll('nav a');
window.addEventListener('scroll',()=>{{
  let cur='';
  secs.forEach(s=>{{if(window.scrollY>=s.offsetTop-80)cur=s.id;}});
  navAs.forEach(a=>{{
    const active=a.getAttribute('href')==='#'+cur;
    a.style.color=active?'#378ADD':'';
    a.style.borderBottomColor=active?'#378ADD':'transparent';
  }});
}});
</script>
</body>
</html>"""
    return HTML


# ════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ════════════════════════════════════════════════════════════

def run_eda(file_path=FILE_PATH, output_path=OUTPUT_PATH):
    print("=" * 62)
    print("  EDA Report Generator v3.0  —  Starting")
    print("=" * 62)

    df = load_data(file_path)

    def step(msg):
        print(f"  [→] {msg}"); return msg

    step("Computing overview & stats...")
    ov   = overview(df, file_path)
    miss = missing_info(df)
    ns   = num_stats(df)
    cs   = cat_stats(df)

    step("Generating base charts...")
    charts = {
        "dtype_pie":    chart_dtype_pie(ov),
        "miss_bar":     chart_missing_bar(df),
        "miss_heatmap": chart_missing_heatmap(df),
        "histograms":   chart_histograms(df),
        "boxplots":     chart_boxplots(df),
        "violins":      chart_violins(df),
        "qq":           chart_qq(df),
        "corr_heatmap": chart_corr_heatmap(df),
        "corr_dots":    chart_corr_dots(df),
        "cat_bars":     chart_cat_bars(cs),
        "scatter_matrix": chart_scatter_matrix(df),
    }

    # Compute high-corr pairs for the report
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols)>=2:
        corr_abs = df[num_cols].corr().abs()
        pairs = []
        for c1,c2 in combinations(num_cols,2):
            v = df[num_cols].corr().loc[c1,c2]
            av = abs(v)
            if av>=CORRELATION_THRESHOLD:
                pairs.append({"c1":c1,"c2":c2,"v":round(float(v),4),"abs":round(av,4)})
        charts["high_corr"] = sorted(pairs, key=lambda x: x["abs"], reverse=True)
    else:
        charts["high_corr"] = []

    step("Generating advanced charts (geom · Sankey · Bubble · Facets)...")
    charts["geom_point"]   = chart_geom_point(df)
    charts["geom_line"]    = chart_geom_line(df)
    charts["geom_bar"]     = chart_geom_bar(df)
    charts["bubble"]       = chart_bubble(df)
    charts["sankey"]       = chart_sankey(df)
    charts["facet_wrap"]   = chart_facet_wrap(df)
    charts["facet_grid"]   = chart_facet_grid(df)

    step("Assembling HTML report...")
    html = build_report(df, ov, miss, ns, cs, charts, file_path)

    out_dir = os.path.dirname(output_path)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n{'='*62}")
    print(f"  ✅  Report saved → {output_path}")
    print(f"      Rows:    {ov['rows']:,}")
    print(f"      Columns: {ov['columns']}  ({ov['n_num']} numeric, {ov['n_cat']} cat, {ov['n_dt']} datetime)")
    print(f"      Missing: {ov['miss_pct']}%    Duplicates: {ov['dups']:,}")
    print(f"{'='*62}\n")

    # Databricks display helper
    try:
        from IPython.display import display, HTML as iHTML
        display(iHTML(f'<a href="{output_path}" target="_blank" '
                      f'style="font-size:14px;font-weight:600">📄 Open EDA Report</a>'))
    except Exception:
        pass

    return output_path


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) >= 2: FILE_PATH = sys.argv[1]
    if len(sys.argv) >= 3: OUTPUT_PATH = sys.argv[2]
    run_eda(FILE_PATH, OUTPUT_PATH)
