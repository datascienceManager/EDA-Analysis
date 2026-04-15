This is a large and complex application. Build as a full standalone HTML file that downloads — too large for the inline widget, so I'll write it directly to the outputs directory.**EDA Studio** — a fully self-contained browser app (~87KB). Just open the downloaded HTML file in any browser and upload your data. Here's everything inside:

**6 fully interactive pages:**

| Page | What's there |
|---|---|
| 📂 Upload | Drag-drop CSV/XLSX, instant preview with summary pills, no server needed |
| 📊 Overview | KPI cards, dtype pie, missing bar chart, 6 mini-histograms |
| 🔢 Columns | Full column table with type badges, missing % visual bars, unique counts |
| 📈 EDA & Stats | Numeric stats table (mean/median/std/Q1/Q3/skewness/outliers), categorical accordions with top-value tables, correlation heatmap |
| 🎨 Chart Builder | 18 chart types selectable, full column mapping, render + download PNG |
| 📉 Trend Analysis | Line/scatter/bar/area with rolling window, geom_label annotated chart, trend KPIs |
| 💡 Summary | Auto-insights, data quality report, outlier summary table |

**18 chart types in the builder:**

`geom_bar` · `geom_line` · `geom_point` · `geom_label` · Pie · Donut · Area · Bubble · Scatter · Horizontal bar · **Sankey** · **facet_wrap** · **facet_grid** · Histogram · Box plot · **Heatmap** · **Waterfall** · **Pareto**

**Auto-generated gallery** builds 7 charts automatically from your data the moment you upload — geom_bar, geom_point scatter, trend line, pie, bubble, histogram, and Pareto.

**Download button** generates a clean self-contained HTML report with all stats, the column table, numeric stats, data preview, and automated observations — ready to share with anyone.
