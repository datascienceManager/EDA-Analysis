Build a comprehensive Python EDA script optimised for Databricks — with a beautiful self-contained HTML report.Here's everything the script does and how to use it in Databricks:

**What the HTML report contains (9 sections):**

1. **Overview** — KPI cards: rows, columns, memory, missing %, duplicate count
2. **Data Types** — pie chart + full column table with dtype badges
3. **Missing Data** — bar chart coloured by severity + per-column missing % with visual progress bars
4. **Numeric Statistics** — mean, median, std, min/max, Q25/Q75, skewness chip, outlier count (IQR), normality test (Shapiro-Wilk)
5. **Categorical Analysis** — accordion per column with cardinality badge, top values table, bar chart
6. **Correlation Matrix** — heatmap + high-correlation pairs panel (flagging multicollinearity)
7. **Distributions & Outliers** — histograms (with mean/median lines), box plots, scatter matrix
8. **Data Preview** — first N rows rendered as table
9. **Automated Insights** — auto-generated observations on skewness, outliers, missing, high cardinality, duplicates

**Databricks usage — 3 ways:**

```python
# Option 1: Paste entire script into a notebook cell, then at the bottom:
run_eda("/dbfs/FileStore/your_data.csv", "/dbfs/tmp/eda_report.html")

# Option 2: Upload the .py file and %run it
%run /Workspace/Shared/eda_report
run_eda("/dbfs/FileStore/churn_data.csv", "/dbfs/tmp/eda_report.html")

# Option 3: CLI
python eda_report.py mydata.csv /tmp/report.html
```

**After running in Databricks, download the HTML report:**
```python
# Copy from DBFS to download
dbutils.fs.cp("dbfs:/tmp/eda_report.html", "file:/tmp/eda_report.html")
# Then Files → Download in the Databricks UI
```

All charts are embedded as base64 PNGs so the HTML file is fully self-contained — no external dependencies needed to open it.
