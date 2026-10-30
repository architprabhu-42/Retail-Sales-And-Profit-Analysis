"""Retail Sales & Profit Intelligence Studio - simple black & white BI dashboard.
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/superstore.xls"
REQUIRED = ["Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Customer Name", "Segment",
            "State", "Region", "Category", "Sub-Category", "Product Name", "Sales", "Quantity", "Discount", "Profit"]
PAL = ["#2E86AB", "#F18F01", "#C73E1D", "#3B8B5A", "#7B4B94", "#E2B13C"]  # chart colours (UI stays black & white)
POS, NEG = "#3B8B5A", "#C73E1D"

st.set_page_config(page_title="Retail BI Studio", layout="wide")

# ---------------- Black & white theme with Poppins ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
html, body, .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stApp li,
.stApp input, .stApp button, .stApp textarea, .stApp table, .stApp div[data-baseweb] {
    font-family: 'Poppins', sans-serif !important;
}
.stApp { background: #FFFFFF; color: #000000; }
section[data-testid="stSidebar"] { background: #000000; }
section[data-testid="stSidebar"] * { color: #FFFFFF; }
section[data-testid="stSidebar"] div[data-baseweb="select"] *,
section[data-testid="stSidebar"] div[data-baseweb="input"] *,
section[data-testid="stSidebar"] [data-testid="stDateInput"] *,
section[data-testid="stSidebar"] button * { color: #000000 !important; }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * { color: #FFFFFF !important; }
section[data-testid="stSidebar"] span[data-baseweb="tag"] { background: #000000 !important; }
section[data-testid="stSidebar"] span[data-baseweb="tag"] * { color: #FFFFFF !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: #1A1A1A; }
section[data-testid="stSidebar"] button { background: #FFFFFF !important; border: 1px solid #FFFFFF; }
.banner { background: #000; color: #fff; padding: 22px 28px; border-radius: 6px; margin-bottom: 18px; }
.banner h1 { color: #fff !important; margin: 0; font-size: 28px; font-weight: 700; }
.banner p { color: #CCCCCC !important; margin: 4px 0 0; font-size: 14px; }
.kpi { border: 2px solid #000; border-radius: 6px; padding: 14px 16px; background: #fff; min-height: 104px; }
.kpi .label { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #555; }
.kpi .value { font-size: 20px; white-space: nowrap; font-weight: 700; color: #000; }
.insight { border-left: 4px solid #000; background: #F5F5F5; padding: 10px 14px; margin: 6px 0; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
.stTabs [aria-selected="true"] { color: #000 !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #000 !important; }
</style>
""", unsafe_allow_html=True)


def style(fig, height=380):
    fig.update_layout(template="simple_white", font=dict(family="Poppins", color="#000"), height=height,
                      plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                      title_font=dict(size=16), margin=dict(l=10, r=10, t=50, b=10),
                      colorway=PAL, legend=dict(orientation="h", y=-0.2))
    return fig


def kpi_row(items):
    for col, (label, value) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div></div>',
                     unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


# ---------------- Data loading & cleaning ----------------
@st.cache_data
def load(file):
    """Return (orders, returns, people). Returns/People are None when the file has no such sheets."""
    src = DATA_PATH if file is None else file
    if file is not None and not file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_csv(file, encoding_errors="replace", dtype={"Postal Code": str}), None, None
    sheets = pd.read_excel(src, sheet_name=None, dtype={"Postal Code": str})
    orders = sheets.get("Orders", next(iter(sheets.values())))
    return orders, sheets.get("Returns"), sheets.get("People")


@st.cache_data
def clean(raw, returns, people):
    df = raw.rename(columns={"Country/Region": "Country", "State/Province": "State"})
    report = {"Rows (raw)": len(df), "Missing values": int(df.isna().sum().sum())}
    dup = df.drop(columns=[c for c in ["Row ID"] if c in df]).duplicated()
    report["Duplicates removed"] = int(dup.sum())
    df = df.loc[~dup].copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
    for c in ["Sales", "Quantity", "Discount", "Profit"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[["Order Date", "Sales", "Profit"]].isna().any(axis=1)
    report["Invalid dates/numbers removed"] = int(bad.sum())
    df = df.loc[~bad].copy()
    report["Ship date before order date"] = int((df["Ship Date"] < df["Order Date"]).sum())
    report["Negative-profit rows (kept)"] = int((df["Profit"] < 0).sum())
    if "Country" in df and "Postal Code" in df:
        us = (df["Country"] == "United States") & df["Postal Code"].notna()
        fix = us & (df["Postal Code"].astype(str).str.len() < 5)
        report["US ZIP codes fixed (leading zero)"] = int(fix.sum())
        df.loc[us, "Postal Code"] = df.loc[us, "Postal Code"].astype(str).str.zfill(5)
    if returns is not None:
        df = df.merge(returns.drop_duplicates("Order ID"), on="Order ID", how="left")
        df["Returned"] = df["Returned"].fillna("No")
        report["Order lines in returned orders"] = int((df["Returned"] == "Yes").sum())
    df["Is Returned"] = (df["Returned"] == "Yes").astype(int) if "Returned" in df else 0
    if people is not None:
        df = df.merge(people, on="Region", how="left")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Discount Band"] = pd.cut(df["Discount"], [-0.01, 0, 0.2, 0.4, 1.0],
                                 labels=["No discount", "1-20%", "21-40%", "41-80%"]).astype(str)
    df["Is Loss"] = (df["Profit"] < 0).astype(int)
    df["Sales x Discount"] = df["Sales"] * df["Discount"]
    report["Rows (clean)"] = len(df)
    return df, report


def perf(d, by):
    g = d.groupby(by).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "nunique"),
                          Units=("Quantity", "sum")).reset_index()
    g["Margin %"] = np.where(g["Sales"] != 0, g["Profit"] / g["Sales"] * 100, np.nan)
    return g.sort_values("Sales", ascending=False)


# ---------------- Sidebar ----------------
st.sidebar.markdown("## RETAIL BI STUDIO")
upload = st.sidebar.file_uploader("Upload your own CSV / Excel (optional)", type=["csv", "xlsx", "xls"])
try:
    raw, returns_sheet, people_sheet = load(upload)
except FileNotFoundError:
    st.error(f"Dataset not found at `{DATA_PATH}`. Put the Superstore workbook there or upload a file.")
    st.stop()
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

missing = [c for c in REQUIRED if c not in raw.columns.str.replace("/Province", "")]
if missing:
    st.error("This file is missing required columns: " + ", ".join(missing))
    st.stop()

data, clean_report = clean(raw, returns_sheet, people_sheet)
has_returns = returns_sheet is not None
st.sidebar.caption(f"Source: {'Uploaded - ' + upload.name if upload else 'Built-in Superstore'} · {len(data):,} rows")

st.sidebar.markdown("### Filters")
if st.sidebar.button("Reset filters"):
    for k in ["dates", "region", "category", "segment"]:
        st.session_state.pop(k, None)
dmin, dmax = data["Order Date"].min().date(), data["Order Date"].max().date()
dates = st.sidebar.date_input("Order date range", (dmin, dmax), min_value=dmin, max_value=dmax, key="dates")
regions = st.sidebar.multiselect("Region", sorted(data["Region"].unique()), key="region")
cats = st.sidebar.multiselect("Category", sorted(data["Category"].unique()), key="category")
segs = st.sidebar.multiselect("Segment", sorted(data["Segment"].unique()), key="segment")

df = data
if isinstance(dates, tuple) and len(dates) == 2:
    df = df[(df["Order Date"].dt.date >= dates[0]) & (df["Order Date"].dt.date <= dates[1])]
for col, sel in [("Region", regions), ("Category", cats), ("Segment", segs)]:
    if sel:
        df = df[df[col].isin(sel)]

# ---------------- Header & KPIs ----------------
st.markdown('<div class="banner"><h1>Retail Sales & Profit Intelligence Studio</h1>'
            '<p>Business Intelligence and Analytics Mini Project · Official Tableau Superstore, US &amp; Canada, 2023–2026 (USD)</p></div>',
            unsafe_allow_html=True)

if df.empty:
    st.warning("No rows match the selected filters. Change or reset the filters.")
    st.stop()

sales, profit, orders = df["Sales"].sum(), df["Profit"].sum(), df["Order ID"].nunique()
kpi_row([("Total Sales", f"${sales:,.0f}"), ("Total Profit", f"${profit:,.0f}"), ("Orders", f"{orders:,}"),
         ("Profit Margin", f"{profit / sales * 100:.1f}%" if sales else "-"),
         ("Avg Order Value", f"${sales / orders:,.0f}" if orders else "-"),
         ("Customers", f"{df['Customer ID'].nunique():,}")]
        + ([("Return Rate", f"{df.loc[df['Is Returned'] == 1, 'Order ID'].nunique() / orders * 100:.1f}%")]
           if has_returns else []))
st.write("")

tabs = st.tabs(["Overview", "Sales & Profit", "Customers", "Returns", "Prediction Models", "Data & Cleaning"])

# ---------------- Overview ----------------
with tabs[0]:
    m = df.groupby("Month")[["Sales", "Profit"]].sum().reset_index()
    fig = go.Figure([go.Scatter(x=m["Month"], y=m["Sales"], name="Sales", line=dict(color=PAL[0], width=2.5)),
                     go.Scatter(x=m["Month"], y=m["Profit"], name="Profit", line=dict(color=PAL[1], width=2.5))])
    st.plotly_chart(style(fig.update_layout(title="Monthly Sales and Profit Trend", yaxis_title="USD")),
                    width="stretch", theme=None)

    c1, c2 = st.columns(2)
    cat = perf(df, "Category")
    c1.plotly_chart(style(px.bar(cat, x="Category", y=["Sales", "Profit"], barmode="group",
                                 title="Sales vs Profit by Category", labels={"value": "USD", "variable": ""},
                                 color_discrete_sequence=[PAL[0], PAL[1]])), width="stretch", theme=None)
    reg = perf(df, "Region").sort_values("Profit", ascending=False)
    c2.plotly_chart(style(px.bar(reg, x="Region", y="Profit", title="Profit by Region", text_auto=",.0f",
                                 color_discrete_sequence=[PAL[0]])), width="stretch", theme=None)

    c1, c2 = st.columns(2)
    top = df.groupby("Product Name")["Sales"].sum().nlargest(10).sort_values().reset_index()
    top["Product"] = top["Product Name"].str.slice(0, 40)
    c1.plotly_chart(style(px.bar(top, x="Sales", y="Product", orientation="h", title="Top 10 Products by Sales",
                                 color_discrete_sequence=[PAL[0]]), 420), width="stretch", theme=None)
    seg = df.groupby("Segment")["Sales"].sum().reset_index()
    c2.plotly_chart(style(px.pie(seg, names="Segment", values="Sales", title="Sales Share by Segment", hole=0.45,
                                 color_discrete_sequence=PAL), 420),
                    width="stretch", theme=None)

    st.markdown("#### Key insights (calculated from the current filters)")
    best_m = m.loc[m["Sales"].idxmax()]
    loss_prod = (df.groupby("Product Name")["Profit"].sum() < 0).sum()
    insight(f"Highest-sales category: <b>{cat.iloc[0]['Category']}</b> (${cat.iloc[0]['Sales']:,.0f})")
    insight(f"Most profitable region: <b>{reg.iloc[0]['Region']}</b> (${reg.iloc[0]['Profit']:,.0f})")
    insight(f"Best sales month: <b>{best_m['Month']:%B %Y}</b> (${best_m['Sales']:,.0f})")
    insight(f"Loss-making products: <b>{loss_prod}</b> of {df['Product Name'].nunique()}")

# ---------------- Sales & Profit ----------------
with tabs[1]:
    sub = perf(df, "Sub-Category").sort_values("Profit")
    sub["Result"] = np.where(sub["Profit"] < 0, "Loss", "Profit")
    fig = px.bar(sub, x="Profit", y="Sub-Category", orientation="h", color="Result",
                 title="Profit by Sub-Category", color_discrete_map={"Profit": POS, "Loss": NEG})
    st.plotly_chart(style(fig, 520), width="stretch", theme=None)

    c1, c2 = st.columns(2)
    stack = df.groupby(["Category", "Segment"])["Sales"].sum().reset_index()
    c1.plotly_chart(style(px.bar(stack, x="Category", y="Sales", color="Segment", title="Sales by Category and Segment",
                                 color_discrete_sequence=PAL)), width="stretch", theme=None)
    band = df.groupby("Discount Band").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reindex(
        ["No discount", "1-20%", "21-40%", "41-80%"]).dropna().reset_index()
    band["Margin %"] = band["Profit"] / band["Sales"] * 100
    c2.plotly_chart(style(px.bar(band, x="Discount Band", y="Margin %", title="Profit Margin by Discount Band",
                                 text_auto=".1f", color_discrete_sequence=[PAL[0]])), width="stretch", theme=None)

    c1, c2 = st.columns(2)
    sample = df.sample(min(len(df), 3000), random_state=1)
    c1.plotly_chart(style(px.scatter(sample, x="Discount", y="Profit", size="Sales", opacity=0.35,
                                     hover_name="Product Name", title="Discount vs Profit (size = Sales)",
                                     color_discrete_sequence=[PAL[0]]), 420), width="stretch", theme=None)
    hm = df.pivot_table(index="Sub-Category", columns="Region", values="Profit", aggfunc="sum")
    c2.plotly_chart(style(px.imshow(hm, text_auto=",.0f", aspect="auto", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                                    title="Profit Heatmap: Sub-Category x Region"), 420), width="stretch", theme=None)
    st.caption("Discount and profit move together in this sample data; this shows association, not proof of cause.")

    st.markdown("#### Loss-making products")
    lp = df.groupby("Product Name").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                                        Quantity=("Quantity", "sum"), Avg_Discount=("Discount", "mean"))
    st.dataframe(lp[lp["Profit"] < 0].sort_values("Profit").style.format(
        {"Sales": "${:,.0f}", "Profit": "${:,.0f}", "Avg_Discount": "{:.0%}"}), width="stretch", height=300)

# ---------------- Customers ----------------
with tabs[2]:
    c1, c2 = st.columns(2)
    sp = perf(df, "Segment")
    c1.plotly_chart(style(px.bar(sp, x="Segment", y=["Sales", "Profit"], barmode="group", title="Segment Performance",
                                 labels={"value": "USD", "variable": ""}, color_discrete_sequence=[PAL[0], PAL[1]])),
                    width="stretch", theme=None)
    tc = df.groupby("Customer Name")[["Sales", "Profit"]].sum().nlargest(10, "Sales").sort_values("Sales").reset_index()
    c2.plotly_chart(style(px.bar(tc, x="Sales", y="Customer Name", orientation="h", title="Top 10 Customers by Sales",
                                 color_discrete_sequence=[PAL[0]])), width="stretch", theme=None)

    st.markdown("#### Customer segmentation (RFM + K-Means, 4 clusters)")
    ref = df["Order Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer ID").agg(Recency=("Order Date", lambda s: (ref - s.max()).days),
                                        Frequency=("Order ID", "nunique"), Monetary=("Sales", "sum"))
    if len(rfm) >= 8:
        km = KMeans(4, n_init=10, random_state=42).fit(StandardScaler().fit_transform(np.log1p(rfm)))
        rfm["Cluster"] = km.labels_
        rank = rfm.groupby("Cluster")["Monetary"].mean().rank(ascending=False).astype(int)
        rfm["Segment"] = rfm["Cluster"].map(rank.map({1: "Champions", 2: "Loyal", 3: "Potential", 4: "Low Value"}))
        order = ["Champions", "Loyal", "Potential", "Low Value"]
        fig = px.scatter(rfm.reset_index(), x="Frequency", y="Monetary", color="Segment", symbol="Segment", log_y=True,
                         category_orders={"Segment": order}, hover_name="Customer ID",
                         color_discrete_sequence=PAL,
                         title="Customer Segments (Frequency vs Monetary)")
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(style(fig, 420), width="stretch", theme=None)
        prof = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().reindex(order)
        prof.insert(0, "Customers", rfm["Segment"].value_counts())
        c2.dataframe(prof.style.format({"Recency": "{:.0f} days", "Frequency": "{:.1f}", "Monetary": "${:,.0f}"}),
                     width="stretch")
    else:
        st.info("Not enough customers in the current filter for clustering.")

# ---------------- Returns ----------------
with tabs[3]:
    if not has_returns:
        st.info("This dataset has no Returns sheet, so returns analysis is not available.")
    else:
        st.caption("Return rate = share of order lines that belong to a returned order (returns are recorded per order).")

        def rr(by):
            g = df.groupby(by).agg(Lines=("Is Returned", "size"), Returned=("Is Returned", "sum")).reset_index()
            g["Return Rate %"] = g["Returned"] / g["Lines"] * 100
            return g.sort_values("Return Rate %", ascending=False)

        c1, c2 = st.columns(2)
        reg_by = ["Region", "Regional Manager"] if "Regional Manager" in df else ["Region"]
        r = rr(reg_by)
        r["Label"] = r[reg_by].astype(str).agg(" - ".join, axis=1)
        c1.plotly_chart(style(px.bar(r, x="Label", y="Return Rate %", text_auto=".1f", title="Return Rate by Region (Manager)",
                                     labels={"Label": ""}, color_discrete_sequence=[PAL[0]])), width="stretch", theme=None)
        c2.plotly_chart(style(px.bar(rr("Category"), x="Category", y="Return Rate %", text_auto=".1f",
                                     title="Return Rate by Category", color_discrete_sequence=[PAL[1]])),
                        width="stretch", theme=None)
        c1, c2 = st.columns(2)
        s = rr("Sub-Category").sort_values("Return Rate %")
        c1.plotly_chart(style(px.bar(s, x="Return Rate %", y="Sub-Category", orientation="h",
                                     title="Return Rate by Sub-Category", color_discrete_sequence=[PAL[0]]), 480),
                        width="stretch", theme=None)
        mr = df.groupby("Month").agg(Lines=("Is Returned", "size"), Returned=("Is Returned", "sum")).reset_index()
        mr["Return Rate %"] = mr["Returned"] / mr["Lines"] * 100
        c2.plotly_chart(style(px.line(mr, x="Month", y="Return Rate %", title="Monthly Return Rate", markers=True,
                                      color_discrete_sequence=[PAL[0]]), 480), width="stretch", theme=None)
        top = r.iloc[0]
        insight(f"Highest return rate: <b>{top['Label']}</b> ({top['Return Rate %']:.1f}% of order lines) "
                f"vs {df['Is Returned'].mean() * 100:.1f}% overall")
        st.markdown("#### Returned orders")
        ro = df[df["Is Returned"] == 1].groupby(["Order ID", "Order Date", "Customer Name", "Region"]).agg(
            Sales=("Sales", "sum"), Profit=("Profit", "sum"), Lines=("Sales", "size")).reset_index()
        st.dataframe(ro.sort_values("Sales", ascending=False).style.format(
            {"Sales": "${:,.0f}", "Profit": "${:,.0f}", "Order Date": "{:%Y-%m-%d}"}), width="stretch", height=300)

# ---------------- Prediction Models ----------------
with tabs[4]:
    @st.cache_data
    def run_models(d):
        num = ["Sales", "Quantity", "Discount", "Sales x Discount"]
        cat_ = ["Category", "Sub-Category", "Region", "Segment", "Ship Mode"]
        pre = ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(handle_unknown="ignore"), cat_)])
        X = d[num + cat_]
        out = {}
        Xtr, Xte, ytr, yte = train_test_split(X, d["Profit"], test_size=0.2, random_state=42)
        p = Pipeline([("p", pre), ("m", LinearRegression())]).fit(Xtr, ytr).predict(Xte)
        out["reg"] = (r2_score(yte, p), mean_absolute_error(yte, p), yte.values, p)
        Xtr, Xte, ytr, yte = train_test_split(X, d["Is Loss"], test_size=0.2, random_state=42, stratify=d["Is Loss"])
        out["clf"] = {}
        for name, m in [("Logistic Regression", LogisticRegression(max_iter=1000)),
                        ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42))]:
            p = Pipeline([("p", pre), ("m", m)]).fit(Xtr, ytr).predict(Xte)
            out["clf"][name] = (accuracy_score(yte, p), precision_score(yte, p), recall_score(yte, p),
                                f1_score(yte, p), confusion_matrix(yte, p))
        return out

    st.caption("Models are trained on the full cleaned dataset (filters do not apply here).")
    res = run_models(data)

    st.markdown("#### Regression: predict order-line profit (Linear Regression)")
    r2, mae, ya, yp = res["reg"]
    kpi_row([("R² (test)", f"{r2:.2f}"), ("MAE", f"${mae:,.2f}"), ("Test rows", f"{len(ya):,}")])
    fig = px.scatter(x=ya, y=yp, labels={"x": "Actual profit", "y": "Predicted profit"}, opacity=0.4,
                     title="Actual vs Predicted Profit", color_discrete_sequence=[PAL[0]])
    fig.add_shape(type="line", x0=ya.min(), y0=ya.min(), x1=ya.max(), y1=ya.max(), line=dict(color=PAL[2], dash="dash"))
    st.plotly_chart(style(fig, 400), width="stretch", theme=None)

    st.markdown("#### Classification: will an order line make a loss?")
    rows = {n: dict(zip(["Accuracy", "Precision", "Recall", "F1"], v[:4])) for n, v in res["clf"].items()}
    st.dataframe(pd.DataFrame(rows).T.style.format("{:.3f}"), width="stretch")
    cols = st.columns(2)
    for col, (name, v) in zip(cols, res["clf"].items()):
        fig = px.imshow(v[4], text_auto=True, color_continuous_scale="Blues", x=["Profit", "Loss"], y=["Profit", "Loss"],
                        labels=dict(x="Predicted", y="Actual"), title=f"Confusion Matrix - {name}")
        col.plotly_chart(style(fig.update_coloraxes(showscale=False), 360), width="stretch", theme=None)

    st.markdown("#### Forecast: monthly sales (trend + seasonality regression)")
    ms = data.set_index("Order Date").resample("MS")["Sales"].sum()
    Xm = pd.get_dummies(pd.DataFrame({"t": np.arange(len(ms)), "m": ms.index.month}), columns=["m"],
                        drop_first=True).astype(float)
    horizon = st.slider("Months to forecast", 3, 12, 6)
    lr = LinearRegression().fit(Xm, ms.values)
    fidx = pd.date_range(ms.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
    Xf = pd.get_dummies(pd.DataFrame({"t": np.arange(len(ms), len(ms) + horizon), "m": fidx.month}),
                        columns=["m"]).reindex(columns=Xm.columns, fill_value=0).astype(float)
    fig = go.Figure([go.Scatter(x=ms.index, y=ms.values, name="Actual", line=dict(color=PAL[0])),
                     go.Scatter(x=fidx, y=lr.predict(Xf), name="Forecast", line=dict(color=PAL[2], dash="dash", width=3))])
    st.plotly_chart(style(fig.update_layout(title="Monthly Sales Forecast", yaxis_title="USD")), width="stretch", theme=None)
    st.caption("Simple model on four years of sample data - treat the forecast as indicative only.")

# ---------------- Data & Cleaning ----------------
with tabs[5]:
    st.markdown("#### Data quality report")
    st.dataframe(pd.Series(clean_report, name="Count").to_frame(), width="stretch")
    c1, c2 = st.columns(2)
    c1.markdown("**Column types (raw data)**")
    c1.dataframe(pd.DataFrame({"Type": raw.dtypes.astype(str), "Missing": raw.isna().sum(),
                               "Unique": raw.nunique()}), width="stretch", height=320)
    c2.markdown("**Distribution of sales (log scale)**")
    c2.plotly_chart(style(px.histogram(np.log10(df["Sales"].clip(lower=0.01)), nbins=40,
                                       labels={"value": "log10(Sales)"}, color_discrete_sequence=[PAL[0]]), 320)
                    .update_layout(showlegend=False, margin=dict(t=10)), width="stretch", theme=None)
    st.markdown(f"**Filtered data preview** ({len(df):,} rows)")
    st.dataframe(df.head(200), width="stretch", height=300)
    st.download_button("Download filtered clean data (CSV)", df.to_csv(index=False).encode(),
                       "superstore_filtered_clean.csv", "text/csv")
