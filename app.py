# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from datetime import datetime

st.set_page_config(page_title="Digital Health & Fitness Analyzer", layout="wide")

# ---------------------------
# Simple in-memory auth (demo)
# ---------------------------
if "users" not in st.session_state:
    # demo user: username: demo@example.com, password: demo123
    st.session_state.users = {"demo@example.com": "demo123"}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

def show_login():
    st.sidebar.title("Welcome — DHFA")
    st.sidebar.write("Sign in to access your dashboard")
    email = st.sidebar.text_input("Email")
    pwd = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if email in st.session_state.users and st.session_state.users[email] == pwd:
            st.session_state.logged_in = True
            st.session_state.username = email
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid credentials (demo user: demo@example.com / demo123)")
    st.sidebar.markdown("---")
    st.sidebar.write("New? Create demo account below")
    new_email = st.sidebar.text_input("New email", key="new_email")
    new_pwd = st.sidebar.text_input("New password", type="password", key="new_pwd")
    if st.sidebar.button("Register"):
        if new_email and new_pwd:
            if new_email in st.session_state.users:
                st.sidebar.error("User exists")
            else:
                st.session_state.users[new_email] = new_pwd
                st.sidebar.success("Account created. Use login form above.")
        else:
            st.sidebar.error("Enter email and password")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.experimental_rerun()

# ---------------------------
# Theme toggle (light/dark)
# ---------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

with st.sidebar:
    st.write("Theme")
    if st.button("Toggle Light/Dark"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.write(f"Current: *{st.session_state.theme}*")
    st.markdown("---")
    if st.session_state.logged_in:
        st.write("Signed in as:")
        st.write(st.session_state.username)
        if st.button("Logout"):
            logout()

# Add small CSS for glassy look (subtle)
if st.session_state.theme == "dark":
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg,#07101a,#072a2f); color: #e6f6f5; }
        .card { background: rgba(255,255,255,0.03); padding: 16px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg,#f4fcff,#f0fff4); color: #08304b; }
        .card { background: rgba(255,255,255,0.75); padding: 16px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Show login if not logged in
# ---------------------------
if not st.session_state.logged_in:
    show_login()
    st.stop()

# ---------------------------
# Main App
# ---------------------------
st.markdown("<h1 style='text-align: left;'>Digital Health & Fitness Analyzer</h1>", unsafe_allow_html=True)
st.markdown(f"*User:* {st.session_state.username}    •    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

col1, col2 = st.columns([2,1])

with col2:
    st.markdown("### Quick actions")
    uploaded_history = st.session_state.get("uploads", [])
    if uploaded_history:
        st.write("Recent uploads:")
        for i, info in enumerate(reversed(uploaded_history[-5:])):
            st.write(f"- {info['name']} ({info['rows']} rows)")
    else:
        st.write("No uploads yet.")
    st.markdown("---")
    st.write("Download sample CSV if you need one:")
    sample_csv = "date,steps,calories,heart_rate,sleep_hours,weight\n2025-01-01,4000,180,70,6.5,62\n2025-01-02,6500,250,75,7,61.5\n2025-01-03,3000,150,68,5.5,62\n"
    st.download_button("Download sample.csv", data=sample_csv, file_name="sample_fitness.csv", mime="text/csv")

with col1:
    st.markdown("### Upload your CSV (columns like: date, steps, calories, heart_rate, sleep_hours, weight)")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV to start analysis (or download the sample CSV).")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # store upload history in session_state
    if "uploads" not in st.session_state:
        st.session_state.uploads = []
    st.session_state.uploads.append({"name": getattr(uploaded_file, "name", "uploaded.csv"), "rows": df.shape[0], "time": datetime.now().isoformat()})

    # Basic cleaning and column detection
    st.markdown("#### Data Preview")
    st.dataframe(df.head(50))

    # detect datetime column
    date_col = None
    for c in df.columns:
        sample = df[c].astype(str).iloc[0]
        try:
            pd.to_datetime(sample)
            date_col = c
            break
        except Exception:
            continue

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    st.markdown("### Quick Summary")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Rows", df.shape[0])
    colB.metric("Columns", df.shape[1])
    colC.metric("Missing values", int(df.isnull().sum().sum()))
    colD.metric("Numeric cols", len(numeric_cols))

    # Description & basic stats
    st.markdown("#### Numeric column statistics")
    if len(numeric_cols):
        desc = df[numeric_cols].describe().T
        st.dataframe(desc)
    else:
        st.info("No numeric columns detected for stats.")

    # Insights (rule-based)
    st.markdown("### Automated Insights")
    insights = []
    # missing value insights
    for c in df.columns:
        miss_pct = df[c].isnull().mean()
        if miss_pct > 0.2:
            insights.append(f"Column *{c}* has high missing rate: {miss_pct:.0%}")
    # unique value single
    for c in df.columns:
        if df[c].nunique() <= 1:
            insights.append(f"Column *{c}* has only {df[c].nunique()} unique value(s).")
    # correlation
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        # find strong correlations
        strong_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                a = corr.columns[i]
                b = corr.columns[j]
                val = corr.loc[a,b]
                if abs(val) > 0.75:
                    strong_pairs.append((a,b,val))
        for a,b,val in strong_pairs:
            insights.append(f"Strong correlation between *{a}* and *{b}* (r = {val:.2f})")
    if not insights:
        st.write("No notable automated insights found. Upload more varied data for richer insights.")
    else:
        for ins in insights:
            st.write("• " + ins)

    # Visualization area
    st.markdown("---")
    st.markdown("## Visualizations")

    # 1) If date column found, show time-series for numeric cols
    if date_col:
        st.markdown(f"*Time series detected using column:* {date_col}")
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            ts_col = st.selectbox("Choose numeric column to show over time", options=numeric_cols, index=0 if numeric_cols else None)
            if ts_col:
                fig, ax = plt.subplots(figsize=(10,4))
                sns.lineplot(data=df.sort_values(date_col), x=date_col, y=ts_col, ax=ax)
                ax.set_title(f"{ts_col} over time")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error plotting time series: {e}")

    # 2) Histograms for numeric columns
    if numeric_cols:
        st.markdown("### Distribution (Histogram)")
        pick = st.selectbox("Choose numeric for histogram", options=numeric_cols, key="hist")
        if pick:
            fig, ax = plt.subplots(figsize=(6,3))
            sns.histplot(df[pick].dropna(), bins=20, kde=True, ax=ax)
            ax.set_title(f"Distribution of {pick}")
            st.pyplot(fig)

    # 3) Scatter: choose two numeric columns
    if len(numeric_cols) >= 2:
        st.markdown("### Scatter — Relationship between two numeric columns")
        x_col = st.selectbox("X axis", options=numeric_cols, key="sx")
        y_col = st.selectbox("Y axis", options=numeric_cols, key="sy")
        if x_col and y_col:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.set_title(f"{y_col} vs {x_col}")
            st.pyplot(fig)

    # 4) Categorical top counts
    if categorical_cols:
        st.markdown("### Top categories (for categorical cols)")
        cat = st.selectbox("Choose categorical column", options=categorical_cols, key="cat")
        if cat:
            vc = df[cat].fillna("(blank)").value_counts().head(20)
            fig, ax = plt.subplots(figsize=(6,3))
            sns.barplot(x=vc.values, y=vc.index, ax=ax)
            ax.set_title(f"Top categories — {cat}")
            st.pyplot(fig)

    # 5) Correlation Heatmap
    if len(numeric_cols) >= 2:
        st.markdown("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="vlag", ax=ax)
        st.pyplot(fig)

    # ---------------------------
    # PDF Report Generation
    # ---------------------------
    st.markdown("---")
    st.markdown("## Generate PDF Report")
    report_name = st.text_input("Report filename", value=f"DHFA_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
    include_plots = st.checkbox("Include plots in PDF", value=True)
    if st.button("Generate & Download PDF"):
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elems = []
            elems.append(Paragraph("Digital Health & Fitness Analyzer — Report", styles['Title']))
            elems.append(Paragraph(f"User: {st.session_state.username}", styles['Normal']))
            elems.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            elems.append(Spacer(1,12))

            # Summary table
            elems.append(Paragraph("Summary Statistics (numeric columns)", styles['Heading2']))
            if len(numeric_cols):
                d = df[numeric_cols].describe().round(3)
                data_tbl = [ ["Column"] + d.columns.tolist() ]
                # transpose style table: limited to few rows
                # we'll show count, mean, std, min, max for each numeric column
                data_for_table = [["Metric"] + d.index.tolist()]
                # convert columns into values per metric
                # Better: make a simple table with columns: column, count, mean, std, min, max
                tbl = [["Column", "count", "mean", "std", "min", "max"]]
                for col in numeric_cols:
                    row = [col,
                           f"{d.loc['count', col]:.0f}",
                           f"{d.loc['mean', col]:.3f}",
                           f"{d.loc['std', col]:.3f}",
                           f"{d.loc['min', col]:.3f}",
                           f"{d.loc['max', col]:.3f}"]
                    tbl.append(row)
                elems.append(Table(tbl))
            else:
                elems.append(Paragraph("No numeric columns found to summarize.", styles['Normal']))

            elems.append(Spacer(1,12))
            elems.append(Paragraph("Automated Insights", styles['Heading2']))
            if insights:
                for ins in insights:
                    elems.append(Paragraph("- " + ins, styles['Normal']))
            else:
                elems.append(Paragraph("No notable insights detected.", styles['Normal']))

            # Include plots: save current matplotlib figs to temp images
            if include_plots:
                # create and save a couple of representative plots
                try:
                    # 1) histogram of first numeric col
                    if numeric_cols:
                        fig1, ax1 = plt.subplots(figsize=(6,3))
                        sns.histplot(df[numeric_cols[0]].dropna(), bins=20, kde=True, ax=ax1)
                        ax1.set_title(f"Distribution of {numeric_cols[0]}")
                        img_buf = io.BytesIO()
                        fig1.tight_layout()
                        fig1.savefig(img_buf, format='PNG')
                        plt.close(fig1)
                        img_buf.seek(0)
                        elems.append(Spacer(1,12))
                        elems.append(Image(img_buf, width=450, height=200))
                    # 2) correlation heatmap
                    if len(numeric_cols) >= 2:
                        fig2, ax2 = plt.subplots(figsize=(6,4))
                        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="vlag", ax=ax2)
                        fig2.tight_layout()
                        img_buf2 = io.BytesIO()
                        fig2.savefig(img_buf2, format='PNG')
                        plt.close(fig2)
                        img_buf2.seek(0)
                        elems.append(Spacer(1,12))
                        elems.append(Image(img_buf2, width=450, height=300))
                except Exception as e:
                    elems.append(Paragraph("Error creating plot images: " + str(e), styles['Normal']))

            doc.build(elems)
            buffer.seek(0)
            st.download_button("Download Report (PDF)", data=buffer, file_name=report_name, mime="application/pdf")
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")

    st.markdown("---")
    st.markdown("## Notes & Next Steps")
    st.write("""
    • This is a single-file Streamlit demo designed to be fast to run and robust.  
    • For production: move auth to a secure DB, store uploads, and use Plotly/kaleido for interactive images.  
    • To add scheduled reports or user storage, we can add a small FastAPI backend and Postgres.
    """)