import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration first
st.set_page_config(
    page_title="Demand Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    # Simulated structure based on your data imports
    df = pd.read_csv("train_processed.csv", parse_dates=["Order Date", "Ship Date"])
    monthly_sales = pd.read_csv("monthly_sales.csv", parse_dates=["Order Date"])
    weekly_sales = pd.read_csv("weekly_sales.csv", parse_dates=["Order Date"])
    forecast_df = pd.read_csv("forecast.csv", parse_dates=["Order Date"])
    comparison_df = pd.read_csv("comparison_prophet.csv")
    metrics_df = pd.read_csv("metrics.csv")
    anomalies = pd.read_csv("anomalies.csv", parse_dates=["Order Date"])
    cluster_df = pd.read_csv("cluster_results.csv")
    
    return (df, monthly_sales, weekly_sales, forecast_df, comparison_df, metrics_df, anomalies, cluster_df)

df, monthly_sales, weekly_sales, forecast_df, comparison_df, metrics_df, anomalies, cluster_df = load_data()

# ---------------- Summary Data ---------------- #
total_sales = df["Sales"].sum()
total_orders = df["Order ID"].nunique()
avg_order_value = df["Sales"].mean()
total_customers = df["Customer ID"].nunique()

# ---------------- Sidebar Navigation ---------------- #
with st.sidebar:
    st.markdown("## ⚙️ Navigation")
    page = st.selectbox(
        "Select Dashboard Page",
        ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Demand Segments"]
    )
    st.markdown("---")
    st.markdown("💡 *Tip: Use filters inside pages to slice data dynamically.*")

# ---------------- SALES OVERVIEW ---------------- #
def sales_overview():
    st.title("📊 Sales Overview")
    st.caption("Gain a bird's-eye view of high-level revenue performance, regional segments, and product trends.")
    st.write("")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Sales", f"${total_sales:,.0f}")
    with col2:
        st.metric("🛒 Orders Placed", f"{total_orders:,}")
    with col3:
        st.metric("📦 Avg Order Value", f"${avg_order_value:,.2f}")
    with col4:
        st.metric("👥 Active Customers", f"{total_customers:,}")

    st.markdown("---")

    # Filters Section in an Expandable Container
    with st.expander("🔍 Filter View Configuration", expanded=True):
        f_col1, f_col2 = st.columns(2)
        selected_year = f_col1.selectbox("Select Year", sorted(df["Year"].unique()))
        selected_region = f_col2.selectbox("Select Region", ["All"] + sorted(df["Region"].unique().tolist()))

    filtered_df = df[df["Year"] == selected_year]
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]

    # Data Preview
    st.markdown("### 📋 Dataset Preview")
    st.dataframe(filtered_df.head(5), use_container_width=True)

    st.markdown("---")

    # Layout for Trends & Categories
    st.markdown("### 📈 Trends & Categorical Distribution")
    g1_col1, g1_col2 = st.columns([2, 1])

    with g1_col1:
        filtered_monthly = monthly_sales[monthly_sales["Order Date"].dt.year == selected_year]
        fig_trend = px.line(
            filtered_monthly,
            x="Order Date",
            y="Sales",
            markers=True,
            title="Monthly Sales Trend",
            color_discrete_sequence=["#1f77b4"],
            template="plotly_dark"
        )
        fig_trend.update_traces(line_shape='spline', line_width=3)
        st.plotly_chart(fig_trend, use_container_width=True)

    with g1_col2:
        category_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()
        fig_pie = px.pie(
            category_sales,
            names="Category",
            values="Sales",
            hole=0.5,
            title="Sales by Category",
            color_discrete_sequence=px.colors.sequential.Teal,
            template="plotly_dark"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # Layout for Regions & Segments
    g2_col1, g2_col2 = st.columns(2)

    with g2_col1:
        region_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()
        fig_region = px.bar(
            region_sales,
            x="Region",
            y="Sales",
            color="Region",
            text_auto=".2s",
            title="Sales by Region",
            color_discrete_sequence=px.colors.sequential.Viridis,
            template="plotly_dark"
        )
        st.plotly_chart(fig_region, use_container_width=True)

    with g2_col2:
        segment_sales = filtered_df.groupby("Segment")["Sales"].sum().reset_index()
        fig_segment = px.bar(
            segment_sales,
            x="Segment",
            y="Sales",
            color="Segment",
            text_auto=".2s",
            title="Sales by Customer Segment",
            color_discrete_sequence=px.colors.sequential.Burg,
            template="plotly_dark"
        )
        st.plotly_chart(fig_segment, use_container_width=True)

    st.markdown("---")

    # Top Products Horizontal Bar
    st.markdown("### 🏆 Top 10 Products Performance")
    top_products = filtered_df.groupby("Product Name")["Sales"].sum().sort_values(ascending=True).tail(10).reset_index()
    fig_prod = px.bar(
        top_products,
        x="Sales",
        y="Product Name",
        orientation="h",
        color="Sales",
        color_continuous_scale="teal",
        template="plotly_dark",
        title="Top 10 Revenue Generating Products"
    )
    fig_prod.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    st.plotly_chart(fig_prod, use_container_width=True)


# ---------------- FORECAST EXPLORER ---------------- #
def forecast_explorer():
    st.title("📈 Prophet Forecast Explorer")
    st.caption("Review deep-learning and statistical forecasts alongside key historical evaluation metrics.")
    st.write("")

    # Model Performance KPIs
    st.markdown("### 📊 Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"{metrics_df.loc[0,'Value']:.2f}")
    with col2:
        st.metric("Root Mean Squared Error (RMSE)", f"{metrics_df.loc[1,'Value']:.2f}")
    with col3:
        st.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics_df.loc[2,'Value']:.2f}%")

    st.markdown("---")

    # Actual vs Predicted Plotly
    comparison_df["ds"] = pd.to_datetime(comparison_df["ds"])
    fig_comp = px.line(
        comparison_df,
        x="ds",
        y=["Actual", "Predicted"],
        markers=True,
        title="Historical Backtest: Actual vs. Predicted Sales",
        color_discrete_sequence=["#2ca02c", "#ff7f0e"],
        template="plotly_dark"
    )
    fig_comp.update_traces(line_shape='spline', marker=dict(size=4))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # Confidence Interval Visual
    st.markdown("### 🔮 Future Horizon Forecast")
    fig_forecast = px.line(
        forecast_df,
        x="Order Date",
        y="Forecast",
        title="Prophet Operational Forecast with Confidence Bounds (80%)",
        color_discrete_sequence=["#00b4d8"],
        template="plotly_dark"
    )
    fig_forecast.add_scatter(
        x=forecast_df["Order Date"], y=forecast_df["Upper"],
        mode="lines", line=dict(width=0), showlegend=False
    )
    fig_forecast.add_scatter(
        x=forecast_df["Order Date"], y=forecast_df["Lower"],
        mode="lines", fill="tonexty", fillcolor="rgba(0, 180, 216, 0.15)",
        line=dict(width=0), name="Confidence Interval"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)


# ---------------- ANOMALY REPORT ---------------- #
def anomaly_report():
    st.title("🚨 Sales Anomaly Report")
    st.info(
        "Outliers and unusual spikes are flagged here using an **Isolation Forest unsupervised machine learning model**. "
        "These target areas typically imply flash promotions, bulk distribution actions, or unexpected disruptions."
    )
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Anomalies Detected", len(anomalies))
    with col2:
        st.metric("Average Outlier Transaction Value", f"${anomalies['Sales'].mean():,.0f}")

    st.markdown("---")

    # Interactive Trend Plotting with Outlier overlays
    st.markdown("### Weekly Outlier Context Timeline")
    fig_anom = px.line(
        weekly_sales,
        x="Order Date",
        y="Sales",
        title="Weekly Baseline with Anomalous Spikes Highlighted",
        color_discrete_sequence=["#cccccc"],
        template="plotly_dark"
    )
    fig_anom.add_scatter(
        x=anomalies["Order Date"],
        y=anomalies["Sales"],
        mode="markers",
        name="Anomaly Trigger",
        marker=dict(color="#ef553b", size=11, symbol="circle-open-dot", line=dict(width=2))
    )
    st.plotly_chart(fig_anom, use_container_width=True)

    st.markdown("---")

    # Raw table filtered list
    st.markdown("### Detailed Outlier Log")
    clean_anom = anomalies[["Order Date", "Sales"]].sort_values("Order Date", ascending=False)
    st.dataframe(clean_anom, use_container_width=True, height=300)


# ---------------- DEMAND SEGMENTS ---------------- #
def demand_segments():
    st.title("📦 Product Demand Segmentation")
    st.markdown(
        "Products are dynamically distributed using **K-Means clustering** across structural parameters like "
        "volume profiles, historical velocity, and demand standard deviations."
    )
    st.write("")

    # Visualizing Dimension Clusters
    st.markdown("### Dimensionality Mapping View (PCA)")
    fig_cluster = px.scatter(
        cluster_df,
        x="PCA1",
        y="PCA2",
        color="Cluster_Name",
        hover_name="Sub-Category",
        title="Product Categories Structural Clusters space",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark"
    )
    fig_cluster.update_traces(marker=dict(size=12, opacity=0.85, line=dict(width=1, color='white')))
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("---")

    # Aggregated mapping metadata list table
    st.markdown("### Segment Metadata Directory Summary")
    clean_cluster = cluster_df[["Sub-Category", "Cluster_Name", "Total_Sales", "Growth_Rate"]]
    st.dataframe(clean_cluster.sort_values(by="Total_Sales", ascending=False), use_container_width=True, height=400)


# Routing Execution Elements
if page == "Sales Overview":
    sales_overview()
elif page == "Forecast Explorer":
    forecast_explorer()
elif page == "Anomaly Report":
    anomaly_report()
elif page == "Demand Segments":
    demand_segments()