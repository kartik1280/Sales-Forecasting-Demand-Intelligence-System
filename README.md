# Sales-Forecasting-Demand-Intelligence-System

## 🚀 Live Dashboard

**[View the interactive Streamlit dashboard →](https://sales-forecasting-demand-intelligence-system-kaniep4e7skwyl3kj.streamlit.app/)**

End-to-end analysis of retail sales data covering exploratory data analysis, time series forecasting, anomaly detection, and product demand segmentation — built to support supply chain and inventory planning decisions.

## 📊 Project Overview

This project analyzes four years (2015–2018) of retail order data to answer key business questions:
- Which product categories and regions drive the most revenue?
- What seasonal patterns exist in sales?
- Can we accurately forecast sales for the next 3 months?
- Which weeks show unusual (anomalous) sales behavior?
- How should products be grouped for smarter inventory management?

## 🗂️ Dataset

- **Source:** `train.csv` (Retail Superstore dataset — United States)
- **Size:** 9,800 rows × 18 columns
- **Key fields:** Order Date, Ship Date, Region, Category, Sub-Category, Product Name, Sales

## 🧭 Project Structure (Tasks)

| Task | Description |
|------|-------------|
| Task 1 | Data cleaning, feature engineering, and exploratory data analysis (EDA) |
| Task 2 | Time series decomposition (trend, seasonality, residuals) + stationarity testing (ADF) |
| Task 3 | Sales forecasting using SARIMA, Prophet, and XGBoost — model comparison |
| Task 4 | Category- and region-level forecasting |
| Task 5 | Anomaly detection using Isolation Forest and rolling Z-score |
| Task 6 | Product demand segmentation via K-Means clustering + PCA |
| Task 7 | Streamlit dashboard for interactive exploration |
| Task 8 | Executive business report (`summary.docx`) for non-technical stakeholders |

## 🔍 Key Findings

- **Top revenue category:** Technology ($827,456), followed by Furniture ($728,659) and Office Supplies ($705,422)
- **Most consistent regional growth:** East region ($127,653 in 2015 → $210,129 in 2018)
- **Average shipping time:** ~3.96 days, consistent across all regions
- **Seasonality:** November is the strongest sales month every year; January/February are consistently the weakest
- **Best forecasting model:** Prophet (MAPE 14.64%), outperforming SARIMA (15.49%) and XGBoost (15.36%)
- **Anomalies detected:** 11 unusual weeks via Isolation Forest, largely concentrated around November–December holiday demand
- **Product segments:** 3 clusters identified — Premium High-Value, High-Volume Core, and Stable Lower-Volume products

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Forecasting:** statsmodels (SARIMA), Prophet, XGBoost
- **Anomaly detection:** scikit-learn (Isolation Forest), rolling Z-score
- **Clustering:** scikit-learn (K-Means, PCA, StandardScaler)
- **Dashboard:** Streamlit
- **Reporting:** python-docx / Word

## 📁 Repository Contents

├── train.csv                # Raw dataset
├── analysis.ipynb            # Full analysis notebook (Tasks 1–7)
├── anomalies.csv             # Exported anomaly records
├── summary.docx               # Executive business report (Task 8)
├── app.py                     # Streamlit dashboard (if included)
└── README.md

## ▶️ How to Run

1. Clone the repo and install dependencies:
```bash
   pip install pandas numpy matplotlib seaborn statsmodels prophet xgboost scikit-learn streamlit
```
2. Open and run the notebook:
```bash
   jupyter notebook analysis.ipynb
```
3. (Optional) Launch the dashboard:
```bash
   streamlit run app.py
```

## 📄 Executive Report

A non-technical, stakeholder-ready summary (`summary.docx`) is included, covering the 3-month sales forecast, top anomalies, product segmentation, and business recommendations — written for supply chain and finance leadership.

## ⚠️ Limitations

Forecasts are based on 48 months of aggregated data, a relatively small sample for statistical time series modeling. Model accuracy is approximately ±15% (MAPE), and unusual one-off sales events can affect forecast reliability. Forecasts should be reviewed monthly and used to support — not replace — business judgment.

## 👤 Author

Kartik Sharma
