<p align="center">
  <img src="https://img.shields.io/badge/BigMart-Sales%20Forecasting-0C8A0C?style=for-the-badge&logo=python&logoColor=white" alt="BigMart Sales Forecasting">
</p>

<h1 align="center">BigMart Sales Forecasting & Customer Insights</h1>

<p align="center">
  <strong>End-to-end Machine Learning pipeline for retail sales forecasting, customer segmentation, and automated business decision-making — with an interactive Streamlit dashboard.</strong>
</p>

<p align="center">
  <a href="#project-overview">Overview</a>
  |
  <a href="#architecture">Architecture</a>
  |
  <a href="#results">Results</a>
  |
  <a href="#quick-start">Quick Start</a>
  |
  <a href="#dashboard">Dashboard</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Scikit-learn" src="https://img.shields.io/badge/Scikit--learn-ML%20Model-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
</p>

<p align="center">
  <img alt="Model" src="https://img.shields.io/badge/Model-Random%20Forest-0C8A0C?style=flat-square">
  <img alt="R2 Score" src="https://img.shields.io/badge/R%C2%B2%20Score-0.9425-2563eb?style=flat-square">
  <img alt="Clustering" src="https://img.shields.io/badge/Clustering-KMeans-7c3aed?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/Status-Complete-00b894?style=flat-square">
</p>

---

<table>
  <tr>
    <td width="55%">
      <h2>Predict Sales. Understand Customers. Make Smart Decisions.</h2>
      <p>
        BigMart Sales Forecasting is a complete data science project that takes raw retail data,
        learns from 20 months of sales history, and predicts future sales with 94% accuracy.
        It also segments customers into actionable groups and gives an automated business decision plan.
      </p>
      <p>
        This repository combines data preprocessing, EDA, KMeans clustering, a Random Forest forecasting model,
        an IF-ELSE decision engine, and an interactive Streamlit web dashboard — all in one pipeline.
      </p>
    </td>
    <td width="45%">
      <table>
        <tr><td><strong>Forecasting</strong></td><td>Random Forest Regressor (R² = 0.94)</td></tr>
        <tr><td><strong>Segmentation</strong></td><td>KMeans Clustering (4 segments)</td></tr>
        <tr><td><strong>Decisions</strong></td><td>IF-ELSE business logic engine</td></tr>
        <tr><td><strong>Dashboard</strong></td><td>Streamlit — localhost:8501</td></tr>
        <tr><td><strong>SQL Layer</strong></td><td>DBeaver + MySQL queries</td></tr>
        <tr><td><strong>Visualizations</strong></td><td>7 matplotlib charts + Tableau</td></tr>
      </table>
    </td>
  </tr>
</table>

---

## Project Overview

<table>
  <tr>
    <td align="center" width="25%">
      <h3>Sales Forecast</h3>
      <p>Predicts next month's sales per product category using 20 months of historical data and lag features.</p>
    </td>
    <td align="center" width="25%">
      <h3>Customer Segments</h3>
      <p>Groups 2,500 customers into Champions, Loyal, At-Risk, and New using RFM analysis + KMeans.</p>
    </td>
    <td align="center" width="25%">
      <h3>Decision System</h3>
      <p>Converts predictions into exact action plans — stock to order, discount to apply, marketing budget to set.</p>
    </td>
    <td align="center" width="25%">
      <h3>Live Dashboard</h3>
      <p>Streamlit app where any retail company can enter data and get forecasts, recommendations, and decisions instantly.</p>
    </td>
  </tr>
</table>

---

## Architecture

mermaid
flowchart LR
  subgraph Data["Raw Data Layer"]
    CSV["9 CSV Files\n(Orders, Customers,\nProducts, Delivery...)"]
  end

  subgraph Pipeline["Python ML Pipeline"]
    DA["01 Data Analysis\nLoad + Clean + Merge"]
    CS["03 Customer Segmentation\nKMeans RFM Clustering"]
    VZ["02 Visualizations\n7 Matplotlib Charts"]
    SF["04 Sales Forecast\nRandom Forest R²=0.94"]
    DS["05 Decision System\nIF-ELSE Business Logic"]
  end

  subgraph Output["Output Layer"]
    DB["Streamlit Dashboard\nlocalhost:8501"]
    SQL["DBeaver SQL Queries\nBusiness Reports"]
    TAB["Tableau\nExecutive Charts"]
  end

  CSV --> DA
  DA --> CS
  DA --> VZ
  CS --> VZ
  DA --> SF
  SF --> DS
  DS --> DB
  VZ --> TAB
  DA --> SQL


### Pipeline Run Order

text
01_data_analysis.py          -> load, clean, merge all 9 CSV files
03_customer_segmentation.py  -> KMeans clustering on RFM features
02_visualizations.py         -> generate 7 PNG chart files
04_sales_forecast.py         -> train Random Forest, save model.pkl
05_decision_system.py        -> test decision engine with scenarios
06_streamlit_app.py          -> launch interactive dashboard


---

## Dataset

9 CSV files from a Blinkit (quick commerce) dataset:

| File | Rows | Key Columns |
|---|---|---|
| blinkit_orders.csv | 5,000 | order_id, customer_id, order_date, order_total |
| blinkit_customers.csv | 2,500 | customer_id, segment, area, avg_order_value |
| blinkit_products.csv | 268 | product_id, category, brand, price, mrp |
| blinkit_order_items.csv | 5,000 | order_id, product_id, quantity, unit_price |
| blinkit_delivery_performance.csv | 5,000 | delivery_status, delivery_time_minutes |
| blinkit_marketing_performance.csv | 5,400 | campaign, spend, revenue, roas |
| blinkit_customer_feedback.csv | 5,000 | rating, sentiment, feedback_text |
| blinkit_inventoryNew.csv | 18,105 | product_id, stock_received, damaged_stock |

*Date range:* March 2023 – November 2024 (20 months)

---

## Key Results

| Metric | Value |
|---|---|
| Total Revenue Analysed | Rs. 49,72,415 |
| Total Orders | 5,000 |
| Total Customers | 2,500 |
| Best Category | Dairy & Breakfast |
| ML Model | Random Forest Regressor |
| R² Score | *0.9425* (94% accuracy) |
| MAE | *2.92 units* per month |
| On-Time Delivery | 69.4% |
| Avg Customer Rating | 3.34 / 5 |

---

## Machine Learning Model

### Features Used (Inputs)

| Feature | Description |
|---|---|
| year, month, quarter | Time period of forecast |
| category_enc | Product category encoded as number |
| lag_1_qty | Units sold last month |
| lag_2_qty | Units sold 2 months ago |
| lag_3_qty | Units sold 3 months ago |
| rolling_3m | 3-month rolling average |
| num_orders | Number of orders placed |
| avg_price | Average selling price |

### Target (Output)

total_qty — Units to sell next month per category

### Model Performance

text
Algorithm   : Random Forest Regressor (200 trees, max_depth=10)
R² Score    : 0.9425   (94.25% of variation explained)
MAE         : 2.92     (average error of ~3 units per month)
RMSE        : 3.61
Cross-Val   : 5-fold, consistent performance


---

## Customer Segmentation

RFM (Recency, Frequency, Monetary) features built for 2,500 customers, then grouped using KMeans (K=4):

| Segment | Count | Avg Spend | Avg Orders | Strategy |
|---|---|---|---|---|
| Champions | 517 | Rs.4,520 | 4.0 | VIP offers, referral program |
| Loyal Customers | 439 | Rs.2,760 | 1.6 | Loyalty points, cross-sell |
| At-Risk Customers | 765 | Rs.1,380 | 2.1 | Win-back campaign |
| New Customers | 451 | Rs.818 | 1.4 | Onboarding discount |

---

## Decision System

Takes the ML prediction and outputs a complete business action plan:

text
INPUT:  Category = Dairy & Breakfast
        Predicted Sales = 520 units
        Current Stock   = 300 units
        Avg Price       = Rs.120
        Cost Price      = Rs.70

OUTPUT:
  STOCK    -> ORDER 272 MORE UNITS immediately (URGENT)
  PRICING  -> NO DISCOUNT — high demand, charge full price
  DELIVERY -> MONITOR DELIVERY — 88% on-time, slightly below target
  MARKETING-> MAINTAIN BUDGET — ROAS 3.5x is acceptable
  PROFIT   -> Expected Rs.26,000 | Revenue change +73.3%


The system also detects Indian festivals automatically (Diwali +40%, Holi +25%, etc.) and adjusts predictions accordingly.

---

## Project Structure

text
files3/
  |
  |-- data/                          <- all 9 CSV files
  |     |-- blinkit_orders.csv
  |     |-- blinkit_customers.csv
  |     |-- blinkit_products.csv
  |     |-- blinkit_order_items.csv
  |     |-- blinkit_delivery_performance.csv
  |     |-- blinkit_marketing_performance.csv
  |     |-- blinkit_customer_feedback.csv
  |     |-- blinkit_inventory.csv
  |     +-- blinkit_inventoryNew.csv
  |
  |-- models/                        <- trained ML model
  |     +-- sales_forecast_model.pkl
  |
  |-- outputs/                       <- generated CSV outputs
  |     |-- full_merged_data.csv
  |     |-- customer_segments.csv
  |     +-- monthly_sales_features.csv
  |
  |-- charts/                        <- 7 PNG chart images
  |     |-- chart1_monthly_revenue.png
  |     |-- chart2_category_revenue.png
  |     |-- chart3_customer_segments.png
  |     |-- chart4_delivery_sentiment.png
  |     |-- chart5_top_products.png
  |     |-- chart6_payment_methods.png
  |     +-- chart7_customer_clusters.png
  |
  |-- 01_data_analysis.py
  |-- 02_visualizations.py
  |-- 03_customer_segmentation.py
  |-- 04_sales_forecast.py
  |-- 05_decision_system.py
  |-- 06_streamlit_app.py            <- interactive dashboard
  |-- 01_create_tables.sql
  |-- 02_analysis_queries.sql
  +-- requirements.txt


---

## Tech Stack

<table>
  <tr>
    <td><strong>Language</strong></td>
    <td>Python 3.10+</td>
  </tr>
  <tr>
    <td><strong>Data Analysis</strong></td>
    <td>Pandas, NumPy</td>
  </tr>
  <tr>
    <td><strong>Machine Learning</strong></td>
    <td>Scikit-learn (Random Forest, KMeans, StandardScaler, LabelEncoder)</td>
  </tr>
  <tr>
    <td><strong>Visualization</strong></td>
    <td>Matplotlib, Seaborn, Tableau</td>
  </tr>
  <tr>
    <td><strong>Dashboard</strong></td>
    <td>Streamlit</td>
  </tr>
  <tr>
    <td><strong>Database</strong></td>
    <td>MySQL / DBeaver (SQLite compatible)</td>
  </tr>
  <tr>
    <td><strong>Tools</strong></td>
    <td>VS Code, Jupyter Notebook, Git</td>
  </tr>
</table>

---

## Quick Start

### 1. Clone the repository

bash
git clone https://github.com/jaydarji2101/BigMart-Sales-Prediction.git
cd BigMart-Sales-Prediction


### 2. Install dependencies

powershell
pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly openpyxl


### 3. Create required folders

powershell
New-Item -ItemType Directory -Force -Path "outputs"
New-Item -ItemType Directory -Force -Path "charts"
New-Item -ItemType Directory -Force -Path "models"


### 4. Add your CSV files

Place all 9 CSV files inside the data/ folder.

### 5. Run the full pipeline

powershell
python 01_data_analysis.py
python 03_customer_segmentation.py
python 02_visualizations.py
python 04_sales_forecast.py
python 05_decision_system.py


### 6. Launch the dashboard

powershell
streamlit run 06_streamlit_app.py


Open browser at:

text
http://localhost:8501


---

## Dashboard

The Streamlit dashboard supports:

- *Any retail company* — type your company name and it updates everywhere
- *Business type* — Online (Blinkit/Zepto), Offline (DMart/BigMart), or Both
- *Full-date forecast* — predicts sales by 15th of the month AND end of month
- *Festival detection* — auto-detects Diwali, Holi, Navratri and boosts demand prediction
- *Channel split* — separate forecasts for online orders vs in-store sales
- *Dataset folder* — type any folder path to load and preview your own CSV files
- *Recommendations* — white text on colored backgrounds (always readable)
- *Decision system* — exact stock units, discount %, marketing budget, expected profit

---

## SQL Analysis

12 business queries in 02_analysis_queries.sql covering:

sql
-- Total revenue
SELECT SUM(order_total) FROM orders;

-- Monthly sales trend
SELECT DATE_FORMAT(order_date, '%Y-%m'), SUM(order_total)
FROM orders GROUP BY 1 ORDER BY 1;

-- Top products by revenue
SELECT product_name, SUM(quantity * unit_price) AS revenue
FROM order_items JOIN products USING(product_id)
GROUP BY product_name ORDER BY revenue DESC LIMIT 10;

-- Customer segment analysis
SELECT customer_segment, COUNT(*), AVG(avg_order_value)
FROM customers GROUP BY customer_segment;


---

## Roadmap

mermaid
flowchart TB
  A["Current Version"] --> B["Time-Series Forecasting\nProphet / ARIMA"]
  B --> C["Real-Time Data\nAPI Integration"]
  C --> D["Geospatial Analysis\nDelivery Zone Heatmaps"]
  D --> E["Recommendation Engine\nCollaborative Filtering"]
  E --> F["Cloud Deployment\nStreamlit Cloud / AWS"]


Planned upgrades:

- Add Prophet or ARIMA for true time-series forecasting.
- Integrate real-time sales data through REST API.
- Add geospatial delivery heatmaps by pincode.
- Build a product recommendation engine using collaborative filtering.
- Deploy dashboard to Streamlit Cloud or AWS.
- Add Power BI integration for enterprise reporting.

---

## Command Center

powershell
# Full pipeline
python 01_data_analysis.py
python 03_customer_segmentation.py
python 02_visualizations.py
python 04_sales_forecast.py
python 05_decision_system.py
streamlit run 06_streamlit_app.py

# Streamlit only (if model already trained)
streamlit run 06_streamlit_app.py


---

## Author

*Jay Darji*
B.E. Information Technology — Apollo Institute of Engineering and Technology, GTU

- GitHub: [jaydarji2101](https://github.com/jaydarji2101)
- LinkedIn: [linkedin.com/in/jay-darjii](https://linkedin.com/in/jay-darjii)
- Email: darjijay2101@gmail.com

---

<p align="center">
  <strong>BigMart Sales Forecasting turns raw retail data into a complete business intelligence system — forecast, segment, decide, and act.</strong>
</p>

<p align="center">
  Built as a Data Science internship project at Fingertips DIS Pvt. Ltd. (Jan 2026 – Apr 2026)
</p>