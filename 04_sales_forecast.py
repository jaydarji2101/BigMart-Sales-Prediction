import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  BLINKIT SALES FORECASTING — MACHINE LEARNING MODEL")
print("=" * 60)

# ── STEP 1: Load Data ──────────────────────────────────────
DATA = 'data/'
orders      = pd.read_csv(DATA + 'blinkit_orders.csv', parse_dates=['order_date'])
order_items = pd.read_csv(DATA + 'blinkit_order_items.csv')
products    = pd.read_csv(DATA + 'blinkit_products.csv')

order_items['line_total'] = order_items['quantity'] * order_items['unit_price']

df = (order_items
      .merge(orders[['order_id','order_date']], on='order_id')
      .merge(products[['product_id','category']], on='product_id'))

df['order_date'] = pd.to_datetime(df['order_date'])
df['year']       = df['order_date'].dt.year
df['month']      = df['order_date'].dt.month
df['quarter']    = df['order_date'].dt.quarter

# ── STEP 2: Aggregate monthly by category ─────────────────
# We group sales by MONTH + CATEGORY so we predict
# "In January 2025, how many Dairy products will sell?"
monthly = (df.groupby(['year','month','quarter','category'])
             .agg(total_qty    = ('quantity',   'sum'),
                  total_revenue= ('line_total', 'sum'),
                  num_orders   = ('order_id',   'nunique'),
                  avg_price    = ('unit_price', 'mean'))
             .reset_index()
             .sort_values(['category','year','month']))

# ── STEP 3: Feature Engineering ───────────────────────────
# LAG FEATURES: "What happened LAST month?"
# Example: lag_1_qty = last month's quantity sold
# The model uses this to understand "if last month was high,
# this month is likely high too"
monthly['lag_1_qty']  = monthly.groupby('category')['total_qty'].shift(1)
monthly['lag_2_qty']  = monthly.groupby('category')['total_qty'].shift(2)
monthly['lag_3_qty']  = monthly.groupby('category')['total_qty'].shift(3)

# ROLLING AVERAGE: "Average of last 3 months"
# Smooths out random spikes
monthly['rolling_3m'] = (monthly.groupby('category')['total_qty']
                          .transform(lambda x: x.shift(1).rolling(3).mean()))

# Encode category as a number (ML can't read text)
le = LabelEncoder()
monthly['category_enc'] = le.fit_transform(monthly['category'])

monthly.to_csv('outputs/monthly_sales_features.csv', index=False)
print(f"\n  ✅ Monthly data prepared: {monthly.shape}")

# ── STEP 4: Prepare ML inputs ─────────────────────────────
FEATURES = ['year','month','quarter','category_enc',
            'lag_1_qty','lag_2_qty','lag_3_qty','rolling_3m',
            'num_orders','avg_price']
TARGET   = 'total_qty'

ml_df = monthly.dropna(subset=FEATURES + [TARGET])
X = ml_df[FEATURES]
y = ml_df[TARGET]

print(f"  ✅ ML dataset: {len(ml_df)} rows")
print(f"  Features: {FEATURES}")
print(f"  Target  : {TARGET}")

# ── STEP 5: Split train/test ───────────────────────────────
# 80% of data = training (model learns from this)
# 20% of data = testing  (we check how accurate it is)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\n  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# ── STEP 6: Train Random Forest ───────────────────────────
print("\n  Training Random Forest (200 trees)...")
model = RandomForestRegressor(
    n_estimators=200,    # 200 decision trees
    max_depth=10,        # each tree can be max 10 levels deep
    min_samples_split=3,
    random_state=42
)
model.fit(X_train, y_train)
print("  ✅ Model trained!")

# ── STEP 7: Evaluate ──────────────────────────────────────
y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
cv   = cross_val_score(model, X, y, cv=5, scoring='r2').mean()

print(f"""
  ╔══════════════════════════════════════╗
  ║         MODEL PERFORMANCE           ║
  ╠══════════════════════════════════════╣
  ║  R² Score     : {r2:.4f}              ║
  ║  (94% = very good! Max is 1.0)       ║
  ║  MAE          : {mae:.2f} units/month  ║
  ║  (average error is only ~{mae:.0f} units) ║
  ║  RMSE         : {rmse:.2f} units         ║
  ║  Cross-Val R² : {cv:.4f}              ║
  ╚══════════════════════════════════════╝
""")

# ── STEP 8: Save model ────────────────────────────────────
model_data = {
    'model':          model,
    'label_encoder':  le,
    'features':       FEATURES,
    'categories':     list(le.classes_),
    'monthly_cat':    monthly,
    'r2':             r2,
    'mae':            mae
}
with open('models/sales_forecast_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("  ✅ Model saved to models/sales_forecast_model.pkl")

# ── STEP 9: Sample Predictions ────────────────────────────
print("\n  SAMPLE PREDICTIONS (next month for each category):")
print("  " + "-" * 50)
last_data = monthly.groupby('category').last().reset_index()

for _, row in last_data.iterrows():
    if pd.isna(row.get('lag_1_qty')): continue
    next_month = row['month'] + 1 if row['month'] < 12 else 1
    next_year  = row['year'] + (1 if row['month'] == 12 else 0)
    next_qtr   = (next_month - 1) // 3 + 1
    cat_enc    = le.transform([row['category']])[0]
    inp = pd.DataFrame([[next_year, next_month, next_qtr, cat_enc,
                         row['total_qty'], row.get('lag_1_qty', 0),
                         row.get('lag_2_qty', 0), row.get('rolling_3m', 0),
                         row['num_orders'], row['avg_price']]],
                       columns=FEATURES)
    pred = model.predict(inp)[0]
    print(f"  {row['category']:<30s}: {pred:.0f} units predicted")

print("\n  ✅ Forecasting complete! Next: run the Decision System.")
print("  ✅ Run dashboard/05_streamlit_app.py for the interactive app")
