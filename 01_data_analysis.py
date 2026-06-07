import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')                   # allows saving without screen
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── COLOUR PALETTE (Blinkit brand colours) ────────────────
GREEN  = '#0C8A0C'
YELLOW = '#F8D210'
ORANGE = '#FF6B35'
BLUE   = '#2196F3'
GREY   = '#F5F5F5'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})

# =============================================================
#  STEP 1: LOAD ALL DATA
#  (Like opening all the notebooks from your bag)
# =============================================================
print("=" * 55)
print("  STEP 1: Loading all CSV files...")
print("=" * 55)

DATA = 'data/'

orders      = pd.read_csv(DATA + 'blinkit_orders.csv',     parse_dates=['order_date'])
order_items = pd.read_csv(DATA + 'blinkit_order_items.csv')
products    = pd.read_csv(DATA + 'blinkit_products.csv')
customers   = pd.read_csv(DATA + 'blinkit_customers.csv',  parse_dates=['registration_date'])
delivery    = pd.read_csv(DATA + 'blinkit_delivery_performance.csv')
feedback    = pd.read_csv(DATA + 'blinkit_customer_feedback.csv', parse_dates=['feedback_date'])
marketing   = pd.read_csv(DATA + 'blinkit_marketing_performance.csv', parse_dates=['date'])
inventory   = pd.read_csv(DATA + 'blinkit_inventoryNew.csv')

datasets = {
    'orders': orders, 'order_items': order_items,
    'products': products, 'customers': customers,
    'delivery': delivery, 'feedback': feedback,
    'marketing': marketing, 'inventory': inventory
}
for name, df in datasets.items():
    print(f"  ✅ {name:15s}: {df.shape[0]:,} rows × {df.shape[1]} columns")

# =============================================================
#  STEP 2: CLEAN DATA
#  (Fix missing values, wrong types, etc.)
# =============================================================
print("\n" + "=" * 55)
print("  STEP 2: Checking & Cleaning Data...")
print("=" * 55)

for name, df in datasets.items():
    missing = df.isnull().sum().sum()
    print(f"  {name:15s}: {missing} missing values")

# Fix delivery reasons_if_delayed — fill blanks with 'None'
delivery['reasons_if_delayed'] = delivery['reasons_if_delayed'].fillna('None')
print("\n  ✅ Filled missing delivery reasons with 'None'")
print("  ✅ All other datasets are clean (no missing values)")

# =============================================================
#  STEP 3: MERGE DATASETS
#  (Like assembling the full picture from puzzle pieces)
#
#  Think of it like this:
#    - order_items knows WHAT was bought (product, qty, price)
#    - orders knows WHO bought it and WHEN
#    - products knows the category and brand
#    - customers knows the area and segment
# =============================================================
print("\n" + "=" * 55)
print("  STEP 3: Merging Datasets...")
print("=" * 55)

order_items['line_total'] = order_items['quantity'] * order_items['unit_price']

full_df = (
    order_items
    .merge(orders,   on='order_id',   how='left')
    .merge(products, on='product_id', how='left')
    .merge(customers[['customer_id','customer_name','customer_segment','area','pincode']],
           on='customer_id', how='left')
)

# Time features
full_df['year']        = full_df['order_date'].dt.year
full_df['month']       = full_df['order_date'].dt.month
full_df['month_name']  = full_df['order_date'].dt.strftime('%b')
full_df['day_of_week'] = full_df['order_date'].dt.dayofweek
full_df['quarter']     = full_df['order_date'].dt.quarter
full_df['is_weekend']  = (full_df['day_of_week'] >= 5).astype(int)
full_df['month_str']   = full_df['order_date'].dt.strftime('%Y-%m')

full_df.to_csv('outputs/full_merged_data.csv', index=False)
print(f"  ✅ Merged dataset: {full_df.shape[0]:,} rows × {full_df.shape[1]} columns")
print("  ✅ Saved to outputs/full_merged_data.csv")

# =============================================================
#  STEP 4: KEY BUSINESS INSIGHTS
#  (The "So what?" numbers every manager wants)
# =============================================================
print("\n" + "=" * 55)
print("  STEP 4: Key Business Insights")
print("=" * 55)

total_revenue = full_df['line_total'].sum()
total_orders  = orders['order_id'].nunique()
total_customers = customers['customer_id'].nunique()
total_products  = products['product_id'].nunique()
avg_order_val   = orders['order_total'].mean()
on_time_pct     = (delivery['delivery_status'] == 'On Time').mean() * 100
avg_rating      = feedback['rating'].mean()

print(f"""
  💰 Total Revenue        : ₹{total_revenue:,.2f}
  🛒 Total Orders         : {total_orders:,}
  👥 Total Customers      : {total_customers:,}
  📦 Total Products       : {total_products:,}
  🧾 Avg Order Value      : ₹{avg_order_val:.2f}
  🚴 On-Time Delivery %   : {on_time_pct:.1f}%
  ⭐ Avg Customer Rating  : {avg_rating:.2f} / 5
""")

# Top category
top_cat = full_df.groupby('category')['line_total'].sum().idxmax()
top_rev = full_df.groupby('category')['line_total'].sum().max()
print(f"  🏆 Best Category        : {top_cat} (₹{top_rev:,.0f})")

# Best month
best_month = full_df.groupby('month_str')['line_total'].sum().idxmax()
print(f"  📅 Best Month           : {best_month}")

print("\n  ✅ Run analysis/02_visualizations.py to see all charts!")
print("  ✅ Run analysis/03_customer_segmentation.py for clustering!")
print("  ✅ Run models/04_sales_forecast.py for ML predictions!")
