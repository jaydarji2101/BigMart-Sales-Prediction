import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING")
print("=" * 55)

# ── Load data ──────────────────────────────────────────────
DATA = 'data/'
orders      = pd.read_csv(DATA + 'blinkit_orders.csv', parse_dates=['order_date'])
order_items = pd.read_csv(DATA + 'blinkit_order_items.csv')
customers   = pd.read_csv(DATA + 'blinkit_customers.csv')

# ── Build RFM features ─────────────────────────────────────
# R = Recency    (How recently did they buy?)
# F = Frequency  (How often do they buy?)
# M = Monetary   (How much do they spend?)

order_items['line_total'] = order_items['quantity'] * order_items['unit_price']
order_data = order_items.merge(
    orders[['order_id','customer_id','order_date']], on='order_id')

max_date = orders['order_date'].max()

rfm = order_data.groupby('customer_id').agg(
    total_spent    = ('line_total',  'sum'),
    total_orders   = ('order_id',   'nunique'),
    avg_basket     = ('line_total',  'mean'),
    last_order_date= ('order_date',  'max')
).reset_index()

rfm['days_since_last'] = (max_date - rfm['last_order_date']).dt.days

print(f"\n  Built RFM table for {len(rfm):,} customers")
print(rfm[['total_spent','total_orders','avg_basket','days_since_last']].describe().round(2))

# ── Scale features (important for KMeans!) ─────────────────
# KMeans uses DISTANCE. If total_spent is 5000 and days is 30,
# the big number (5000) will dominate. Scaling fixes this.
features = ['total_spent','total_orders','avg_basket','days_since_last']
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features].fillna(0))

# ── Find best K (number of clusters) ─────────────────────
# Silhouette score: higher = clusters are more distinct (better)
print("\n  Finding best number of clusters...")
sil_scores = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, km.labels_)
    print(f"    K={k}: Silhouette = {sil_scores[k]:.4f}")

# We use K=4 for business interpretability (4 clear segments)
BEST_K = 4
print(f"\n  ✅ Using K={BEST_K} clusters (best for business story)")

# ── Fit final model ────────────────────────────────────────
km = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
rfm['cluster'] = km.fit_predict(X_scaled)

# ── Label clusters based on their behaviour ────────────────
cluster_stats = rfm.groupby('cluster').agg(
    count        = ('customer_id', 'count'),
    avg_spent    = ('total_spent',    'mean'),
    avg_orders   = ('total_orders',   'mean'),
    avg_recency  = ('days_since_last','mean'),
    avg_basket   = ('avg_basket',     'mean')
).reset_index().sort_values('avg_spent', ascending=False)

LABELS = {
    cluster_stats.iloc[0]['cluster']: '👑 Champions',
    cluster_stats.iloc[1]['cluster']: '💛 Loyal Customers',
    cluster_stats.iloc[2]['cluster']: '💤 At-Risk Customers',
    cluster_stats.iloc[3]['cluster']: '🆕 New Customers',
}
rfm['segment'] = rfm['cluster'].map(LABELS)

# ── Print segment analysis ─────────────────────────────────
print("\n" + "=" * 55)
print("  CUSTOMER SEGMENT PROFILES")
print("=" * 55)
for seg, data in rfm.groupby('segment', sort=False):
    print(f"""
  {seg}
  ─────────────────────────────
  Customers   : {len(data):,}
  Avg Spent   : ₹{data['total_spent'].mean():,.0f}
  Avg Orders  : {data['total_orders'].mean():.1f}
  Avg Basket  : ₹{data['avg_basket'].mean():,.0f}
  Last Bought : {data['days_since_last'].mean():.0f} days ago
  """)

# ── Business Recommendations per segment ──────────────────
print("=" * 55)
print("  BUSINESS RECOMMENDATIONS BY SEGMENT")
print("=" * 55)
print("""
  👑 Champions
     → Send exclusive early access & VIP offers
     → Ask for referrals ("Refer a friend, get ₹200")
     → Highest priority for customer service

  💛 Loyal Customers
     → Launch loyalty points program
     → Offer subscription discounts
     → Cross-sell related categories

  💤 At-Risk Customers
     → Send WIN-BACK campaign: "We miss you! ₹100 off"
     → Investigate why they stopped (feedback survey)
     → Offer free delivery for next 3 orders

  🆕 New Customers
     → Send onboarding email series
     → Offer "First 3 orders" discount
     → Show them your bestsellers
""")

# Save
rfm.to_csv('outputs/customer_segments.csv', index=False)
print("  ✅ Segments saved to outputs/customer_segments.csv")
