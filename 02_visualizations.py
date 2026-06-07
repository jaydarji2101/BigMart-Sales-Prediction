import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os


GREEN  = '#0C8A0C'
YELLOW = '#F8D210'
ORANGE = '#FF6B35'
BLUE   = '#2196F3'
RED    = '#E53935'
GREY   = '#F5F5F5'

CATEGORY_COLORS = [GREEN, YELLOW, ORANGE, BLUE,
                   '#9C27B0', '#E91E63', '#00BCD4',
                   '#795548', '#607D8B', '#FF5722', '#4CAF50']

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'figure.facecolor'  : 'white',
})

OUT = 'charts/'

# ── Load data ──────────────────────────────────────────────
DATA = 'data/'
orders      = pd.read_csv(DATA + 'blinkit_orders.csv',     parse_dates=['order_date'])
order_items = pd.read_csv(DATA + 'blinkit_order_items.csv')
products    = pd.read_csv(DATA + 'blinkit_products.csv')
customers   = pd.read_csv(DATA + 'blinkit_customers.csv')
delivery    = pd.read_csv(DATA + 'blinkit_delivery_performance.csv')
feedback    = pd.read_csv(DATA + 'blinkit_customer_feedback.csv')

order_items['line_total'] = order_items['quantity'] * order_items['unit_price']
full = (order_items
        .merge(orders,   on='order_id')
        .merge(products, on='product_id')
        .merge(customers[['customer_id', 'customer_segment', 'area']], on='customer_id'))
full['month_str'] = full['order_date'].dt.strftime('%Y-%m')

# ════════════════════════════════════════════════════════════
#  CHART 1 — Monthly Revenue Trend
# ════════════════════════════════════════════════════════════
monthly = full.groupby('month_str')['line_total'].sum().reset_index()
monthly.columns = ['month', 'revenue']

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(range(len(monthly)), monthly['revenue'],
                alpha=0.12, color=GREEN)
ax.plot(range(len(monthly)), monthly['revenue'],
        color=GREEN, linewidth=2.5, marker='o', markersize=5)
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly['month'], rotation=45, ha='right', fontsize=8)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x/1e6:.1f}M'))
ax.set_title('📈 Monthly Revenue Trend', fontsize=15, fontweight='bold', pad=14)
ax.set_ylabel('Revenue (₹)')
ax.set_facecolor(GREY)
ax.grid(axis='y', alpha=0.35, linestyle='--')

# Annotate peak
peak_i = monthly['revenue'].idxmax()
ax.annotate(
    f"Peak ₹{monthly.loc[peak_i,'revenue']/1e6:.2f}M",
    xy=(peak_i, monthly.loc[peak_i, 'revenue']),
    xytext=(peak_i, monthly.loc[peak_i, 'revenue'] * 1.08),
    ha='center', fontsize=8, color=GREEN, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2))

plt.tight_layout()
plt.savefig(OUT + 'chart1_monthly_revenue.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ Chart 1 saved — Monthly Revenue Trend')

# ════════════════════════════════════════════════════════════
#  CHART 2 — Revenue by Category
# ════════════════════════════════════════════════════════════
cat_rev = (full.groupby('category')['line_total']
           .sum().sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(cat_rev.index, cat_rev.values,
               color=CATEGORY_COLORS[:len(cat_rev)], edgecolor='white')
ax.set_title('🛒 Revenue by Product Category',
             fontsize=15, fontweight='bold', pad=14)
ax.set_xlabel('Total Revenue (₹)')
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x/1e6:.1f}M'))
ax.set_facecolor(GREY)
ax.grid(axis='x', alpha=0.3, linestyle='--')
for bar in bars:
    w = bar.get_width()
    ax.text(w + w * 0.01, bar.get_y() + bar.get_height() / 2,
            f'₹{w/1e6:.2f}M', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'chart2_category_revenue.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ Chart 2 saved — Category Revenue')

# ════════════════════════════════════════════════════════════
#  CHART 3 — Customer Segments (pie + avg order value bar)
# ════════════════════════════════════════════════════════════
seg_counts = customers['customer_segment'].value_counts()
seg_colors = [GREEN, YELLOW, ORANGE, BLUE]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

wedges, texts, autotexts = ax1.pie(
    seg_counts.values, labels=seg_counts.index,
    autopct='%1.1f%%', colors=seg_colors, startangle=90,
    pctdistance=0.75, wedgeprops=dict(linewidth=2, edgecolor='white'))
for at in autotexts:
    at.set_fontsize(11); at.set_fontweight('bold')
ax1.set_title('👥 Customer Segment Distribution',
              fontsize=13, fontweight='bold')

seg_val = (customers.groupby('customer_segment')['avg_order_value']
           .mean().sort_values(ascending=False))
color_map = dict(zip(seg_counts.index, seg_colors))
b2 = ax2.bar(seg_val.index,
             seg_val.values,
             color=[color_map.get(s, '#999') for s in seg_val.index],
             edgecolor='white')
ax2.set_title('💰 Avg Order Value by Segment',
              fontsize=13, fontweight='bold')
ax2.set_ylabel('Avg Order Value (₹)')
ax2.set_facecolor(GREY)
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
for bar in b2:
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 5,
             f'₹{bar.get_height():,.0f}',
             ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'chart3_customer_segments.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ Chart 3 saved — Customer Segments')

# ════════════════════════════════════════════════════════════
#  CHART 4 — Delivery Status + Sentiment
# ════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

del_stat = delivery['delivery_status'].value_counts()
status_colors = {
    'On Time': GREEN, 'Slightly Delayed': YELLOW,
    'Delayed': ORANGE, 'Cancelled': RED
}
c4 = [status_colors.get(s, BLUE) for s in del_stat.index]
ax1.bar(del_stat.index, del_stat.values, color=c4, edgecolor='white')
ax1.set_title('🚴 Delivery Status Breakdown',
              fontsize=13, fontweight='bold')
ax1.set_ylabel('Number of Orders')
ax1.set_facecolor(GREY)
for i, (idx, val) in enumerate(del_stat.items()):
    ax1.text(i, val + 15,
             f'{val:,}\n({val/len(delivery)*100:.1f}%)',
             ha='center', fontsize=9, fontweight='bold')

sent = feedback['sentiment'].value_counts()
sent_colors = {'Positive': GREEN, 'Neutral': YELLOW, 'Negative': RED}
c5 = [sent_colors.get(s, ORANGE) for s in sent.index]
ax2.bar(sent.index, sent.values, color=c5, edgecolor='white')
ax2.set_title('💬 Customer Sentiment',
              fontsize=13, fontweight='bold')
ax2.set_ylabel('Number of Reviews')
ax2.set_facecolor(GREY)
for i, (idx, val) in enumerate(sent.items()):
    ax2.text(i, val + 15,
             f'{val:,}\n({val/len(feedback)*100:.1f}%)',
             ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'chart4_delivery_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ Chart 4 saved — Delivery & Sentiment')

# ════════════════════════════════════════════════════════════
#  CHART 5 — Top 10 Products by Revenue
# ════════════════════════════════════════════════════════════
top_prod = (full.groupby('product_name')['line_total']
            .sum().nlargest(10).sort_values())

fig, ax = plt.subplots(figsize=(12, 7))
bar_colors = [GREEN if i >= 7 else YELLOW if i >= 4 else ORANGE
              for i in range(len(top_prod))]
bars = ax.barh(top_prod.index, top_prod.values,
               color=bar_colors, edgecolor='white')
ax.set_title('🏆 Top 10 Products by Revenue',
             fontsize=15, fontweight='bold', pad=14)
ax.set_xlabel('Total Revenue (₹)')
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
ax.set_facecolor(GREY)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 100, bar.get_y() + bar.get_height() / 2,
            f'₹{w:,.0f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUT + 'chart5_top_products.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ Chart 5 saved — Top Products')

# ════════════════════════════════════════════════════════════
#  CHART 6 — Payment Method Preference
# ════════════════════════════════════════════════════════════
pay = orders['payment_method'].value_counts()
fig, ax = plt.subplots(figsize=(9, 5))
pay_colors = [GREEN, YELLOW, ORANGE, BLUE, '#9C27B0']
bars = ax.bar(pay.index, pay.values,
              color=pay_colors[:len(pay)], edgecolor='white')
ax.set_title('💳 Payment Method Preference',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Orders')
ax.set_facecolor(GREY)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f'{bar.get_height():,}\n({bar.get_height()/len(orders)*100:.1f}%)',
            ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'chart6_payment_methods.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ Chart 6 saved — Payment Methods')

# ════════════════════════════════════════════════════════════
#  CHART 7 — Customer Cluster Scatter
# ════════════════════════════════════════════════════════════
# ── CHART 7 — Customer Clusters ──────────────────────
try:
    # Try both possible file locations
    seg_path = 'outputs/customer_segments.csv'
    if not os.path.exists(seg_path):
        seg_path = '../outputs/customer_segments.csv'

    seg_df = pd.read_csv(seg_path)

    # Auto-detect which column has the segment names
    # Sometimes it is called 'segment_label', sometimes 'segment'
    if 'segment_label' in seg_df.columns:
        seg_col = 'segment_label'
    elif 'segment' in seg_df.columns:
        seg_col = 'segment'
    else:
        print('⚠️  Chart 7 skipped — no segment column found')
        print(f'    Columns found: {seg_df.columns.tolist()}')
        raise ValueError('No segment column')

    # Also check if total_orders and total_spent columns exist
    if 'total_orders' not in seg_df.columns:
        seg_df['total_orders'] = 1
    if 'total_spent' not in seg_df.columns:
        seg_df['total_spent'] = 0

    seg_colors_map = {
        '👑 Champions'        : GREEN,
        '💛 Loyal Customers'  : YELLOW,
        '💤 At-Risk Customers': ORANGE,
        '🆕 New Customers'    : BLUE,
    }

    # Also handle plain names without emojis
    plain_colors = {
        'Champions'       : GREEN,
        'Loyal Customers' : YELLOW,
        'At-Risk Customers': ORANGE,
        'New Customers'   : BLUE,
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    unique_segs = seg_df[seg_col].unique()
    assigned_colors = {}
    color_list = [GREEN, YELLOW, ORANGE, BLUE]
    for i, s in enumerate(unique_segs):
        matched = False
        for key, col in {**seg_colors_map, **plain_colors}.items():
            if key in str(s):
                assigned_colors[s] = col
                matched = True
                break
        if not matched:
            assigned_colors[s] = color_list[i % len(color_list)]

    for seg in unique_segs:
        col  = assigned_colors.get(seg, BLUE)
        mask = seg_df[seg_col] == seg
        if mask.any():
            axes[0].scatter(
                seg_df.loc[mask, 'total_orders'],
                seg_df.loc[mask, 'total_spent'],
                c=col, label=str(seg),
                alpha=0.55, s=35, edgecolors='none'
            )

    axes[0].set_xlabel('Total Orders')
    axes[0].set_ylabel('Total Spent (Rs.)')
    axes[0].set_title('Customer Segments — Orders vs Spend',
                      fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].set_facecolor(GREY)
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'Rs.{x:,.0f}'))

    counts     = seg_df[seg_col].value_counts()
    colors_bar = [assigned_colors.get(s, BLUE) for s in counts.index]
    axes[1].bar(range(len(counts)), counts.values,
                color=colors_bar, edgecolor='white')
    axes[1].set_xticks(range(len(counts)))
    axes[1].set_xticklabels(
        [str(s)[:20] for s in counts.index],
        fontsize=8, rotation=15, ha='right'
    )
    axes[1].set_title('Customers per Segment',
                      fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Number of Customers')
    axes[1].set_facecolor(GREY)
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 3, str(v),
                     ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT + 'chart7_customer_clusters.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('✅ Chart 7 saved — Customer Clusters')

except FileNotFoundError:
    print('⚠️  Chart 7 skipped — run 03_customer_segmentation.py first')
    print('    Then rerun 02_visualizations.py')
except Exception as e:
    print(f'⚠️  Chart 7 skipped — {e}')

print('\n🎉 All charts saved to charts/')
print('   Open them with any image viewer or include in your report.')
