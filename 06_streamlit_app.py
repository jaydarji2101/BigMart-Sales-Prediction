"""
=============================================================
  BIGMART SALES FORECASTING & CUSTOMER INSIGHTS
  File: 06_streamlit_app.py

  HOW TO RUN:
      cd "D:\Fingertips\PMMS\Big Mart Sales Prediction\Blinkit\files2"
      streamlit run 06_streamlit_app.py

  FIXES IN THIS VERSION:
  1. Generic title - works for BigMart, DMart, Blinkit, Zepto, etc.
  2. Company name input in sidebar - customizable
  3. Upload your own CSV datasets + preview in Data Preview tab
  4. Full-date forecast: sales by 15th AND by end of month
  5. Indian festival detection with demand multiplier
  6. Online / Offline / Both channel support
  7. Recommendations - WHITE TEXT on COLORED background (fully visible)
  8. Decision System - WHITE TEXT on DARK background (fully visible)
  9. Generic footer for all retail types
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import datetime
import calendar
import importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Auto-create folders
for _f in ['outputs', 'charts', 'models', 'data']:
    os.makedirs(_f, exist_ok=True)

# ── Load 05_decision_system.py safely ─────────────────────
def _load_dec():
    for p in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '05_decision_system.py'),
        os.path.join(os.getcwd(), '05_decision_system.py'),
        '05_decision_system.py'
    ]:
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("dm", p)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None

_dm = _load_dec()

# ── Indian Festival Calendar ───────────────────────────────
FESTIVALS = {
    1:  ("New Year / Makar Sankranti", 1.15),
    2:  ("Valentine's Day",            1.10),
    3:  ("Holi",                       1.25),
    4:  ("Ugadi / Gudi Padwa",         1.10),
    8:  ("Raksha Bandhan / Janmashtami", 1.20),
    9:  ("Navratri / Onam",            1.20),
    10: ("Diwali / Durga Puja",        1.40),
    11: ("Diwali / Bhai Dooj",         1.30),
    12: ("Christmas / New Year Eve",   1.20),
}

def detect_festival(month):
    return FESTIVALS.get(month, None)

# ── Recommendations ────────────────────────────────────────
def get_recs(cat, pred_qty, rating, ot_pct, roas, channel, fest):
    r = []
    if fest:
        fn, fm = fest
        r.append((f"FESTIVAL ALERT: {fn} in this month! Demand boosted {int((fm-1)*100)}%. Stock up early. NO discounts.", "rec-orange"))
    if pred_qty > 500:
        r.append(("HIGH DEMAND DETECTED — Increase stock levels immediately", "rec-red"))
        r.append(("AVOID HEAVY DISCOUNTS — Product sells well at full price", "rec-green"))
    elif pred_qty > 200:
        r.append(("MODERATE DEMAND — Maintain current stock levels", "rec-green"))
        r.append(("LIGHT OFFER OK (5-8%) to boost volume slightly", "rec-blue"))
    else:
        r.append(("LOW DEMAND — Do not over-stock, risk of wastage", "rec-yellow"))
        r.append(("RUN PROMOTIONS — Flash sale or bundle offer recommended", "rec-yellow"))
        r.append(("INVESTIGATE — Check if quality or pricing issue exists", "rec-red"))

    if channel == 'Online':
        r.append(("ONLINE FOCUS — Use app push notifications and home page banners", "rec-blue"))
    elif channel == 'Offline':
        r.append(("STORE FOCUS — Improve shelf placement and in-aisle display signage", "rec-blue"))
    else:
        r.append(("OMNI-CHANNEL — Sync online and offline stock to avoid out-of-stock", "rec-blue"))

    if rating < 3.0:
        r.append(("LOW RATINGS ALERT — Urgently review product quality and supplier", "rec-red"))
        r.append(("RESPOND TO COMPLAINTS — Reply to all 1-2 star reviews within 24 hours", "rec-red"))
    elif rating >= 4.5:
        r.append(("EXCELLENT RATINGS — Feature in Top Picks and Best Seller section", "rec-green"))
    else:
        r.append(("AVERAGE RATINGS — Send post-purchase survey to find improvements", "rec-blue"))

    if channel != 'Offline':
        if ot_pct < 80:
            r.append((f"DELIVERY BELOW TARGET ({ot_pct:.0f}%) — Add more delivery partners urgently", "rec-red"))
        elif ot_pct >= 95:
            r.append((f"DELIVERY EXCELLENT ({ot_pct:.0f}%) — Use as USP in marketing ads", "rec-green"))

    if roas > 4.0:
        r.append((f"HIGH ROAS ({roas:.1f}x) — INCREASE ad budget by 50% immediately", "rec-green"))
    elif roas < 2.0:
        r.append((f"POOR ROAS ({roas:.1f}x) — PAUSE this campaign, test new creative", "rec-red"))
    else:
        r.append((f"MODERATE ROAS ({roas:.1f}x) — Optimize ad creatives to improve results", "rec-blue"))
    return r

# ── Decision System ────────────────────────────────────────
def get_decisions(cat, pred_qty, stock, avg_p, cost_p,
                  ot_pct, rating, roas, channel, fest,
                  f_start, f_end):
    decs = []
    fest_mult = 1.0
    if fest:
        fn, fest_mult = fest
        pred_qty = int(pred_qty * fest_mult)

    safety     = int(pred_qty * 0.10)
    target_stk = pred_qty + safety
    gap        = target_stk - stock
    mid_units  = int(pred_qty * 0.47)

    # Stock
    if gap > 0:
        urg = "URGENT — Order within 48 hours" if gap > 200 else "Order within 1 week"
        decs.append(("STOCK",
            f"ORDER {gap:,} MORE UNITS before {f_start.strftime('%d %b %Y')}",
            f"Predicted: {pred_qty:,} | Current: {stock:,} | Gap: {gap:,} | Target (10% buffer): {target_stk:,} | {urg}"))
    elif gap < -100:
        decs.append(("STOCK",
            f"REDUCE NEXT ORDER — Excess stock = {abs(gap):,} units",
            f"Predicted: {pred_qty:,} | Current: {stock:,} | Excess: {abs(gap):,} | Risk of wastage"))
    else:
        decs.append(("STOCK",
            "STOCK IS OPTIMAL — No restocking action needed",
            f"Current stock ({stock:,}) is within 10% of forecast ({pred_qty:,})"))

    # Pricing
    disc = 0
    if fest:
        fn, _ = fest
        decs.append(("PRICING",
            f"NO DISCOUNT during {fn} — Charge full price Rs.{avg_p:.0f}",
            f"Festival demand is high. Discounting = lost profit. Revenue at full price: Rs.{pred_qty * avg_p:,.0f}"))
    elif pred_qty > 500:
        decs.append(("PRICING",
            f"NO DISCOUNT — High demand. Maintain Rs.{avg_p:.0f}",
            f"Demand is {pred_qty:,} units. No incentive needed. Revenue: Rs.{pred_qty * avg_p:,.0f}"))
    elif pred_qty > 200:
        eff = avg_p * 0.94; disc = 6
        decs.append(("PRICING",
            f"APPLY 6% DISCOUNT — Price: Rs.{avg_p:.0f} -> Rs.{eff:.0f}",
            f"Moderate demand. Small offer boosts conversion. Revenue at Rs.{eff:.0f}: Rs.{int(pred_qty * eff):,}"))
    else:
        eff = avg_p * 0.85; disc = 15
        decs.append(("PRICING",
            f"FLASH SALE 15% OFF — Rs.{avg_p:.0f} -> Rs.{eff:.0f} for 48 hours",
            f"Low demand. Promotion needed. Revenue at sale price: Rs.{int(pred_qty * eff):,}"))

    # Delivery/Store
    if channel == 'Offline':
        decs.append(("STORE OPS",
            "Ensure shelves fully stocked by 9 AM every day",
            f"Track daily footfall and in-store conversion rate. Place {cat} in high-traffic aisle."))
    elif ot_pct < 80:
        decs.append(("DELIVERY",
            f"ADD DELIVERY PARTNERS — On-time rate {ot_pct:.1f}% is below 80%",
            f"Assign {max(2, int((80-ot_pct)/5))} more partners to this zone. Target: 90%+ in 2 weeks."))
    elif ot_pct < 90:
        decs.append(("DELIVERY",
            f"MONITOR DELIVERY — {ot_pct:.1f}% on-time is slightly below target",
            "No major change. Track daily. Investigate routes with repeated delays."))
    else:
        decs.append(("DELIVERY",
            f"DELIVERY EXCELLENT — {ot_pct:.1f}% on-time. Highlight in ads.",
            "Maintain partner incentives. Use delivery speed as key marketing message."))

    # Marketing
    if roas >= 4.0:
        decs.append(("MARKETING",
            f"INCREASE AD BUDGET +50% — ROAS {roas:.1f}x is excellent",
            f"Every Rs.1 returns Rs.{roas:.1f}. Recommended budget increase: Rs.{int(pred_qty * avg_p * 0.05):,}"))
    elif roas >= 2.5:
        decs.append(("MARKETING",
            f"MAINTAIN BUDGET — ROAS {roas:.1f}x is acceptable",
            "A/B test new creative formats. Try video vs static. Improve targeting."))
    else:
        decs.append(("MARKETING",
            f"CUT AD BUDGET -30% — ROAS {roas:.1f}x is poor",
            "Pause current campaign. Test influencer / WhatsApp marketing instead."))

    # Customer
    if rating >= 4.5:
        decs.append(("CUSTOMERS",
            "COLLECT REVIEWS — Feature 5-star customers in ads",
            "Send WhatsApp/email for review 2 hours after delivery/purchase."))
    elif rating >= 3.5:
        decs.append(("CUSTOMERS",
            "REQUEST FEEDBACK — Send post-purchase survey to all buyers",
            "Use feedback to find top 3 improvement areas this month."))
    else:
        decs.append(("CUSTOMERS",
            f"URGENT — Rating {rating:.1f}/5 is low. Respond to ALL reviews in 24h",
            "Offer Rs.50 coupon to 1-2 star customers. Escalate to product team."))

    # Financials
    eff_p  = avg_p * (1 - disc/100)
    f_rev  = round(pred_qty * eff_p, 0)
    f_cost = round(pred_qty * cost_p, 0)
    f_prof = round(f_rev - f_cost, 0)
    m_rev  = round(mid_units * eff_p, 0)
    decs.append(("FINANCIALS",
        f"Revenue by {f_start.replace(day=15).strftime('%d-%m-%Y')}: Rs.{m_rev:,.0f} ({mid_units:,} units) | "
        f"Full month by {f_end.strftime('%d-%m-%Y')}: Rs.{f_rev:,.0f} ({pred_qty:,} units)",
        f"Expected profit: Rs.{f_prof:,.0f} | Margin/unit: Rs.{eff_p - cost_p:.2f} | Discount: {disc}%"))

    return decs, disc, pred_qty, f_rev, f_prof

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BigMart Sales Forecasting & Insights",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS — All text fully visible ──────────────────────────
st.markdown("""
<style>
.rec-box { padding:11px 16px; margin:6px 0; font-size:0.93em;
           font-weight:500; line-height:1.65; border-left:5px solid transparent; }
.rec-orange { background:#BF360C; color:#FFFFFF !important; border-left-color:#FF8A65; }
.rec-red    { background:#B71C1C; color:#FFFFFF !important; border-left-color:#EF9A9A; }
.rec-green  { background:#1B5E20; color:#FFFFFF !important; border-left-color:#A5D6A7; }
.rec-yellow { background:#E65100; color:#FFFFFF !important; border-left-color:#FFE082; }
.rec-blue   { background:#0D47A1; color:#FFFFFF !important; border-left-color:#90CAF9; }
.dec-box { background:#1A237E; color:#FFFFFF !important; padding:12px 16px;
           margin:7px 0; border-left:5px solid #7986CB; }
.dec-label { font-size:0.73em; text-transform:uppercase; letter-spacing:.07em;
             color:#C5CAE9 !important; margin-bottom:4px; font-weight:700; }
.dec-main  { font-size:0.95em; font-weight:700; color:#FFFFFF !important; margin:3px 0; }
.dec-sub   { font-size:0.82em; color:#B0BEC5 !important; line-height:1.5; margin-top:3px; }
.date-card { padding:16px; border-radius:10px; color:#FFFFFF !important; text-align:center; }
.dc-date   { font-size:0.78em; opacity:0.85; margin-bottom:4px; }
.dc-units  { font-size:1.85em; font-weight:700; }
.dc-lbl    { font-size:0.80em; opacity:0.75; }
.dc-rev    { font-size:1.0em; font-weight:600; margin-top:5px; }
.fest-banner { background:linear-gradient(90deg,#E65100,#FF6F00); color:#FFFFFF !important;
               padding:14px 22px; border-radius:10px; font-size:1.02em;
               font-weight:700; text-align:center; margin-bottom:16px; }
.ch-card   { padding:14px; border-radius:10px; color:#FFFFFF !important; margin-bottom:8px; }
.ch-lbl    { font-size:0.80em; opacity:0.82; }
.ch-val    { font-size:1.6em; font-weight:700; }
.ch-rev    { font-size:0.90em; font-weight:600; margin-top:3px; }
.ch-tip    { font-size:0.78em; opacity:0.75; margin-top:5px; }
.prev-hdr  { background:#263238; color:#ECEFF1 !important; padding:9px 14px;
             border-radius:6px; font-weight:600; font-size:0.88em; margin-bottom:6px; }
.stButton>button { background:linear-gradient(90deg,#1B5E20,#2E7D32) !important;
                   color:#FFFFFF !important; font-size:16px !important;
                   font-weight:700 !important; border-radius:10px !important;
                   border:none !important; padding:13px !important; width:100% !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏢 Company Settings")
    company_name  = st.text_input("Company / Store Name", value="BigMart",
        help="e.g. BigMart, DMart, Blinkit, Zepto, InstaMart, Swiggy Instamart")
    business_type = st.selectbox("Business Type", [
        "Online (Quick Commerce)",
        "Offline (Physical Store)",
        "Both Online + Offline"
    ], index=2)
    channel = {"Online (Quick Commerce)":"Online",
                "Offline (Physical Store)":"Offline",
                "Both Online + Offline":"Both"}[business_type]

    st.markdown("---")
    st.markdown("### 📂 Upload Your Datasets")
    st.caption(
        "Upload one or more CSV files. Works whether you have 1 file or 9 files. "
        "Preview them in the Data Preview tab."
    )
    uploaded_files_list = st.file_uploader(
        "Drop your CSV files here",
        type='csv',
        accept_multiple_files=True,
        key='multi_upload',
        help=(
            "You can upload ANY number of CSV files at once.\n"
            "One big file with all data? ✅ Works fine.\n"
            "9 separate files like Blinkit? ✅ Works fine.\n"
            "Mix of both? ✅ Works fine."
        )
    )
    # Convert list to dict by filename for easy lookup
    up_map = {}
    if uploaded_files_list:
        for uf in uploaded_files_list:
            up_map[uf.name] = uf
        st.success(f"✅ {len(uploaded_files_list)} file(s) uploaded successfully!")

    # Keep old variable names as None so rest of code doesn't break
    up_orders    = None
    up_products  = None
    up_customers = None
    up_inventory = None

    st.markdown("---")
    st.markdown("### 📊 Sales Data Input")

    CATS = ['Baby Care','Cold Drinks & Juices','Dairy & Breakfast',
            'Fruits & Vegetables','Grocery & Staples','Household Care',
            'Instant & Frozen Food','Personal Care','Pet Care',
            'Pharmacy','Snacks & Munchies','Electronics',
            'Clothing & Apparel','Home & Kitchen','Sports & Fitness']
    category = st.selectbox("Product Category", CATS, index=2)

    st.markdown("**Forecast Month**")
    cy, cm = st.columns(2)
    now = datetime.datetime.now()
    with cy:
        pred_year = st.selectbox("Year", [now.year, now.year+1])
    with cm:
        MN = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        pmn = st.selectbox("Month", MN, index=now.month % 12)
        pred_month = MN.index(pmn) + 1

    last_day       = calendar.monthrange(pred_year, pred_month)[1]
    forecast_start = datetime.date(pred_year, pred_month, 1)
    forecast_end   = datetime.date(pred_year, pred_month, last_day)
    forecast_mid   = datetime.date(pred_year, pred_month, 15)
    pred_qtr       = (pred_month - 1) // 3 + 1
    fest_info      = detect_festival(pred_month)

    st.markdown("**Last 3 Months Sales**")
    c1, c2 = st.columns(2)
    with c1:
        lm  = st.number_input("Last month",   0, 99999, 400, 10)
        m2  = st.number_input("2 months ago", 0, 99999, 380, 10)
    with c2:
        m3  = st.number_input("3 months ago", 0, 99999, 360, 10)
        stk = st.number_input("Current stock",0, 999999,300, 10)

    if channel == 'Both':
        st.markdown("**Channel Split**")
        on_pct  = st.slider("Online orders %", 0, 100, 60, 5)
        off_pct = 100 - on_pct
        st.caption(f"Online: {on_pct}%  |  In-store: {off_pct}%")
    else:
        on_pct  = 100 if channel == 'Online' else 0
        off_pct = 100 - on_pct

    st.markdown("**💰 Pricing**")
    pc1, pc2 = st.columns(2)
    with pc1:
        avg_p  = st.number_input("Avg Selling Price (Rs.)", min_value=1,
                                  max_value=99999, value=120, step=1,
                                  help="The price at which you sell to customer")
    with pc2:
        cost_p = st.number_input("Cost Price (Rs.)", min_value=1,
                                  max_value=99999, value=70, step=1,
                                  help="The price at which you buy/produce")
    if avg_p <= cost_p:
        st.warning("⚠️ Selling price is less than or equal to cost price!")
    else:
        margin_show = avg_p - cost_p
        margin_pct  = round((margin_show / avg_p) * 100, 1)
        st.caption(f"Gross margin: Rs.{margin_show} per unit ({margin_pct}%)")

    st.markdown("**📊 Operations**")
    op1, op2 = st.columns(2)
    with op1:
        if channel != 'Offline':
            ot_pct = st.number_input("On-Time Delivery %", min_value=0.0,
                                      max_value=100.0, value=88.0, step=0.5,
                                      help="% of orders delivered on time")
        else:
            ot_pct = 100.0
            st.info("🏪 Offline — delivery N/A")
    with op2:
        rating = st.number_input("Avg Customer Rating", min_value=1.0,
                                  max_value=5.0, value=4.2, step=0.1,
                                  help="Average star rating out of 5")

    roas = st.number_input("Marketing ROAS", min_value=0.1, max_value=20.0,
                            value=3.5, step=0.1,
                            help="Revenue earned per Rs.1 spent on ads. Above 4.0 = excellent")
    st.markdown("---")
    predict_btn = st.button("🔮 PREDICT & ANALYZE", use_container_width=True)

# ══════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background:linear-gradient(90deg,#1B5E20,#2E7D32,#F9A825);
            padding:22px 28px; border-radius:14px; margin-bottom:18px;'>
  <h1 style='color:white; margin:0; font-size:1.85em; font-weight:700;'>
      🛒 {company_name} — Sales Forecasting & Customer Insights
  </h1>
  <p style='color:rgba(255,255,255,0.92); margin:7px 0 0 0; font-size:1.01em;'>
      Predict Sales · Business Recommendations · Automated Decision System · {business_type}
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════
t1, t2, t3 = st.tabs(["🔮 Forecast & Decisions", "📂 Data Preview", "ℹ️ How to Use"])

# ──────────────────────────────────────────────────────────
#  TAB 2 — DATA PREVIEW
# ──────────────────────────────────────────────────────────
with t2:
    st.markdown("### 📂 Dataset Preview")
    st.info("Upload CSV files in the sidebar to see your own data. "
            "If nothing uploaded, demo Blinkit data is shown.")

    # ── Helper function to preview any dataframe ───────────
    def preview_df(label, df, tag):
        st.markdown(
            f"<div class='prev-hdr'>📄 {label} — "
            f"{df.shape[0]:,} rows × {df.shape[1]} cols | {tag}</div>",
            unsafe_allow_html=True
        )
        cs, ct = st.columns([1, 3])
        with cs:
            st.markdown("**Quick stats**")
            nulls  = df.isnull().sum().sum()
            num_c  = df.select_dtypes(include='number').shape[1]
            txt_c  = df.shape[1] - num_c
            st.markdown(f"""
- Rows: **{df.shape[0]:,}**
- Columns: **{df.shape[1]}**
- Missing values: **{nulls}**
- Numeric columns: **{num_c}**
- Text columns: **{txt_c}**
            """)
            st.markdown("**Column names:**")
            for col in df.columns:
                st.markdown(f"• `{col}`")
        with ct:
            st.markdown("**First 10 rows:**")
            st.dataframe(df.head(10), use_container_width=True)
        with st.expander(f"📈 Full summary statistics — {label}"):
            st.dataframe(
                df.describe(include='all').round(2),
                use_container_width=True
            )
        st.markdown("---")

    # ── Show UPLOADED files ────────────────────────────────
    if uploaded_files_list:
        st.markdown(f"### ✅ Your Uploaded Files ({len(uploaded_files_list)} files)")
        st.info(
            "These are YOUR uploaded CSV files. "
            "All files are shown below regardless of their name or structure."
        )
        for uf in uploaded_files_list:
            try:
                df_up = pd.read_csv(uf)
                # Auto-detect what type of file this might be
                cols_lower = [c.lower() for c in df_up.columns]
                if any(x in cols_lower for x in ['order_id', 'order_date', 'order_total']):
                    file_type = "Orders data"
                elif any(x in cols_lower for x in ['product_id', 'product_name', 'category']):
                    file_type = "Products data"
                elif any(x in cols_lower for x in ['customer_id', 'customer_name', 'customer_segment']):
                    file_type = "Customers data"
                elif any(x in cols_lower for x in ['stock', 'inventory', 'quantity']):
                    file_type = "Inventory data"
                elif any(x in cols_lower for x in ['delivery', 'delivery_time', 'delivery_status']):
                    file_type = "Delivery data"
                elif any(x in cols_lower for x in ['campaign', 'roas', 'impressions', 'clicks']):
                    file_type = "Marketing data"
                elif any(x in cols_lower for x in ['rating', 'feedback', 'sentiment']):
                    file_type = "Feedback data"
                else:
                    file_type = "General data"
                preview_df(
                    f"{uf.name} ({file_type})",
                    df_up,
                    "✅ Your uploaded data"
                )
            except Exception as e:
                st.error(f"Could not read {uf.name}: {e}")

    else:
        # ── Show DEFAULT Blinkit demo files ────────────────
        default_files = {
            'Orders':    'data/blinkit_orders.csv',
            'Products':  'data/blinkit_products.csv',
            'Customers': 'data/blinkit_customers.csv',
            'Inventory': 'data/blinkit_inventoryNew.csv',
        }
        shown_any = False
        for label, path in default_files.items():
            if os.path.exists(path):
                if not shown_any:
                    st.markdown("#### 📊 Demo Data — Blinkit Sample Datasets")
                    st.info(
                        "No files uploaded yet. Showing Blinkit demo data below. "
                        "Upload your own CSV files in the sidebar to replace this."
                    )
                    shown_any = True
                preview_df(label, pd.read_csv(path), "📋 Demo data")

        if not shown_any:
            st.warning(
                "No uploaded files and no demo data found. "
                "Please upload CSV files using the sidebar uploader."
            )
# ──────────────────────────────────────────────────────────
#  TAB 3 — HOW TO USE
# ──────────────────────────────────────────────────────────
with t3:
    st.markdown("### ℹ️ How to Use This Dashboard")
    st.markdown("""
**This dashboard works for ANY retail company — online, offline, or both.**

Supported companies:
- Online Quick Commerce: Blinkit, Zepto, InstaMart, Swiggy Instamart
- Offline Retail: DMart, BigMart, Reliance Smart, Spencer's
- Both channels: Any company with physical store + online delivery

---
**Steps:**

1. **Set company name** in sidebar → appears in title and all reports
2. **Choose business type** → Online / Offline / Both. Changes delivery metrics, recommendations, and decisions
3. **Upload CSV datasets** (optional) → Preview in Data Preview tab. Demo data shown if nothing uploaded
4. **Enter last 3 months sales** + current stock + prices + operations data
5. **Select forecast month** → System detects Indian festivals automatically and boosts demand
6. **Click PREDICT & ANALYZE**

**What you get:**
- Sales by 15th of the month (mid-month) and full month end date
- Stock order: exact units to order, urgency level, target stock
- Business Recommendations — colored boxes, fully visible white text
- Decision System — dark boxes, fully visible white text, exact numbers
- Channel split (if Both): online vs in-store separate forecasts
- Financial summary: revenue, cost, profit, margin, discount

**Festival Detection (automatic):**
| Month | Festival | Demand Boost |
|---|---|---|
| October | Diwali / Durga Puja | +40% |
| November | Diwali / Bhai Dooj | +30% |
| March | Holi | +25% |
| August | Raksha Bandhan | +20% |
| September | Navratri | +20% |
| December | Christmas / New Year | +20% |
| January | New Year | +15% |
    """)

# ──────────────────────────────────────────────────────────
#  TAB 1 — FORECAST & DECISIONS
# ──────────────────────────────────────────────────────────
with t1:

    @st.cache_resource
    def load_model():
        for p in ['models/sales_forecast_model.pkl',
                  'sales_forecast_model.pkl',
                  'models/sales_forecast_model.pkl']:
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    return pickle.load(f)
        return None

    md = load_model()

    # Festival banner
    if fest_info:
        fn, fm = fest_info
        st.markdown(f"""<div class='fest-banner'>
            🎉 FESTIVAL ALERT: {fn} is in {pmn} {pred_year}!
            Demand boosted by {int((fm-1)*100)}%. Stock up early. Do NOT offer discounts.
        </div>""", unsafe_allow_html=True)

    # Base prediction
    rolling = (lm + m2 + m3) / 3
    n_orders = max(1, int(rolling // 3))
    pred_qty = int(rolling * 1.05)
    model_tag = "3-month trend estimate"

    if md:
        try:
            le, rf, feats = md['label_encoder'], md['model'], md['features']
            ce = le.transform([category])[0] if category in le.classes_ else 0
            inp = pd.DataFrame([[pred_year, pred_month, pred_qtr, ce,
                                  lm, m2, m3, rolling, n_orders, avg_p]],
                               columns=feats)
            pred_qty  = int(rf.predict(inp)[0])
            model_tag = "Random Forest (R²=0.94)"
        except Exception as e:
            st.warning(f"Using trend estimate. ({e})")

    # Apply festival
    fm_mult = 1.0
    if fest_info:
        _, fm_mult = fest_info
        pred_qty = int(pred_qty * fm_mult)

    # Get recs and decisions
    recs = get_recs(category, pred_qty, rating, ot_pct, roas, channel, fest_info)
    decs, disc, full_u, full_rev, full_prof = get_decisions(
        category, int(pred_qty/max(fm_mult,1)), stk,
        avg_p, cost_p, ot_pct, rating, roas,
        channel, fest_info, forecast_start, forecast_end)

    mid_u   = int(full_u * 0.47)
    eff_p   = avg_p * (1 - disc/100)
    mid_rev = round(mid_u * eff_p, 0)
    cost_t  = round(full_u * cost_p, 0)
    curr_r  = stk * avg_p
    rev_chg = round(((full_rev - curr_r) / max(curr_r,1)) * 100, 1)
    safety  = int(full_u * 0.10)
    t_stk   = full_u + safety
    gap     = t_stk - stk
    margin  = round(eff_p - cost_p, 2)

    # Forecast period header
    st.markdown(f"""
    <div style='background:#263238;color:#ECEFF1;padding:11px 16px;
                border-radius:8px;margin-bottom:14px;font-size:0.91em;'>
        📅 <strong>{forecast_start.strftime('%d %b %Y')}</strong>
        → <strong>{forecast_end.strftime('%d %b %Y')}</strong>
        &nbsp;|&nbsp; Model: {model_tag}
        &nbsp;|&nbsp; {business_type}
        {f'&nbsp;|&nbsp; Festival: {fest_info[0]} (x{fm_mult:.1f})' if fest_info else ''}
    </div>""", unsafe_allow_html=True)

    # Date forecast cards
    st.markdown("### 📅 Full-Date Sales Forecast")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(f"""<div class='date-card' style='background:#1A237E;'>
            <div class='dc-date'>By {forecast_mid.strftime('%d-%m-%Y')} (mid-month)</div>
            <div class='dc-units'>{mid_u:,} units</div>
            <div class='dc-lbl'>approx 47% of monthly target</div>
            <div class='dc-rev'>Rs.{mid_rev:,.0f} expected revenue</div>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown(f"""<div class='date-card' style='background:#1B5E20;'>
            <div class='dc-date'>By {forecast_end.strftime('%d-%m-%Y')} (full month)</div>
            <div class='dc-units'>{full_u:,} units</div>
            <div class='dc-lbl'>Full month prediction</div>
            <div class='dc-rev'>Rs.{full_rev:,.0f} expected revenue</div>
        </div>""", unsafe_allow_html=True)
    with d3:
        sc = '#B71C1C' if gap > 0 else '#1B5E20'
        sm = f"Order {gap:,} more units" if gap > 0 else f"Excess {abs(gap):,} units"
        su = "URGENT — restock before month start" if gap > 200 else ("Restock within 1 week" if gap > 0 else "Stock sufficient")
        st.markdown(f"""<div class='date-card' style='background:{sc};'>
            <div class='dc-date'>Stock Decision</div>
            <div class='dc-units' style='font-size:1.25em;'>{sm}</div>
            <div class='dc-lbl'>Target: {t_stk:,} units (10% buffer)</div>
            <div class='dc-rev'>{su}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # KPI row
    st.markdown("### 📊 Key Metrics")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("Predicted Sales", f"{full_u:,} units", f"{full_u-lm:+,} vs last month")
    with k2: st.metric("Stock Gap", f"{gap:+,} units", "Order now!" if gap>0 else "Stock OK", delta_color="inverse" if gap>0 else "normal")
    with k3: st.metric("Expected Revenue", f"Rs.{full_rev:,.0f}", f"{rev_chg:+.1f}%")
    with k4: st.metric("Expected Profit", f"Rs.{full_prof:,.0f}", f"Rs.{margin:.0f}/unit")
    with k5: st.metric("Discount", f"{disc}%", "No discount" if disc==0 else f"{disc}% offer")

    st.markdown("---")

    # Channel split
    if channel == 'Both':
        st.markdown("### 📱🏪 Channel Split")
        cc1, cc2 = st.columns(2)
        on_u  = int(full_u * on_pct / 100)
        off_u = full_u - on_u
        with cc1:
            st.markdown(f"""<div class='ch-card' style='background:#0D47A1;'>
                <div class='ch-lbl'>Online Orders ({on_pct}%)</div>
                <div class='ch-val'>{on_u:,} units</div>
                <div class='ch-rev'>Rs.{on_u*eff_p:,.0f} revenue</div>
                <div class='ch-tip'>App push, cashback, express delivery ads</div>
            </div>""", unsafe_allow_html=True)
        with cc2:
            st.markdown(f"""<div class='ch-card' style='background:#4A148C;'>
                <div class='ch-lbl'>In-Store Sales ({off_pct}%)</div>
                <div class='ch-val'>{off_u:,} units</div>
                <div class='ch-rev'>Rs.{off_u*eff_p:,.0f} revenue</div>
                <div class='ch-tip'>Shelf placement, footfall, in-store signage</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("---")

    # Chart + Financial | Recs + Decisions
    lc, rc = st.columns([1, 1])

    with lc:
        st.markdown("### 📈 Sales Trend + Forecast")
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [f"3 Mo Ago\n({m3:,})", f"2 Mo Ago\n({m2:,})",
                  f"Last Mo\n({lm:,})", f"Forecast\n({full_u:,})"]
        vals   = [m3, m2, lm, full_u]
        cols   = ['#546E7A','#78909C','#2E7D32','#F9A825']
        bars   = ax.bar(labels, vals, color=cols, edgecolor='#1E272E', width=0.55)
        ax.set_facecolor('#1E272E'); fig.patch.set_facecolor('#1E272E')
        ax.tick_params(colors='white', labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor('#546E7A')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+max(vals)*0.025,
                    f'{v:,}', ha='center', fontsize=10, fontweight='bold', color='white')
        ft = f" | {fest_info[0]} x{fm_mult}" if fest_info else ""
        ax.set_title(f"{category} — {pmn} {pred_year}{ft}",
                     fontweight='bold', color='white', pad=8, fontsize=10)
        ax.set_ylabel('Units', color='white')
        ax.grid(axis='y', color='#546E7A', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("### 💰 Financial Summary")
        fin_df = pd.DataFrame({
            'Metric': [
                'Forecast Period',
                f'Units by {forecast_mid.strftime("%d-%m-%Y")}',
                f'Units by {forecast_end.strftime("%d-%m-%Y")}',
                'Avg Selling Price', 'Discount Applied', 'Effective Price',
                'Expected Revenue', 'Expected Cost', 'Expected Profit',
                'Margin per Unit', 'Revenue Change', 'Festival Boost'
            ],
            'Value': [
                f"{forecast_start.strftime('%d %b')} - {forecast_end.strftime('%d %b %Y')}",
                f"{mid_u:,} units",
                f"{full_u:,} units",
                f"Rs.{avg_p:.2f}", f"{disc}%", f"Rs.{eff_p:.2f}",
                f"Rs.{full_rev:,.2f}", f"Rs.{cost_t:,.2f}", f"Rs.{full_prof:,.2f}",
                f"Rs.{margin:.2f}", f"{rev_chg:+.1f}%",
                f"x{fm_mult:.1f} ({fest_info[0]})" if fest_info else "None"
            ]
        })
        st.dataframe(fin_df, hide_index=True, use_container_width=True)

    with rc:
        st.markdown("### 📋 Business Recommendations")
        st.caption("High-level suggestions — what to consider and why.")
        for txt, cls in recs:
            st.markdown(f"<div class='rec-box {cls}'>{txt}</div>",
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🤖 Automated Decision System")
        st.caption("Exact actions with specific numbers, dates, and expected results.")
        for label, main, sub in decs:
            st.markdown(f"""<div class='dec-box'>
                <div class='dec-label'>{label}</div>
                <div class='dec-main'>{main}</div>
                <div class='dec-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Executive Summary
    st.markdown("---")
    fs = f" | {fest_info[0]}: x{fm_mult} boost" if fest_info else ""
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1B5E20,#33691E);
                color:white;padding:20px 26px;border-radius:12px;text-align:center;'>
        <h3 style='margin:0 0 10px 0;font-size:1.1em;'>
            EXECUTIVE SUMMARY — {company_name} | {category} | {pmn} {pred_year}
        </h3>
        <p style='margin:0;font-size:0.97em;line-height:2;'>
            Predicted: <strong>{full_u:,} units</strong> |
            Stock: <strong>{'Order '+str(gap)+' units' if gap>0 else 'Sufficient'}</strong> |
            Discount: <strong>{disc}%</strong> |
            Revenue: <strong>Rs.{full_rev:,.0f}</strong> |
            Profit: <strong>Rs.{full_prof:,.0f}</strong> |
            Change: <strong>{rev_chg:+.1f}%</strong>{fs}
        </p>
    </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────
st.markdown(f"""
<hr style='border-color:#333;margin-top:28px;'/>
<p style='text-align:center;color:#888;font-size:0.80em;line-height:2;'>
    🛒 <strong>{company_name} Sales Intelligence Platform</strong>
    &nbsp;|&nbsp; Built with Python · Scikit-learn · Streamlit
    &nbsp;|&nbsp; Supports: Online (Blinkit · Zepto · InstaMart) &amp; Offline (DMart · BigMart · Reliance)
    &nbsp;|&nbsp; BigMart Sales Forecasting &amp; Customer Insights System
</p>
""", unsafe_allow_html=True)
