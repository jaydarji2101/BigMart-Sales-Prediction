import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
#  PART 1: BUSINESS RECOMMENDATION ENGINE (Simple Level)
#  Returns human-friendly suggestions based on conditions
# ══════════════════════════════════════════════════════════

def business_recommendations(category, predicted_qty, avg_rating,
                              on_time_pct, marketing_roas):
    """
    Takes key metrics and returns bullet-point recommendations.
    
    Parameters:
    -----------
    category       : str   — product category name
    predicted_qty  : int   — predicted units for next month
    avg_rating     : float — average customer rating (1-5)
    on_time_pct    : float — % of orders delivered on time
    marketing_roas : float — Return on Ad Spend (e.g. 3.5 = ₹3.5 earned per ₹1 spent)
    
    Returns:
    --------
    list of recommendation strings
    """
    recommendations = []

    # ── Demand-based recommendations ─────────────────────
    if predicted_qty > 500:
        recommendations.append("🔥 HIGH DEMAND DETECTED → Increase stock levels immediately")
        recommendations.append("🚫 AVOID HEAVY DISCOUNTS → Product sells well at current price")
        recommendations.append("📦 PRIORITY RESTOCKING → This category needs weekly stock check")
    elif predicted_qty > 200:
        recommendations.append("📈 MODERATE DEMAND → Maintain current stock levels")
        recommendations.append("💡 LIGHT OFFERS OK → Small discount (5-10%) can boost volume")
    else:
        recommendations.append("📉 LOW DEMAND → Do not over-stock, risk of waste")
        recommendations.append("🎯 RUN PROMOTIONS → Flash sale or bundle offer recommended")
        recommendations.append("🔍 INVESTIGATE → Check if quality/pricing issue exists")

    # ── Quality-based recommendations ─────────────────────
    if avg_rating < 3.0:
        recommendations.append("⚠️  LOW RATINGS → Review product quality and supplier")
        recommendations.append("📧 RESPOND TO FEEDBACK → Send apology + coupon to low-raters")
    elif avg_rating >= 4.5:
        recommendations.append("⭐ EXCELLENT RATINGS → Feature in 'Top Picks' section")
        recommendations.append("📣 USE IN MARKETING → Highlight as customer favourite")

    # ── Delivery-based recommendations ─────────────────────
    if on_time_pct < 80:
        recommendations.append("🚴 DELIVERY ISSUES → Add more delivery partners in this area")
        recommendations.append("🗺️  OPTIMIZE ROUTES → Use dark store / hub-and-spoke model")
    elif on_time_pct >= 95:
        recommendations.append("✅ DELIVERY EXCELLENT → Use as USP in ads ('10-minute delivery')")

    # ── Marketing-based recommendations ─────────────────────
    if marketing_roas > 4.0:
        recommendations.append(f"💰 HIGH ROAS ({marketing_roas:.1f}x) → INCREASE marketing budget")
        recommendations.append("📊 SCALE ADS → Double ad spend on this campaign")
    elif marketing_roas < 2.0:
        recommendations.append(f"💸 LOW ROAS ({marketing_roas:.1f}x) → PAUSE or rethink this campaign")
        recommendations.append("🔄 TEST NEW CHANNEL → Try Instagram Reels or Google Shopping")

    return recommendations


# ══════════════════════════════════════════════════════════
#  PART 2: DECISION SYSTEM (Full Action Plan with IF-ELSE)
#  This gives SPECIFIC numbers and actions, not just words
# ══════════════════════════════════════════════════════════

def decision_system(category, predicted_qty, current_stock,
                    avg_selling_price, cost_price, on_time_pct,
                    avg_rating, marketing_roas):
    """
    Full decision engine. Takes inputs, returns a complete action plan.

    Example Input:
        category       = "Dairy & Breakfast"
        predicted_qty  = 500
        current_stock  = 300
        avg_price      = 120.0
        cost_price     = 70.0
        on_time_pct    = 85.0
        avg_rating     = 4.2
        marketing_roas = 3.5

    Example Output:
        📦 Order 250 units (target stock = 550)
        🚫 No discount — demand is high
        📈 Expected revenue = ₹60,000
        💰 Profit impact = +₹12,500 (+14%)
    """

    decisions = {}
    margin = avg_selling_price - cost_price

    # ── 1. STOCK DECISION ─────────────────────────────────
    safety_buffer  = int(predicted_qty * 0.10)   # 10% buffer for surprises
    target_stock   = predicted_qty + safety_buffer
    stock_gap      = target_stock - current_stock

    if stock_gap > 0:
        decisions['stock_action']   = f"📦 ORDER {stock_gap:,} MORE UNITS immediately"
        decisions['target_stock']   = f"Target stock level: {target_stock:,} units"
        decisions['stock_reason']   = f"Predicted demand = {predicted_qty:,}, current = {current_stock:,}, gap = {stock_gap:,}"
        decisions['stock_urgency']  = "🔴 URGENT" if stock_gap > 200 else "🟡 NORMAL"
    elif stock_gap < -100:
        excess = abs(stock_gap)
        decisions['stock_action']  = f"⚠️  REDUCE ORDER — Excess stock = {excess:,} units"
        decisions['target_stock']  = f"Target stock level: {target_stock:,} units"
        decisions['stock_reason']  = f"Predicted demand = {predicted_qty:,} but current stock = {current_stock:,}"
        decisions['stock_urgency'] = "🟢 LOW URGENCY"
    else:
        decisions['stock_action']  = "✅ STOCK IS OPTIMAL — No restocking needed"
        decisions['target_stock']  = f"Current stock ({current_stock:,}) matches demand"
        decisions['stock_urgency'] = "🟢 OK"

    # ── 2. PRICING / DISCOUNT DECISION ────────────────────
    if predicted_qty > 500:
        decisions['pricing_action'] = "🚫 NO DISCOUNT — High demand, customers will pay full price"
        decisions['pricing_reason'] = "Offering discount now = losing money unnecessarily"
        discount_pct = 0
    elif predicted_qty > 200:
        decisions['pricing_action'] = "💡 SMALL DISCOUNT OK (5-8%) to boost volume slightly"
        decisions['pricing_reason'] = "Moderate demand — gentle push will increase conversions"
        discount_pct = 6
    else:
        decisions['pricing_action'] = "🔥 RUN FLASH SALE — 15% discount for 48 hours"
        decisions['pricing_reason'] = "Low demand — needs promotion to clear stock"
        discount_pct = 15

    # ── 3. DELIVERY DECISION ──────────────────────────────
    if on_time_pct < 80:
        decisions['delivery_action'] = "🚴 ADD DELIVERY PARTNERS — current on-time rate too low"
        decisions['delivery_target'] = "Target: 90%+ on-time delivery within 2 weeks"
    elif on_time_pct < 90:
        decisions['delivery_action'] = "📍 MONITOR DELIVERY — slightly below target"
        decisions['delivery_target'] = "No major change needed; track daily"
    else:
        decisions['delivery_action'] = "✅ DELIVERY EXCELLENT — highlight in ads"
        decisions['delivery_target'] = "Maintain current delivery partner performance"

    # ── 4. MARKETING DECISION ─────────────────────────────
    if marketing_roas >= 4.0:
        budget_change = "+50%"
        decisions['marketing_action'] = f"📢 INCREASE AD BUDGET by 50% — ROAS={marketing_roas:.1f}x is excellent"
        decisions['marketing_channel']= "Double down on best-performing channel"
    elif marketing_roas >= 2.5:
        budget_change = "+10%"
        decisions['marketing_action'] = f"📊 MAINTAIN BUDGET — ROAS={marketing_roas:.1f}x is acceptable"
        decisions['marketing_channel']= "Optimize ad creatives to improve ROAS"
    else:
        budget_change = "-30%"
        decisions['marketing_action'] = f"🛑 CUT AD BUDGET by 30% — ROAS={marketing_roas:.1f}x is poor"
        decisions['marketing_channel']= "Pause current campaign; test new creative"

    # ── 5. CUSTOMER ENGAGEMENT DECISION ───────────────────
    if avg_rating >= 4.5:
        decisions['customer_action'] = "⭐ COLLECT REVIEWS — feature 5-star customers in ads"
    elif avg_rating >= 3.5:
        decisions['customer_action'] = "📧 REQUEST FEEDBACK — send post-delivery survey"
    else:
        decisions['customer_action'] = "🆘 URGENT: Respond to all negative reviews in 24 hours"

    # ── 6. FINANCIAL IMPACT ───────────────────────────────
    eff_price      = avg_selling_price * (1 - discount_pct/100)
    revenue_next   = predicted_qty * eff_price
    cost_next      = predicted_qty * cost_price
    profit_next    = revenue_next - cost_next
    curr_revenue   = current_stock * avg_selling_price
    revenue_change = ((revenue_next - curr_revenue) / max(curr_revenue, 1)) * 100

    decisions['financial'] = {
        'predicted_sales_units': predicted_qty,
        'discount_applied_pct' : discount_pct,
        'expected_revenue'     : round(revenue_next, 2),
        'expected_cost'        : round(cost_next, 2),
        'expected_profit'      : round(profit_next, 2),
        'revenue_change_pct'   : round(revenue_change, 1),
        'margin_per_unit'      : round(margin * (1 - discount_pct/100), 2)
    }

    return decisions


# ══════════════════════════════════════════════════════════
#  MAIN: Run the full system with sample data
# ══════════════════════════════════════════════════════════

def print_decision_report(category, predicted_qty, current_stock,
                           avg_price, cost_price, on_time_pct,
                           avg_rating, marketing_roas):
    print("\n" + "█" * 60)
    print(f"  📊 BLINKIT DECISION SYSTEM REPORT")
    print(f"  Category: {category}")
    print("█" * 60)

    print(f"""
  INPUT PARAMETERS:
  ─────────────────────────────────────────
  Predicted Sales (Next Month) : {predicted_qty:,} units
  Current Stock Available      : {current_stock:,} units
  Avg Selling Price            : ₹{avg_price:.2f}
  Cost Price                   : ₹{cost_price:.2f}
  On-Time Delivery Rate        : {on_time_pct:.1f}%
  Customer Rating              : {avg_rating:.1f} / 5.0
  Marketing ROAS               : {marketing_roas:.1f}x
""")

    # Recommendations
    recs = business_recommendations(
        category, predicted_qty, avg_rating, on_time_pct, marketing_roas)
    print("  📋 BUSINESS RECOMMENDATIONS:")
    print("  ─────────────────────────────────────────")
    for r in recs:
        print(f"  {r}")

    # Decisions
    dec = decision_system(
        category, predicted_qty, current_stock, avg_price,
        cost_price, on_time_pct, avg_rating, marketing_roas)
    fin = dec['financial']

    print(f"""
  🤖 AUTOMATED DECISION SYSTEM:
  ─────────────────────────────────────────
  STOCK    : {dec['stock_action']}
  ({dec['stock_urgency']}) {dec['stock_reason']}

  PRICING  : {dec['pricing_action']}
  Reason   : {dec['pricing_reason']}

  DELIVERY : {dec['delivery_action']}
  Target   : {dec['delivery_target']}

  MARKETING: {dec['marketing_action']}
  Action   : {dec['marketing_channel']}

  CUSTOMERS: {dec['customer_action']}

  💰 FINANCIAL IMPACT FORECAST:
  ─────────────────────────────────────────
  Predicted Units     : {fin['predicted_sales_units']:,}
  Discount Applied    : {fin['discount_applied_pct']}%
  Expected Revenue    : ₹{fin['expected_revenue']:,.2f}
  Expected Cost       : ₹{fin['expected_cost']:,.2f}
  Expected Profit     : ₹{fin['expected_profit']:,.2f}
  Revenue Change      : {fin['revenue_change_pct']:+.1f}%
  Margin per Unit     : ₹{fin['margin_per_unit']:.2f}
""")
    print("█" * 60)


# ── Run sample scenarios ───────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BLINKIT DECISION SYSTEM — 3 REAL SCENARIOS")
    print("=" * 60)

    # Scenario 1: High-demand category
    print_decision_report(
        category       = "Dairy & Breakfast",
        predicted_qty  = 520,
        current_stock  = 300,
        avg_price      = 125.0,
        cost_price     = 75.0,
        on_time_pct    = 88.0,
        avg_rating     = 4.3,
        marketing_roas = 3.8
    )

    # Scenario 2: Low-demand category
    print_decision_report(
        category       = "Pet Care",
        predicted_qty  = 85,
        current_stock  = 200,
        avg_price      = 350.0,
        cost_price     = 180.0,
        on_time_pct    = 92.0,
        avg_rating     = 4.6,
        marketing_roas = 1.8
    )

    # Scenario 3: Moderate demand with delivery issues
    print_decision_report(
        category       = "Fruits & Vegetables",
        predicted_qty  = 310,
        current_stock  = 250,
        avg_price      = 80.0,
        cost_price     = 45.0,
        on_time_pct    = 74.0,
        avg_rating     = 3.2,
        marketing_roas = 4.5
    )
