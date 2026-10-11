"""
app.py

Full Streamlit app for the E-Commerce Return Prediction pipeline.

Tabs:
    1. Predict            - score a single order via a form, with a plain-English
                             explanation, cost-based decision engine, and live what-if sliders
    2. Batch Prediction    - upload a CSV, download scored results
    3. EDA Dashboard       - explore return patterns in the dataset
    4. Returns Map         - India map colored by city return rate
    5. BigQuery Pipeline   - live data pulled from BigQuery mart layer (optional)

Run locally:
    python train_model.py     # produces best_model.pkl, encoders.pkl
    streamlit run app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from train_model import engineer_features, encode_features, FEATURE_COLS, CATEGORICAL_COLS
from decision_engine import decide_single_order, add_decision_columns, INTERVENTION_COST
from explain import get_explainer, explain_order

st.set_page_config(page_title="E-Commerce Return Prediction", page_icon="🛒", layout="wide")

DATA_PATH = "ecommerce_product_returns.csv"
MODEL_PATH = "best_model.pkl"
ENCODERS_PATH = "encoders.pkl"
def _resolve_project_id():
    pid = os.environ.get("GCP_PROJECT_ID")
    if pid:
        return pid
    try:
        return st.secrets["gcp_service_account"]["project_id"]
    except Exception:
        return "your-gcp-project-id"


PROJECT_ID = _resolve_project_id()


@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        st.warning("No trained model found — training now (first launch only)...")
        import train_model
        train_model.main()
    bundle = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return bundle["model"], bundle["model_name"], encoders


@st.cache_resource
def load_explainer(_model):
    return get_explainer(_model)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["order_date"])


def risk_tier(prob):
    if prob >= 0.7:
        return "HIGH", "🔴"
    elif prob >= 0.4:
        return "MEDIUM", "🟡"
    return "LOW", "🟢"


def render_gauge(prob):
    """Renders a gauge chart like the risk-gauge screenshot, using Plotly."""
    tier, _ = risk_tier(prob)
    tier_colors = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": "Return Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": tier_colors[tier]},
            "steps": [
                {"range": [0, 40], "color": "#dcfce7"},
                {"range": [40, 70], "color": "#fef9c3"},
                {"range": [70, 100], "color": "#fee2e2"},
            ],
        },
    ))
    fig.update_layout(height=350, margin=dict(t=50, b=10, l=30, r=30))
    return fig


model, model_name, encoders = load_model_bundle()
df = load_data()
shap_explainer = load_explainer(model)

st.title("🛒 E-Commerce Product Return Prediction")
st.caption(f"Best model in use: **{model_name}**")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Predict", "📁 Batch Prediction", "📊 EDA Dashboard",
    "🎚️ What-If Explorer", "🗺️ Returns Map", "☁️ BigQuery Pipeline",
])

# ---------------- TAB 1: Single prediction ----------------
with tab1:
    st.subheader("Score a Single Order")
    col1, col2, col3 = st.columns(3)

    with col1:
        customer_age = st.number_input("Customer Age", 18, 80, 30)
        customer_type = st.selectbox("Customer Type", sorted(df["customer_type"].unique()))
        customer_city = st.selectbox("Customer City", sorted(df["customer_city"].unique()))
        product_category = st.selectbox("Product Category", sorted(df["product_category"].unique()))

    with col2:
        subcats = sorted(df[df["product_category"] == product_category]["product_subcategory"].unique())
        product_subcategory = st.selectbox("Product Subcategory", subcats)
        product_price = st.number_input("Product Price (₹)", 100.0, 100000.0, 2500.0)
        discount_percent = st.slider("Discount %", 0, 80, 20)
        order_quantity = st.number_input("Order Quantity", 1, 10, 1)

    with col3:
        payment_method = st.selectbox("Payment Method", sorted(df["payment_method"].unique()))
        shipping_type = st.selectbox("Shipping Type", sorted(df["shipping_type"].unique()))
        delivery_days = st.slider("Delivery Days", 1, 15, 5)
        seller_rating = st.slider("Seller Rating", 2.5, 5.0, 4.0)
        customer_rating = st.slider("Customer Rating", 1, 5, 4)

    col4, col5 = st.columns(2)
    with col4:
        warranty_available = st.selectbox("Warranty Available", ["Yes", "No"])
    with col5:
        product_size_available = st.selectbox("Size Available", ["Yes", "No"])

    if st.button("Predict Return Risk", type="primary"):
        order_date = pd.Timestamp.now()
        input_df = pd.DataFrame([{
            "order_date": order_date, "customer_age": customer_age, "customer_type": customer_type,
            "customer_city": customer_city, "product_category": product_category,
            "product_subcategory": product_subcategory, "product_price": product_price,
            "discount_percent": discount_percent, "order_quantity": order_quantity,
            "payment_method": payment_method, "shipping_type": shipping_type,
            "delivery_days": delivery_days, "seller_rating": seller_rating,
            "customer_rating": customer_rating, "warranty_available": warranty_available,
            "product_size_available": product_size_available,
        }])
        input_df = engineer_features(input_df)
        input_enc, _ = encode_features(input_df, encoders=encoders, fit=False)
        prob = model.predict_proba(input_enc[FEATURE_COLS])[0, 1]
        tier, emoji = risk_tier(prob)
        verdict = "Will Return" if prob >= 0.5 else "Not Returned"

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Prediction")
            st.markdown(f"## {verdict}")
        with c2:
            st.caption("Return Probability")
            st.markdown(f"## {prob*100:.1f}%")
        with c3:
            st.caption("")
            st.markdown(f"## {emoji} {tier} RISK")

        st.plotly_chart(render_gauge(prob), use_container_width=True)

        signals = []
        if discount_percent >= 40:
            signals.append("High discount (≥40%)")
        if delivery_days > 5:
            signals.append("Late delivery (>5 days)")
        if seller_rating < 3.5:
            signals.append("Low seller rating (<3.5)")
        if payment_method == "Cash on Delivery":
            signals.append("Cash on Delivery payment")
        if signals:
            st.warning("⚠️ Signals detected: " + " · ".join(signals))
        else:
            st.success("✅ No major risk signals detected")

        # ---- Decision Engine: is this order worth acting on? ----
        st.markdown("---")
        st.markdown("### 💰 Business Decision")
        decision = decide_single_order(prob, product_price)

        d1, d2, d3 = st.columns(3)
        d1.metric("Expected Loss if No Action", f"₹{decision['expected_loss']:.0f}")
        d2.metric("Cost to Intervene", f"₹{decision['intervention_cost']:.0f}")
        d3.metric("Net Benefit if Acted", f"₹{decision['net_benefit_if_acted']:.0f}")

        if decision["worth_intervening"]:
            st.success(
                f"✅ **Worth intervening** — expected savings of "
                f"₹{decision['net_benefit_if_acted']:.0f} outweigh the "
                f"₹{decision['intervention_cost']:.0f} cost of acting "
                f"(e.g. a confirmation call, size-guide prompt, or packaging check)."
            )
        else:
            st.info(
                f"➖ **Not worth intervening** — the expected loss "
                f"(₹{decision['expected_loss']:.0f}) doesn't justify the "
                f"₹{decision['intervention_cost']:.0f} cost of proactive action. "
                f"Let this order proceed normally."
            )
        st.caption(
            "Decision = predicted return probability × (restocking/shipping cost + "
            "margin lost on this order's price), compared against the fixed cost of "
            "intervening. See decision_engine.py to adjust these cost assumptions."
        )

        # ---- SHAP Explanation: WHY did the model flag this order? ----
        st.markdown("---")
        st.markdown("### 🔍 Why This Prediction?")
        try:
            explanation = explain_order(shap_explainer, input_enc[FEATURE_COLS], top_n=3)
            st.write(explanation["explanation"])
            shap_df = pd.DataFrame(explanation["top_features"])
            fig_shap = px.bar(
                shap_df, x="impact", y="feature", orientation="h",
                title="Top factors driving this prediction",
                color="impact", color_continuous_scale=["#22c55e", "#ef4444"],
            )
            fig_shap.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")

# ---------------- TAB 2: Batch prediction ----------------
with tab2:
    st.subheader("Batch Score a CSV of Orders")
    st.caption("Upload a CSV with the same columns as the training data (minus is_returned/return_reason).")
    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
        batch_df = pd.read_csv(uploaded, parse_dates=["order_date"])
        batch_feat = engineer_features(batch_df)
        batch_enc, _ = encode_features(batch_feat, encoders=encoders, fit=False)
        batch_df["predicted_return_probability"] = model.predict_proba(batch_enc[FEATURE_COLS])[:, 1]
        batch_df["risk_tier"] = batch_df["predicted_return_probability"].apply(lambda p: risk_tier(p)[0])
        batch_df = add_decision_columns(batch_df)

        n_worth = int(batch_df["worth_intervening"].sum())
        total_savings = batch_df.loc[batch_df["worth_intervening"], "net_benefit_if_acted"].sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Orders Scored", len(batch_df))
        m2.metric("Worth Intervening On", n_worth)
        m3.metric("Total Potential Savings", f"₹{total_savings:,.0f}")

        st.dataframe(batch_df)
        st.download_button(
            "Download Scored CSV",
            batch_df.to_csv(index=False).encode("utf-8"),
            "scored_orders.csv",
            "text/csv",
        )

# ---------------- TAB 3: EDA ----------------
with tab3:
    st.subheader("📊 Exploratory Data Analysis")

    # ---- Return Reasons donut chart ----
    st.markdown("#### Return Reasons")
    reasons = df[df["is_returned"] == 1]["return_reason"].value_counts()
    donut_colors = [
        "#1f6fd8", "#7fc4f7", "#e8342a", "#f6a9a5",
        "#1e9c8c", "#7ee6ac", "#f28c1e", "#f7d08a", "#7b2fd6",
    ]
    fig_reasons = px.pie(
        values=reasons.values, names=reasons.index, hole=0.35,
        color_discrete_sequence=donut_colors,
    )
    fig_reasons.update_traces(textinfo="percent", textfont_size=14)
    fig_reasons.update_layout(legend=dict(orientation="v", x=1.05, y=0.5))
    st.plotly_chart(fig_reasons, use_container_width=True)

    st.markdown("---")

    BLUE = "#1f6fd8"

    # ---- KPI Header Row ----
    k1, k2, k3, k4 = st.columns(4)
    overall_rate = df["is_returned"].mean() * 100
    k1.metric("Overall Return Rate", f"{overall_rate:.1f}%")
    best_cat = df.groupby("product_category")["is_returned"].mean().idxmin()
    worst_cat = df.groupby("product_category")["is_returned"].mean().idxmax()
    k2.metric("Lowest-Risk Category", best_cat)
    k3.metric("Highest-Risk Category", worst_cat)
    k4.metric("Total Orders Analyzed", f"{len(df):,}")

    st.markdown("---")

    # ---- Interactive dimension explorer (one chart, many breakdowns) ----
    st.markdown("#### 🔎 Explore Return Rate By...")
    dimension_options = {
        "Product Category": "product_category",
        "Product Subcategory": "product_subcategory",
        "Customer City": "customer_city",
        "Customer Type": "customer_type",
        "Payment Method": "payment_method",
        "Shipping Type": "shipping_type",
    }
    chosen_label = st.selectbox("Choose a dimension to break down return rate by:", list(dimension_options.keys()))
    chosen_col = dimension_options[chosen_label]

    explore_data = (df.groupby(chosen_col)["is_returned"].agg(["mean", "count"]) * [100, 1]).round(2)
    explore_data.columns = ["return_rate", "order_count"]
    explore_data = explore_data.sort_values("return_rate", ascending=False).reset_index()

    fig_explore = px.bar(
        explore_data, x=chosen_col, y="return_rate", color="return_rate",
        color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
        hover_data={"order_count": True},
        labels={"return_rate": "Return Rate (%)"},
        title=f"Return Rate by {chosen_label}",
    )
    fig_explore.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(fig_explore, use_container_width=True)

    st.markdown("---")

    # ---- Treemap: category > subcategory sized by orders, colored by return rate ----
    st.markdown("#### 🌳 Category Breakdown (size = order volume, color = return rate)")
    treemap_data = df.groupby(["product_category", "product_subcategory"]).agg(
        order_count=("order_id", "count"), return_rate=("is_returned", "mean")
    ).reset_index()
    fig_tree = px.treemap(
        treemap_data, path=["product_category", "product_subcategory"], values="order_count",
        color="return_rate", color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
        hover_data={"return_rate": ":.1%"},
    )
    fig_tree.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("---")

    col_sun, col_heat = st.columns(2)

    with col_sun:
        st.markdown("#### ☀️ Payment Flow → Return Outcome")
        sun_data = df.copy()
        sun_data["outcome"] = sun_data["is_returned"].map({0: "Kept", 1: "Returned"})
        fig_sun = px.sunburst(
            sun_data, path=["payment_method", "outcome"],
            color="outcome", color_discrete_map={"Kept": "#1f6fd8", "Returned": "#ef4444"},
        )
        fig_sun.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_sun, use_container_width=True)

    with col_heat:
        st.markdown("#### 🌡️ Correlation Heatmap")
        numeric_cols = ["customer_age", "product_price", "discount_percent", "order_quantity",
                         "delivery_days", "seller_rating", "customer_rating", "is_returned"]
        corr = df[numeric_cols].corr().round(2)
        fig_heat = px.imshow(
            corr, text_auto=True, color_continuous_scale=["#ef4444", "#ffffff", "#1f6fd8"],
            zmin=-1, zmax=1, aspect="auto",
        )
        fig_heat.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_heat, use_container_width=True)

# ---------------- TAB 4: What-If Explorer ----------------
with tab4:
    st.subheader("🎚️ Live What-If Explorer")
    st.caption("Drag these sliders to see the predicted return risk update instantly.")

    wif_col1, wif_col2 = st.columns(2)
    with wif_col1:
        wif_discount = st.slider("Discount %", 0, 80, 20, key="wif_discount")
        wif_delivery = st.slider("Delivery Days", 1, 15, 5, key="wif_delivery")
    with wif_col2:
        wif_price = st.slider("Product Price (₹)", 100, 90000, 5000, key="wif_price")
        wif_seller_rating = st.slider("Seller Rating", 2.5, 5.0, 4.0, key="wif_seller")

    wif_input = pd.DataFrame([{
        "order_date": pd.Timestamp.now(), "customer_age": 30, "customer_type": "New",
        "customer_city": df["customer_city"].mode()[0], "product_category": df["product_category"].mode()[0],
        "product_subcategory": df["product_subcategory"].mode()[0], "product_price": wif_price,
        "discount_percent": wif_discount, "order_quantity": 1,
        "payment_method": df["payment_method"].mode()[0], "shipping_type": df["shipping_type"].mode()[0],
        "delivery_days": wif_delivery, "seller_rating": wif_seller_rating,
        "customer_rating": 4, "warranty_available": "Yes", "product_size_available": "Yes",
    }])
    wif_feat = engineer_features(wif_input)
    wif_enc, _ = encode_features(wif_feat, encoders=encoders, fit=False)
    wif_prob = model.predict_proba(wif_enc[FEATURE_COLS])[0, 1]
    wif_tier, wif_emoji = risk_tier(wif_prob)

    wc1, wc2 = st.columns([1, 2])
    with wc1:
        st.metric("Live Return Risk", f"{wif_prob*100:.1f}%", f"{wif_emoji} {wif_tier}")
        wif_decision = decide_single_order(wif_prob, wif_price)
        if wif_decision["worth_intervening"]:
            st.success(f"✅ Worth intervening (net ₹{wif_decision['net_benefit_if_acted']:.0f})")
        else:
            st.info(f"➖ Not worth intervening (net ₹{wif_decision['net_benefit_if_acted']:.0f})")
    with wc2:
        st.plotly_chart(render_gauge(wif_prob), use_container_width=True)

# ---------------- TAB 5: Returns Map ----------------
with tab5:

    CITY_COORDS = {
        "Delhi": (28.7041, 77.1025), "Mumbai": (19.0760, 72.8777), "Bangalore": (12.9716, 77.5946),
        "Chennai": (13.0827, 80.2707), "Kolkata": (22.5726, 88.3639), "Hyderabad": (17.3850, 78.4867),
        "Pune": (18.5204, 73.8567), "Ahmedabad": (23.0225, 72.5714), "Jaipur": (26.9124, 75.7873),
        "Lucknow": (26.8467, 80.9462), "Surat": (21.1702, 72.8311), "Nagpur": (21.1458, 79.0882),
        "Indore": (22.7196, 75.8577), "Bhopal": (23.2599, 77.4126), "Patna": (25.5941, 85.1376),
        "Chandigarh": (30.7333, 76.7794), "Coimbatore": (11.0168, 76.9558), "Jalandhar": (31.3260, 75.5762),
    }
    city_stats = df.groupby("customer_city").agg(
        return_rate=("is_returned", "mean"), order_count=("order_id", "count")
    ).reset_index()
    city_stats["lat"] = city_stats["customer_city"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    city_stats["lon"] = city_stats["customer_city"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
    city_stats = city_stats.dropna(subset=["lat", "lon"])

    try:
        fig_map = px.scatter_map(
            city_stats, lat="lat", lon="lon", size="order_count", color="return_rate",
            color_continuous_scale="RdYlGn_r", hover_name="customer_city",
            hover_data={"return_rate": ":.2%", "order_count": True, "lat": False, "lon": False},
            zoom=3.5, height=550, title="Return Rate by City (darker red = higher return rate)",
        )
        fig_map.update_layout(map_style="open-street-map", margin=dict(t=50, b=0, l=0, r=0))
    except AttributeError:
        # Older Plotly versions use scatter_mapbox / mapbox_style instead
        fig_map = px.scatter_mapbox(
            city_stats, lat="lat", lon="lon", size="order_count", color="return_rate",
            color_continuous_scale="RdYlGn_r", hover_name="customer_city",
            hover_data={"return_rate": ":.2%", "order_count": True, "lat": False, "lon": False},
            zoom=3.5, height=550, title="Return Rate by City (darker red = higher return rate)",
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin=dict(t=50, b=0, l=0, r=0))
    st.plotly_chart(fig_map, use_container_width=True)

# ---------------- TAB 6: BigQuery pipeline ----------------
with tab6:
    st.subheader("☁️ Live Data from BigQuery")
    st.caption("Reads mart.fact_orders + mart.fact_predictions once the pipeline has been run.")

    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account

        @st.cache_resource
        def get_bq_client():
            if "gcp_service_account" in st.secrets:
                creds = service_account.Credentials.from_service_account_info(
                    dict(st.secrets["gcp_service_account"])
                )
                return bigquery.Client(project=PROJECT_ID, credentials=creds)
            raw_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if raw_json:
                creds = service_account.Credentials.from_service_account_info(json.loads(raw_json))
                return bigquery.Client(project=PROJECT_ID, credentials=creds)
            return bigquery.Client(project=PROJECT_ID)

        if st.button("Load from BigQuery"):
            client = get_bq_client()
            bq_df = client.query(f"SELECT * FROM `{PROJECT_ID}.mart.fact_orders`").to_dataframe()
            try:
                preds = client.query(f"SELECT * FROM `{PROJECT_ID}.mart.fact_predictions`").to_dataframe()
                bq_df = bq_df.merge(preds, on="order_id", how="left")
            except Exception:
                pass
            st.success(f"Loaded {len(bq_df)} rows from BigQuery")
            st.dataframe(bq_df.head(50))
    except Exception as e:
        st.error(f"BigQuery not configured yet: {e}")
        st.info("See DEPLOY_SETUP.md for setting up credentials as a secret.")
