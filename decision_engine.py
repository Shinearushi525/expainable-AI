"""
decision_engine.py

Turns a raw return-probability into a business DECISION: is this order
worth taking action on, given its price and the cost of intervening?

This is the same pattern used in real fraud/risk systems (credit cards,
insurance, lending): a probability alone isn't a decision — you compare
the EXPECTED COST of doing nothing against the COST of acting, and only
act when it's worth it.

------------------------------------------------------------------------
THE COST MODEL (edit these numbers to match real assumptions)
------------------------------------------------------------------------
INTERVENTION_COST
    What it costs the business to proactively act on a flagged order —
    e.g. a confirmation call/SMS, a size-guide popup, a packaging check.
    Assumed to be a small FIXED cost regardless of order size.

RESTOCKING_SHIPPING_COST
    The fixed cost incurred if an order IS returned — reverse shipping,
    warehouse handling, restocking labor. Same for every order.

MARGIN_LOSS_RATE
    The fraction of the product's price that's lost if it's returned
    (damaged packaging, discounted resale, write-off risk). Scales with
    the order's price — a returned ₹15,000 laptop loses more real money
    than a returned ₹200 t-shirt, even at the same risk %.

------------------------------------------------------------------------
THE DECISION RULE
------------------------------------------------------------------------
expected_loss_if_no_action = P(return) * (RESTOCKING_SHIPPING_COST
                                           + MARGIN_LOSS_RATE * product_price)

worth_intervening = expected_loss_if_no_action > INTERVENTION_COST

In words: "Only spend money proactively checking on an order if the
expected money saved by preventing its return is bigger than the cost
of checking in the first place."
"""

import pandas as pd

# ---- Cost assumptions (INR) — adjust to match a real business case ----
# NOTE: these were calibrated against this dataset's actual price distribution
# (median order ≈ ₹6,100, mean ≈ ₹15,800). At INTERVENTION_COST=50, almost every
# order clears the bar (98%+) because average order value is high relative to a
# trivially cheap intervention — that's not a useful decision boundary. ₹800
# represents a more realistic cost for meaningful proactive action (e.g. a phone
# call + manual order review, not just an automated SMS), and produces a
# decision that actually discriminates (~31% of orders flagged as worth acting on).
INTERVENTION_COST = 800.0         # cost to proactively act on a flagged order
RESTOCKING_SHIPPING_COST = 150.0  # fixed cost if a return actually happens
MARGIN_LOSS_RATE = 0.15           # % of product price lost on a return


def add_decision_columns(df: pd.DataFrame,
                          prob_col: str = "predicted_return_probability",
                          price_col: str = "product_price") -> pd.DataFrame:
    """
    Adds three columns to a dataframe that already has a predicted
    return probability and a product price:

        expected_loss        -> ₹ expected loss if we do nothing
        net_benefit_if_acted -> ₹ expected_loss - intervention_cost
        worth_intervening     -> True/False recommendation
    """
    df = df.copy()
    df["expected_loss"] = df[prob_col] * (
        RESTOCKING_SHIPPING_COST + MARGIN_LOSS_RATE * df[price_col]
    )
    df["net_benefit_if_acted"] = df["expected_loss"] - INTERVENTION_COST
    df["worth_intervening"] = df["net_benefit_if_acted"] > 0
    return df


def decide_single_order(probability: float, product_price: float) -> dict:
    """Same logic, for a single order (used by the Streamlit Predict tab)."""
    expected_loss = probability * (
        RESTOCKING_SHIPPING_COST + MARGIN_LOSS_RATE * product_price
    )
    net_benefit = expected_loss - INTERVENTION_COST
    return {
        "expected_loss": round(expected_loss, 2),
        "intervention_cost": INTERVENTION_COST,
        "net_benefit_if_acted": round(net_benefit, 2),
        "worth_intervening": net_benefit > 0,
    }


if __name__ == "__main__":
    # Quick manual sanity check
    examples = [
        {"probability": 0.72, "product_price": 15000},  # expensive, high risk
        {"probability": 0.72, "product_price": 200},    # cheap, high risk
        {"probability": 0.20, "product_price": 15000},  # expensive, low risk
    ]
    for ex in examples:
        result = decide_single_order(**ex)
        print(f"price=₹{ex['product_price']:<7} prob={ex['probability']:<5} -> {result}")
