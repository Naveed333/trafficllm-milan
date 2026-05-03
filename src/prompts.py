SYSTEM = (
    "You are a Milan hourly traffic forecaster. "
    "Hourly loads are non-negative floats with strong 24-hour periodicity. "
    "Follow the format of the examples: produce an initial prediction, "
    "4-aspect self-feedback, then a refined prediction. "
    "The line beginning 'Refined prediction:' must contain exactly 24 "
    "comma-separated non-negative numbers and nothing else."
)

# ── p_ques (appended after p_input at test time) ──────────────────────────────
PQUES = (
    "Produce, in this exact format:\n"
    "Initial prediction: <24 comma-separated numbers>\n"
    "Feedback:\n"
    "  Q1 (overall performance): What is the Mean Absolute Error you would expect "
    "for these predictions, given the historical pattern? Show arithmetic; output a "
    "single number.\n"
    "  Q2 (periodical performance): For the past 24 loads and your predicted 24 "
    "loads, what are their projected functions derived from the combination of sine "
    "and cosine functions (period=24h)? Report (a_sin, a_cos) for each and state "
    "whether they align.\n"
    "  Q3 (prediction format): Do the predictions align with the format of the "
    "historical loads and provide a complete prediction for each timestamp?\n"
    "  Q4 (prediction method): What is the prediction method applied in the current "
    "iteration, and which more accurate method (numerical, machine learning, or "
    "hybrid) would you use next?\n"
    "Refined prediction: <24 comma-separated numbers>"
)

# ── Phase-A prompt templates (multi-call iterative refinement with real GT) ───

def p_input_timestamped(x_times, x_values):
    rows = "\n".join(f"  {t}, {v:.2f}" for t, v in zip(x_times, x_values))
    return f"Past 24 hourly loads (timestamp UTC, value):\n{rows}"


def p_feed(x_times, x_values, y_times, y_values, y_hat, iteration):
    x_str = ", ".join(f"{v:.2f}" for v in x_values)
    y_str = ", ".join(f"{v:.2f}" for v in y_values)
    yhat_str = ", ".join(f"{v:.2f}" for v in y_hat)
    return (
        f"Past 24 loads: {x_str}\n"
        f"Ground-truth next 24 loads: {y_str}\n"
        f"Current prediction ŷ_{iteration}: {yhat_str}\n\n"
        "Q1 (overall performance): What is the Mean Absolute Error of the predictions? "
        "Compute MAE = mean(|ŷ − y|). Show arithmetic; output the number.\n"
        "Q2 (periodical performance): For ground truths and predictions, what are their "
        "projected functions derived from the combination of sine and cosine functions "
        "(period=24)? Report (a_sin, a_cos) coefficients for each; state whether "
        "predicted periodicity matches ground-truth periodicity.\n"
        "Q3 (prediction format): Do the predictions align with the format of the ground "
        "truths and provide a complete prediction for each timestamp?\n"
        "Q4 (prediction method): What is the prediction method applied in the current "
        "iteration? Propose a more accurate method (numerical, machine learning, or "
        "hybrid; e.g., Seasonal ARIMA, LSTM+ARIMA hybrid)."
    )


PVALIDATE = (
    "Please review the previous answers and find potential mistakes, "
    "particularly in the MAE arithmetic in Q1 and the sin/cos projection in Q2. "
    "List each mistake on its own line, or write 'none' if there are none."
)

PCRITIQUE = (
    "Please correct the answers based on the identified mistakes. "
    "Re-emit Q1–Q4 in the same structure with corrected numbers."
)

def p_refine(iteration):
    return (
        "Please refine predictions based on the previous thorough feedback. "
        "To enhance performance, more accurate time series prediction methods should "
        "be considered, including numerical methods, machine learning methods, and "
        "hybrid methods. The prediction should match the function of the real Milan "
        "hourly traffic. The prediction should be complete and match the format of "
        "the real Milan hourly traffic. "
        f"Output 24 comma-separated non-negative numbers on a single line "
        f"(iteration {iteration + 1} refined prediction)."
    )


def render_demo_chain(train_idx, x_times, x_values, y_hat_initial,
                      feedback_corrected, y_hat_refined):
    x_rows = "\n".join(f"  {t}, {v:.2f}" for t, v in zip(x_times, x_values))
    init_str = ", ".join(f"{v:.2f}" for v in y_hat_initial)
    refined_str = ", ".join(f"{v:.2f}" for v in y_hat_refined)
    return (
        f"[Example (train idx {train_idx})]\n"
        f"Past 24 hourly loads (timestamp UTC, value):\n{x_rows}\n"
        f"Initial prediction: {init_str}\n"
        f"Feedback:\n{feedback_corrected}\n"
        f"Refined prediction: {refined_str}"
    )
