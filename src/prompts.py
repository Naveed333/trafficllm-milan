SYSTEM = (
    "You are a Milan hourly traffic forecaster. "
    "Hourly loads are non-negative floats with strong 24-hour periodicity. "
    "Follow the format shown in the examples. "
    "When asked to predict, output exactly 24 comma-separated non-negative numbers."
)

# ── p_ques (appended after p_input at test time) ──────────────────────────────
# Per paper Section II-B: p_ques "involves inquiries about future traffic" only.
# Q1-Q4 feedback belongs to p_feed (Phase A only). At test time the chain
# format is learned from p_exam demo examples, not from explicit instructions here.
PQUES = (
    "What are the predicted hourly traffic loads for the next 24 hours? "
    "Output exactly 24 comma-separated non-negative numbers on a single line."
)

# ── Phase-A prompt templates (multi-call iterative refinement with real GT) ───

def p_input_timestamped(x_times, x_values):
    rows = "\n".join(f"  {t}, {v:.2f}" for t, v in zip(x_times, x_values))
    return f"Past 24 hourly loads (timestamp UTC, value):\n{rows}"


def p_feed(x_times, x_values, abs_errors, mae_val, gt_sin_cos, y_hat, iteration):
    """
    abs_errors   : list of |ŷ_i - y_i| (pre-computed, GT not exposed raw)
    mae_val      : scalar MAE (pre-computed)
    gt_sin_cos   : (a_sin, a_cos) fitted on GT (pre-computed, GT not exposed raw)
    """
    x_str = ", ".join(f"{v:.2f}" for v in x_values)
    yhat_str = ", ".join(f"{v:.2f}" for v in y_hat)
    err_str = ", ".join(f"{e:.2f}" for e in abs_errors)
    a_sin_gt, a_cos_gt = gt_sin_cos
    return (
        f"Past 24 loads: {x_str}\n"
        f"Current prediction ŷ_{iteration}: {yhat_str}\n"
        f"Per-step absolute errors |ŷ - y| (t=0..23): {err_str}\n"
        f"MAE = {mae_val:.4f}\n\n"
        "Q1 (overall performance): The MAE above is exact. Identify which timesteps "
        "have the largest errors and describe the pattern of over- or under-prediction.\n"
        f"Q2 (periodical performance): The ground-truth sine/cosine projection "
        f"(period=24, t=0..23) has coefficients a_sin={a_sin_gt:.4f}, a_cos={a_cos_gt:.4f}. "
        f"Fit the current prediction to the same form A·sin(2π·t/24)+B·cos(2π·t/24) "
        f"and report its (a_sin, a_cos). State whether the predicted periodicity "
        f"matches ground-truth periodicity.\n"
        "Q3 (prediction format): Do the predictions provide a complete non-negative "
        "value for each of the 24 timestamps?\n"
        "Q4 (prediction method): What method was used in the current iteration? "
        "Propose a more accurate method (e.g., Seasonal ARIMA, LSTM+ARIMA hybrid)."
    )


PVALIDATE = (
    "Please review the previous answers and find potential mistakes, "
    "particularly in the MAE arithmetic in Q1 and the sin/cos projection in Q2. "
    "List each mistake on its own line, or write 'none' if there are none."
)

PCRITIQUE = (
    "Please correct the answers based on the identified mistakes. "
    "Re-emit Q1–Q4 in the same structure with corrected numbers. "
    "Be concise: for Q1 state only the final MAE value (no per-row table). "
    "Keep the entire response under 150 words."
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
