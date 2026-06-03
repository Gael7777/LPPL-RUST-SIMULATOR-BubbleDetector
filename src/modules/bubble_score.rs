/// Paper-inspired Bubble Score combining LPPL residual signal with hype/sentiment.
/// epsilon_norm : in [-1, 1] from running-max normalization (Eq. 8 in the paper)
/// hype & sentiment are normalized ~ [-1, 1] or z-scores.
pub fn compute_bubble_score(
    epsilon_norm: f64,
    hype: f64,
    sentiment: f64,
    alpha1: f64,
    alpha2: f64,
) -> f64 {
    if epsilon_norm >= 0.0 {
        epsilon_norm + alpha1 * hype + alpha2 * sentiment
    } else {
        epsilon_norm - alpha1 * hype + alpha2 * sentiment
    }
}

/// Causal running-max normalization (paper Eq. 8): ε_norm(t) = ε(t) / max_{s≤t} |ε(s)|.
/// Values before `start_idx` are left at 0.
pub fn normalize_running_max(raw: &[f64], start_idx: usize) -> Vec<f64> {
    let n = raw.len();
    let mut out = vec![0.0; n];
    let mut running_max = 0.0f64;
    for i in start_idx..n {
        running_max = running_max.max(raw[i].abs());
        out[i] = if running_max > 1e-12 {
            (raw[i] / running_max).clamp(-1.0, 1.0)
        } else {
            0.0
        };
    }
    out
}

/// Last-day ε_norm from a single fit window using running max **within that window only**
/// (used in fast mode between full-series rebuilds).
pub fn normalize_last_residual_running_style(residuals: &[f64]) -> f64 {
    if residuals.is_empty() {
        return 0.0;
    }
    let running_max = residuals.iter().map(|r| r.abs()).fold(0.0f64, f64::max);
    let last = *residuals.last().unwrap();
    if running_max > 1e-12 {
        (last / running_max).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Legacy z-score normalization (deprecated — kept for API stability).
#[deprecated(note = "Use normalize_running_max or paper signal pipeline instead")]
pub fn normalize_last_residual(residuals: &[f64]) -> f64 {
    normalize_last_residual_running_style(residuals)
}
