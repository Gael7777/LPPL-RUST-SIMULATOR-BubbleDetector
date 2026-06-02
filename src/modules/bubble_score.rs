/// Paper-inspired Bubble Score combining LPPL residual signal with hype/sentiment.
/// epsilon_norm : typically last residual / rolling std(residuals)  (positive = overpriced relative to LPPL)
/// hype & sentiment are normalized ~ [-1, 1] or z-scores.
pub fn compute_bubble_score(
    epsilon_norm: f64,
    hype: f64,
    sentiment: f64,
    alpha1: f64, // weight on hype
    alpha2: f64, // weight on sentiment
) -> f64 {
    if epsilon_norm >= 0.0 {
        epsilon_norm + alpha1 * hype + alpha2 * sentiment
    } else {
        epsilon_norm - alpha1 * hype - alpha2 * sentiment
    }
}

/// Normalize a residual series (last value relative to its own std)
pub fn normalize_last_residual(residuals: &[f64]) -> f64 {
    if residuals.len() < 5 {
        return 0.0;
    }
    let n = residuals.len();
    let last = residuals[n - 1];

    let mean = residuals.iter().sum::<f64>() / n as f64;
    let var = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let std = (var + 1e-12).sqrt();

    last / std
}
