/// Very simple sentiment proxy from returns (positive momentum = mild positive).
/// In full version this would call FinBERT ONNX on headlines.
pub fn compute_simple_sentiment(daily_returns: &[f64]) -> Vec<f64> {
    daily_returns
        .iter()
        .map(|r| r.signum() * 0.15 + (r * 0.3).clamp(-0.4, 0.4)) // mild signal
        .collect()
}

/// Neutral sentiment (used when news not available)
pub fn neutral_sentiment(n: usize) -> Vec<f64> {
    vec![0.0; n]
}
