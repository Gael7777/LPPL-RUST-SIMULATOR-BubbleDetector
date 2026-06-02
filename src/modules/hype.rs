/// Compute a simple hype proxy from volume series.
/// Returns a vector of normalized hype values (z-score like over rolling window).
/// High recent volume relative to history => positive hype.
pub fn compute_volume_hype(volumes: &[f64], window: usize) -> Vec<f64> {
    let n = volumes.len();
    if n == 0 {
        return vec![];
    }
    let w = window.min(n).max(5);

    let mut hype = vec![0.0; n];

    for i in 0..n {
        let start = i.saturating_sub(w);
        let slice = &volumes[start..=i];

        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / slice.len() as f64;
        let std = (var + 1e-9).sqrt();

        let z = (volumes[i] - mean) / std;
        // squash a bit and shift so neutral ~ 0
        hype[i] = (z * 0.6).tanh();
    }
    hype
}

// Optional: market-relative hype (volume / avg SPX volume or sector).
// For single-ticker backtests we just use the absolute volume z-score above.
