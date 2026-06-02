use nalgebra::{Cholesky, DMatrix, DVector, LU};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
use std::f64::consts::PI;
use chrono::NaiveDate;

/// 7-parameter LPPL model parameters (HLPPL / HLPLL)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LpplParams {
    pub tc: f64,    // critical time (in same units as t, usually > t_max)
    pub m: f64,     // power law exponent (0 < m < 1)
    pub omega: f64, // angular frequency of log-periodic oscillations (>0)
    pub a: f64,     // constant
    pub b: f64,     // power law amplitude (usually <0 for bubbles)
    pub c: f64,     // log-periodic amplitude
    pub phi: f64,   // phase
}

impl LpplParams {
    /// Predict log-price at time t (tau = tc - t clipped > epsilon)
    pub fn predict_log_price(&self, t: f64) -> f64 {
        let tau = (self.tc - t).max(1e-8);
        let tau_m = tau.powf(self.m);
        let osc = (self.omega * tau.ln() + self.phi).cos();
        self.a + self.b * tau_m + self.c * tau_m * osc
    }

    /// Damping condition often required for valid LPPL bubble fits
    pub fn is_valid(&self) -> bool {
        self.tc > 0.0
            && self.m > 0.0
            && self.m < 1.0
            && self.omega > 0.5
            && self.omega < 30.0
            && (self.c.abs() < self.b.abs() * (1.0 + self.m * self.m).sqrt() + 1e-6)
    }
}

#[derive(Debug, Clone)]
pub struct LpplFit {
    pub params: LpplParams,
    pub sse: f64,
    pub residuals: Vec<f64>,
    pub times: Vec<f64>,
    pub n_points: usize,
}

/// Fit the LPPL model to log-prices using pure multi-start random search + closed-form
/// linear solve for (A,B,C). The `seed` makes the search deterministic/reproducible:
/// the same seed + same data always yields the same best parameters (and thus same
/// bubble scores and strategy results).
///
/// No external optimizer crate (avoids LAPACK/Windows linking issues).
///
/// times: increasing sequence, typically 0.0 .. N (trading days index)
/// log_prices: ln(close) or ln(adj_close), same length, positive prices.
pub fn fit_lppl(log_prices: &[f64], times: &[f64], seed: u64) -> Result<LpplFit, String> {
    if log_prices.len() != times.len() || log_prices.len() < 30 {
        return Err("Need at least 30 points for LPPL fit".to_string());
    }

    let n = log_prices.len();
    let t_max = times[n - 1];
    let t_span = t_max - times[0];

    // Reasonable search bounds for tc (critical time slightly in future)
    let tc_min = t_max + 1.0;
    let tc_max = t_max + (t_span * 0.4).max(30.0);

    let mut rng: StdRng = StdRng::seed_from_u64(seed);
    let n_samples = 1200usize; // plenty for demo / backtest speed

    let mut best_params: Option<LpplParams> = None;
    let mut best_sse = f64::INFINITY;

    for _ in 0..n_samples {
        let tc = rng.gen_range(tc_min..tc_max);
        let m = rng.gen_range(0.05..0.95);
        let omega = rng.gen_range(3.0..18.0);
        let phi = rng.gen_range(-PI..PI);

        let (a, b, c, sse) = linear_solve_abc(log_prices, times, tc, m, omega, phi);

        let candidate = LpplParams {
            tc,
            m,
            omega,
            a,
            b,
            c,
            phi,
        };

        if sse < best_sse && candidate.is_valid() {
            best_sse = sse;
            best_params = Some(candidate);
        }
    }

    // If no valid found, relax omega/m a bit and retry a few
    if best_params.is_none() {
        for _ in 0..300 {
            let tc = rng.gen_range(tc_min..tc_max);
            let m = rng.gen_range(0.01..0.99);
            let omega = rng.gen_range(1.0..22.0);
            let phi = rng.gen_range(-PI..PI);

            let (a, b, c, sse) = linear_solve_abc(log_prices, times, tc, m, omega, phi);
            let candidate = LpplParams { tc, m, omega, a, b, c, phi };
            if sse < best_sse && candidate.m > 0.0 && candidate.m < 1.0 && candidate.omega > 0.1 {
                best_sse = sse;
                best_params = Some(candidate);
            }
        }
    }

    let params = best_params.ok_or_else(|| "LPPL multi-start search failed to find valid fit".to_string())?;

    // Recompute residuals with best params
    let mut residuals = Vec::with_capacity(n);
    for i in 0..n {
        let fitted = params.predict_log_price(times[i]);
        residuals.push(log_prices[i] - fitted);
    }

    Ok(LpplFit {
        params,
        sse: best_sse,
        residuals,
        times: times.to_vec(),
        n_points: n,
    })
}

/// Compute A,B,C via linear least squares (nalgebra) for fixed (tc,m,omega,phi).
/// Returns (A, B, C, sse)
fn linear_solve_abc(
    log_prices: &[f64],
    times: &[f64],
    tc: f64,
    m: f64,
    omega: f64,
    phi: f64,
) -> (f64, f64, f64, f64) {
    let n = log_prices.len();
    let mut x = DMatrix::<f64>::zeros(n, 3);
    let mut y = DVector::<f64>::zeros(n);

    for i in 0..n {
        let tau = (tc - times[i]).max(1e-8);
        let tm = tau.powf(m);
        let osc = (omega * tau.ln() + phi).cos();
        x[(i, 0)] = 1.0;
        x[(i, 1)] = tm;
        x[(i, 2)] = tm * osc;
        y[i] = log_prices[i];
    }

    // Normal equations + Cholesky (or LU fallback)
    let xtx = x.transpose() * &x;
    let xty = x.transpose() * &y;

    let beta = if let Some(chol) = Cholesky::new(xtx.clone()) {
        chol.solve(&xty)
    } else {
        let lu: LU<f64, nalgebra::Dyn, nalgebra::Dyn> = xtx.lu();
        lu.solve(&xty).unwrap_or(DVector::zeros(3))
    };

    let a = beta[0];
    let b = beta[1];
    let c = beta[2];

    // SSE
    let fitted = &x * &beta;
    let res = y - fitted;
    let sse = res.dot(&res);

    (a, b, c, sse)
}

/// Convenience: fit on a slice of PriceBars (uses log adj_close, trading-day time index)
pub fn fit_lppl_on_bars(bars: &[super::data::PriceBar], start_idx: usize, end_idx: usize, seed: u64) -> Result<LpplFit, String> {
    let slice = &bars[start_idx..end_idx];
    if slice.len() < 30 {
        return Err("window too small".into());
    }

    // Use trading day index as time (robust, avoids calendar gaps)
    let times: Vec<f64> = (0..slice.len()).map(|i| i as f64).collect();
    let log_prices: Vec<f64> = slice
        .iter()
        .map(|b| (b.adj_close.max(0.01)).ln())
        .collect();

    fit_lppl(&log_prices, &times, seed)
}

// ============================================================================
// ENHANCED LPPLS / HLPPL BUBBLE ANALYSIS (Multi-Window, Strict JLS Filters)
// From advanced documentation: rolling windows, physics constraints, Bubble Confidence Index (% valid fits),
// risk levels, predicted critical dates for future bubble prediction.
// Used for both historical analysis and live "current sentiment" at end of data.
// ============================================================================

/// Configuration for the strict filtering of valid LPPLS fits (JLS criteria).
#[derive(Debug, Clone)]
pub struct LpplFilterConfig {
    pub m_min: f64,
    pub m_max: f64,
    pub omega_min: f64,
    pub omega_max: f64,
    pub require_b_negative: bool, // B < 0 for upward accelerating bubble
    pub min_tc_offset_days: usize, // tc must be at least this many days in the future
}

impl Default for LpplFilterConfig {
    fn default() -> Self {
        Self {
            m_min: 0.1,
            m_max: 0.9,
            omega_min: 4.5,
            omega_max: 13.0,
            require_b_negative: true,
            min_tc_offset_days: 3,
        }
    }
}

/// Result of a multi-window LPPLS bubble confidence analysis.
#[derive(Debug, Clone)]
pub struct BubbleAnalysisResult {
    pub analysis_date: NaiveDate,
    pub current_price: f64,
    pub total_windows_tested: usize,
    pub valid_fits: usize,
    pub bubble_confidence_index: f64, // 0-100%
    pub risk_level: String, // "LOW RISK", "MODERATE RISK", "HIGH RISK", "CRITICAL BUBBLE REGIME"
    pub predicted_critical_dates: Vec<NaiveDate>,
    pub median_predicted_date: Option<NaiveDate>,
    /// Optional: list of (window_start_date, tc_date) for valid fits, for detailed export.
    pub valid_fit_details: Vec<(NaiveDate, NaiveDate)>,
}

impl BubbleAnalysisResult {
    pub fn risk_description(&self) -> &'static str {
        match self.risk_level.as_str() {
            "LOW RISK" => "Normal organic trend or basic noise. No systematic herding signatures detected.",
            "MODERATE RISK" => "Localized acceleration detected. Keep an eye on parabolic velocity changes.",
            "HIGH RISK" => "Significant critical mass forming. The trend shows super-exponential characteristics.",
            "CRITICAL BUBBLE REGIME" => "Extreme herd synchronization. Acceleration pattern is structurally unstable.",
            _ => "Unknown risk level.",
        }
    }
}

/// Compute a robust LPPLS Bubble Confidence Index using multi-window rolling fits.
/// Sweeps windows from lookback_max down to lookback_min days before current_idx, step by step_days.
/// For each, fits LPPLS (using provided seed for reproducibility), applies strict JLS filters.
/// Returns the % of windows that pass, list of predicted critical dates, risk assessment, etc.
/// This is the "extensive" version for predicting future bubbles and assessing current sentiment.
pub fn compute_bubble_confidence(
    bars: &[super::data::PriceBar],
    current_idx: usize,
    lookback_min: usize,
    lookback_max: usize,
    step_days: usize,
    filter: &LpplFilterConfig,
    seed: u64,
) -> Result<BubbleAnalysisResult, String> {
    if current_idx >= bars.len() || lookback_max > current_idx || bars.is_empty() {
        return Err("Invalid current_idx or lookback for bubble analysis".into());
    }
    if lookback_min < 30 {
        return Err("lookback_min must be at least 30 for stable LPPL".into());
    }

    let current_date = bars[current_idx].date;
    let current_price = bars[current_idx].adj_close;

    let mut valid_fits = 0usize;
    let mut total_windows = 0usize;
    let mut predicted_dates: Vec<NaiveDate> = vec![];
    let mut valid_details: Vec<(NaiveDate, NaiveDate)> = vec![];

    // Sweep start indices backward
    let mut start = current_idx.saturating_sub(lookback_max);
    while start <= current_idx.saturating_sub(lookback_min) {
        let window_len = current_idx - start + 1;
        if window_len < 30 {
            start += step_days;
            continue;
        }

        total_windows += 1;

        // Fit on this sub-window [start ..= current_idx]
        match fit_lppl_on_bars(bars, start, current_idx + 1, seed) {
            Ok(fit) => {
                let p = fit.params;
                // tc in the window's local time (0 at start, window_len-1 at current)
                let tc_abs = start as f64 + p.tc; // absolute trading-day index
                let is_tc_valid = tc_abs > current_idx as f64 + filter.min_tc_offset_days as f64;
                let is_m_valid = p.m >= filter.m_min && p.m <= filter.m_max;
                let is_omega_valid = p.omega >= filter.omega_min && p.omega <= filter.omega_max;
                let is_b_valid = if filter.require_b_negative { p.b < 0.0 } else { true };
                let is_structurally_valid = p.is_valid(); // existing damping etc.

                if is_tc_valid && is_m_valid && is_omega_valid && is_b_valid && is_structurally_valid {
                    valid_fits += 1;

                    // Extrapolate to calendar date
                    let days_ahead = (tc_abs - current_idx as f64).round() as i64;
                    if let Some(pred_date) = current_date.checked_add_signed(chrono::Duration::days(days_ahead)) {
                        predicted_dates.push(pred_date);
                        valid_details.push((bars[start].date, pred_date));
                    }
                }
            }
            Err(_) => {
                // ignore bad windows
            }
        }

        start += step_days;
    }

    let confidence = if total_windows > 0 {
        (valid_fits as f64 / total_windows as f64) * 100.0
    } else {
        0.0
    };

    let risk_level = if confidence < 15.0 {
        "LOW RISK".to_string()
    } else if confidence < 45.0 {
        "MODERATE RISK".to_string()
    } else if confidence < 75.0 {
        "HIGH RISK".to_string()
    } else {
        "CRITICAL BUBBLE REGIME".to_string()
    };

    let median_predicted_date = if !predicted_dates.is_empty() {
        predicted_dates.sort();
        let mid = predicted_dates.len() / 2;
        Some(predicted_dates[mid])
    } else {
        None
    };

    Ok(BubbleAnalysisResult {
        analysis_date: current_date,
        current_price,
        total_windows_tested: total_windows,
        valid_fits,
        bubble_confidence_index: confidence,
        risk_level,
        predicted_critical_dates: predicted_dates,
        median_predicted_date,
        valid_fit_details: valid_details,
    })
}

/// Helper: run the strict JLS filter check on a single already-fitted LpplParams (from any fit).
/// Returns (is_valid, reasons) for diagnostics. Aligns with gemini-data-LPPLS.md + Sornette/JLS literature:
/// 0.1 <= m <=0.9, appropriate omega, B<0 for upward, tc sufficiently in future, plus base damping.
pub fn is_strict_jls_valid(p: &LpplParams, t_current: f64, filter: &LpplFilterConfig) -> (bool, Vec<String>) {
    let mut reasons = vec![];
    let mut ok = true;

    if p.tc <= t_current + filter.min_tc_offset_days as f64 {
        ok = false; reasons.push("tc not far enough in future".into());
    }
    if p.m < filter.m_min || p.m > filter.m_max {
        ok = false; reasons.push(format!("m={:.3} outside [{},{}]", p.m, filter.m_min, filter.m_max));
    }
    if p.omega < filter.omega_min || p.omega > filter.omega_max {
        ok = false; reasons.push(format!("omega={:.2} outside [{},{}]", p.omega, filter.omega_min, filter.omega_max));
    }
    if filter.require_b_negative && p.b >= 0.0 {
        ok = false; reasons.push("B>=0 (not upward accelerating)".into());
    }
    if !p.is_valid() {
        ok = false; reasons.push("failed base damping/structural validity".into());
    }
    (ok, reasons)
}

/// Compute bubble confidence over an *ensemble* of seeds for robustness (recommended for live/prediction).
/// Returns (mean_confidence, std_confidence, per_seed_results, aggregated predicted dates etc).
/// If seeds is empty, falls back to single-seed using `seed`.
pub fn compute_bubble_confidence_ensemble(
    bars: &[super::data::PriceBar],
    current_idx: usize,
    lookback_min: usize,
    lookback_max: usize,
    step_days: usize,
    filter: &LpplFilterConfig,
    seeds: &[u64],
) -> Result<(BubbleAnalysisResult, f64 /*mean*/, f64 /*std*/, Vec<BubbleAnalysisResult>), String> {
    let effective_seeds: Vec<u64> = if seeds.is_empty() { vec![42] } else { seeds.to_vec() };
    let mut results: Vec<BubbleAnalysisResult> = vec![];
    for &sd in &effective_seeds {
        let r = compute_bubble_confidence(bars, current_idx, lookback_min, lookback_max, step_days, filter, sd)?;
        results.push(r);
    }
    let n = results.len() as f64;
    let mean_c = results.iter().map(|r| r.bubble_confidence_index).sum::<f64>() / n;
    let var = results.iter().map(|r| (r.bubble_confidence_index - mean_c).powi(2)).sum::<f64>() / n.max(1.0);
    let std_c = var.sqrt();

    // Aggregate: use the first (or mean-ish) for main fields; merge predicted dates across seeds for richer distro
    let mut agg = results[0].clone();
    agg.bubble_confidence_index = mean_c;
    let mut all_preds: Vec<NaiveDate> = results.iter().flat_map(|r| r.predicted_critical_dates.clone()).collect();
    all_preds.sort();
    all_preds.dedup();
    agg.predicted_critical_dates = all_preds;
    if !agg.predicted_critical_dates.is_empty() {
        let mid = agg.predicted_critical_dates.len() / 2;
        agg.median_predicted_date = Some(agg.predicted_critical_dates[mid]);
    }
    // risk from mean conf
    agg.risk_level = if mean_c < 15.0 { "LOW RISK".into() } else if mean_c < 45.0 { "MODERATE RISK".into() } else if mean_c < 75.0 { "HIGH RISK".into() } else { "CRITICAL BUBBLE REGIME".into() };

    Ok((agg, mean_c, std_c, results))
}
