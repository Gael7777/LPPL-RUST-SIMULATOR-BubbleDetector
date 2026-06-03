use nalgebra::{Cholesky, DMatrix, DVector, LU};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
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

/// Notebook / paper-aligned bounds and search settings for LPPL fitting.
#[derive(Debug, Clone)]
pub struct LpplFitConfig {
    pub m_min: f64,
    pub m_max: f64,
    pub omega_min: f64,
    pub omega_max: f64,
    pub b_min: f64,
    pub b_max: f64,
    pub tc_future_min: usize,
    pub tc_future_max: usize,
    pub num_random_starts: usize,
    pub num_polish_steps: usize,
    pub require_b_negative: bool,
    pub require_damping: bool,
}

impl Default for LpplFitConfig {
    fn default() -> Self {
        Self {
            m_min: 0.1,
            m_max: 0.9,
            omega_min: 2.0,
            omega_max: 20.0,
            b_min: -1.0,
            b_max: -0.01,
            tc_future_min: 5,
            tc_future_max: 250,
            num_random_starts: 80,
            num_polish_steps: 32,
            require_b_negative: true,
            require_damping: true,
        }
    }
}

impl LpplParams {
    fn passes_fit_filters(&self, cfg: &LpplFitConfig) -> bool {
        if self.m < cfg.m_min || self.m > cfg.m_max {
            return false;
        }
        if self.omega < cfg.omega_min || self.omega > cfg.omega_max {
            return false;
        }
        if cfg.require_b_negative && self.b >= 0.0 {
            return false;
        }
        if self.b < cfg.b_min || self.b > cfg.b_max {
            return false;
        }
        if cfg.require_damping && !self.is_valid() {
            return false;
        }
        true
    }
}

/// Fit LPPL with default paper-aligned config (1-based time indices).
pub fn fit_lppl(log_prices: &[f64], times: &[f64], seed: u64) -> Result<LpplFit, String> {
    fit_lppl_with_config(log_prices, times, seed, &LpplFitConfig::default())
}

/// Multi-start search over (tc, m, omega) with OLS for (A, B, C1, C2) — same structure as the notebook.
/// Time indices should be 1..=W (notebook convention).
pub fn fit_lppl_with_config(
    log_prices: &[f64],
    times: &[f64],
    seed: u64,
    cfg: &LpplFitConfig,
) -> Result<LpplFit, String> {
    if log_prices.len() != times.len() || log_prices.len() < 30 {
        return Err("Need at least 30 points for LPPL fit".to_string());
    }

    let n = log_prices.len();
    let t_max = times[n - 1];
    let tc_min = t_max + cfg.tc_future_min as f64;
    let tc_max = t_max + cfg.tc_future_max as f64;

    let mut rng: StdRng = StdRng::seed_from_u64(seed);
    let mut best_params: Option<LpplParams> = None;
    let mut best_sse = f64::INFINITY;

    for _ in 0..cfg.num_random_starts {
        let tc = rng.gen_range(tc_min..tc_max);
        let m = rng.gen_range(cfg.m_min..cfg.m_max);
        let omega = rng.gen_range(cfg.omega_min..cfg.omega_max);
        try_candidate(log_prices, times, tc, m, omega, cfg, &mut best_params, &mut best_sse);
    }

    if let Some(p) = best_params {
        polish_fit(log_prices, times, &p, cfg, &mut best_params, &mut best_sse);
    }

    let params = best_params.ok_or_else(|| "LPPL multi-start search failed to find valid fit".to_string())?;

    let mut residuals = Vec::with_capacity(n);
    for i in 0..n {
        residuals.push(log_prices[i] - params.predict_log_price(times[i]));
    }

    Ok(LpplFit {
        params,
        sse: best_sse,
        residuals,
        times: times.to_vec(),
        n_points: n,
    })
}

fn try_candidate(
    log_prices: &[f64],
    times: &[f64],
    tc: f64,
    m: f64,
    omega: f64,
    cfg: &LpplFitConfig,
    best_params: &mut Option<LpplParams>,
    best_sse: &mut f64,
) {
    if tc <= times[times.len() - 1] {
        return;
    }
    let (params, sse) = linear_solve_cos_sin(log_prices, times, tc, m, omega);
    if sse < *best_sse && params.passes_fit_filters(cfg) {
        *best_sse = sse;
        *best_params = Some(params);
    }
}

fn polish_fit(
    log_prices: &[f64],
    times: &[f64],
    start: &LpplParams,
    cfg: &LpplFitConfig,
    best_params: &mut Option<LpplParams>,
    best_sse: &mut f64,
) {
    let mut cur = *start;
    let mut cur_sse = *best_sse;
    let t_max = times[times.len() - 1];
    let tc_min = t_max + cfg.tc_future_min as f64;
    let tc_max = t_max + cfg.tc_future_max as f64;

    let deltas: &[(f64, f64, f64)] = &[
        (2.0, 0.0, 0.0),
        (-2.0, 0.0, 0.0),
        (0.0, 0.02, 0.0),
        (0.0, -0.02, 0.0),
        (0.0, 0.0, 0.3),
        (0.0, 0.0, -0.3),
    ];

    for _ in 0..cfg.num_polish_steps {
        let mut improved = false;
        for &(dtc, dm, domega) in deltas {
            let tc = (cur.tc + dtc).clamp(tc_min, tc_max);
            let m = (cur.m + dm).clamp(cfg.m_min, cfg.m_max);
            let omega = (cur.omega + domega).clamp(cfg.omega_min, cfg.omega_max);
            if tc <= t_max {
                continue;
            }
            let (params, sse) = linear_solve_cos_sin(log_prices, times, tc, m, omega);
            if sse < cur_sse && params.passes_fit_filters(cfg) {
                cur = params;
                cur_sse = sse;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    if cur_sse < *best_sse {
        *best_sse = cur_sse;
        *best_params = Some(cur);
    }
}

/// OLS for A, B, C1, C2 with cos/sin log-periodic terms (notebook-compatible).
fn linear_solve_cos_sin(
    log_prices: &[f64],
    times: &[f64],
    tc: f64,
    m: f64,
    omega: f64,
) -> (LpplParams, f64) {
    let n = log_prices.len();
    let mut x = DMatrix::<f64>::zeros(n, 4);
    let mut y = DVector::<f64>::zeros(n);

    for i in 0..n {
        let tau = (tc - times[i]).max(1e-8);
        let tm = tau.powf(m);
        let lnt = tau.ln();
        x[(i, 0)] = 1.0;
        x[(i, 1)] = tm;
        x[(i, 2)] = tm * (omega * lnt).cos();
        x[(i, 3)] = tm * (omega * lnt).sin();
        y[i] = log_prices[i];
    }

    let xtx = x.transpose() * &x;
    let xty = x.transpose() * &y;
    let beta = if let Some(chol) = Cholesky::new(xtx.clone()) {
        chol.solve(&xty)
    } else {
        let lu: LU<f64, nalgebra::Dyn, nalgebra::Dyn> = xtx.lu();
        lu.solve(&xty).unwrap_or(DVector::zeros(4))
    };

    let a = beta[0];
    let b = beta[1];
    let c1 = beta[2];
    let c2 = beta[3];
    let c = (c1 * c1 + c2 * c2).sqrt();
    let phi = if c > 1e-12 { c2.atan2(c1) } else { 0.0 };

    let fitted = &x * &beta;
    let sse = (&y - &fitted).dot(&(&y - &fitted));

    (
        LpplParams {
            tc,
            m,
            omega,
            a,
            b,
            c,
            phi,
        },
        sse,
    )
}

/// Fit on price bars; time index 1..=W per notebook Task 13.
pub fn fit_lppl_on_bars(
    bars: &[super::data::PriceBar],
    start_idx: usize,
    end_idx: usize,
    seed: u64,
    cfg: &LpplFitConfig,
) -> Result<LpplFit, String> {
    let slice = &bars[start_idx..end_idx];
    if slice.len() < 30 {
        return Err("window too small".into());
    }
    let times: Vec<f64> = (1..=slice.len()).map(|i| i as f64).collect();
    let log_prices: Vec<f64> = slice
        .iter()
        .map(|b| (b.adj_close.max(0.01)).ln())
        .collect();
    fit_lppl_with_config(&log_prices, &times, seed, cfg)
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
        match fit_lppl_on_bars(bars, start, current_idx + 1, seed, &LpplFitConfig::default()) {
            Ok(fit) => {
                let p = fit.params;
                // tc in window-local time (1..=W); map to absolute bar index
                let tc_abs = start as f64 + p.tc - 1.0;
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

    if p.tc <= t_current + 1.0 + filter.min_tc_offset_days as f64 {
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
