use nalgebra::{Cholesky, DMatrix, DVector, LU};
use rand::Rng;
use std::f64::consts::PI;

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

/// Fit the LPPL model to log-prices using pure multi-start random search + closed-form linear solve for (A,B,C).
/// No external optimizer crate (avoids LAPACK/Windows linking issues).
///
/// times: increasing sequence, typically 0.0 .. N (trading days index)
/// log_prices: ln(close) or ln(adj_close), same length, positive prices.
pub fn fit_lppl(log_prices: &[f64], times: &[f64]) -> Result<LpplFit, String> {
    if log_prices.len() != times.len() || log_prices.len() < 30 {
        return Err("Need at least 30 points for LPPL fit".to_string());
    }

    let n = log_prices.len();
    let t_max = times[n - 1];
    let t_span = t_max - times[0];

    // Reasonable search bounds for tc (critical time slightly in future)
    let tc_min = t_max + 1.0;
    let tc_max = t_max + (t_span * 0.4).max(30.0);

    let mut rng = rand::thread_rng();
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
pub fn fit_lppl_on_bars(bars: &[super::data::PriceBar], start_idx: usize, end_idx: usize) -> Result<LpplFit, String> {
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

    fit_lppl(&log_prices, &times)
}
