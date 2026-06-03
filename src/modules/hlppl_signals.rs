//! HLPPL signal pipeline aligned with the research notebook (Tasks 15–17).
//!
//! - Overlapping LPPL windows → mean raw residual ε(t) per day
//! - Causal running-max normalization → ε_norm ∈ [-1, 1]
//! - Volume-hype + return-sentiment proxies fused into BubbleScore (Eq. 14)

use super::bubble_score::{compute_bubble_score, normalize_running_max};
use super::data::PriceBar;
use super::hype::compute_volume_hype;
use super::lppl::fit_lppl_on_bars;
use super::sentiment::compute_simple_sentiment;
use crate::modules::backtest::{BacktestConfig, SignalMode};

/// Per-bar HLPPL features used by the backtester and live views.
#[derive(Debug, Clone)]
pub struct HlpplSignalSeries {
    pub raw_epsilon: Vec<f64>,
    pub eps_norm: Vec<f64>,
    pub hype: Vec<f64>,
    pub sentiment: Vec<f64>,
    pub bubble_score: Vec<f64>,
    /// First bar index with a valid LPPL residual (usually `lookback_days - 1`).
    pub signal_start: usize,
}

/// Build the full paper-style signal path (overlapping windows + running-max norm).
pub fn build_paper_signal_series(
    bars: &[PriceBar],
    cfg: &BacktestConfig,
) -> Result<HlpplSignalSeries, String> {
    let n = bars.len();
    let w = cfg.lookback_days;
    if n < w {
        return Err(format!("Need at least {} bars for LPPL window, have {}", w, n));
    }

    let fit_cfg = cfg.lppl_fit_config();
    let mut raw_sum = vec![0.0; n];
    let mut counts = vec![0u32; n];

    let stride = cfg.window_stride.max(1);
    let mut end = w - 1;
    while end < n {
        let start = end + 1 - w;
        let seed = cfg
            .random_seed
            .wrapping_add((start as u64).wrapping_mul(131))
            .wrapping_add((end as u64).wrapping_mul(17));

        if let Ok(fit) = fit_lppl_on_bars(bars, start, end + 1, seed, &fit_cfg) {
            for (k, &resid) in fit.residuals.iter().enumerate() {
                let idx = start + k;
                if idx < n {
                    raw_sum[idx] += resid;
                    counts[idx] += 1;
                }
            }
        }
        end += stride;
    }

    let mut raw_epsilon = vec![0.0; n];
    for i in 0..n {
        if counts[i] > 0 {
            raw_epsilon[i] = raw_sum[i] / counts[i] as f64;
        }
    }

    let signal_start = w - 1;
    let eps_norm = normalize_running_max(&raw_epsilon, signal_start);

    let hype = build_hype_series(bars, 60);
    let sentiment = build_sentiment_series(bars);

    let mut bubble_score = vec![0.0; n];
    for i in signal_start..n {
        bubble_score[i] = compute_bubble_score(
            eps_norm[i],
            hype[i],
            sentiment[i],
            cfg.alpha1_hype,
            cfg.alpha2_sentiment,
        );
    }

    Ok(HlpplSignalSeries {
        raw_epsilon,
        eps_norm,
        hype,
        sentiment,
        bubble_score,
        signal_start,
    })
}

/// Fast path: single rolling window, refit schedule handled by caller (legacy).
pub fn build_signal_series(bars: &[PriceBar], cfg: &BacktestConfig) -> Result<HlpplSignalSeries, String> {
    match cfg.signal_mode {
        SignalMode::Paper => build_paper_signal_series(bars, cfg),
        SignalMode::Fast => Err("Fast signal mode is filled incrementally in run_backtest".into()),
    }
}

fn build_hype_series(bars: &[PriceBar], window: usize) -> Vec<f64> {
    let vols: Vec<f64> = bars.iter().map(|b| b.volume).collect();
    let h = compute_volume_hype(&vols, window);
    if h.len() == bars.len() {
        h
    } else {
        let mut out = vec![0.0; bars.len()];
        let offset = bars.len().saturating_sub(h.len());
        for (i, &v) in h.iter().enumerate() {
            out[offset + i] = v;
        }
        out
    }
}

fn build_sentiment_series(bars: &[PriceBar]) -> Vec<f64> {
    let rets: Vec<f64> = bars
        .windows(2)
        .map(|w| {
            if w[0].adj_close > 0.0 {
                w[1].adj_close / w[0].adj_close - 1.0
            } else {
                0.0
            }
        })
        .collect();
    let mut s = compute_simple_sentiment(&rets);
    s.insert(0, 0.0);
    if s.len() < bars.len() {
        s.resize(bars.len(), 0.0);
    }
    s
}

pub fn hype_at(bars: &[PriceBar], i: usize, window: usize) -> f64 {
    let start = i.saturating_sub(window);
    let vols: Vec<f64> = bars[start..=i].iter().map(|b| b.volume).collect();
    *compute_volume_hype(&vols, window.min(vols.len()).max(5))
        .last()
        .unwrap_or(&0.0)
}

pub fn sentiment_at(bars: &[PriceBar], i: usize, window: usize) -> f64 {
    let start = i.saturating_sub(window);
    let rets: Vec<f64> = bars[start..=i]
        .windows(2)
        .map(|w| {
            if w[0].adj_close > 0.0 {
                w[1].adj_close / w[0].adj_close - 1.0
            } else {
                0.0
            }
        })
        .collect();
    *compute_simple_sentiment(&rets).last().unwrap_or(&0.0)
}
