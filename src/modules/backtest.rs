use crate::modules::bubble_score::{compute_bubble_score, normalize_last_residual};
use crate::modules::data::PriceBar;
use crate::modules::hype::compute_volume_hype;
use crate::modules::lppl::{fit_lppl_on_bars, LpplParams};
use crate::modules::sentiment::compute_simple_sentiment;
use chrono::NaiveDate;

/// Internal: last fitted LPPL model, used to "project" a live eps_norm on days when we do not
/// perform a full (expensive) re-fit. This is a key improvement for responsiveness:
/// the expensive nonlinear search for (tc, m, omega, phi) happens only every `refit_every` days,
/// but every day we re-measure the *current* deviation of today's log-price from the last
/// fitted curve, and we refresh hype/sentiment with the latest bar. This reduces the staleness
/// that was previously causing the strategy to stay in a position for many days after market
/// regime changed.
struct LiveModel {
    params: LpplParams,
    residual_std: f64,
    t_end: f64,
    refit_i: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PositionBias {
    /// Can take both long and short positions based on the sign/magnitude of the bubble score.
    #[default]
    LongShort,
    /// Long-only version: treat negative scores as "flat". Useful for "detect higher-going momentum".
    LongOnly,
    /// Short-only version: treat positive scores as "flat". Useful for "detect lower-going momentum".
    ShortOnly,
}

#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub lookback_days: usize,   // e.g. 252 or 300 trading days for each LPPL window
    pub refit_every: usize,     // refit LPPL every N bars (20-60 typical)
    pub long_threshold: f64,    // bubble_score > this => long
    pub short_threshold: f64,   // bubble_score < -this => short
    pub cost_bps: f64,          // round-trip cost in basis points (e.g. 10 = 0.10%)
    pub max_position: f64,      // 1.0 = fully invested long/short

    /// Controls whether the strategy is allowed to go long, short, or both.
    /// Lets the user run "long only momentum" or "short only" variants of the bubble signal.
    pub position_bias: PositionBias,

    /// If true, flip the meaning of the score: high positive bubble_score is interpreted as
    /// "overextended / bubble risk" (favor short or flat instead of long). This is often more
    /// aligned with the original "bubble detection" literature (high score = danger of crash).
    pub invert_signal: bool,

    /// Seed for the random number generator used in the LPPL multi-start fitting.
    /// Using a fixed seed (default 42) makes every run with the same inputs 100% reproducible
    /// (same fits → same scores → same positions → same equity curve).
    /// Change the seed if you want different random explorations of the parameter space.
    pub random_seed: u64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            lookback_days: 300,
            refit_every: 25,
            long_threshold: 0.8,
            short_threshold: 0.8,
            cost_bps: 12.0,
            max_position: 1.0,
            position_bias: PositionBias::LongShort,
            invert_signal: false,
            random_seed: 42,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DailySignal {
    pub date: NaiveDate,
    pub close: f64,
    pub daily_return: f64,
    pub volume: f64,
    pub eps_norm: f64,
    pub hype_volume: f64,
    pub sentiment: f64,
    pub bubble_score: f64,
    pub position: f64, // -1.0 short, 0 flat, +1.0 long
    pub trade: bool,
}

#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub ticker: String,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub n_days: usize,
    pub signals: Vec<DailySignal>,
    pub equity: Vec<f64>,
    /// Buy & Hold equity curve (same length/alignment as `equity`), scaled to same starting capital as the strategy.
    /// Plotted in the GUI for direct visual comparison.
    pub bh_equity: Vec<f64>,
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub num_trades: usize,
    pub buy_hold_return: f64,
    pub buy_hold_sharpe: f64,
}

/// Run walk-forward LPPL + BubbleScore backtest on a single ticker's bars.
pub fn run_backtest(
    ticker: &str,
    bars: &[PriceBar],
    cfg: &BacktestConfig,
) -> Result<BacktestResult, String> {
    if bars.len() < cfg.lookback_days + 30 {
        return Err(format!(
            "Not enough data for {}: {} bars < required {}",
            ticker,
            bars.len(),
            cfg.lookback_days + 30
        ));
    }

    let n = bars.len();
    let mut signals: Vec<DailySignal> = Vec::with_capacity(n);
    let mut equity = vec![1.0];
    let mut bh_equity = vec![1.0];
    let mut position: f64 = 0.0;
    let mut num_trades = 0usize;
    let cost = cfg.cost_bps / 10000.0;

    let mut daily_rets: Vec<f64> = Vec::new();
    let mut bh_rets: Vec<f64> = Vec::new();

    let mut last_signal_date = bars[0].date;

    let mut live_model: Option<LiveModel> = None;

    for i in cfg.lookback_days..n {
        let do_refit = (i - cfg.lookback_days) % cfg.refit_every == 0 || live_model.is_none();

        let mut bubble_score = 0.0;
        let mut eps_norm = 0.0;
        let mut hype_volume = 0.0;
        let mut sentiment = 0.0;

        // Always compute *fresh* hype and sentiment using the most recent data up to bar i.
        // This makes the non-LPPL components live every day (cheap to do).
        {
            let hype_w = 60usize;
            let hstart = i.saturating_sub(hype_w);
            let vol_win: Vec<f64> = bars[hstart..=i].iter().map(|b| b.volume).collect();
            let hvals = compute_volume_hype(&vol_win, hype_w);
            hype_volume = *hvals.last().unwrap_or(&0.0);

            let ret_win: Vec<f64> = bars[hstart..=i]
                .windows(2)
                .map(|w| {
                    let p = w[0].adj_close;
                    if p > 0.0 { w[1].adj_close / p - 1.0 } else { 0.0 }
                })
                .collect();
            let svals = compute_simple_sentiment(&ret_win);
            sentiment = *svals.last().unwrap_or(&0.0);
        }

        if do_refit {
            // Full (expensive) LPPL re-fit + historical eps_norm from the fit window.
            match fit_lppl_on_bars(bars, i - cfg.lookback_days, i, cfg.random_seed) {
                Ok(fit) => {
                    eps_norm = normalize_last_residual(&fit.residuals);

                    // Compute the std used for normalization so we can project later.
                    let nres = fit.residuals.len();
                    let m: f64 = fit.residuals.iter().sum::<f64>() / nres as f64;
                    let v: f64 = fit.residuals.iter().map(|r| (r - m).powi(2)).sum::<f64>() / nres as f64;
                    let rstd = (v + 1e-12).sqrt();

                    live_model = Some(LiveModel {
                        params: fit.params,
                        residual_std: rstd,
                        t_end: (fit.n_points.saturating_sub(1)) as f64,
                        refit_i: i,
                    });

                    bubble_score = compute_bubble_score(eps_norm, hype_volume, sentiment, 0.7, 0.3);
                }
                Err(e) => {
                    log::warn!("LPPL fit failed at {} for {}: {}", bars[i].date, ticker, e);
                    bubble_score = 0.0;
                }
            }
        } else if let Some(m) = &live_model {
            // === Key mathematical improvement for accuracy / lower staleness ===
            // Instead of freezing the entire score (including the LPPL mispricing) for `refit_every` days,
            // we keep the *last fitted curve shape* (tc,m,omega,phi,A,B,C) and every day we re-evaluate
            // the *current* residual of today's log-price against that fixed curve, using a forward time index.
            // Hype and sentiment are already refreshed above with today's bar.
            // This gives a "live" bubble_score every day while only paying the cost of the heavy random
            // search + OLS periodically. The signal reacts much faster to price action after the last model fit.
            let dt = (i - m.refit_i) as f64;
            let t = m.t_end + dt;
            let logp = (bars[i].adj_close.max(0.01)).ln();
            let fitted = m.params.predict_log_price(t);
            let resid = logp - fitted;
            eps_norm = resid / m.residual_std;

            bubble_score = compute_bubble_score(eps_norm, hype_volume, sentiment, 0.7, 0.3);
        } else {
            // fallback (should not happen)
            bubble_score = 0.0;
        }

        // Decide new position, applying invert + bias
        let raw = if bubble_score > cfg.long_threshold {
            cfg.max_position
        } else if bubble_score < -cfg.short_threshold {
            -cfg.max_position
        } else {
            0.0
        };

        let mut target_pos = if cfg.invert_signal { -raw } else { raw };

        target_pos = match cfg.position_bias {
            PositionBias::LongOnly => target_pos.max(0.0),
            PositionBias::ShortOnly => target_pos.min(0.0),
            PositionBias::LongShort => target_pos,
        };

        let trade = (target_pos - position).abs() > 1e-6;
        if trade {
            num_trades += 1;
        }

        // Apply cost on trade
        let mut daily_ret = 0.0;
        if i > 0 {
            let prev_close = bars[i - 1].adj_close;
            let ret = if prev_close > 0.0 {
                (bars[i].adj_close / prev_close) - 1.0
            } else {
                0.0
            };

            daily_ret = position * ret;
            if trade {
                daily_ret -= cost.abs(); // simple one-way cost on change
            }
            bh_rets.push(ret);
            daily_rets.push(daily_ret);

            // B&H equity (for plotting comparison)
            let bh_new = *bh_equity.last().unwrap_or(&1.0) * (1.0 + ret);
            bh_equity.push(bh_new);
        }

        let new_equity = equity.last().copied().unwrap_or(1.0) * (1.0 + daily_ret);
        equity.push(new_equity);

        signals.push(DailySignal {
            date: bars[i].date,
            close: bars[i].adj_close,
            daily_return: daily_ret,
            volume: bars[i].volume,
            eps_norm,
            hype_volume,
            sentiment,
            bubble_score,
            position: target_pos,
            trade,
        });

        position = target_pos;
        last_signal_date = bars[i].date;
    }

    // Metrics
    let total_return = equity.last().copied().unwrap_or(1.0) - 1.0;

    let ann_return = if daily_rets.len() > 5 {
        let mean_ret = daily_rets.iter().sum::<f64>() / daily_rets.len() as f64;
        (1.0 + mean_ret).powf(252.0) - 1.0
    } else {
        total_return
    };

    let sharpe = if daily_rets.len() > 5 {
        let n = daily_rets.len() as f64;
        let mean = daily_rets.iter().sum::<f64>() / n;
        let var = daily_rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();
        if std > 1e-12 {
            mean / std * (252.0_f64).sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };

    let max_dd = compute_max_drawdown(&equity);

    let bh_total = if !bh_rets.is_empty() {
        bh_rets.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0
    } else {
        0.0
    };

    let bh_sharpe = if bh_rets.len() > 5 {
        let n = bh_rets.len() as f64;
        let mean = bh_rets.iter().sum::<f64>() / n;
        let var = bh_rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();
        if std > 1e-12 {
            mean / std * 252.0_f64.sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };

    Ok(BacktestResult {
        ticker: ticker.to_string(),
        start_date: bars[cfg.lookback_days].date,
        end_date: last_signal_date,
        n_days: signals.len(),
        signals,
        equity,
        bh_equity,
        total_return,
        annualized_return: ann_return,
        sharpe,
        max_drawdown: max_dd,
        num_trades,
        buy_hold_return: bh_total,
        buy_hold_sharpe: bh_sharpe,
    })
}

fn compute_max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = equity[0];
    let mut max_dd = 0.0;
    for &e in equity {
        if e > peak {
            peak = e;
        }
        let dd = (peak - e) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}
