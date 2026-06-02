use crate::modules::bubble_score::{compute_bubble_score, normalize_last_residual};
use crate::modules::data::PriceBar;
use crate::modules::hype::compute_volume_hype;
use crate::modules::lppl::fit_lppl_on_bars;
use crate::modules::sentiment::compute_simple_sentiment;
use chrono::NaiveDate;

#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub lookback_days: usize,   // e.g. 252 or 300 trading days for each LPPL window
    pub refit_every: usize,     // refit LPPL every N bars (20-60 typical)
    pub long_threshold: f64,    // bubble_score > this => long
    pub short_threshold: f64,   // bubble_score < -this => short
    pub cost_bps: f64,          // round-trip cost in basis points (e.g. 10 = 0.10%)
    pub max_position: f64,      // 1.0 = fully invested long/short
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
    let mut equity = vec![1.0]; // start with $1
    let mut position: f64 = 0.0;
    let mut num_trades = 0usize;
    let cost = cfg.cost_bps / 10000.0; // one-way for simplicity; adjust if roundtrip

    let mut daily_rets: Vec<f64> = Vec::new();
    let mut bh_rets: Vec<f64> = Vec::new();

    let mut last_signal_date = bars[0].date;

    for i in cfg.lookback_days..n {
        let do_refit = (i - cfg.lookback_days) % cfg.refit_every == 0;

        let mut bubble_score = 0.0;
        let mut eps_norm = 0.0;
        let mut hype_volume = 0.0;
        let mut sentiment = 0.0;

        if do_refit {
            // Fit LPPL on the window [i-lookback .. i)
            match fit_lppl_on_bars(bars, i - cfg.lookback_days, i) {
                Ok(fit) => {
                    eps_norm = normalize_last_residual(&fit.residuals);

                    // Hype from volume in same window
                    let vol_slice: Vec<f64> = bars[i - cfg.lookback_days..i]
                        .iter()
                        .map(|b| b.volume)
                        .collect();
                    let hype_vals = compute_volume_hype(&vol_slice, 60);
                    hype_volume = *hype_vals.last().unwrap_or(&0.0);

                    // Sentiment proxy from daily returns in same window.
                    let ret_slice: Vec<f64> = bars[i - cfg.lookback_days..i]
                        .windows(2)
                        .map(|w| {
                            let prev = w[0].adj_close;
                            let curr = w[1].adj_close;
                            if prev > 0.0 {
                                (curr / prev) - 1.0
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let sent_vals = compute_simple_sentiment(&ret_slice);
                    sentiment = *sent_vals.last().unwrap_or(&0.0);

                    bubble_score = compute_bubble_score(
                        eps_norm,
                        hype_volume,
                        sentiment,
                        0.7, // alpha1
                        0.3, // alpha2
                    );
                }
                Err(e) => {
                    log::warn!("LPPL fit failed at {} for {}: {}", bars[i].date, ticker, e);
                    bubble_score = 0.0;
                }
            }
        } else if let Some(prev) = signals.last() {
            bubble_score = prev.bubble_score;
            eps_norm = prev.eps_norm;
            hype_volume = prev.hype_volume;
            sentiment = prev.sentiment;
        }

        // Decide new position
        let target_pos = if bubble_score > cfg.long_threshold {
            cfg.max_position
        } else if bubble_score < -cfg.short_threshold {
            -cfg.max_position
        } else {
            0.0
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
