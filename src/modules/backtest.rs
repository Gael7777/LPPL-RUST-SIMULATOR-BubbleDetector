use crate::modules::bubble_score::{compute_bubble_score, normalize_last_residual_running_style, normalize_running_max};
use crate::modules::data::PriceBar;
use crate::modules::hlppl_signals::{build_signal_series, hype_at, sentiment_at};
use crate::modules::lppl::{compute_bubble_confidence_ensemble, fit_lppl_on_bars, LpplFilterConfig, LpplFitConfig, LpplParams};
use chrono::NaiveDate;

/// Fast-mode only: project daily residual from last fitted LPPL curve (between refits).
struct LiveModel {
    params: LpplParams,
    running_max_abs: f64,
    t_end: f64,
    refit_i: usize,
}

/// How LPPL residuals feed the BubbleScore (paper notebook vs. legacy fast path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SignalMode {
    /// Overlapping windows + mean ε + causal running-max ε_norm (notebook Tasks 15–16).
    #[default]
    Paper,
    /// Single window, refit every N days + forward projection (faster).
    Fast,
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

/// Run mode for the HLPPL engine. Allows the same backbone to power:
/// - Classic historical walk-forward backtests + trading sims (with optional C1 overlay)
/// - Pure future bubble prediction (multi-window C1 + tc forecasts, for "will it peak soon?")
/// - Live current sentiment snapshots (for real-time trading signals using the LPPLS eq + C1)
/// - Hybrids
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RunMode {
    /// Full historical backtest: walk-forward LPPL fits, live eps projection, bubble_score,
    /// positions from thresholds + bias/invert/confidence filter, equity curve + B&H compare.
    /// C1 / bubble analysis runs per bar if `enable_bubble_analysis`.
    #[default]
    HistoricalBacktest,
    /// Focus on forward-looking bubble prediction: at end of data (or query point), run the
    /// extensive multi-window LPPLS sweep (strict JLS filters per gemini-doc), report
    /// Bubble Confidence Index (%), risk level, distribution of predicted critical times (tc).
    /// No (or minimal) trading simulation/equity. Ideal for "predicting future bubbles".
    FutureBubblePrediction,
    /// Compute "live" / current sentiment snapshot for trading now: C1 + last score/pos
    /// (or projected) at the most recent bar. Fast path for ongoing monitoring / alerts.
    /// Useful for live trading on current sentiment with the LPPLS equation.
    LiveCurrentSentiment,
    /// Do both: full historical sim (for validation/research) + the prediction/sentiment outputs.
    HybridAnalysis,
}

#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Paper / notebook LPPL rolling window W (default 250 trading days).
    pub lookback_days: usize,
    /// Fast mode only: refit LPPL every N bars.
    pub refit_every: usize,
    /// Paper mode: fit every window end with this stride (1 = full notebook, 5 = faster).
    pub window_stride: usize,
    /// Paper vs fast residual pipeline (see `SignalMode`).
    pub signal_mode: SignalMode,
    pub alpha1_hype: f64,
    pub alpha2_sentiment: f64,
    pub lppl_random_starts: usize,
    pub long_threshold: f64,    // bubble_score > this => long (paper scale ~0.3–0.7)
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

    // === NEW EXTENSIVE LPPLS / HLPPL BUBBLE ANALYSIS OPTIONS (from advanced multi-window JLS) ===
    /// Enable full multi-window bubble confidence analysis at refit points (for historical) or at end (live).
    /// This runs the rolling window sweep, applies strict filters, computes % valid fits as "bubble index".
    pub enable_bubble_analysis: bool,

    /// For multi-window analysis: min lookback days for the shortest window (e.g. 60).
    pub analysis_lookback_min: usize,
    /// Max lookback for the longest window (e.g. 260).
    pub analysis_lookback_max: usize,
    /// Step in days when sweeping the window starts (e.g. 5).
    pub analysis_step_days: usize,

    /// LPPL filter parameters for "valid" fits (strict JLS criteria for high confidence bubble flag).
    pub filter_m_min: f64,
    pub filter_m_max: f64,
    pub filter_omega_min: f64,
    pub filter_omega_max: f64,
    pub filter_require_b_negative: bool,
    pub filter_min_tc_offset_days: usize,

    /// How to use the bubble confidence for the trading strategy / live sentiment.
    /// If true and confidence > threshold, force position to 0 (risk management: avoid in high bubble regime).
    /// Can be combined with invert etc.
    pub use_confidence_for_flat: bool,
    pub confidence_flat_threshold: f64, // e.g. 50.0 or 75.0

    // === RUN MODE & EXTENSIVE FUTURE / LIVE FEATURES (from gemini-data-LPPLS.md + literature C1) ===
    /// Selects the primary computation goal. See RunMode docs. Default HistoricalBacktest for backward compat.
    /// Switch to FutureBubblePrediction or LiveCurrentSentiment to emphasize tc forecasts and C1 % over trading sim.
    pub run_mode: RunMode,

    /// Ensemble seeds for robust multi-seed C1 computation (Bubble Confidence).
    /// If non-empty, compute_bubble_confidence (and per-bar analyses) will be run for each seed and averaged.
    /// This gives more stable %valid and reduces sensitivity to any single random search.
    /// Example: vec![42, 43, 44, 45] for 4-seed ensemble. Empty => use single `random_seed`.
    pub ensemble_seeds: Vec<u64>,

    /// For future prediction outputs: when reporting "will tc fall in next N days?", use this horizon.
    /// Also affects "prob within horizon" rough estimate from the valid tc distribution.
    pub predict_horizon_days: usize,

    /// If true (and in historical/hybrid), use the per-day bubble_confidence to scale position size
    /// (e.g. pos *= clamp(conf/100.0, 0.2, 1.0) or similar). Experimental risk-aware sizing.
    pub use_confidence_for_sizing: bool,
}

impl BacktestConfig {
    /// LPPL nonlinear search bounds (notebook-style; separate from strict JLS C1 filters).
    pub fn lppl_fit_config(&self) -> LpplFitConfig {
        LpplFitConfig {
            num_random_starts: self.lppl_random_starts,
            ..LpplFitConfig::default()
        }
    }
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            lookback_days: 250,
            refit_every: 25,
            window_stride: 5,
            signal_mode: SignalMode::Paper,
            alpha1_hype: 0.7,
            alpha2_sentiment: 0.3,
            lppl_random_starts: 80,
            long_threshold: 0.55,
            short_threshold: 0.55,
            cost_bps: 12.0,
            max_position: 1.0,
            position_bias: PositionBias::LongShort,
            invert_signal: false,
            random_seed: 42,

            enable_bubble_analysis: false,
            analysis_lookback_min: 60,
            analysis_lookback_max: 260,
            analysis_step_days: 5,
            filter_m_min: 0.1,
            filter_m_max: 0.9,
            filter_omega_min: 4.5,
            filter_omega_max: 13.0,
            filter_require_b_negative: true,
            filter_min_tc_offset_days: 3,
            use_confidence_for_flat: false,
            confidence_flat_threshold: 50.0,

            run_mode: RunMode::HistoricalBacktest,
            ensemble_seeds: vec![],
            predict_horizon_days: 60,
            use_confidence_for_sizing: false,
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

    /// NEW: Bubble Confidence Index (0-100%) at this point in time, from multi-window analysis (if enabled).
    /// High value = many recent windows show valid LPPLS bubble signatures (herding, super-exp growth).
    /// Can be used for risk management (e.g. force flat when high).
    pub bubble_confidence: f64,
}

/// Rich result for a dedicated future bubble prediction run (RunMode::FutureBubblePrediction or part of hybrid).
/// Captures the multi-window C1 from the gemini-doc / JLS literature: % of windows with strict valid fits,
/// risk categorization, and the distribution of predicted critical times (tc) for "when might it peak/burst".
#[derive(Debug, Clone)]
pub struct FutureBubblePrediction {
    pub ticker: String,
    pub analysis_date: NaiveDate,
    pub current_price: f64,
    pub total_windows_tested: usize,
    pub valid_fits: usize,
    pub bubble_confidence_index: f64, // 0-100
    pub risk_level: String,
    pub risk_description: String,
    /// All calendar dates extrapolated from valid fits' tc (sorted)
    pub predicted_critical_dates: Vec<NaiveDate>,
    pub median_predicted_date: Option<NaiveDate>,
    /// Rough "days from analysis_date to median tc"
    pub median_days_to_tc: Option<i64>,
    /// Fraction of valid predicted tcs that fall within `predict_horizon_days` of analysis_date
    pub prob_tc_within_horizon: f64,
    pub ensemble_seeds_used: Vec<u64>,
    pub ensemble_mean_confidence: f64,
    pub ensemble_std_confidence: f64,
    /// Sample of valid fit details (window_start, tc_date) for diagnostics/export
    pub sample_valid_fits: Vec<(NaiveDate, NaiveDate)>,
}

/// Snapshot for live/current sentiment trading (RunMode::LiveCurrentSentiment).
/// Combines the latest bubble_score/position (from score or prior) with the C1 risk view.
/// The `recommendation` and `actionable_note` synthesize bias/invert/confidence/ C1 for "trade now?" use.
#[derive(Debug, Clone)]
pub struct LiveSentimentSnapshot {
    pub ticker: String,
    pub date: NaiveDate,
    pub current_price: f64,
    pub bubble_score: f64,
    pub bubble_confidence: f64, // C1 %
    pub risk_level: String,
    pub position: f64, // the trading pos that would be taken (after all filters/bias/invert/sizing)
    pub recommendation: String, // e.g. "BUY / GO LONG (score driven; low C1 ok)"
    pub actionable_note: String, // e.g. "High bubble conf (72%) - consider reducing or inverting for risk"
    pub median_predicted_peak: Option<NaiveDate>,
    pub median_days_to_tc: Option<i64>,
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

    let paper_signals = if cfg.signal_mode == SignalMode::Paper {
        log::info!(
            "Building paper HLPPL signals (W={}, stride={}) for {}…",
            cfg.lookback_days,
            cfg.window_stride,
            ticker
        );
        Some(build_signal_series(bars, cfg)?)
    } else {
        None
    };

    for i in cfg.lookback_days..n {
        let (bubble_score, eps_norm, hype_volume, sentiment) = if let Some(ref series) = paper_signals {
            (
                series.bubble_score[i],
                series.eps_norm[i],
                series.hype[i],
                series.sentiment[i],
            )
        } else {
            let do_refit = (i - cfg.lookback_days) % cfg.refit_every == 0 || live_model.is_none();
            let hype_volume = hype_at(bars, i, 60);
            let sentiment = sentiment_at(bars, i, 60);

            if do_refit {
                let fit_cfg = cfg.lppl_fit_config();
                let start = i + 1 - cfg.lookback_days;
                match fit_lppl_on_bars(bars, start, i + 1, cfg.random_seed, &fit_cfg) {
                    Ok(fit) => {
                        let window_eps = normalize_running_max(&fit.residuals, 0);
                        let eps_norm = *window_eps.last().unwrap_or(&0.0);
                        let running_max = fit
                            .residuals
                            .iter()
                            .map(|r| r.abs())
                            .fold(0.0f64, f64::max);

                        live_model = Some(LiveModel {
                            params: fit.params,
                            running_max_abs: running_max.max(1e-12),
                            t_end: fit.times.last().copied().unwrap_or(cfg.lookback_days as f64),
                            refit_i: i,
                        });

                        let bubble_score = compute_bubble_score(
                            eps_norm,
                            hype_volume,
                            sentiment,
                            cfg.alpha1_hype,
                            cfg.alpha2_sentiment,
                        );
                        (bubble_score, eps_norm, hype_volume, sentiment)
                    }
                    Err(e) => {
                        log::warn!("LPPL fit failed at {} for {}: {}", bars[i].date, ticker, e);
                        (0.0, 0.0, hype_volume, sentiment)
                    }
                }
            } else if let Some(m) = live_model.as_mut() {
                let dt = (i - m.refit_i) as f64;
                let t = m.t_end + dt;
                let logp = (bars[i].adj_close.max(0.01)).ln();
                let resid = logp - m.params.predict_log_price(t);
                m.running_max_abs = m.running_max_abs.max(resid.abs());
                let eps_norm = if m.running_max_abs > 1e-12 {
                    (resid / m.running_max_abs).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                let bubble_score = compute_bubble_score(
                    eps_norm,
                    hype_volume,
                    sentiment,
                    cfg.alpha1_hype,
                    cfg.alpha2_sentiment,
                );
                (bubble_score, eps_norm, hype_volume, sentiment)
            } else {
                (0.0, 0.0, hype_volume, sentiment)
            }
        };

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

        // NEW: compute bubble confidence if enabled (multi-window at this point in time)
        // Supports ensemble seeds for robustness (from gemini doc best practice). Single seed fallback.
        // This gives the "current sentiment" bubble index for risk overlay or live trading.
        let bubble_confidence = if cfg.enable_bubble_analysis {
            let filter = LpplFilterConfig {
                m_min: cfg.filter_m_min,
                m_max: cfg.filter_m_max,
                omega_min: cfg.filter_omega_min,
                omega_max: cfg.filter_omega_max,
                require_b_negative: cfg.filter_require_b_negative,
                min_tc_offset_days: cfg.filter_min_tc_offset_days,
            };
            let seeds = if cfg.ensemble_seeds.is_empty() { vec![cfg.random_seed] } else { cfg.ensemble_seeds.clone() };
            // Use ensemble helper for mean conf (cheaper to call per bar only if wanted; expensive but for research ok)
            match compute_bubble_confidence_ensemble(
                bars,
                i,
                cfg.analysis_lookback_min,
                cfg.analysis_lookback_max,
                cfg.analysis_step_days,
                &filter,
                &seeds,
            ) {
                Ok((_analysis, mean_c, _std_c, _per_seed)) => {
                    if cfg.use_confidence_for_flat && mean_c > cfg.confidence_flat_threshold {
                        target_pos = 0.0;
                    }
                    // Optional proportional sizing from conf (extensive feature)
                    if cfg.use_confidence_for_sizing && target_pos.abs() > 1e-9 {
                        let scale = (mean_c / 100.0).clamp(0.2, 1.0);
                        target_pos *= scale;
                    }
                    mean_c
                }
                Err(_) => 0.0,
            }
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
            bubble_confidence,
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

/// Standalone: run a pure future bubble prediction using the extensive multi-window LPPLS + strict JLS C1
/// (no full trading walk-forward equity unless hybrid). Uses ensemble if configured.
/// This is the "predicting future bubbles" path emphasized in the gemini documentation.
pub fn run_future_bubble_prediction(
    ticker: &str,
    bars: &[PriceBar],
    cfg: &BacktestConfig,
) -> Result<(FutureBubblePrediction, Option<BacktestResult>), String> {
    if bars.is_empty() {
        return Err("No bars for prediction".into());
    }
    let current_idx = bars.len() - 1;

    let filter = LpplFilterConfig {
        m_min: cfg.filter_m_min,
        m_max: cfg.filter_m_max,
        omega_min: cfg.filter_omega_min,
        omega_max: cfg.filter_omega_max,
        require_b_negative: cfg.filter_require_b_negative,
        min_tc_offset_days: cfg.filter_min_tc_offset_days,
    };
    let seeds = if cfg.ensemble_seeds.is_empty() { vec![cfg.random_seed] } else { cfg.ensemble_seeds.clone() };

    let (agg, mean_c, std_c, _per) = compute_bubble_confidence_ensemble(
        bars, current_idx, cfg.analysis_lookback_min, cfg.analysis_lookback_max, cfg.analysis_step_days, &filter, &seeds
    )?;

    let days_to = agg.median_predicted_date.map(|d| (d - agg.analysis_date).num_days());
    let horizon = cfg.predict_horizon_days as i64;
    let within = if !agg.predicted_critical_dates.is_empty() {
        let cnt = agg.predicted_critical_dates.iter().filter(|&&d| (d - agg.analysis_date).num_days() <= horizon && (d - agg.analysis_date).num_days() > 0 ).count();
        cnt as f64 / agg.predicted_critical_dates.len() as f64
    } else { 0.0 };

    let pred = FutureBubblePrediction {
        ticker: ticker.to_string(),
        analysis_date: agg.analysis_date,
        current_price: agg.current_price,
        total_windows_tested: agg.total_windows_tested,
        valid_fits: agg.valid_fits,
        bubble_confidence_index: mean_c,
        risk_level: agg.risk_level.clone(),
        risk_description: agg.risk_description().to_string(),
        predicted_critical_dates: agg.predicted_critical_dates.clone(),
        median_predicted_date: agg.median_predicted_date,
        median_days_to_tc: days_to,
        prob_tc_within_horizon: within,
        ensemble_seeds_used: seeds.clone(),
        ensemble_mean_confidence: mean_c,
        ensemble_std_confidence: std_c,
        sample_valid_fits: agg.valid_fit_details.clone(),
    };

    // If hybrid, also produce the full backtest result for validation
    let maybe_bt = if cfg.run_mode == RunMode::HybridAnalysis {
        Some(run_backtest(ticker, bars, cfg)?)
    } else {
        None
    };

    Ok((pred, maybe_bt))
}

/// Standalone live/current sentiment snapshot (for "live trading on current sentiment").
/// Fast: only analyzes the *end* of the series (latest bar as "now").
/// Respects full config (bias, invert, confidence flat/sizing, ensemble).
pub fn compute_live_sentiment(
    ticker: &str,
    bars: &[PriceBar],
    cfg: &BacktestConfig,
) -> Result<LiveSentimentSnapshot, String> {
    if bars.len() < 30 {
        return Err("insufficient bars for live sentiment".into());
    }
    let current_idx = bars.len() - 1;
    let _current_bar = &bars[current_idx];

    // Compute a "current" bubble_score using last window + live proj style (or simple last)
    // For purity, reuse a mini last-window fit + hype/sent on tail.
    let score = if cfg.signal_mode == SignalMode::Paper {
        match build_signal_series(bars, cfg) {
            Ok(s) => s.bubble_score[current_idx],
            Err(_) => 0.0,
        }
    } else {
        let w = cfg.lookback_days.min(current_idx);
        let start_fit = current_idx + 1 - w;
        let fit_cfg = cfg.lppl_fit_config();
        match fit_lppl_on_bars(bars, start_fit, current_idx + 1, cfg.random_seed, &fit_cfg) {
            Ok(fit) => {
                let e = normalize_last_residual_running_style(&fit.residuals);
                let hypev = hype_at(bars, current_idx, 60);
                let sv = sentiment_at(bars, current_idx, 60);
                compute_bubble_score(e, hypev, sv, cfg.alpha1_hype, cfg.alpha2_sentiment)
            }
            Err(_) => 0.0,
        }
    };

    // raw pos from score
    let raw = if score > cfg.long_threshold { cfg.max_position } else if score < -cfg.short_threshold { -cfg.max_position } else { 0.0 };
    let mut pos = if cfg.invert_signal { -raw } else { raw };
    pos = match cfg.position_bias {
        PositionBias::LongOnly => pos.max(0.0),
        PositionBias::ShortOnly => pos.min(0.0),
        PositionBias::LongShort => pos,
    };

    // C1 via ensemble at end
    let filter = LpplFilterConfig { m_min: cfg.filter_m_min, m_max: cfg.filter_m_max, omega_min: cfg.filter_omega_min, omega_max: cfg.filter_omega_max, require_b_negative: cfg.filter_require_b_negative, min_tc_offset_days: cfg.filter_min_tc_offset_days };
    let seeds = if cfg.ensemble_seeds.is_empty() { vec![cfg.random_seed] } else { cfg.ensemble_seeds.clone() };
    let (analysis, mean_c, _s, _ ) = compute_bubble_confidence_ensemble(bars, current_idx, cfg.analysis_lookback_min, cfg.analysis_lookback_max, cfg.analysis_step_days, &filter, &seeds)?;

    if cfg.use_confidence_for_flat && mean_c > cfg.confidence_flat_threshold {
        pos = 0.0;
    }
    if cfg.use_confidence_for_sizing && pos.abs() > 1e-9 {
        pos *= (mean_c / 100.0).clamp(0.2, 1.0);
    }

    let rec = if pos > 0.5 {
        "BUY / GO LONG (positive bubble score / momentum)"
    } else if pos < -0.5 {
        "SELL / GO SHORT (negative score / overextension or momentum down)"
    } else {
        "HOLD / FLAT"
    };
    let note = if mean_c > cfg.confidence_flat_threshold {
        format!("High C1 ({:.1}%) - bubble risk elevated; risk override applied", mean_c)
    } else if mean_c > 45.0 {
        format!("Elevated bubble confidence ({:.1}%) - monitor for tc cluster", mean_c)
    } else {
        "C1 low/moderate - normal regime per LPPLS filters".into()
    };

    let days = analysis.median_predicted_date.map(|d| (d - analysis.analysis_date).num_days());

    Ok(LiveSentimentSnapshot {
        ticker: ticker.to_string(),
        date: analysis.analysis_date,
        current_price: analysis.current_price,
        bubble_score: score,
        bubble_confidence: mean_c,
        risk_level: analysis.risk_level.clone(),
        position: pos,
        recommendation: rec.to_string(),
        actionable_note: note,
        median_predicted_peak: analysis.median_predicted_date,
        median_days_to_tc: days,
    })
}
