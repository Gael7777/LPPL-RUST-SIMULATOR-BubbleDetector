//! High-level HLPPL logic engine.
//!
//! This module provides a clean, UI- and library-friendly facade over the core
//! backtesting logic. It is the recommended way to drive the HLPPL engine from
//! external code, new frontends, or scripts when you want "the logic" isolated
//! from any particular CLI/TUI/GUI.
//!
//! The low-level building blocks (fetch_yahoo_history, run_backtest, etc.) are
//! still re-exported from the crate root for advanced / fine-grained use.

use chrono::NaiveDate;

use crate::modules::backtest::{BacktestConfig, BacktestResult, run_backtest, run_future_bubble_prediction, compute_live_sentiment, FutureBubblePrediction, LiveSentimentSnapshot, RunMode};
use crate::modules::data::{fetch_yahoo_history, PriceBar};
use crate::modules::lppl::{compute_bubble_confidence, BubbleAnalysisResult, LpplFilterConfig};
use crate::modules::utils::export_backtest_artifacts;

/// Snapshot of "current sentiment" for live trading or at a specific date.
/// Combines traditional bubble_score + the new multi-window LPPLS confidence index (C1).
/// The recommendation incorporates bias, invert, confidence filter, and run mode context.
/// Extended for future prediction use (tc, days to peak, risk).
#[derive(Clone, Debug)]
pub struct CurrentSentiment {
    pub date: NaiveDate,
    pub bubble_score: f64,
    pub bubble_confidence: f64, // 0-100% from extensive multi-window analysis (C1)
    pub position: f64,
    pub recommendation: String, // "BUY / GO LONG", "SELL / GO SHORT", "HOLD / FLAT"
    pub risk_level: String,
    pub median_predicted_peak: Option<NaiveDate>,
    /// From RunMode-aware: for prediction modes this carries extra context
    pub mode_note: String,
}

/// A single completed (or open-to-end) trade leg, derived strictly from the
/// strategy's position changes and the equity curve. Used by both the TUI and
/// the GUI explorers.
#[derive(Clone, Debug)]
pub struct Trade {
    pub entry_date: NaiveDate,
    pub exit_date: NaiveDate,
    pub direction: String, // "LONG" or "SHORT"
    pub bars_held: usize,
    pub ret_pct: f64,
    pub pnl_usd: f64,
}

/// Convenient high-level wrapper for running HLPPL backtests.
///
/// Holds parameters, cached price data, the latest result, and derived trade
/// legs. Both `hlpll-explorer` (TUI) and `hlpll-gui` (native) as well as any
/// future consumers can drive this struct instead of duplicating fetch/run/trade
/// extraction code.
///
/// Example (library use):
/// ```ignore
/// let mut eng = HlpplEngine::new(
///     "CAR",
///     NaiveDate::from_ymd_opt(2023,1,1).unwrap(),
///     NaiveDate::from_ymd_opt(2024,12,31).unwrap(),
///     BacktestConfig { lookback_days: 180, random_seed: 42, ..Default::default() },
///     // random_seed controls the RNG for LPPL's multi-start nonlinear search (tc,m,omega,phi samples).
///     // The LPPL eq is nonlinear/hard to optimize globally, so random restarts explore; seed makes deterministic.
///     10_000.0,
/// );
/// eng.fetch()?;
/// eng.run()?;
/// println!("Final equity: ${:.0}", eng.final_capital());
/// ```
#[derive(Clone)]
pub struct HlpplEngine {
    pub ticker: String,
    pub start: NaiveDate,
    pub end: NaiveDate,
    pub config: BacktestConfig,
    pub initial_capital: f64,

    pub bars: Option<Vec<PriceBar>>,
    pub result: Option<BacktestResult>,
    pub trades: Vec<Trade>,

    /// Human readable status of the last fetch attempt (for UIs).
    pub fetch_status: String,

    // Extensive new: dedicated prediction & live sentiment results (populated by mode-aware runs)
    pub last_future_prediction: Option<FutureBubblePrediction>,
    pub last_live_sentiment: Option<LiveSentimentSnapshot>,
}

impl HlpplEngine {
    pub fn new(
        ticker: &str,
        start: NaiveDate,
        end: NaiveDate,
        config: BacktestConfig,
        initial_capital: f64,
    ) -> Self {
        Self {
            ticker: ticker.to_uppercase(),
            start,
            end,
            config,
            initial_capital,
            bars: None,
            result: None,
            trades: vec![],
            fetch_status: "No data loaded.".to_string(),
            last_future_prediction: None,
            last_live_sentiment: None,
        }
    }

    /// Fetch (or re-fetch) data from Yahoo for the current ticker + date range.
    /// Updates `bars` and `fetch_status`. Does **not** run the backtest.
    pub fn fetch(&mut self) -> Result<usize, String> {
        self.fetch_status = format!("Fetching {} {} → {} ...", self.ticker, self.start, self.end);

        let bars = fetch_yahoo_history(&self.ticker, self.start, self.end)
            .map_err(|e| format!("Yahoo fetch failed for {}: {}", self.ticker, e))?;

        let n = bars.len();
        if n == 0 {
            self.fetch_status = "Yahoo returned zero bars.".to_string();
            self.bars = None;
            return Err(self.fetch_status.clone());
        }

        let first = bars.first().unwrap().date;
        let last = bars.last().unwrap().date;
        let last_close = bars.last().unwrap().adj_close;

        self.bars = Some(bars);
        self.fetch_status = format!(
            "Yahoo OK ✓ {} bars | {} → {} | last ${:.2}",
            n, first, last, last_close
        );

        // Invalidate previous result when data changes
        self.result = None;
        self.trades.clear();
        self.last_future_prediction = None;
        self.last_live_sentiment = None;

        Ok(n)
    }

    /// Run the full walk-forward LPPL + bubble score + strategy simulation
    /// **strictly** using the same code path as the classic backtester
    /// (`run_backtest`).
    ///
    /// This forces HistoricalBacktest semantics for backward compat (full equity, trades).
    /// For future prediction or live sentiment, prefer `run_with_mode` or the dedicated getters.
    /// Requires that `fetch()` has succeeded. Recomputes the derived `trades`
    /// list for the current `initial_capital`.
    pub fn run(&mut self) -> Result<(), String> {
        self.config.run_mode = RunMode::HistoricalBacktest;
        let bars = self
            .bars
            .as_ref()
            .ok_or_else(|| "No price data. Call fetch() first.".to_string())?;

        if bars.len() < self.config.lookback_days + 30 {
            return Err(format!(
                "Not enough bars (have {}, need at least {})",
                bars.len(),
                self.config.lookback_days + 30
            ));
        }

        let res = run_backtest(&self.ticker, bars, &self.config)?;
        self.result = Some(res);
        self.recompute_trades();
        Ok(())
    }

    /// Run according to the engine's current `config.run_mode`.
    /// - Historical / Hybrid: full backtest (result + trades populated)
    /// - FutureBubblePrediction: populates last_future_prediction (and result if hybrid)
    /// - LiveCurrentSentiment: populates last_live_sentiment
    /// Always safe after fetch().
    pub fn run_with_mode(&mut self) -> Result<(), String> {
        let bars = self
            .bars
            .as_ref()
            .ok_or_else(|| "No price data. Call fetch() first.".to_string())?;

        match self.config.run_mode {
            RunMode::HistoricalBacktest => {
                if bars.len() < self.config.lookback_days + 30 {
                    return Err(format!("Not enough bars (have {}, need at least {})", bars.len(), self.config.lookback_days + 30));
                }
                let res = run_backtest(&self.ticker, bars, &self.config)?;
                self.result = Some(res);
                self.recompute_trades();
            }
            RunMode::FutureBubblePrediction | RunMode::HybridAnalysis => {
                let (pred, maybe_res) = run_future_bubble_prediction(&self.ticker, bars, &self.config)?;
                self.last_future_prediction = Some(pred);
                if let Some(r) = maybe_res {
                    self.result = Some(r);
                    self.recompute_trades();
                }
            }
            RunMode::LiveCurrentSentiment => {
                let snap = compute_live_sentiment(&self.ticker, bars, &self.config)?;
                self.last_live_sentiment = Some(snap);
            }
        }
        Ok(())
    }

    /// Current final capital for the simulated investment (initial * equity multiplier).
    pub fn final_capital(&self) -> f64 {
        if let Some(res) = &self.result {
            if let Some(&last_mult) = res.equity.last() {
                return self.initial_capital * last_mult;
            }
        }
        self.initial_capital
    }

    /// Total P&L in dollars for the current simulation.
    pub fn total_pnl_usd(&self) -> f64 {
        self.final_capital() - self.initial_capital
    }

    /// Export all artifacts (signals CSV, equity CSVs, PNG) using the shared
    /// engine utilities. Requires a successful `run()`.
    pub fn export_artifacts(&self, outdir: &str) -> Result<(), String> {
        let res = self
            .result
            .as_ref()
            .ok_or_else(|| "No result to export. Run the simulation first.".to_string())?;

        export_backtest_artifacts(res, outdir, self.initial_capital)
            .map_err(|e| format!("Export failed: {}", e))
    }

    /// Rebuild the `trades` list from the current result + capital.
    /// Called automatically by `run()`.
    pub fn recompute_trades(&mut self) {
        self.trades.clear();

        let Some(res) = &self.result else { return; };
        let sigs = &res.signals;
        let eq = &res.equity;
        if sigs.is_empty() || eq.len() < 2 {
            return;
        }

        let n = sigs.len();
        let mut i = 0usize;

        while i < n {
            if !sigs[i].trade {
                i += 1;
                continue;
            }
            let dir = sigs[i].position;
            if dir.abs() < 0.5 {
                i += 1;
                continue;
            }

            let entry_i = i;

            let mut j = i + 1;
            while j < n && !sigs[j].trade {
                j += 1;
            }
            let exit_i = if j < n { j } else { n - 1 };

            let e_entry = eq[entry_i + 1];
            let e_exit = eq[exit_i + 1];

            let pnl_usd = (e_exit - e_entry) * self.initial_capital;
            let ret_pct = if e_entry > 1e-12 {
                (e_exit / e_entry) - 1.0
            } else {
                0.0
            };

            self.trades.push(Trade {
                entry_date: sigs[entry_i].date,
                exit_date: sigs[exit_i].date,
                direction: if dir > 0.0 { "LONG".into() } else { "SHORT".into() },
                bars_held: exit_i - entry_i + 1,
                ret_pct,
                pnl_usd,
            });

            i = j;
        }
    }

    /// Helper for UIs: get a reference to the signals for the current result (if any).
    pub fn signals(&self) -> Option<&[crate::modules::backtest::DailySignal]> {
        self.result.as_ref().map(|r| r.signals.as_slice())
    }

    // === NEW EXTENSIVE FEATURES FOR BUBBLE PREDICTION & LIVE SENTIMENT ===

    /// Run the advanced multi-window LPPLS/HLPPL Bubble Confidence analysis on the latest data.
    /// This implements the rolling window sweep with strict JLS filters from the documentation.
    /// Ideal for "predicting future bubbles" (gives median predicted peak date) and
    /// "current sentiment" at the end of the series (confidence % + risk level).
    /// Updates no internal result, just returns the analysis. Use alongside or instead of run().
    pub fn run_bubble_analysis(&mut self) -> Result<BubbleAnalysisResult, String> {
        if self.bars.is_none() {
            self.fetch()?;
        }
        let bars = self.bars.as_ref().unwrap();
        if bars.is_empty() {
            return Err("No bars loaded for bubble analysis".into());
        }
        let current_idx = bars.len() - 1;

        let filter = LpplFilterConfig {
            m_min: self.config.filter_m_min,
            m_max: self.config.filter_m_max,
            omega_min: self.config.filter_omega_min,
            omega_max: self.config.filter_omega_max,
            require_b_negative: self.config.filter_require_b_negative,
            min_tc_offset_days: self.config.filter_min_tc_offset_days,
        };

        compute_bubble_confidence(
            bars,
            current_idx,
            self.config.analysis_lookback_min,
            self.config.analysis_lookback_max,
            self.config.analysis_step_days,
            &filter,
            self.config.random_seed,
        )
    }

    /// Get a clean "Current Sentiment" snapshot for live trading decisions or at the end date.
    /// This runs the bubble analysis (if not cached) and combines with the last computed
    /// bubble_score / position from a prior run() if available.
    /// The recommendation respects all config (bias, invert, confidence flat threshold, etc.).
    /// Perfect for "live trading on current sentiment with this equation".
    /// Now mode-aware via last_live or last_prediction if run_with_mode was used.
    pub fn get_current_sentiment(&mut self) -> Result<CurrentSentiment, String> {
        // Prefer rich live snapshot if available
        if let Some(snap) = &self.last_live_sentiment {
            return Ok(CurrentSentiment {
                date: snap.date,
                bubble_score: snap.bubble_score,
                bubble_confidence: snap.bubble_confidence,
                position: snap.position,
                recommendation: snap.recommendation.clone(),
                risk_level: snap.risk_level.clone(),
                median_predicted_peak: snap.median_predicted_peak,
                mode_note: snap.actionable_note.clone(),
            });
        }
        if let Some(pred) = &self.last_future_prediction {
            let rec = if pred.bubble_confidence_index > self.config.confidence_flat_threshold {
                "HOLD / FLAT (high C1 bubble risk from prediction)"
            } else if pred.risk_level.contains("CRITICAL") || pred.risk_level.contains("HIGH") {
                "CAUTION / MONITOR (elevated future bubble probability)"
            } else {
                "NEUTRAL (no strong future tc cluster)"
            };
            return Ok(CurrentSentiment {
                date: pred.analysis_date,
                bubble_score: 0.0,
                bubble_confidence: pred.bubble_confidence_index,
                position: 0.0,
                recommendation: rec.to_string(),
                risk_level: pred.risk_level.clone(),
                median_predicted_peak: pred.median_predicted_date,
                mode_note: format!("Future pred: median peak ~{:?} (conf {:.1}%)", pred.median_predicted_date, pred.bubble_confidence_index),
            });
        }

        let analysis = self.run_bubble_analysis()?;

        let (score, pos, base_rec) = if let Some(res) = &self.result {
            if let Some(last) = res.signals.last() {
                let r = if last.position > 0.5 {
                    "BUY / GO LONG"
                } else if last.position < -0.5 {
                    "SELL / GO SHORT"
                } else {
                    "HOLD / FLAT"
                };
                (last.bubble_score, last.position, r.to_string())
            } else {
                (0.0, 0.0, "HOLD / FLAT".to_string())
            }
        } else {
            (0.0, 0.0, "HOLD / FLAT".to_string())
        };

        // Apply confidence filter for final rec if configured
        let final_rec = if self.config.use_confidence_for_flat && analysis.bubble_confidence_index > self.config.confidence_flat_threshold {
            "HOLD / FLAT (high bubble confidence - risk management override)".to_string()
        } else {
            base_rec
        };

        Ok(CurrentSentiment {
            date: analysis.analysis_date,
            bubble_score: score,
            bubble_confidence: analysis.bubble_confidence_index,
            position: pos,
            recommendation: final_rec,
            risk_level: analysis.risk_level.clone(),
            median_predicted_peak: analysis.median_predicted_date,
            mode_note: if self.config.run_mode != RunMode::HistoricalBacktest {
                format!("Mode: {:?} | C1={:.1}%", self.config.run_mode, analysis.bubble_confidence_index)
            } else { "".into() },
        })
    }

    /// Run (or re-run) the dedicated future bubble prediction using current config (ensemble, horizons, filters).
    /// Populates `last_future_prediction`. For pure prediction or hybrid.
    pub fn get_future_prediction(&mut self) -> Result<FutureBubblePrediction, String> {
        if self.bars.is_none() {
            self.fetch()?;
        }
        let bars = self.bars.as_ref().unwrap();
        let (pred, maybe_res) = run_future_bubble_prediction(&self.ticker, bars, &self.config)?;
        self.last_future_prediction = Some(pred.clone());
        if let Some(r) = maybe_res {
            self.result = Some(r);
            self.recompute_trades();
        }
        Ok(pred)
    }

    /// Compute a live/current sentiment snapshot for "trade now" decisions.
    /// Populates `last_live_sentiment`. Uses the latest bar as "today".
    pub fn get_live_sentiment(&mut self) -> Result<LiveSentimentSnapshot, String> {
        if self.bars.is_none() {
            self.fetch()?;
        }
        let bars = self.bars.as_ref().unwrap();
        let snap = compute_live_sentiment(&self.ticker, bars, &self.config)?;
        self.last_live_sentiment = Some(snap.clone());
        Ok(snap)
    }
}
