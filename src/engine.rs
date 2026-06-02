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

use crate::modules::backtest::{BacktestConfig, BacktestResult, run_backtest};
use crate::modules::data::{fetch_yahoo_history, PriceBar};
use crate::modules::utils::export_backtest_artifacts;

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
///     BacktestConfig { lookback_days: 180, ..Default::default() },
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

        Ok(n)
    }

    /// Run the full walk-forward LPPL + bubble score + strategy simulation
    /// **strictly** using the same code path as the classic backtester
    /// (`run_backtest`).
    ///
    /// Requires that `fetch()` has succeeded. Recomputes the derived `trades`
    /// list for the current `initial_capital`.
    pub fn run(&mut self) -> Result<(), String> {
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
}
