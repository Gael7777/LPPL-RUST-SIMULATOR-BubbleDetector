//! HLPLL Explorer - Interactive TUI for manual parameter tweaking, Yahoo API testing,
//! bubble score visualization, and $10k trade simulation strictly following the strategy.
//!
//! Run with: cargo run --bin hlpll-explorer [--release]
//!
//! This is intentionally a *separate* binary from the main CLI backtester to keep the
//! core research tool headless while providing a rich interactive interface for
//! exploration, visualization, and what-if analysis with full user control.

use chrono::NaiveDate;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Wrap,
    },
    prelude::Marker,
    Frame, Terminal,
};
use std::error::Error;
use std::io::{self, stdout};

use hlpll_backtester::engine::{HlpplEngine, Trade};
use hlpll_backtester::modules::backtest::{BacktestConfig, BacktestResult, DailySignal};
use hlpll_backtester::modules::data::PriceBar;

const NUM_FIELDS: usize = 13; // main params + run_mode(10), ensemble(11), predict_horizon(12). Advanced analysis strings are parsed on [R] even if not all tab-focusable.

struct App {
    // --- Editable parameter strings (UI source of truth for the form) ---
    ticker: String,
    start_str: String,
    end_str: String,
    window_str: String,
    refit_str: String,
    long_str: String,
    short_str: String,
    cost_str: String,
    capital_str: String,
    random_seed_str: String,

    // NEW extensive
    enable_bubble_analysis_str: String,
    analysis_lookback_min_str: String,
    analysis_lookback_max_str: String,
    analysis_step_str: String,
    filter_m_min_str: String,
    filter_m_max_str: String,
    filter_omega_min_str: String,
    filter_omega_max_str: String,
    confidence_flat_threshold_str: String,
    use_confidence_for_flat: bool, // simple bool for TUI

    // Run mode + ensemble for the most extensive features
    run_mode_str: String, // "historical" | "prediction" | "live" | "hybrid"
    ensemble_seeds_str: String,
    predict_horizon_str: String,

    focused: usize, // 0..NUM_FIELDS-1

    // --- The actual separated logic engine (HLPPL core) ---
    // The TUI is just a view + controller over this.
    engine: HlpplEngine,

    // --- Cached for viz (derived from engine.result) ---
    // We keep local copies only for the TUI's immediate rendering convenience.
    bars: Option<Vec<PriceBar>>,
    result: Option<BacktestResult>,
    trades: Vec<Trade>,

    // --- Viz state (full user control) ---
    view_start: usize,
    view_len: usize,
    cursor: usize, // index into signals for details
    show_price: bool,
    show_score: bool,
    show_equity: bool,

    // --- UI state ---
    status: String,
    last_error: Option<String>,
    outdir: String,
    show_help: bool,
    running: bool,
}

impl App {
    fn new() -> Self {
        // Default params (also used to seed the high-level engine)
        let ticker = "CAR".to_string();
        let start_str = "2022-01-01".to_string();
        let end_str = "2024-12-31".to_string();
        let window_str = "250".to_string();
        let refit_str = "25".to_string();
        let long_str = "0.55".to_string();
        let short_str = "0.55".to_string();
        let cost_str = "8".to_string();
        let capital_str = "10000".to_string();
        let random_seed_str = "42".to_string();

        // NEW
        let enable_bubble_analysis_str = "true".to_string();
        let analysis_lookback_min_str = "60".to_string();
        let analysis_lookback_max_str = "260".to_string();
        let analysis_step_str = "5".to_string();
        let filter_m_min_str = "0.1".to_string();
        let filter_m_max_str = "0.9".to_string();
        let filter_omega_min_str = "4.5".to_string();
        let filter_omega_max_str = "13.0".to_string();
        let confidence_flat_threshold_str = "50.0".to_string();

        // Parse once for the engine (fall back to sensible defaults on bad strings)
        let start = NaiveDate::parse_from_str(&start_str, "%Y-%m-%d").unwrap_or_else(|_| NaiveDate::from_ymd_opt(2022, 1, 1).unwrap());
        let end = NaiveDate::parse_from_str(&end_str, "%Y-%m-%d").unwrap_or_else(|_| NaiveDate::from_ymd_opt(2024, 12, 31).unwrap());
        let window: usize = window_str.trim().parse().unwrap_or(180);
        let refit: usize = refit_str.trim().parse().unwrap_or(20);
        let long_t: f64 = long_str.trim().parse().unwrap_or(0.65);
        let short_t: f64 = short_str.trim().parse().unwrap_or(0.65);
        let cost: f64 = cost_str.trim().parse().unwrap_or(8.0);
        let cap: f64 = capital_str.trim().parse().unwrap_or(10000.0);
        let seed: u64 = random_seed_str.trim().parse().unwrap_or(42);

        // analysis parses (were partially present)
        let enable_analysis: bool = enable_bubble_analysis_str.trim().parse().unwrap_or(true);
        let min_lb: usize = analysis_lookback_min_str.trim().parse().unwrap_or(60);
        let max_lb: usize = analysis_lookback_max_str.trim().parse().unwrap_or(260);
        let step: usize = analysis_step_str.trim().parse().unwrap_or(5);
        let mmin: f64 = filter_m_min_str.trim().parse().unwrap_or(0.1);
        let mmax: f64 = filter_m_max_str.trim().parse().unwrap_or(0.9);
        let omin: f64 = filter_omega_min_str.trim().parse().unwrap_or(4.5);
        let omax: f64 = filter_omega_max_str.trim().parse().unwrap_or(13.0);
        let conf_thresh: f64 = confidence_flat_threshold_str.trim().parse().unwrap_or(50.0);

        let cfg = BacktestConfig {
            lookback_days: window,
            refit_every: refit,
            long_threshold: long_t,
            short_threshold: short_t,
            cost_bps: cost,
            max_position: 1.0,
            random_seed: seed,

            enable_bubble_analysis: enable_analysis,
            analysis_lookback_min: min_lb,
            analysis_lookback_max: max_lb,
            analysis_step_days: step,
            filter_m_min: mmin,
            filter_m_max: mmax,
            filter_omega_min: omin,
            filter_omega_max: omax,
            filter_require_b_negative: true,
            filter_min_tc_offset_days: 3,
            use_confidence_for_flat: true,
            confidence_flat_threshold: conf_thresh,
            ..Default::default()
        };

        let engine = HlpplEngine::new(&ticker, start, end, cfg, cap);

        Self {
            ticker,
            start_str,
            end_str,
            window_str,
            refit_str,
            long_str,
            short_str,
            cost_str,
            capital_str,
            random_seed_str,
            enable_bubble_analysis_str,
            analysis_lookback_min_str,
            analysis_lookback_max_str,
            analysis_step_str,
            filter_m_min_str,
            filter_m_max_str,
            filter_omega_min_str,
            filter_omega_max_str,
            confidence_flat_threshold_str,
            use_confidence_for_flat: true,

            run_mode_str: "historical".into(),
            ensemble_seeds_str: "42".into(),
            predict_horizon_str: "60".into(),
            focused: 0,
            engine,
            bars: None,
            result: None,
            trades: vec![],
            view_start: 0,
            view_len: 300,
            cursor: 0,
            show_price: true,
            show_score: true,
            show_equity: true,
            status: "Welcome to HLPLL Explorer. Tweak params, [F]etch (tests Yahoo), [R]un (uses shared engine), arrows pan, ? help, [Q]uit.".to_string(),
            last_error: None,
            outdir: "results".to_string(),
            show_help: false,
            running: true,
        }
    }

    fn field_label(idx: usize) -> &'static str {
        match idx {
            0 => "Ticker",
            1 => "Start (YYYY-MM-DD)",
            2 => "End (YYYY-MM-DD)",
            3 => "Window (trading days)",
            4 => "Refit every N days",
            5 => "Long thresh (score > )",
            6 => "Short thresh (score < -)",
            7 => "Cost (bps one-way)",
            8 => "Initial Capital ($)",
            9 => "RNG seed (LPPL fits)",
            10 => "Run mode (hist/pred/live/hybrid)",
            11 => "Ensemble seeds (csv)",
            12 => "Predict horizon days",
            _ => "Adv. (edit in sync or GUI)",
        }
    }

    fn current_field_mut(&mut self) -> &mut String {
        match self.focused {
            0 => &mut self.ticker,
            1 => &mut self.start_str,
            2 => &mut self.end_str,
            3 => &mut self.window_str,
            4 => &mut self.refit_str,
            5 => &mut self.long_str,
            6 => &mut self.short_str,
            7 => &mut self.cost_str,
            8 => &mut self.capital_str,
            9 => &mut self.random_seed_str,
            10 => &mut self.run_mode_str,
            11 => &mut self.ensemble_seeds_str,
            12 => &mut self.predict_horizon_str,
            _ => &mut self.ticker,
        }
    }

    /// Parse the current form strings, validate, and **sync** them into the
    /// `HlpplEngine` (the single source of truth for the logic).
    /// Returns the parsed capital for convenience.
    fn sync_params_to_engine(&mut self) -> Result<f64, String> {
        let ticker = self.ticker.trim().to_uppercase();
        if ticker.is_empty() {
            return Err("Ticker cannot be empty".into());
        }

        let start = NaiveDate::parse_from_str(self.start_str.trim(), "%Y-%m-%d")
            .map_err(|_| "Invalid start date (use YYYY-MM-DD)".to_string())?;
        let end = NaiveDate::parse_from_str(self.end_str.trim(), "%Y-%m-%d")
            .map_err(|_| "Invalid end date (use YYYY-MM-DD)".to_string())?;
        if end <= start {
            return Err("End date must be after start".into());
        }

        let window: usize = self
            .window_str
            .trim()
            .parse()
            .map_err(|_| "Window must be integer >= 60".to_string())?;
        if window < 60 {
            return Err("Window too small for stable LPPL (min 60)".into());
        }

        let refit: usize = self
            .refit_str
            .trim()
            .parse()
            .map_err(|_| "Refit must be positive integer".to_string())?;
        if refit == 0 {
            return Err("refit_every must be >= 1".into());
        }

        let long_t: f64 = self
            .long_str
            .trim()
            .parse()
            .map_err(|_| "Long thresh must be number".to_string())?;
        let short_t: f64 = self
            .short_str
            .trim()
            .parse()
            .map_err(|_| "Short thresh must be number".to_string())?;
        if long_t < 0.0 || short_t < 0.0 {
            return Err("Thresholds should be >= 0".into());
        }

        let cost: f64 = self
            .cost_str
            .trim()
            .parse()
            .map_err(|_| "Cost must be number (bps)".to_string())?;

        let cap: f64 = self
            .capital_str
            .trim()
            .parse()
            .map_err(|_| "Capital must be positive number".to_string())?;
        if cap <= 0.0 {
            return Err("Initial capital must be > 0".into());
        }

        let seed: u64 = self
            .random_seed_str
            .trim()
            .parse()
            .unwrap_or(42);

        let enable_analysis: bool = self.enable_bubble_analysis_str.trim().parse().unwrap_or(true);
        let min_lb: usize = self.analysis_lookback_min_str.trim().parse().unwrap_or(60);
        let max_lb: usize = self.analysis_lookback_max_str.trim().parse().unwrap_or(260);
        let step: usize = self.analysis_step_str.trim().parse().unwrap_or(5);
        let mmin: f64 = self.filter_m_min_str.trim().parse().unwrap_or(0.1);
        let mmax: f64 = self.filter_m_max_str.trim().parse().unwrap_or(0.9);
        let omin: f64 = self.filter_omega_min_str.trim().parse().unwrap_or(4.5);
        let omax: f64 = self.filter_omega_max_str.trim().parse().unwrap_or(13.0);
        let conf_thresh: f64 = self.confidence_flat_threshold_str.trim().parse().unwrap_or(50.0);

        let cfg = BacktestConfig {
            lookback_days: window,
            refit_every: refit,
            long_threshold: long_t,
            short_threshold: short_t,
            cost_bps: cost,
            max_position: 1.0,
            random_seed: seed,

            enable_bubble_analysis: enable_analysis,
            analysis_lookback_min: min_lb,
            analysis_lookback_max: max_lb,
            analysis_step_days: step,
            filter_m_min: mmin,
            filter_m_max: mmax,
            filter_omega_min: omin,
            filter_omega_max: omax,
            filter_require_b_negative: true,
            filter_min_tc_offset_days: 3,
            use_confidence_for_flat: self.use_confidence_for_flat,
            confidence_flat_threshold: conf_thresh,

            run_mode: match self.run_mode_str.as_str() { "prediction"=>hlpll_backtester::RunMode::FutureBubblePrediction, "live"=>hlpll_backtester::RunMode::LiveCurrentSentiment, "hybrid"=>hlpll_backtester::RunMode::HybridAnalysis, _=>hlpll_backtester::RunMode::HistoricalBacktest },
            ensemble_seeds: if self.ensemble_seeds_str.trim().is_empty(){vec![]}else{self.ensemble_seeds_str.split(',').filter_map(|s|s.trim().parse().ok()).collect()},
            predict_horizon_days: self.predict_horizon_str.trim().parse().unwrap_or(60),
            ..Default::default()
        };

        // Sync into the separated engine (this is the important modular step)
        self.engine.ticker = ticker;
        self.engine.start = start;
        self.engine.end = end;
        self.engine.config = cfg;
        self.engine.initial_capital = cap;

        Ok(cap)
    }

    /// Nudge numeric field up/down (great for quick tweaking without typing).
    fn nudge(&mut self, dir: i32) {
        let step = match self.focused {
            3 => 10i32,   // window
            4 => 5,       // refit
            5 | 6 => 5,   // thresh *0.01
            7 => 2,       // cost bps
            8 => 1000,    // capital
            _ => return,
        };

        match self.focused {
            3 => {
                if let Ok(mut v) = self.window_str.trim().parse::<i32>() {
                    v = (v + dir * step).max(60).min(2000);
                    self.window_str = v.to_string();
                }
            }
            4 => {
                if let Ok(mut v) = self.refit_str.trim().parse::<i32>() {
                    v = (v + dir * step).max(1).min(100);
                    self.refit_str = v.to_string();
                }
            }
            5 => {
                if let Ok(v) = self.long_str.trim().parse::<f64>() {
                    let nv = (v + (dir as f64) * 0.05).max(0.0).min(5.0);
                    self.long_str = format!("{:.2}", nv);
                }
            }
            6 => {
                if let Ok(v) = self.short_str.trim().parse::<f64>() {
                    let nv = (v + (dir as f64) * 0.05).max(0.0).min(5.0);
                    self.short_str = format!("{:.2}", nv);
                }
            }
            7 => {
                if let Ok(v) = self.cost_str.trim().parse::<f64>() {
                    let nv = (v + (dir as f64) * 1.0).max(0.0).min(100.0);
                    self.cost_str = format!("{:.1}", nv);
                }
            }
            8 => {
                if let Ok(v) = self.capital_str.trim().parse::<i32>() {
                    let nv = (v + dir * step).max(1000).min(1_000_000);
                    self.capital_str = nv.to_string();
                }
            }
            9 => {
                if let Ok(v) = self.random_seed_str.trim().parse::<u64>() {
                    let nv = (v as i64 + dir as i64 * 1).max(0).min(u64::MAX as i64) as u64;
                    self.random_seed_str = nv.to_string();
                }
            }
            _ => {}
        }
    }

    fn do_fetch(&mut self) {
        if let Err(e) = self.sync_params_to_engine() {
            self.last_error = Some(e);
            return;
        }

        self.engine.fetch_status = format!(
            "Fetching {} {} → {} from Yahoo...",
            self.engine.ticker, self.engine.start, self.engine.end
        );
        self.status = "Contacting Yahoo Finance API (no key required)...".to_string();

        match self.engine.fetch() {
            Ok(_n) => {
                // Mirror into TUI-local caches for the existing rendering code
                self.bars = self.engine.bars.clone();
                // engine.fetch() already set a nice fetch_status inside itself
                self.status = "Data ready (via shared HlpplEngine). Press [R] to run simulation using the isolated logic engine.".to_string();
                self.result = None;
                self.trades.clear();
                self.last_error = None;
            }
            Err(e) => {
                self.bars = None;
                self.last_error = Some(e);
                self.status = "Fix ticker/dates and press [F] again.".to_string();
            }
        }
    }

    fn do_run(&mut self) {
        if let Err(e) = self.sync_params_to_engine() {
            self.last_error = Some(e);
            return;
        }

        if self.engine.bars.is_none() {
            // Auto-fetch using the engine
            if let Err(e) = self.engine.fetch() {
                self.last_error = Some(e);
                self.bars = None;
                return;
            }
            self.bars = self.engine.bars.clone();
        }

        self.status = "Running via HlpplEngine (mode-aware: historical / future prediction / live sentiment)...".to_string();
        self.last_error = None;

        match self.engine.run_with_mode() {
            Ok(()) => {
                // Pull results back for TUI viz
                self.result = self.engine.result.clone();
                self.trades = self.engine.trades.clone();

                let cap = self.engine.initial_capital;
                let mode = self.engine.config.run_mode;
                let mut extra = String::new();
                if let Some(p) = &self.engine.last_future_prediction {
                    extra = format!(" | C1={:.1}% {} median_tc~{} ({}d)", p.bubble_confidence_index, p.risk_level, p.median_predicted_date.map(|d|d.to_string()).unwrap_or("N/A".into()), p.median_days_to_tc.unwrap_or(0));
                }
                if let Some(s) = &self.engine.last_live_sentiment {
                    extra = format!(" | LIVE: {} C1={:.1}%", s.recommendation, s.bubble_confidence);
                }
                self.status = format!(
                    "Run complete (mode {:?}). {} trades | Final ${:.0} on ${:.0} start.{}{}  j/k cursor, [E] export.",
                    mode,
                    self.trades.len(),
                    self.engine.final_capital(),
                    cap,
                    extra,
                    if self.result.is_none() { " (prediction/live: see info below)" } else { "" }
                );

                if let Some(r) = &self.result {
                    let n = r.signals.len();
                    self.view_start = 0;
                    self.view_len = n.min(450).max(50);
                    self.cursor = n.saturating_sub(1).min(n / 2);
                }
            }
            Err(e) => {
                self.last_error = Some(format!("Engine run error: {}", e));
                self.status = "Run failed. See error. Try adjusting window / dates / thresholds.".to_string();
            }
        }
    }

    fn final_dollar(&self) -> f64 {
        self.engine.final_capital()
    }

    fn do_export(&mut self) {
        // Delegate export to the engine (which already knows the capital and has the result)
        match self.engine.export_artifacts(&self.outdir) {
            Ok(_) => {
                if let Some(res) = &self.result {
                    let ticker = &res.ticker;
                    let cap = self.engine.initial_capital;
                    self.status = format!(
                        "Exported via engine → {}/{}_signals.csv + equity CSVs ({}k) + PNG.",
                        self.outdir, ticker, cap / 1000.0
                    );
                } else {
                    self.status = "Exported.".to_string();
                }
                self.last_error = None;
            }
            Err(e) => {
                self.last_error = Some(e);
            }
        }
    }

    fn reset_view(&mut self) {
        if let Some(res) = &self.result {
            let n = res.signals.len();
            self.view_start = 0;
            self.view_len = n.min(500);
            self.cursor = (n / 2).min(n.saturating_sub(1));
        }
    }

    fn pan(&mut self, delta: isize) {
        if let Some(res) = &self.result {
            let n = res.signals.len();
            let new_start = (self.view_start as isize + delta)
                .max(0)
                .min((n as isize - self.view_len as isize).max(0)) as usize;
            self.view_start = new_start;
            // keep cursor visible
            self.ensure_cursor_visible();
        }
    }

    fn zoom(&mut self, factor: f32) {
        if let Some(res) = &self.result {
            let n = res.signals.len();
            let center = self.view_start + self.view_len / 2;
            let new_len = ((self.view_len as f32 * factor) as usize).clamp(20, n);
            self.view_len = new_len;
            let half = new_len / 2;
            self.view_start = center.saturating_sub(half).min(n.saturating_sub(new_len));
            self.ensure_cursor_visible();
        }
    }

    fn ensure_cursor_visible(&mut self) {
        if let Some(res) = &self.result {
            let n = res.signals.len();
            let end = self.view_start + self.view_len;
            if self.cursor < self.view_start {
                self.view_start = self.cursor;
            } else if self.cursor >= end {
                self.view_start = self.cursor.saturating_sub(self.view_len.saturating_sub(1)).min(n.saturating_sub(self.view_len));
            }
            if self.view_start + self.view_len > n {
                self.view_start = n.saturating_sub(self.view_len);
            }
        }
    }

    fn move_cursor(&mut self, delta: isize) {
        if let Some(res) = &self.result {
            let n = res.signals.len();
            self.cursor = ((self.cursor as isize + delta).max(0).min(n as isize - 1)) as usize;
            self.ensure_cursor_visible();
        }
    }

    fn toggle_series(&mut self, which: u8) {
        match which {
            0 => self.show_price = !self.show_price,
            1 => self.show_score = !self.show_score,
            2 => self.show_equity = !self.show_equity,
            _ => {}
        }
    }

    /// Handle a key. Returns true if we should keep running.
    fn handle_key(&mut self, code: KeyCode) -> bool {
        if self.show_help {
            if matches!(code, KeyCode::Char('?') | KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter) {
                self.show_help = false;
            }
            return true;
        }

        match code {
            // Special action keys (before generic Char)
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                self.running = false;
                return false;
            }
            KeyCode::Char('?') => {
                self.show_help = true;
            }
            KeyCode::Char('f') | KeyCode::Char('F') => self.do_fetch(),
            KeyCode::Char('r') | KeyCode::Char('R') => self.do_run(),
            KeyCode::Char('e') | KeyCode::Char('E') => self.do_export(),
            KeyCode::Char('0') => self.reset_view(),
            KeyCode::Char('p') | KeyCode::Char('P') => self.toggle_series(0),
            KeyCode::Char('s') | KeyCode::Char('S') => self.toggle_series(1),
            KeyCode::Char('u') | KeyCode::Char('U') => self.toggle_series(2), // eqUity

            // Nudge / tweak (also Char but specific)
            KeyCode::Char('+') | KeyCode::Char('=') => self.nudge(1),
            KeyCode::Char('-') | KeyCode::Char('_') => self.nudge(-1),

            // Viz nav (non-Char or specific)
            KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('H') => self.pan(-20),
            KeyCode::Right | KeyCode::Char('l') | KeyCode::Char('L') => self.pan(20),
            KeyCode::PageUp => self.pan(-80),
            KeyCode::PageDown => self.pan(80),
            KeyCode::Char('[') => self.zoom(1.25),
            KeyCode::Char(']') => self.zoom(0.75),
            KeyCode::Char('j') | KeyCode::Char('J') => self.move_cursor(1),
            KeyCode::Char('k') | KeyCode::Char('K') => self.move_cursor(-1),
            KeyCode::Home => {
                self.cursor = 0;
                self.ensure_cursor_visible();
            }
            KeyCode::End => {
                if let Some(res) = &self.result {
                    self.cursor = res.signals.len().saturating_sub(1);
                    self.ensure_cursor_visible();
                }
            }

            // Focus
            KeyCode::Tab => {
                self.focused = (self.focused + 1) % NUM_FIELDS;
            }
            KeyCode::BackTab => {
                self.focused = (self.focused + NUM_FIELDS - 1) % NUM_FIELDS;
            }

            KeyCode::Enter => {
                // convenient: Enter on params often means "apply + run"
                self.do_run();
            }

            // Edit current field (generic text input) — last Char arm
            KeyCode::Char(c) => {
                if c.is_ascii_graphic() || c == ' ' {
                    self.current_field_mut().push(c);
                }
            }
            KeyCode::Backspace => {
                self.current_field_mut().pop();
            }

            _ => {}
        }
        true
    }

    fn current_signal(&self) -> Option<&DailySignal> {
        self.result.as_ref().and_then(|r| r.signals.get(self.cursor))
    }
}

fn ui(f: &mut Frame, app: &App) {
    let size = f.size();

    // Overall layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(8),     // body
            Constraint::Length(4),  // status + help line
        ])
        .split(size);

    // Header
    let header = Paragraph::new(Line::from(vec![
        Span::styled(" HLPLL / HLPPL  Interactive Explorer ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw("  |  "),
        Span::styled("manual tweak • Yahoo test • bubble viz • $10k strict sim", Style::default().fg(Color::Gray)),
    ]))
    .block(Block::default().borders(Borders::ALL).title("grok-lppl-rust"));
    f.render_widget(header, chunks[0]);

    // Body: left params | right viz+info
    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(28), Constraint::Percentage(72)])
        .split(chunks[1]);

    // === LEFT: Parameters + data status ===
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(12), Constraint::Length(6)])
        .split(body_chunks[0]);

    // Params
    let mut param_lines: Vec<Line> = vec![];
    for i in 0..NUM_FIELDS {
        let label = App::field_label(i);
        let val = match i {
            0 => &app.ticker,
            1 => &app.start_str,
            2 => &app.end_str,
            3 => &app.window_str,
            4 => &app.refit_str,
            5 => &app.long_str,
            6 => &app.short_str,
            7 => &app.cost_str,
            8 => &app.capital_str,
            9 => &app.random_seed_str,
            10 => &app.run_mode_str,
            11 => &app.ensemble_seeds_str,
            12 => &app.predict_horizon_str,
            _ => &app.ticker,
        };
        let is_focus = i == app.focused;
        let prefix = if is_focus { "▶ " } else { "  " };
        let style = if is_focus {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        param_lines.push(Line::from(vec![
            Span::raw(prefix),
            Span::styled(format!("{:20}", label), style),
            Span::raw(" "),
            Span::styled(val.clone(), style),
        ]));
    }
    let params = Paragraph::new(param_lines)
        .block(
            Block::default()
                .title(" Parameters (Tab/Shift-Tab, type, +/- nudge, Enter=Run) ")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(params, left_chunks[0]);

    // Data / API status
    let data_text = vec![
        Line::from(Span::styled("Yahoo / Data", Style::default().add_modifier(Modifier::BOLD))),
        Line::from(app.engine.fetch_status.as_str()),
        Line::from(format!("Bars cached: {}", app.bars.as_ref().map_or(0, |b| b.len()))),
        Line::from(""),
        Line::from(Span::styled("Tip: Change ticker/dates then [F] to re-test API.", Style::default().fg(Color::DarkGray))),
    ];
    let data_box = Paragraph::new(data_text)
        .block(Block::default().title(" Data Status ").borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    f.render_widget(data_box, left_chunks[1]);

    // === RIGHT: Viz + metrics + trades ===
    let right = body_chunks[1];
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),   // summary metrics
            Constraint::Min(6),      // charts
            Constraint::Length(7),   // trades + cursor detail
        ])
        .split(right);

    // Metrics bar
    let mut metric_spans = vec![Span::raw("Strategy on 10k:  ")];
    if let Some(res) = &app.result {
        let final_usd = app.final_dollar();
        let pnl = final_usd - app.engine.initial_capital;
        let pnl_str = format!("${:.0}  ({:+.1}%)", final_usd, (pnl / app.engine.initial_capital) * 100.0);
        metric_spans.push(Span::styled(
            pnl_str,
            Style::default().fg(if pnl >= 0.0 { Color::Green } else { Color::Red }).add_modifier(Modifier::BOLD),
        ));
        metric_spans.push(Span::raw(format!(
            "   |  AnnRet {:.1}%  Sharpe {:.2}  MaxDD {:.1}%  Trades {}  vs B&H {:.1}%",
            res.annualized_return * 100.0,
            res.sharpe,
            res.max_drawdown * 100.0,
            res.num_trades,
            res.buy_hold_return * 100.0
        )));
    } else {
        metric_spans.push(Span::styled("no simulation yet — press [R]", Style::default().fg(Color::DarkGray)));
    }
    let metrics = Paragraph::new(Line::from(metric_spans))
        .block(Block::default().title(" 10k Trade Simulation (strictly follows thresholds & costs) ").borders(Borders::ALL));
    f.render_widget(metrics, right_chunks[0]);

    // Charts area — three stacked charts for proper independent scaling + clear bubble indicator
    let chart_area = right_chunks[1];
    if let Some(res) = &app.result {
        let n = res.signals.len();
        let vs = app.view_start.min(n.saturating_sub(1));
        let vl = app.view_len.min(n.saturating_sub(vs)).max(1);

        // Split chart_area into three horizontal strips
        let chart_subs = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(40), // price (with regime coloring)
                Constraint::Percentage(35), // bubble score (the key indicator)
                Constraint::Percentage(25), // equity $
            ])
            .split(chart_area);

        let price_area = chart_subs[0];
        let score_area = chart_subs[1];
        let equity_area = chart_subs[2];

        // Data slices for current view
        let (long_p, short_p, flat_p) = if app.show_price {
            hlpll_backtester::modules::utils::build_regime_price_segments(&res.signals, vs, vl)
        } else {
            (vec![], vec![], vec![])
        };
        let score_pts: Vec<(f64, f64)> = if app.show_score {
            hlpll_backtester::modules::utils::build_score_series(&res.signals, vs, vl)
        } else {
            vec![]
        };
        let eq_pts: Vec<(f64, f64)> = if app.show_equity {
            hlpll_backtester::modules::utils::build_equity_series(res, vs, vl, app.engine.initial_capital)
        } else {
            vec![]
        };

        let x_bounds = [0.0, vl as f64];

        // 1. PRICE chart (colored by strategy position — the "bubble regime" indicator)
        {
            let mut pds: Vec<Dataset> = vec![];
            if !long_p.is_empty() {
                pds.push(Dataset::default().name("LONG").data(&long_p).graph_type(GraphType::Line).marker(Marker::Braille).style(Style::default().fg(Color::Green)));
            }
            if !short_p.is_empty() {
                pds.push(Dataset::default().name("SHORT").data(&short_p).graph_type(GraphType::Line).marker(Marker::Braille).style(Style::default().fg(Color::Red)));
            }
            if !flat_p.is_empty() {
                pds.push(Dataset::default().name("flat").data(&flat_p).graph_type(GraphType::Line).marker(Marker::Braille).style(Style::default().fg(Color::Gray)));
            }
            // compute y range for price view
            let view_prices: Vec<f64> = res.signals[vs..vs+vl.min(res.signals.len())].iter().map(|s| s.close).collect();
            let pmin = if view_prices.is_empty() { 0.0 } else { view_prices.iter().cloned().fold(f64::INFINITY, f64::min) };
            let pmax = if view_prices.is_empty() { 1.0 } else { view_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max) };
            let prange = if (pmax - pmin).abs() > 1e-6 { [pmin*0.995, pmax*1.005] } else { [pmin-1.0, pmax+1.0] };

            let price_chart = Chart::new(pds)
                .block(Block::default().title(format!(" PRICE (green=LONG / red=SHORT / gray=flat)  view {}..{}", vs+1, vs+vl)).borders(Borders::ALL))
                .x_axis(Axis::default().bounds(x_bounds).style(Style::default().fg(Color::DarkGray)))
                .y_axis(Axis::default().bounds(prange).labels(vec!["min".into(), "max".into()]).style(Style::default().fg(Color::DarkGray)));
            f.render_widget(price_chart, price_area);
        }

        // 2. BUBBLE SCORE — the core indicator visualization
        {
            let mut sds: Vec<Dataset> = vec![];
            if !score_pts.is_empty() {
                sds.push(Dataset::default().name("bubble_score").data(&score_pts).graph_type(GraphType::Line).marker(Marker::Dot).style(Style::default().fg(Color::Yellow)));
            }
            let lt: f64 = app.long_str.trim().parse().unwrap_or(0.75);
            let st: f64 = app.short_str.trim().parse().unwrap_or(0.75);
            let w = vl as f64;
            let ltp: Vec<(f64,f64)> = vec![(0.0, lt), (w, lt)];
            let stp: Vec<(f64,f64)> = vec![(0.0, -st), (w, -st)];
            sds.push(Dataset::default().name("+long").data(&ltp).graph_type(GraphType::Line).style(Style::default().fg(Color::LightGreen)));
            sds.push(Dataset::default().name("-short").data(&stp).graph_type(GraphType::Line).style(Style::default().fg(Color::LightRed)));

            let smin = if score_pts.is_empty() { -st-0.5 } else { score_pts.iter().map(|&(_,y)| y).fold(0.0, f64::min).min(-st-0.2) };
            let smax = if score_pts.is_empty() { lt+0.5 } else { score_pts.iter().map(|&(_,y)| y).fold(0.0, f64::max).max(lt+0.2) };

            let score_chart = Chart::new(sds)
                .block(Block::default().title(" BUBBLE SCORE (yellow) + thresholds (green/red lines) — long when above green, short below red ").borders(Borders::ALL))
                .x_axis(Axis::default().bounds(x_bounds).style(Style::default().fg(Color::DarkGray)))
                .y_axis(Axis::default().bounds([smin, smax]).labels(vec![format!("{:.1}", smin).into(), "0".into(), format!("{:.1}", smax).into()]).style(Style::default().fg(Color::DarkGray)));
            f.render_widget(score_chart, score_area);
        }

        // 3. $ EQUITY curve for the 10k simulation
        {
            let mut eds: Vec<Dataset> = vec![];
            if !eq_pts.is_empty() {
                eds.push(Dataset::default().name("$10k equity").data(&eq_pts).graph_type(GraphType::Line).marker(Marker::Braille).style(Style::default().fg(Color::Cyan)));
            }
            let emin = if eq_pts.is_empty() { app.engine.initial_capital } else { eq_pts.iter().map(|&(_,y)| y).fold(f64::INFINITY, f64::min) };
            let emax = if eq_pts.is_empty() { app.engine.initial_capital } else { eq_pts.iter().map(|&(_,y)| y).fold(f64::NEG_INFINITY, f64::max) };
            let erng = if (emax-emin).abs() > 1.0 { [emin*0.99, emax*1.01] } else { [emin-10.0, emax+10.0] };

            let eq_chart = Chart::new(eds)
                .block(Block::default().title(format!(" EQUITY (starting ${:.0}) — strictly follows strategy positions & costs ", app.engine.initial_capital)).borders(Borders::ALL))
                .x_axis(Axis::default().bounds(x_bounds).style(Style::default().fg(Color::DarkGray)))
                .y_axis(Axis::default().bounds(erng).labels(vec!["start".into(), "now".into()]).style(Style::default().fg(Color::DarkGray)));
            f.render_widget(eq_chart, equity_area);
        }

        // Legend line under charts
        let legend = Paragraph::new(Line::from(vec![
            Span::raw("Toggles: [p]rice [s]core [u]equity   Pan: ←→ hl / PgUpDn   Zoom: [ ]   Cursor: j k   Fit: 0   Export: [E]"),
        ]));
        // place it in a 1-line rect at bottom of chart_area if space
        let leg_rect = Rect { x: chart_area.x, y: chart_area.y + chart_area.height.saturating_sub(1), width: chart_area.width, height: 1 };
        f.render_widget(legend, leg_rect);
    } else {
        // Support for new modes: show prediction or live info when no full backtest result
        if let Some(p) = &app.engine.last_future_prediction {
            let info = format!(
                "FUTURE BUBBLE PREDICTION (C1 from multi-window JLS/LPPLS)\n\nDate: {}\nPrice: {:.2}\nC1 Confidence: {:.1}% (ensemble {:?}, std {:.1})\nRisk: {} — {}\nMedian predicted critical/peak: {} ({} days ahead)\nProb tc within horizon ({}d): {:.1}%\nValid fits: {}/{} windows\n\n(Use this for 'will there be a bubble peak soon?' research. See README + gemini-data-LPPLS.md)",
                p.analysis_date, p.current_price, p.bubble_confidence_index, p.ensemble_seeds_used, p.ensemble_std_confidence,
                p.risk_level, p.risk_description,
                p.median_predicted_date.map(|d| d.to_string()).unwrap_or("N/A".into()), p.median_days_to_tc.unwrap_or(0),
                app.engine.config.predict_horizon_days, p.prob_tc_within_horizon * 100.0,
                p.valid_fits, p.total_windows_tested
            );
            let pred_box = Paragraph::new(info)
                .block(Block::default().title(" FUTURE BUBBLE PREDICTION — extensive C1 + tc (no equity chart in pure prediction mode) ").borders(Borders::ALL))
                .wrap(Wrap { trim: true });
            f.render_widget(pred_box, chart_area);
        } else if let Some(s) = &app.engine.last_live_sentiment {
            let info = format!(
                "LIVE CURRENT SENTIMENT (for trading on current LPPLS + C1 now)\n\nAs of: {}  Price: {:.2}\nBubble score: {:.3}\nC1: {:.1}%  Risk: {}\n\nRECOMMENDATION: {}\n\nActionable: {}\nMedian predicted peak: {} (~{}d)\n\n(For live trading signals. Run in 'live' or 'hybrid' mode. Cross-check with other data. Not financial advice.)",
                s.date, s.current_price, s.bubble_score, s.bubble_confidence, s.risk_level,
                s.recommendation, s.actionable_note,
                s.median_predicted_peak.map(|d|d.to_string()).unwrap_or("N/A".into()), s.median_days_to_tc.unwrap_or(0)
            );
            let live_box = Paragraph::new(info)
                .block(Block::default().title(" LIVE CURRENT SENTIMENT — equation + C1 for 'trade now?' ").borders(Borders::ALL))
                .wrap(Wrap { trim: true });
            f.render_widget(live_box, chart_area);
        } else {
            let placeholder = Paragraph::new(
                "No simulation results.\n\nPress [F] to fetch... [R] to run (uses random_seed for LPPL multi-start reproducibility).\n\nNEW: Set 'Run mode' field (historical/prediction/live/hybrid), 'Ensemble seeds', 'Predict horizon' then [R].\nFor prediction: see C1% + risk + median tc dates here.\nFor live: see current BUY/SELL/HOLD sentiment + actionable note.\n\nThe RNG seed (and ensemble) controls the random sampling of LPPL nonlinear params (tc/m/omega/phi) for the fit search (needed b/c non-convex opt; fixed seed => reproducible). \n\nCursor (after historical/hybrid run) shows clear RECOMMENDATION: BUY/SELL/HOLD at that date based on final position after score vs thresh (with bias/invert + C1 filter).\n\nCharts (historical): PRICE (colored by regime), BUBBLE SCORE (steps b/c refits), EQUITY vs B&H.\nTweak fields (incl seed/mode) + rerun live.",
            )
            .block(Block::default().title(" Visualization — Bubble Indicator + Strict $10k Trade Sim + Future/Live (full user control) ").borders(Borders::ALL))
            .wrap(Wrap { trim: true });
            f.render_widget(placeholder, chart_area);
        }
    }

    // Bottom right: Cursor detail + Trade log
    let detail_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(right_chunks[2]);

    // Cursor / current bubble reading
    let mut cursor_lines = vec![Line::from(Span::styled("Cursor / Current Bubble Reading", Style::default().add_modifier(Modifier::BOLD)))];
    if let Some(sig) = app.current_signal() {
        let pos_str = if sig.position > 0.5 {
            "LONG"
        } else if sig.position < -0.5 {
            "SHORT"
        } else {
            "FLAT"
        };
        let pos_col = if sig.position > 0.5 {
            Color::Green
        } else if sig.position < -0.5 {
            Color::Red
        } else {
            Color::Gray
        };
        cursor_lines.push(Line::from(format!(
            "{}  close=${:.2}  vol={:.0}",
            sig.date, sig.close, sig.volume
        )));
        cursor_lines.push(Line::from(vec![
            Span::raw("bubble_score: "),
            Span::styled(format!("{:.3}", sig.bubble_score), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(format!("   → {}", pos_str)),
        ]));
        cursor_lines.push(Line::from(format!(
            "eps_norm={:.3}  hype={:.3}  sent={:.3}",
            sig.eps_norm, sig.hype_volume, sig.sentiment
        )));
        let rec = if sig.position > 0.5 {
            ("BUY / GO LONG (bullish)", Color::Green)
        } else if sig.position < -0.5 {
            ("SELL / GO SHORT (bearish/risk)", Color::Red)
        } else {
            ("HOLD / NEUTRAL (flat)", Color::Gray)
        };
        cursor_lines.push(Line::from(vec![
            Span::raw("RECOMMENDATION: "),
            Span::styled(rec.0, Style::default().fg(rec.1).add_modifier(Modifier::BOLD)),
        ]));
        cursor_lines.push(Line::from(vec![
            Span::raw("position: "),
            Span::styled(pos_str, Style::default().fg(pos_col).add_modifier(Modifier::BOLD)),
            Span::raw(if sig.trade { "   (TRADE day — cost applied)" } else { "" }),
        ]));
    } else {
        cursor_lines.push(Line::from("Move cursor with j/k after running a sim."));
    }
    let cursor_box = Paragraph::new(cursor_lines)
        .block(Block::default().title(" Live Bubble Indicator + RECOMMENDATION (cursor j/k or mouse in GUI) ").borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    f.render_widget(cursor_box, detail_chunks[0]);

    // Trades
    let mut trade_lines = vec![Line::from(Span::styled(
        format!("Trade Log ({} legs, $ PnL on {} start)", app.trades.len(), app.engine.initial_capital),
        Style::default().add_modifier(Modifier::BOLD),
    ))];
    if app.trades.is_empty() {
        trade_lines.push(Line::from("No completed legs yet (or still running)."));
    } else {
        // Show up to 5
        for t in app.trades.iter().rev().take(5) {
            let sign = if t.pnl_usd >= 0.0 { "+" } else { "" };
            trade_lines.push(Line::from(format!(
                "{}→{} {} {}d  {} {:.1}%  ${}{:.0}",
                t.entry_date,
                t.exit_date,
                t.direction,
                t.bars_held,
                if t.direction == "LONG" { "▲" } else { "▼" },
                t.ret_pct * 100.0,
                sign,
                t.pnl_usd
            )));
        }
        if app.trades.len() > 5 {
            trade_lines.push(Line::from(format!("... {} more (see exported CSV)", app.trades.len() - 5)));
        }
    }
    let trade_box = Paragraph::new(trade_lines)
        .block(Block::default().title(" Strict Strategy Trades (derived from position changes) ").borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    f.render_widget(trade_box, detail_chunks[1]);

    // Footer / status
    let footer_text = if let Some(err) = &app.last_error {
        Line::from(vec![
            Span::styled("ERROR: ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::styled(err, Style::default().fg(Color::Red)),
        ])
    } else {
        Line::from(Span::styled(&app.status, Style::default().fg(Color::Cyan)))
    };
    let footer = Paragraph::new(footer_text)
        .block(Block::default().borders(Borders::TOP))
        .wrap(Wrap { trim: true });
    f.render_widget(footer, chunks[2]);

    // Tiny key reminder (overwrites a bit of footer area conceptually)
    // We render a compact help line at very bottom via a small rect if space.
}

/// Main TUI loop.
fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, mut app: App) -> Result<(), Box<dyn Error>> {
    while app.running {
        terminal.draw(|f| ui(f, &app))?;

        if event::poll(std::time::Duration::from_millis(120))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    app.handle_key(key.code);
                }
            }
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let app = App::new();

    // Run
    let res = run_app(&mut terminal, app);

    // Restore
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(e) = res {
        eprintln!("Explorer error: {}", e);
    }

    println!("Explorer exited. Results (if exported) are in ./results/ .");
    println!("You can also run the classic CLI: cargo run --release -- --help");
    Ok(())
}
