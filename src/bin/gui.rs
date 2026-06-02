//! hlpll-gui — Native Windows desktop UI (egui + eframe) for the HLPPL backtester.
//!
//! This is a *completely separate* frontend from the TUI explorer and the CLI.
//! It uses the shared `HlpplEngine` (the isolated logic engine) so the core
//! strategy, LPPL fitting, Yahoo fetching, bubble scoring, trade extraction,
//! and export logic are 100% reused and not duplicated.
//!
//! Build & run (proper .exe):
//!   cargo run --release --bin hlpll-gui --no-default-features --features gui
//!
//! The resulting target/release/hlpll-gui.exe is a self-contained native app
//! with nice interactive charts, sliders, live updates, etc.

use chrono::NaiveDate;
use eframe::egui;
use egui_plot::{GridMark, HLine, Line, Plot, PlotPoints, VLine};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

use hlpll_backtester::engine::HlpplEngine;
use hlpll_backtester::modules::backtest::BacktestConfig;
use hlpll_backtester::modules::utils::{
    build_equity_series, build_regime_price_segments, build_score_series,
};

/// Messages sent from the UI thread to a worker thread.
enum Command {
    Fetch {
        ticker: String,
        start: NaiveDate,
        end: NaiveDate,
    },
    Run {
        ticker: String,
        start: NaiveDate,
        end: NaiveDate,
        config: BacktestConfig,
        capital: f64,
    },
}

/// Messages coming back from the worker (results or errors).
enum Update {
    FetchDone {
        engine: HlpplEngine,
    },
    RunDone {
        engine: HlpplEngine,
    },
    Error(String),
}

struct HlpplGuiApp {
    // The one true source for all HLPPL logic (modular & isolated)
    engine: HlpplEngine,

    // UI form state (strings + numbers for immediate editing)
    ticker: String,
    start_str: String,
    end_str: String,
    window: usize,
    refit_every: usize,
    long_thresh: f64,
    short_thresh: f64,
    cost_bps: f64,
    initial_capital: f64,

    // Async worker plumbing (so long LPPL fits and network never block the UI)
    cmd_tx: Sender<Command>,
    update_rx: Receiver<Update>,
    busy: bool,
    status: String,
    last_error: Option<String>,

    // Plot / cursor state (full user control)
    selected_idx: usize,
    show_price: bool,
    show_score: bool,
    show_equity: bool,

    // Local mirrors for the current view (updated when engine produces a result)
    view_start: usize,
    view_len: usize,

    /// Whether the help / legend / explanation window is open
    show_help: bool,

    // New strategy mode controls (exposed nicely in the UI)
    position_bias: hlpll_backtester::PositionBias,
    invert_signal: bool,
    random_seed: u64,  // exposed so user can control reproducibility of LPPL random search

    track_on_hover: bool, // for live cursor tracker on price chart

    // === NEW EXTENSIVE BUBBLE ANALYSIS CONFIG (multi-window, strict filters, confidence for trading) ===
    enable_bubble_analysis: bool,
    analysis_lookback_min: usize,
    analysis_lookback_max: usize,
    analysis_step_days: usize,
    filter_m_min: f64,
    filter_m_max: f64,
    filter_omega_min: f64,
    filter_omega_max: f64,
    filter_require_b_negative: bool,
    filter_min_tc_offset_days: usize,
    use_confidence_for_flat: bool,
    confidence_flat_threshold: f64,

    // === RUN MODE + ENSEMBLE + PREDICTION (most extensive support for future bubbles + live sentiment) ===
    run_mode: hlpll_backtester::RunMode,
    ensemble_seeds_str: String, // comma sep e.g. "42,43,44" for robust C1
    predict_horizon: usize,
}

impl HlpplGuiApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Reasonable starting point (user can tweak everything)
        let ticker = "CAR".to_string();
        let start_str = "2022-01-01".to_string();
        let end_str = "2024-12-31".to_string();
        let window = 180usize;
        let refit_every = 20usize;
        let long_thresh = 0.65;
        let short_thresh = 0.65;
        let cost_bps = 8.0;
        let initial_capital = 10_000.0;

        let start = NaiveDate::parse_from_str(&start_str, "%Y-%m-%d")
            .unwrap_or_else(|_| NaiveDate::from_ymd_opt(2022, 1, 1).unwrap());
        let end = NaiveDate::parse_from_str(&end_str, "%Y-%m-%d")
            .unwrap_or_else(|_| NaiveDate::from_ymd_opt(2024, 12, 31).unwrap());

        let cfg = BacktestConfig {
            lookback_days: window,
            refit_every,
            long_threshold: long_thresh,
            short_threshold: short_thresh,
            cost_bps,
            max_position: 1.0,
            ..Default::default()
        };

        let engine = HlpplEngine::new(&ticker, start, end, cfg, initial_capital);

        // Worker thread + channels
        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();
        let (update_tx, update_rx) = mpsc::channel::<Update>();

        // Spawn a single background worker. It owns the heavy lifting.
        std::thread::spawn(move || {
            while let Ok(cmd) = cmd_rx.recv() {
                match cmd {
                    Command::Fetch { ticker, start, end } => {
                        let mut eng = HlpplEngine::new(
                            &ticker,
                            start,
                            end,
                            BacktestConfig::default(),
                            10_000.0,
                        );
                        match eng.fetch() {
                            Ok(_) => {
                                let _ = update_tx.send(Update::FetchDone { engine: eng });
                            }
                            Err(e) => {
                                let _ = update_tx.send(Update::Error(e));
                            }
                        }
                    }
                    Command::Run {
                        ticker,
                        start,
                        end,
                        config,
                        capital,
                    } => {
                        let mut eng = HlpplEngine::new(&ticker, start, end, config, capital);
                        // We always (re)fetch inside the worker so the user can hit "Run" directly.
                        if let Err(e) = eng.fetch() {
                            let _ = update_tx.send(Update::Error(format!("Fetch failed: {}", e)));
                            continue;
                        }
                        // Use mode-aware run so Prediction/Live/Hybrid populate the right engine fields (future_prediction etc)
                        match eng.run_with_mode() {
                            Ok(()) => {
                                let _ = update_tx.send(Update::RunDone { engine: eng });
                            }
                            Err(e) => {
                                let _ = update_tx.send(Update::Error(e));
                            }
                        }
                    }
                }
            }
        });

        Self {
            engine,
            ticker,
            start_str,
            end_str,
            window,
            refit_every,
            long_thresh,
            short_thresh,
            cost_bps,
            initial_capital,
            cmd_tx,
            update_rx,
            busy: false,
            status: "Ready. Edit parameters on the left, then click 'Run Simulation' for a full strict backtest + $10k equity curve.".to_string(),
            last_error: None,
            selected_idx: 0,
            show_price: true,
            show_score: true,
            show_equity: true,
            view_start: 0,
            view_len: 400,
            show_help: false,
            position_bias: hlpll_backtester::PositionBias::LongShort,
            invert_signal: false,
            random_seed: 42,
            track_on_hover: true,

            enable_bubble_analysis: true,
            analysis_lookback_min: 60,
            analysis_lookback_max: 260,
            analysis_step_days: 5,
            filter_m_min: 0.1,
            filter_m_max: 0.9,
            filter_omega_min: 4.5,
            filter_omega_max: 13.0,
            filter_require_b_negative: true,
            filter_min_tc_offset_days: 3,
            use_confidence_for_flat: true,
            confidence_flat_threshold: 50.0,

            run_mode: hlpll_backtester::RunMode::HistoricalBacktest,
            ensemble_seeds_str: "42".into(),
            predict_horizon: 60,
        }
    }

    /// Push current UI values into a fresh config and send a Run command.
    fn trigger_run(&mut self) {
        let start = match NaiveDate::parse_from_str(&self.start_str, "%Y-%m-%d") {
            Ok(d) => d,
            Err(_) => {
                self.last_error = Some("Bad start date".into());
                return;
            }
        };
        let end = match NaiveDate::parse_from_str(&self.end_str, "%Y-%m-%d") {
            Ok(d) => d,
            Err(_) => {
                self.last_error = Some("Bad end date".into());
                return;
            }
        };

        let cfg = BacktestConfig {
            lookback_days: self.window,
            refit_every: self.refit_every,
            long_threshold: self.long_thresh,
            short_threshold: self.short_thresh,
            cost_bps: self.cost_bps,
            max_position: 1.0,
            position_bias: self.position_bias,
            invert_signal: self.invert_signal,
            random_seed: self.random_seed,

            enable_bubble_analysis: self.enable_bubble_analysis,
            analysis_lookback_min: self.analysis_lookback_min,
            analysis_lookback_max: self.analysis_lookback_max,
            analysis_step_days: self.analysis_step_days,
            filter_m_min: self.filter_m_min,
            filter_m_max: self.filter_m_max,
            filter_omega_min: self.filter_omega_min,
            filter_omega_max: self.filter_omega_max,
            filter_require_b_negative: self.filter_require_b_negative,
            filter_min_tc_offset_days: self.filter_min_tc_offset_days,
            use_confidence_for_flat: self.use_confidence_for_flat,
            confidence_flat_threshold: self.confidence_flat_threshold,

            run_mode: self.run_mode,
            ensemble_seeds: if self.ensemble_seeds_str.trim().is_empty() { vec![] } else { self.ensemble_seeds_str.split(',').filter_map(|s| s.trim().parse().ok()).collect() },
            predict_horizon_days: self.predict_horizon,
            use_confidence_for_sizing: false, // UI can add checkbox later
        };

        self.busy = true;
        self.last_error = None;
        self.status = "Running LPPL fits + bubble scoring + strategy in background (UI stays responsive)...".to_string();

        let _ = self.cmd_tx.send(Command::Run {
            ticker: self.ticker.clone(),
            start,
            end,
            config: cfg,
            capital: self.initial_capital,
        });
    }

    fn trigger_fetch(&mut self) {
        let start = match NaiveDate::parse_from_str(&self.start_str, "%Y-%m-%d") {
            Ok(d) => d,
            Err(_) => {
                self.last_error = Some("Bad start date".into());
                return;
            }
        };
        let end = match NaiveDate::parse_from_str(&self.end_str, "%Y-%m-%d") {
            Ok(d) => d,
            Err(_) => {
                self.last_error = Some("Bad end date".into());
                return;
            }
        };

        self.busy = true;
        self.last_error = None;
        self.status = "Testing Yahoo Finance API for the chosen security + dates...".to_string();

        let _ = self.cmd_tx.send(Command::Fetch {
            ticker: self.ticker.clone(),
            start,
            end,
        });
    }

    /// Drain any updates that arrived from the worker thread.
    fn poll_updates(&mut self) {
        while let Ok(update) = self.update_rx.try_recv() {
            self.busy = false;
            match update {
                Update::FetchDone { engine } => {
                    self.engine = engine;
                    self.status = self.engine.fetch_status.clone();
                    if self.engine.bars.is_some() {
                        self.status.push_str("  — Now click 'Run Simulation' to execute the full strategy.");
                    }
                }
                Update::RunDone { engine } => {
                    self.engine = engine;
                    // Sync UI numbers from the engine that just succeeded (in case worker normalized anything)
                    self.initial_capital = self.engine.initial_capital;
                    self.ticker = self.engine.ticker.clone();

                    if let Some(res) = &self.engine.result {
                        let n = res.signals.len();
                        self.view_start = 0;
                        self.view_len = n.min(500).max(30);
                        self.selected_idx = n / 2;
                    }
                    self.status = format!(
                        "Done. Final equity ${:.0} ({:+.1}%) | {} trades | engine used for 100% of the logic.",
                        self.engine.final_capital(),
                        (self.engine.total_pnl_usd() / self.engine.initial_capital) * 100.0,
                        self.engine.trades.len()
                    );
                }
                Update::Error(msg) => {
                    self.last_error = Some(msg.clone());
                    self.status = "Error — see message on the left.".to_string();
                }
            }
        }
    }

    fn sync_engine_from_ui(&mut self) {
        // Keep the engine's config in sync with the sliders/texts for the next Run.
        let start = NaiveDate::parse_from_str(&self.start_str, "%Y-%m-%d")
            .unwrap_or(self.engine.start);
        let end = NaiveDate::parse_from_str(&self.end_str, "%Y-%m-%d")
            .unwrap_or(self.engine.end);

        self.engine.ticker = self.ticker.to_uppercase();
        self.engine.start = start;
        self.engine.end = end;
        self.engine.config.lookback_days = self.window;
        self.engine.config.refit_every = self.refit_every;
        self.engine.config.long_threshold = self.long_thresh;
        self.engine.config.short_threshold = self.short_thresh;
        self.engine.config.cost_bps = self.cost_bps;
        self.engine.initial_capital = self.initial_capital;

        // New modes
        self.engine.config.position_bias = self.position_bias;
        self.engine.config.invert_signal = self.invert_signal;
        self.engine.config.random_seed = self.random_seed;

        // NEW bubble analysis extensive settings
        self.engine.config.enable_bubble_analysis = self.enable_bubble_analysis;
        self.engine.config.analysis_lookback_min = self.analysis_lookback_min;
        self.engine.config.analysis_lookback_max = self.analysis_lookback_max;
        self.engine.config.analysis_step_days = self.analysis_step_days;
        self.engine.config.filter_m_min = self.filter_m_min;
        self.engine.config.filter_m_max = self.filter_m_max;
        self.engine.config.filter_omega_min = self.filter_omega_min;
        self.engine.config.filter_omega_max = self.filter_omega_max;
        self.engine.config.filter_require_b_negative = self.filter_require_b_negative;
        self.engine.config.filter_min_tc_offset_days = self.filter_min_tc_offset_days;
        self.engine.config.use_confidence_for_flat = self.use_confidence_for_flat;
        self.engine.config.confidence_flat_threshold = self.confidence_flat_threshold;

        self.engine.config.run_mode = self.run_mode;
        self.engine.config.ensemble_seeds = if self.ensemble_seeds_str.trim().is_empty() { vec![] } else { self.ensemble_seeds_str.split(',').filter_map(|s| s.trim().parse().ok()).collect() };
        self.engine.config.predict_horizon_days = self.predict_horizon;
    }
}

impl eframe::App for HlpplGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_updates();

        // Request repaint while busy so the spinner animates
        if self.busy {
            ctx.request_repaint();
        }

        // Top bar
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("HLPPL Backtesting Explorer — Native GUI");
                ui.separator();
                if self.busy {
                    ui.spinner();
                    ui.label("Working...");
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Export artifacts (CSV + PNG)").clicked() {
                        if let Err(e) = self.engine.export_artifacts("results") {
                            self.last_error = Some(e);
                        } else {
                            self.status = "Exported to ./results/ using the shared engine.".to_string();
                        }
                    }
                    if ui.button("Help / Legends / Explanation").clicked() {
                        self.show_help = true;
                    }
                    if ui.button("Run Simulation").clicked() && !self.busy {
                        self.sync_engine_from_ui();
                        self.trigger_run();
                    }
                    if ui.button("Test Yahoo API").clicked() && !self.busy {
                        self.trigger_fetch();
                    }
                    if ui.button("Run Bubble Analysis (predict/live)").clicked() && !self.busy {
                        self.sync_engine_from_ui();
                        // Run analysis in worker or direct; for simplicity direct here (fast enough)
                        match self.engine.run_bubble_analysis() {
                            Ok(analysis) => {
                                self.status = format!(
                                    "Bubble Analysis: {:.1}% confidence | {} | Median peak ~{}",
                                    analysis.bubble_confidence_index,
                                    analysis.risk_level,
                                    analysis.median_predicted_date.map(|d| d.to_string()).unwrap_or("N/A".into())
                                );
                                // For full sentiment, also run get if wanted
                            }
                            Err(e) => { self.last_error = Some(e); }
                        }
                    }
                });
            });
        });

        // Left control panel
        egui::SidePanel::left("controls").resizable(true).show(ctx, |ui| {
            ui.heading("Parameters (live)");

            egui::Grid::new("params").num_columns(2).spacing([4.0, 4.0]).show(ui, |ui| {
                ui.label("Ticker");
                ui.text_edit_singleline(&mut self.ticker);
                ui.end_row();

                ui.label("Start");
                ui.text_edit_singleline(&mut self.start_str);
                ui.end_row();

                ui.label("End");
                ui.text_edit_singleline(&mut self.end_str);
                ui.end_row();

                ui.label("Window (days)").on_hover_text("LPPL lookback in trading days. Each fit uses this many prior bars. Bigger = more stable but less responsive.");
                ui.add(egui::DragValue::new(&mut self.window).speed(5).range(60..=2000));
                ui.end_row();

                ui.label("Refit every").on_hover_text("Re-fit the LPPL model only every N days. On other days the bubble score is held constant (you see the flat steps in chart 2).");
                ui.add(egui::DragValue::new(&mut self.refit_every).speed(1).range(1..=100));
                ui.end_row();

                ui.label("Long threshold").on_hover_text("If bubble_score > this value the strategy goes LONG (+1).");
                ui.add(egui::DragValue::new(&mut self.long_thresh).speed(0.01).range(0.0..=5.0));
                ui.end_row();

                ui.label("Short threshold").on_hover_text("If bubble_score < -this value the strategy goes SHORT (-1).");
                ui.add(egui::DragValue::new(&mut self.short_thresh).speed(0.01).range(0.0..=5.0));
                ui.end_row();

                ui.label("Cost (bps)").on_hover_text("One-way transaction cost in basis points, subtracted on any day the position changes.");
                ui.add(egui::DragValue::new(&mut self.cost_bps).speed(0.5).range(0.0..=100.0));
                ui.end_row();

                ui.label("Initial Capital $").on_hover_text("Starting portfolio $ for the simulation. All PnL and equity numbers are scaled to this value.");
                ui.add(egui::DragValue::new(&mut self.initial_capital).speed(1000.0).range(1000.0..=10_000_000.0));
                ui.end_row();

                // New: strategy mode controls for "long only", "short only", invert etc.
                ui.label("Position mode").on_hover_text("LongOnly: never short (good for detecting higher-going momentum using positive bubble scores). ShortOnly: never long. LongShort = classic.");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.position_bias, hlpll_backtester::PositionBias::LongShort, "L/S");
                    ui.selectable_value(&mut self.position_bias, hlpll_backtester::PositionBias::LongOnly, "Long only");
                    ui.selectable_value(&mut self.position_bias, hlpll_backtester::PositionBias::ShortOnly, "Short only");
                });
                ui.end_row();

                ui.label("");
                ui.checkbox(&mut self.invert_signal, "Invert (high score = danger)")
                    .on_hover_text("If checked, a high positive bubble score is treated as 'overextended / crash risk' and produces a negative (short) raw signal. Often more in line with the original bubble-detection literature than pure momentum continuation.");
                ui.end_row();

                ui.label("RNG seed (LPPL fits)").on_hover_text("Seed for the random search inside each LPPL fit. Fixed seed (e.g. 42) makes every 'Run Simulation' with identical params produce exactly the same results (fully reproducible). Change it to explore different possible LPPL fits.");
                ui.add(egui::DragValue::new(&mut self.random_seed).speed(1));
                ui.end_row();

                // NEW extensive bubble analysis controls (multi-window JLS for confidence index, future prediction, live sentiment)
                ui.label("Enable Bubble Analysis").on_hover_text("Run multi-window LPPLS sweep with strict filters at run time. Computes Bubble Confidence Index (0-100%), risk level, predicted critical/peak dates for future bubble prediction. Essential for 'live current sentiment'.");
                ui.checkbox(&mut self.enable_bubble_analysis, "");
                ui.end_row();

                ui.label("Analysis lookback min/max").on_hover_text("Range of historical window lengths (in trading days) to sweep for the confidence index. E.g. 60 to 260 days. More windows = more robust % valid fits.");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.analysis_lookback_min).speed(5).range(30..=500));
                    ui.label("/");
                    ui.add(egui::DragValue::new(&mut self.analysis_lookback_max).speed(5).range(60..=1000));
                });
                ui.end_row();

                ui.label("Conf. flat thresh / use for risk").on_hover_text("If enabled and Bubble Confidence > this %, force FLAT (risk mgmt override for high bubble regime). Great for live trading filter on current sentiment.");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.confidence_flat_threshold).speed(1.0).range(0.0..=100.0));
                    ui.checkbox(&mut self.use_confidence_for_flat, "use");
                });
                ui.end_row();

                // === Run mode + ensemble + prediction horizon (extensive new) ===
                ui.label("Run Mode").on_hover_text("Historical: full backtest+equity (classic). Prediction: pure future tc + C1 % (no equity). Live: current sentiment snapshot for trading now. Hybrid: both.");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.run_mode, hlpll_backtester::RunMode::HistoricalBacktest, "Historical");
                    ui.selectable_value(&mut self.run_mode, hlpll_backtester::RunMode::FutureBubblePrediction, "Prediction");
                    ui.selectable_value(&mut self.run_mode, hlpll_backtester::RunMode::LiveCurrentSentiment, "Live");
                    ui.selectable_value(&mut self.run_mode, hlpll_backtester::RunMode::HybridAnalysis, "Hybrid");
                });
                ui.end_row();

                ui.label("Ensemble seeds (C1 robust)").on_hover_text("Comma-separated seeds for multi-seed C1 average (e.g. 42,43,44). Makes confidence less sensitive to RNG. Leave '42' for single.");
                ui.text_edit_singleline(&mut self.ensemble_seeds_str);
                ui.end_row();

                ui.label("Predict horizon (days)").on_hover_text("For prediction reports: 'prob of tc within this many days' from the valid critical times distro.");
                ui.add(egui::DragValue::new(&mut self.predict_horizon).speed(5).range(5..=365));
                ui.end_row();

                // Full strict filters (were partially missing in UI)
                ui.label("JLS m / omega filters").on_hover_text("Strict physics constraints for 'valid bubble' in C1 computation (see gemini-data-LPPLS.md). Only windows meeting all count toward the % confidence.");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.filter_m_min).speed(0.05).range(0.0..=0.99).fixed_decimals(2));
                    ui.label("<=m<=");
                    ui.add(egui::DragValue::new(&mut self.filter_m_max).speed(0.05).range(0.0..=0.99).fixed_decimals(2));
                    ui.label("|");
                    ui.add(egui::DragValue::new(&mut self.filter_omega_min).speed(0.5).range(0.0..=30.0).fixed_decimals(1));
                    ui.label("<=w<=");
                    ui.add(egui::DragValue::new(&mut self.filter_omega_max).speed(0.5).range(0.0..=30.0).fixed_decimals(1));
                });
                ui.end_row();

                ui.label("B<0 / tc offset").on_hover_text("B negative = upward super-exp bubble. tc must be at least N days after current for a 'future' prediction.");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.filter_require_b_negative, "B<0");
                    ui.label("tc +");
                    ui.add(egui::DragValue::new(&mut self.filter_min_tc_offset_days).speed(1).range(0..=30));
                });
                ui.end_row();
            });

            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Quick: CAR 2022-2024").clicked() {
                    self.ticker = "CAR".into();
                    self.start_str = "2022-01-01".into();
                    self.end_str = "2024-12-31".into();
                    self.window = 180;
                    self.refit_every = 20;
                    self.long_thresh = 0.65;
                    self.short_thresh = 0.65;
                }
                if ui.button("Quick: AMTX").clicked() {
                    self.ticker = "AMTX".into();
                    self.start_str = "2021-01-01".into();
                    self.end_str = "2024-12-31".into();
                }
            });

            ui.add_space(8.0);

            if ui.button("Apply params & Run full backtest").clicked() && !self.busy {
                self.sync_engine_from_ui();
                self.trigger_run();
            }

            ui.add_space(12.0);
            ui.label("Status:");
            ui.label(&self.status);

            if let Some(err) = &self.last_error {
                ui.colored_label(egui::Color32::RED, err);
            }

            ui.separator();
            ui.label("This GUI and the TUI both drive the exact same HlpplEngine.\nCore LPPL / bubble / strategy code lives only in the library.");
        });

        // Central area — charts + details
        egui::CentralPanel::default().show(ctx, |ui| {
            let result = self.engine.result.as_ref();

            // Big summary numbers
            ui.horizontal(|ui| {
                let final_cap = self.engine.final_capital();
                let pnl = self.engine.total_pnl_usd();
                let pnl_pct = if self.engine.initial_capital > 0.0 {
                    (pnl / self.engine.initial_capital) * 100.0
                } else {
                    0.0
                };

                ui.label(egui::RichText::new("10k (or custom) Strategy Equity:").strong());
                let color = if pnl >= 0.0 { egui::Color32::GREEN } else { egui::Color32::RED };
                ui.colored_label(color, format!("${:.0}", final_cap));
                ui.colored_label(color, format!("  ({:+.1}%)", pnl_pct));

                if let Some(res) = result {
                    ui.separator();
                    ui.label(format!(
                        "Ann. {:.1}%  |  Sharpe {:.2}  |  MaxDD {:.1}%  |  Trades: {}",
                        res.annualized_return * 100.0,
                        res.sharpe,
                        res.max_drawdown * 100.0,
                        res.num_trades
                    ));
                    ui.separator();
                    ui.label(format!("Buy & Hold: {:.1}%", res.buy_hold_return * 100.0));
                }
            });

            // === EXTENSIVE: dedicated Prediction / Live Sentiment panel (C1, tc, risk) ===
            if let Some(pred) = &self.engine.last_future_prediction {
                ui.colored_label(if pred.bubble_confidence_index > 50.0 { egui::Color32::RED } else { egui::Color32::GREEN },
                    format!("PREDICTION: C1 {:.1}% | {} | median tc ~{} ({}d) | P(within {}d) {:.0}% | ensemble {:?}",
                        pred.bubble_confidence_index, pred.risk_level,
                        pred.median_predicted_date.map(|d|d.to_string()).unwrap_or("N/A".into()),
                        pred.median_days_to_tc.unwrap_or(0), self.predict_horizon, pred.prob_tc_within_horizon*100.0, pred.ensemble_seeds_used));
            }
            if let Some(snap) = &self.engine.last_live_sentiment {
                let c = if snap.position > 0.5 { egui::Color32::GREEN } else if snap.position < -0.5 { egui::Color32::RED } else { egui::Color32::GRAY };
                ui.colored_label(c, format!("LIVE SENTIMENT: {} | C1={:.1}% | {}", snap.recommendation, snap.bubble_confidence, snap.actionable_note));
            }

            ui.separator();

            if result.is_none() && self.engine.last_future_prediction.is_none() && self.engine.last_live_sentiment.is_none() {
                ui.label("No simulation yet. Use the controls on the left and click 'Run Simulation' (or the predict/live buttons).");
                ui.label("Choose Run Mode (Historical/Prediction/Live/Hybrid) to control whether you get full equity backtest, future tc bubble forecasts (C1), or live trading sentiment snapshot.");
                return;
            }

            let res = match result {
                Some(r) => r,
                None => {
                    ui.label("(Pure prediction or live mode active — charts/equity below require a Historical or Hybrid run. See the prediction panel above for C1, risk level, median tc, and actionable sentiment.)");
                    return;
                }
            };
            let n = res.signals.len();
            let vs = self.view_start.min(n.saturating_sub(1));
            let vl = self.view_len.min(n.saturating_sub(vs)).max(1);

            // Toggles
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_price, "Price (regime colored)");
                ui.checkbox(&mut self.show_score, "Bubble Score");
                ui.checkbox(&mut self.show_equity, "$ Equity");
                ui.checkbox(&mut self.track_on_hover, "Track mouse hover (live cursor on price chart)");
                ui.separator();
                if ui.button("Fit view").clicked() {
                    self.view_start = 0;
                    self.view_len = n.min(600);
                }
                ui.add(egui::Slider::new(&mut self.view_len, 30..=n).text("view width"));
            });

            let (long_p, short_p, flat_p) = if self.show_price {
                build_regime_price_segments(&res.signals, vs, vl)
            } else {
                (vec![], vec![], vec![])
            };
            let score_pts = if self.show_score {
                build_score_series(&res.signals, vs, vl)
            } else {
                vec![]
            };
            let eq_pts = if self.show_equity {
                build_equity_series(res, vs, vl, self.engine.initial_capital)
            } else {
                vec![]
            };

            // Precompute date strings for the current view slice so we can show real calendar dates on the x-axis
            // (instead of abstract 0..500 trading-day indices). This makes "over time" obvious.
            let view_dates: Arc<Vec<String>> = Arc::new(
                (0..vl).map(|i| {
                    let gidx = vs + i;
                    res.signals
                        .get(gidx)
                        .map(|s| s.date.format("%Y-%m-%d").to_string())
                        .unwrap_or_else(|| i.to_string())
                })
                .collect()
            );

            let _x_max = vl as f64;

            // Helper to create a date formatter closure for the current view (used by all three plots)
            let make_date_formatter = || {
                let dates = Arc::clone(&view_dates);
                move |mark: GridMark, _range: &std::ops::RangeInclusive<f64>| -> String {
                    let x = mark.value;
                    let i = x.round() as usize;
                    dates.get(i).cloned().unwrap_or_else(|| format!("{:.0}", x))
                }
            };

            // === 1. PRICE CHART (regime-colored) ===
            ui.strong("1. Price (segments colored by the active strategy position that day — the visual 'bubble regime')");
            if self.show_price {
                let _price_resp = Plot::new("price_plot")
                    .height(170.0)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .allow_zoom(false)
                    .allow_boxed_zoom(false)
                    .x_axis_label("Date")
                    .y_axis_label("Adj. Close $")
                    .legend(egui_plot::Legend::default().position(egui_plot::Corner::RightTop))
                    .x_axis_formatter(make_date_formatter())
                    .show(ui, |plot_ui| {
                        if !long_p.is_empty() {
                            let pts: Vec<[f64; 2]> = long_p.into_iter().map(|(x, y)| [x, y]).collect();
                            plot_ui.line(Line::new(PlotPoints::from(pts)).name("Price (LONG regime)").color(egui::Color32::GREEN));
                        }
                        if !short_p.is_empty() {
                            let pts: Vec<[f64; 2]> = short_p.into_iter().map(|(x, y)| [x, y]).collect();
                            plot_ui.line(Line::new(PlotPoints::from(pts)).name("Price (SHORT regime)").color(egui::Color32::RED));
                        }
                        if !flat_p.is_empty() {
                            let pts: Vec<[f64; 2]> = flat_p.into_iter().map(|(x, y)| [x, y]).collect();
                            plot_ui.line(Line::new(PlotPoints::from(pts)).name("Price (FLAT)").color(egui::Color32::GRAY));
                        }
                        let cur_x = (self.selected_idx.saturating_sub(vs)) as f64;
                        plot_ui.vline(VLine::new(cur_x).color(egui::Color32::YELLOW).width(1.5).name("Cursor"));

                        // Live mouse tracker for cursor (if enabled): moving mouse over price chart updates the selected day in real-time for buy/sell/hold inspection
                        if self.track_on_hover {
                            if let Some(coord) = plot_ui.pointer_coordinate() {
                                let rel = (coord.x.round() as isize).max(0) as usize;
                                self.selected_idx = (vs + rel).min(vs + vl.saturating_sub(1));
                            }
                        } else {
                            // Fallback to click/drag only
                            if let Some(coord) = plot_ui.pointer_coordinate() {
                                if plot_ui.response().dragged() || plot_ui.response().clicked() {
                                    let rel = (coord.x.round() as isize).max(0) as usize;
                                    self.selected_idx = (vs + rel).min(vs + vl.saturating_sub(1));
                                }
                            }
                        }
                    });
            }

            // === 2. BUBBLE SCORE (the core indicator the strategy actually follows) ===
            ui.strong("2. Bubble Score (yellow) — recomputed only every 'refit every' days (hence the flat steps). Green/red lines = your Long / -Short thresholds.");
            if self.show_score {
                let lt = self.long_thresh;
                let st = self.short_thresh;

                Plot::new("score_plot")
                    .height(150.0)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .allow_zoom(false)
                    .allow_boxed_zoom(false)
                    .x_axis_label("Date")
                    .y_axis_label("Bubble Score")
                    .legend(egui_plot::Legend::default().position(egui_plot::Corner::RightTop))
                    .x_axis_formatter(make_date_formatter())
                    .show(ui, |plot_ui| {
                        if !score_pts.is_empty() {
                            let pts: Vec<[f64; 2]> = score_pts.into_iter().map(|(x, y)| [x, y]).collect();
                            plot_ui.line(Line::new(PlotPoints::from(pts)).name("Bubble Score (held const. between refits)").color(egui::Color32::YELLOW));
                        }
                        plot_ui.hline(HLine::new(lt).color(egui::Color32::LIGHT_GREEN).width(1.0).name("Long entry threshold"));
                        plot_ui.hline(HLine::new(-st).color(egui::Color32::LIGHT_RED).width(1.0).name("Short entry threshold"));

                        let cur_x = (self.selected_idx.saturating_sub(vs)) as f64;
                        plot_ui.vline(VLine::new(cur_x).color(egui::Color32::YELLOW).width(1.5).name("Cursor"));
                    });
            }

            // === 3. EQUITY CURVE (with Buy & Hold overlay for comparison) ===
            ui.strong("3. Portfolio Equity ($) — cyan = strategy (your rules + costs). Gray = Buy & Hold (same security, no timing). Use this to judge if the bubble signal added value.");
            if self.show_equity {
                // bh_pts use the same relative x (formatter will turn them into dates) and the bh_equity from the improved engine
                let bh_pts: Vec<(f64, f64)> = {
                    let bh = &res.bh_equity;
                    let v_end = vs + vl;
                    (vs..v_end)
                        .filter_map(|k| {
                            let x = (k - vs) as f64;
                            bh.get(k + 1).map(|&e| (x, e * self.engine.initial_capital))
                        })
                        .collect()
                };

                Plot::new("equity_plot")
                    .height(130.0)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .allow_zoom(false)
                    .allow_boxed_zoom(false)
                    .x_axis_label("Date")
                    .y_axis_label("Portfolio Value $")
                    .legend(egui_plot::Legend::default().position(egui_plot::Corner::RightTop))
                    .x_axis_formatter(make_date_formatter())
                    .show(ui, |plot_ui| {
                        if !eq_pts.is_empty() {
                            let pts: Vec<[f64; 2]> = eq_pts.into_iter().map(|(x, y)| [x, y]).collect();
                            plot_ui.line(Line::new(PlotPoints::from(pts)).name("$ Equity (strategy)").color(egui::Color32::from_rgb(0, 200, 255)));
                        }
                        if !bh_pts.is_empty() {
                            let pts: Vec<[f64; 2]> = bh_pts.into_iter().map(|(x, y)| [x, y]).collect();
                            plot_ui.line(
                                Line::new(PlotPoints::from(pts))
                                    .name("Buy & Hold (comparison)")
                                    .color(egui::Color32::from_rgb(170, 170, 170))
                                    .width(1.5),
                            );
                        }
                        let cur_x = (self.selected_idx.saturating_sub(vs)) as f64;
                        plot_ui.vline(VLine::new(cur_x).color(egui::Color32::YELLOW).width(1.5).name("Cursor"));
                    });
            }

            // Cursor / current reading + simple trade list
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Cursor (yellow line = inspected day; hover price chart or drag slider for live buy/sell/hold):").on_hover_text("The cursor (or live mouse tracker) picks a specific trading day within the date range. It shows the model's clear RECOMMENDATION: BUY (go long on bullish bubble signal), SELL (go short on bearish/overextended), or HOLD (neutral). Use it to get precise sentiment at any date in the simulation, not just end.");
                let max_idx = n.saturating_sub(1);
                ui.add(egui::Slider::new(&mut self.selected_idx, 0..=max_idx));

                if let Some(sig) = res.signals.get(self.selected_idx) {
                    let (rec_str, rec_color) = if sig.position > 0.5 {
                        ("BUY / GO LONG (bullish from positive bubble score)", egui::Color32::GREEN)
                    } else if sig.position < -0.5 {
                        ("SELL / GO SHORT (bearish / bubble risk from negative score)", egui::Color32::RED)
                    } else {
                        ("HOLD / NEUTRAL (flat, no strong signal)", egui::Color32::GRAY)
                    };
                    ui.colored_label(
                        rec_color,
                        format!("RECOMMENDATION AT {}: {}", sig.date, rec_str)
                    );
                    let pos_str = if sig.position > 0.5 { "LONG" } else if sig.position < -0.5 { "SHORT" } else { "FLAT" };
                    ui.colored_label(
                        if sig.position > 0.5 { egui::Color32::GREEN } else if sig.position < -0.5 { egui::Color32::RED } else { egui::Color32::GRAY },
                        format!("  (pos={}  score={:.3}  close=${:.2})", pos_str, sig.bubble_score, sig.close)
                    );
                    ui.label(format!("  components: eps={:.2} hype={:.2} sent={:.2}  {}", sig.eps_norm, sig.hype_volume, sig.sentiment, if sig.trade { "(TRADE day)" } else { "" }));
                }
            });

            // Trade log (last 8)
            ui.collapsing("Trade log (derived strictly from position changes + equity curve)", |ui| {
                egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                    for t in self.engine.trades.iter().rev().take(8) {
                        let sign = if t.pnl_usd >= 0.0 { "+" } else { "" };
                        ui.label(format!(
                            "{} → {}  {}  {} bars  ret {:+.1}%   PnL {}{:.0}$",
                            t.entry_date, t.exit_date, t.direction, t.bars_held, t.ret_pct * 100.0, sign, t.pnl_usd
                        ));
                    }
                    if self.engine.trades.len() > 8 {
                        ui.label("...");
                    }
                });
            });
        });

        // Bottom status
        egui::TopBottomPanel::bottom("bottom").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label("All computation goes through the shared HlpplEngine — the GUI is only presentation.");
                });
            });
        });

        // Help / explanation popup window (curated details, legends, cursor explanation, controls)
        if self.show_help {
            egui::Window::new("HLPPL Backtesting Explorer — Help, Legends & Explanations")
                .open(&mut self.show_help)
                .default_width(620.0)
                .default_height(520.0)
                .resizable(true)
                .scroll(true)
                .show(ctx, |ui| {
                    ui.heading("Quick Start");
                    ui.label("1. Set Ticker + Start/End dates (or use Quick preset buttons).");
                    ui.label("2. Tweak Window (LPPL fit lookback), Refit every, Long/Short thresholds, Cost (bps), Initial Capital.");
                    ui.label("3. Click 'Test Yahoo API' to validate data availability for your security/time (shows bar count, last price).");
                    ui.label("4. Click 'Run Simulation' (or the top button) — this runs the exact same engine as the CLI/TUI: full walk-forward LPPL fits + bubble score + strict long/short/flat rules + costs. The 'RNG seed' controls reproducibility of the LPPL random search (same seed + params = identical results every run).");
                    ui.label("5. Use the 'view width' slider + 'Fit view' to zoom/pan the history. Drag the cursor slider (or click+drag on the top Price chart) to inspect any day.");

                    ui.separator();
                    ui.heading("Left Panel Controls (detailed)");
                    ui.monospace("Ticker / Start / End: The security and calendar range. Data is fetched from Yahoo Finance public chart API (no key). Changing these requires re-Test or re-Run.");
                    ui.monospace("Window (days): LPPL lookback length in trading days. Each refit fits the 7-param LPPL model on the prior N bars. Larger = more stable but lags more.");
                    ui.monospace("Refit every: Recompute the (expensive) LPPL + score only every N days. On non-refit days the bubble_score, eps_norm, hype, sentiment are held constant → you see flat steps in the middle chart.");
                    ui.monospace("Long threshold / Short threshold: Position = +1 (long) if bubble_score > Long thresh; -1 (short) if bubble_score < -Short thresh; else 0 (flat).");
                    ui.monospace("Cost (bps): One-way transaction cost in basis points (0.01% = 1 bps) subtracted from the daily strategy return on any day the position changes.");
                    ui.monospace("Initial Capital $: Starting portfolio value for the simulation. All equity numbers and per-trade PnL are scaled to this. Default 10 000.");

                    ui.separator();
                    ui.heading("Charts & Legends (what you are looking at)");
                    ui.strong("Top chart — Price (regime colored)");
                    ui.label("Line segments of the adjusted close price. Color = the strategy position that was active on that day (after the signal was computed):");
                    ui.colored_label(egui::Color32::GREEN, "GREEN segments: strategy was LONG (+1) that day");
                    ui.colored_label(egui::Color32::RED, "RED segments: strategy was SHORT (-1)");
                    ui.colored_label(egui::Color32::GRAY, "GRAY / white: strategy was FLAT (0)");
                    ui.label("This is the visual 'bubble regime' indicator — when the model sees a strong positive bubble reading it goes long (green price line).");

                    ui.strong("Middle chart — Bubble Score (the core indicator) + thresholds");
                    ui.label("The yellow steppy line is the bubble_score. Because scores are only recomputed on refit days, you see flat horizontal steps between refits (this is by design and visible in all outputs).");
                    ui.label("Green horizontal line ≈ your Long threshold. Red ≈ -Short threshold.");
                    ui.label("When the yellow line is above the green line the strategy holds LONG. Below the red line → SHORT. Between → FLAT.");
                    ui.label("The score itself is eps_norm (LPPL residual normalized) + 0.7*hype_volume (volume attention proxy) + 0.3*sentiment (return proxy), with the piecewise sign flip when eps_norm < 0.");

                    ui.strong("Bottom chart — Equity ($)");
                    ui.label("The portfolio value over time when you start with your chosen Initial Capital and strictly follow every position change (including cost drag on trade days).");
                    ui.label("Compare the final value and the shape vs what a simple buy-and-hold of the same security would have done.");

                    ui.separator();
                    ui.heading("Cursor (yellow vertical line + bottom slider + details line)");
                    ui.label("The yellow line marks one specific trading day in the current view window. The slider lets you scrub through the entire backtest history.");
                    ui.label("The text line below the slider shows the exact values the engine produced for that day:");
                    ui.monospace("date  POSITION  bubble_score  close price  |  eps_norm  hype_volume  sent  (TRADE if position changed that day)");
                    ui.label("Use the cursor to answer 'why was the strategy long/short/flat on this day?' or 'what did the LPPL residual + hype look like right before the big move?'. Clicking the top price chart also moves the cursor (nice for visual inspection).");

                    ui.separator();
                    ui.heading("Other UI elements");
                    ui.label("'Fit view' + 'view width' slider: Control how much history is shown in the three charts (independent of the cursor). 'Fit view' shows the whole simulation.");
                    ui.label("The summary bar at the very top gives the headline performance numbers for the whole run (using your capital).");
                    ui.label("Trade log: Expand to see every round-trip or leg the strategy actually took, with realized $ PnL on your capital (computed from the equity curve segments between position changes).");

                    ui.separator();
                    ui.heading("The Random Seed in LPPL (why randomness & reproducibility)");
                    ui.label("The LPPL equation ln(p) = A + B*tau^m + C*tau^m * cos(omega*ln(tau)+phi) has 4 nonlinear params (tc,m,omega,phi) that are difficult to fit analytically (non-convex loss surface with many local minima).");
                    ui.label("We use random multi-start search: sample 1200 random (tc,m,omega,phi) in bounds using the RNG seeded by 'random_seed', for each solve linear A/B/C via OLS, pick best valid (m in (0,1), omega range, damping condition).");
                    ui.label("Randomness is needed for global-ish exploration without heavy solvers. The seed makes it 100% reproducible: same seed+data => identical 'random' samples => same best fit => same bubble_score => same positions => same backtest results.");
                    ui.label("Trust for backtesting? With fixed seed, yes for apples-to-apples comparison of params/strategies (reproducible 'what-if'). Different seeds give slightly different models (robustness check). But it's still an approx model + proxies; not crystal ball. For live 'invest now?', treat as research signal only, not sole decider. Always combine with fundamentals, risk mgmt, multiple indicators. Past != future. This is NOT financial advice.");

                    ui.separator();
                    ui.heading("Strategy in one sentence (strictly followed)");
                    ui.label("On every day after the warm-up window: fit (or reuse) LPPL on the prior Window bars → compute eps_norm + volume hype + return sentiment → bubble_score → if score > long_thresh go long, if < -short_thresh go short, else flat. Apply asset return × position, minus cost if you changed position that day. Compound the equity. That's it. No extra rules.");

                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("All numbers and decisions come from the shared HlpplEngine in the library — the GUI only displays and lets you tweak inputs.").italics());
                });
        }
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 780.0])
            .with_title("HLPPL Backtesting Explorer (Native GUI)"),
        ..Default::default()
    };

    eframe::run_native(
        "HLPPL Explorer",
        native_options,
        Box::new(|cc| Ok(Box::new(HlpplGuiApp::new(cc)))),
    )
}
