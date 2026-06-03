//! HLPLL / HLPPL Bubble Backtester Library
//!
//! **Isolated logic engine** for the Hyped Log-Periodic Power Law (HLPPL/HLPLL) bubble
//! detection + walk-forward backtesting strategy.
//!
//! Extensive support (2026 updates from gemini LPPLS doc + literature):
//! - RunMode: HistoricalBacktest (classic equity sim), FutureBubblePrediction (C1 % + tc forecasts for "will bubble peak?"),
//!   LiveCurrentSentiment (current snapshot for live trading signals), Hybrid.
//! - Multi-window LPPLS Confidence (C1) with strict JLS filters (m 0.1-0.9, omega 4.5-13, B<0, tc future offset) + ensemble seeds for robustness.
//! - Standalone prediction + live fns, per-day C1 in signals, risk levels, median tc, prob-within-horizon.
//! - PositionBias + invert + conf flat/sizing filters work across modes.
//! Use HlpplEngine + run_with_mode / get_future_prediction / get_live_sentiment for all frontends and library consumers.
//!
//! ## For library / engine-only users (recommended for isolation)
//! ```ignore
//! use hlpll_backtester::{HlpplEngine, BacktestConfig, fetch_yahoo_history};
//! use chrono::NaiveDate;
//!
//! let mut engine = HlpplEngine::new(
//!     "CAR",
//!     NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(),
//!     NaiveDate::from_ymd_opt(2024, 12, 31).unwrap(),
//!     BacktestConfig::default(),
//!     10_000.0,
//! );
//! engine.fetch()?;
//! engine.run()?;
//! println!("Final $: {:.0}", engine.final_capital());
//! ```
//!
//! The low-level pieces (`fetch_yahoo_history`, `run_backtest`, LPPL fitter, etc.)
//! are also re-exported if you need full control.
//!
//! Frontends (CLI, TUI explorer, native GUI) live in `src/bin/` and only depend on
//! this engine + their own UI crates (feature-gated).

pub mod engine;

// The internal implementation modules.
// External users are encouraged to use the clean re-exports at the crate root
// and the high-level `HlpplEngine`. The `modules` path is public mainly so the
// various front-end binaries (CLI / TUI / GUI) can share the implementation
// without duplication. When using this crate purely as a library, prefer the
// top-level symbols and `HlpplEngine`.
pub mod modules;

// === Clean public API surface for the isolated HLPPL logic engine ===

pub use engine::{CurrentSentiment, HlpplEngine, Trade};

pub use modules::backtest::{BacktestConfig, BacktestResult, DailySignal, PositionBias, RunMode, run_backtest, FutureBubblePrediction, LiveSentimentSnapshot, run_future_bubble_prediction, compute_live_sentiment};
pub use modules::data::{bars_to_dataframe, fetch_yahoo_history, load_prices_parquet, PriceBar};
pub use modules::lppl::{
    compute_bubble_confidence, compute_bubble_confidence_ensemble, is_strict_jls_valid, fit_lppl, fit_lppl_on_bars,
    BubbleAnalysisResult, LpplFilterConfig, LpplFit, LpplFitConfig, LpplParams,
};
pub use modules::backtest::SignalMode;
pub use modules::hlppl_signals::{build_signal_series, HlpplSignalSeries};
pub use modules::bubble_score::{compute_bubble_score, normalize_running_max, normalize_last_residual_running_style};
pub use modules::utils::{
    build_equity_series, build_regime_price_segments, build_score_series, export_backtest_artifacts,
    plot_equity_curve, print_summary, save_series_csv, save_signals_csv,
};

pub use modules::hype::compute_volume_hype;
pub use modules::sentiment::compute_simple_sentiment;
