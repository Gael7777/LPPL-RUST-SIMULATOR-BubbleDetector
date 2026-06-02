use chrono::NaiveDate;
use clap::Parser;
use env_logger::Env;
use log::info;
use std::error::Error;

use hlpll_backtester::modules::backtest::{run_backtest, BacktestConfig, RunMode, PositionBias, run_future_bubble_prediction, compute_live_sentiment};
use hlpll_backtester::modules::data::fetch_yahoo_history;
use hlpll_backtester::modules::utils::{plot_equity_curve, print_summary, save_series_csv, save_signals_csv};

#[derive(Parser, Debug)]
#[command(author, version, about = "HLPLL / HLPPL Bubble Backtester (Rust) - extensive: historical backtests, future bubble prediction (C1 + tc), live sentiment trading signals")]
struct Args {
    /// Comma-separated tickers, e.g. "HOUS,AMTX,CAR,CSGP"
    #[arg(long, default_value = "HOUS,AMTX,CAR")]
    tickers: String,

    /// Start date YYYY-MM-DD
    #[arg(long, default_value = "2018-01-01")]
    start: String,

    /// End date YYYY-MM-DD
    #[arg(long, default_value = "2024-12-31")]
    end: String,

    /// LPPL lookback window (trading days)
    #[arg(long, default_value_t = 300)]
    window: usize,

    /// Refit LPPL every N days
    #[arg(long, default_value_t = 25)]
    refit_every: usize,

    /// Long threshold for bubble score
    #[arg(long, default_value_t = 0.75)]
    long_thresh: f64,

    /// Short threshold for bubble score
    #[arg(long, default_value_t = 0.75)]
    short_thresh: f64,

    /// Transaction cost in basis points (one way)
    #[arg(long, default_value_t = 10.0)]
    cost_bps: f64,

    /// Output directory for CSVs and plots
    #[arg(long, default_value = "results")]
    outdir: String,

    /// Random seed for LPPL fitting (fixed seed makes runs reproducible)
    #[arg(long, default_value_t = 42)]
    random_seed: u64,

    /// Optional: print buy/sell/hold recommendation at this specific date (YYYY-MM-DD, must be in range)
    #[arg(long)]
    query_date: Option<String>,

    // === EXTENSIVE NEW OPTIONS (from gemini LPPLS doc + full feature set) ===
    /// Run mode: historical (default, full sim+equity), prediction (future tc + C1 only), live (current sentiment snapshot), hybrid
    #[arg(long, default_value = "historical", value_parser = ["historical", "prediction", "live", "hybrid"])]
    mode: String,

    /// Position bias: longshort | longonly | shortonly (for momentum vs pure bubble-risk use)
    #[arg(long, default_value = "longshort", value_parser = ["longshort", "longonly", "shortonly"])]
    position_bias: String,

    /// Invert signal: high positive score = danger (short/flat). Often aligns better with bubble warning literature.
    #[arg(long, default_value_t = false)]
    invert: bool,

    /// Enable multi-window LPPLS bubble confidence (C1 %) + tc prediction in outputs
    #[arg(long, default_value_t = true)]
    enable_bubble_analysis: bool,

    /// Analysis window sweep min/max/step (trading days) for C1 multi-window
    #[arg(long, default_value_t = 60)]
    analysis_min: usize,
    #[arg(long, default_value_t = 260)]
    analysis_max: usize,
    #[arg(long, default_value_t = 5)]
    analysis_step: usize,

    /// Strict JLS filter params for valid bubble fits (C1). See gemini-data-LPPLS.md
    #[arg(long, default_value_t = 0.1)]
    filter_m_min: f64,
    #[arg(long, default_value_t = 0.9)]
    filter_m_max: f64,
    #[arg(long, default_value_t = 4.5)]
    filter_omega_min: f64,
    #[arg(long, default_value_t = 13.0)]
    filter_omega_max: f64,
    #[arg(long, default_value_t = true)]
    filter_b_negative: bool,
    #[arg(long, default_value_t = 3)]
    filter_tc_offset: usize,

    /// If C1 > this, force flat in trading (risk mgmt for live/historical)
    #[arg(long, default_value_t = 50.0)]
    conf_flat_thresh: f64,
    #[arg(long, default_value_t = false)]
    use_conf_flat: bool,

    /// Use C1 to scale position size proportionally (experimental, 0.2-1.0 clamp)
    #[arg(long, default_value_t = false)]
    use_conf_sizing: bool,

    /// Ensemble seeds for robust C1 (comma sep, e.g. "42,43,44"). Empty = single random_seed.
    #[arg(long, default_value = "")]
    ensemble_seeds: String,

    /// Horizon in days for "prob tc within horizon" in prediction reports
    #[arg(long, default_value_t = 60)]
    predict_horizon: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    let tickers: Vec<&str> = args
        .tickers
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let start = NaiveDate::parse_from_str(&args.start, "%Y-%m-%d")?;
    let end = NaiveDate::parse_from_str(&args.end, "%Y-%m-%d")?;

    std::fs::create_dir_all(&args.outdir)?;

    // Map CLI strings to enums
    let run_mode = match args.mode.as_str() {
        "prediction" => RunMode::FutureBubblePrediction,
        "live" => RunMode::LiveCurrentSentiment,
        "hybrid" => RunMode::HybridAnalysis,
        _ => RunMode::HistoricalBacktest,
    };
    let pos_bias = match args.position_bias.as_str() {
        "longonly" => PositionBias::LongOnly,
        "shortonly" => PositionBias::ShortOnly,
        _ => PositionBias::LongShort,
    };
    let ensemble: Vec<u64> = if args.ensemble_seeds.trim().is_empty() {
        vec![]
    } else {
        args.ensemble_seeds.split(',').filter_map(|s| s.trim().parse::<u64>().ok()).collect()
    };

    let cfg = BacktestConfig {
        lookback_days: args.window,
        refit_every: args.refit_every,
        long_threshold: args.long_thresh,
        short_threshold: args.short_thresh,
        cost_bps: args.cost_bps as f64,
        max_position: 1.0,
        random_seed: args.random_seed,
        position_bias: pos_bias,
        invert_signal: args.invert,
        enable_bubble_analysis: args.enable_bubble_analysis,
        analysis_lookback_min: args.analysis_min,
        analysis_lookback_max: args.analysis_max,
        analysis_step_days: args.analysis_step,
        filter_m_min: args.filter_m_min,
        filter_m_max: args.filter_m_max,
        filter_omega_min: args.filter_omega_min,
        filter_omega_max: args.filter_omega_max,
        filter_require_b_negative: args.filter_b_negative,
        filter_min_tc_offset_days: args.filter_tc_offset,
        use_confidence_for_flat: args.use_conf_flat,
        confidence_flat_threshold: args.conf_flat_thresh,
        use_confidence_for_sizing: args.use_conf_sizing,
        run_mode,
        ensemble_seeds: ensemble,
        predict_horizon_days: args.predict_horizon,
        ..Default::default()
    };

    info!(
        "Starting HLPLL backtester for {} tickers | {} -> {}",
        tickers.len(),
        start,
        end
    );

    for ticker in &tickers {
        info!("Fetching data for {} ...", ticker);
        let bars = match fetch_yahoo_history(ticker, start, end) {
            Ok(b) => {
                info!("  {} bars loaded for {}", b.len(), ticker);
                b
            }
            Err(e) => {
                eprintln!("Failed to fetch {}: {}. Skipping.", ticker, e);
                continue;
            }
        };

        info!(
            "Running {} for {} (mode={:?}, window={}, refit_every={}, seed={}, ensemble={:?}) ...",
            if cfg.run_mode == RunMode::HistoricalBacktest { "backtest" } else { "analysis" },
            ticker, cfg.run_mode, cfg.lookback_days, cfg.refit_every, cfg.random_seed, cfg.ensemble_seeds
        );

        // Mode dispatch: use dedicated extensive paths when not pure historical
        match cfg.run_mode {
            RunMode::FutureBubblePrediction | RunMode::HybridAnalysis => {
                match run_future_bubble_prediction(ticker, &bars, &cfg) {
                    Ok((pred, maybe_res)) => {
                        println!("\n=== FUTURE BUBBLE PREDICTION (C1 + tc from multi-window JLS/LPPLS) ===");
                        println!("Ticker: {} | Analysis date: {}", pred.ticker, pred.analysis_date);
                        println!("Current price: {:.2}", pred.current_price);
                        println!("Windows tested: {} | Valid strict fits: {}", pred.total_windows_tested, pred.valid_fits);
                        println!("BUBBLE CONFIDENCE INDEX (mean over ensemble): {:.2}%  (std {:.2})", pred.bubble_confidence_index, pred.ensemble_std_confidence);
                        println!("Risk level: {} - {}", pred.risk_level, pred.risk_description);
                        if let Some(med) = pred.median_predicted_date {
                            println!("Median predicted critical/peak date: {} (approx {} days ahead)", med, pred.median_days_to_tc.unwrap_or(0));
                        }
                        println!("Prob tc within {}d horizon: {:.1}%", cfg.predict_horizon_days, pred.prob_tc_within_horizon * 100.0);
                        println!("Ensemble seeds: {:?}", pred.ensemble_seeds_used);
                        if !pred.predicted_critical_dates.is_empty() {
                            println!("Sample predicted tcs: {:?}", &pred.predicted_critical_dates.iter().take(5).collect::<Vec<_>>());
                        }
                        println!("(See gemini-data-LPPLS.md for interpretation of C1 and filters)");

                        // Also save a prediction summary csv
                        let _pred_path = format!("{}/{}_prediction.csv", args.outdir, ticker);
                        // simple: reuse series or manual - for now log + signals if hybrid
                        if let Some(r) = &maybe_res {
                            print_summary(r);
                            let sig_dates: Vec<_> = r.signals.iter().map(|s| s.date).collect();
                            let _ = save_signals_csv(&format!("{}/{}_signals.csv", args.outdir, ticker), &r.signals);
                            if let Err(e) = save_series_csv(&format!("{}/{}_equity.csv", args.outdir, ticker), &sig_dates, &r.equity[1..].to_vec(), "equity") {
                                eprintln!("equity save: {}", e);
                            }
                        }
                        // minimal prediction export note
                        info!("Prediction report printed. Hybrid also saved signals/equity if applicable.");
                    }
                    Err(e) => eprintln!("Future prediction failed for {}: {}", ticker, e),
                }
            }
            RunMode::LiveCurrentSentiment => {
                match compute_live_sentiment(ticker, &bars, &cfg) {
                    Ok(snap) => {
                        println!("\n=== LIVE CURRENT SENTIMENT (for trading now, using LPPLS + C1) ===");
                        println!("Ticker: {} | As of: {} | Price: {:.2}", snap.ticker, snap.date, snap.current_price);
                        println!("Bubble score: {:.3} | C1 confidence: {:.1}% | Risk: {}", snap.bubble_score, snap.bubble_confidence, snap.risk_level);
                        println!("RECOMMENDATION: {}", snap.recommendation);
                        println!("Actionable: {}", snap.actionable_note);
                        if let Some(pk) = snap.median_predicted_peak {
                            println!("Median predicted peak (from C1): {} (~{}d)", pk, snap.median_days_to_tc.unwrap_or(0));
                        }
                        println!("(Use for live signals; combine with other risk tools. Not advice.)");
                    }
                    Err(e) => eprintln!("Live sentiment failed for {}: {}", ticker, e),
                }
            }
            RunMode::HistoricalBacktest => {
                match run_backtest(ticker, &bars, &cfg) {
                    Ok(res) => {
                        print_summary(&res);

                        // Query specific date recommendation if provided
                        if let Some(qstr) = &args.query_date {
                            if let Ok(qdate) = NaiveDate::parse_from_str(qstr, "%Y-%m-%d") {
                                if let Some(sig) = res.signals.iter().find(|s| s.date == qdate) {
                                    let rec = if sig.position > 0.5 {
                                        "BUY / GO LONG (bullish signal from bubble score)"
                                    } else if sig.position < -0.5 {
                                        "SELL / GO SHORT (bearish / overextension signal)"
                                    } else {
                                        "HOLD / FLAT (no strong directional signal)"
                                    };
                                    println!("Recommendation at {}: {} (bubble_score={:.3}, position={}, C1={:.1}%)", qdate, rec, sig.bubble_score, sig.position, sig.bubble_confidence);
                                } else {
                                    println!("Date {} not in backtest signals range for {}.", qdate, ticker);
                                }
                            } else {
                                eprintln!("Invalid --query-date format, use YYYY-MM-DD");
                            }
                        }

                        // Save equity curve
                        let equity_path = format!("{}/{}_equity.csv", args.outdir, ticker);
                        let sig_dates: Vec<_> = res.signals.iter().map(|s| s.date).collect();
                        let eq_for_save = if res.equity.len() == res.signals.len() + 1 {
                            res.equity[1..].to_vec()
                        } else {
                            res.equity.clone()
                        };
                        if let Err(e) = save_series_csv(&equity_path, &sig_dates, &eq_for_save, "equity") {
                            eprintln!("Failed to save equity: {}", e);
                        } else {
                            info!("Saved {}", equity_path);
                        }

                        // Save detailed signals for diagnostics (now includes bubble_confidence)
                        let score_path = format!("{}/{}_signals.csv", args.outdir, ticker);
                        if let Err(e) = save_signals_csv(&score_path, &res.signals) {
                            eprintln!("Failed to save signals: {}", e);
                        } else {
                            info!("Saved {}", score_path);
                        }

                        // Plot equity curve
                        let plot_path = format!("{}/{}_equity.png", args.outdir, ticker);
                        if let Err(e) = plot_equity_curve(&sig_dates, &eq_for_save, &plot_path, ticker) {
                            eprintln!("Plot failed for {}: {}", ticker, e);
                        } else {
                            info!("Saved plot {}", plot_path);
                        }
                    }
                    Err(e) => {
                        eprintln!("Backtest failed for {}: {}", ticker, e);
                    }
                }
            }
        }
    }

    info!("All done. Results (and predictions if mode) in {}", args.outdir);
    Ok(())
}


