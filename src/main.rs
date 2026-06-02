use chrono::NaiveDate;
use clap::Parser;
use env_logger::Env;
use log::info;
use std::error::Error;

use hlpll_backtester::modules::backtest::{run_backtest, BacktestConfig};
use hlpll_backtester::modules::data::fetch_yahoo_history;
use hlpll_backtester::modules::utils::{plot_equity_curve, print_summary, save_series_csv, save_signals_csv};

#[derive(Parser, Debug)]
#[command(author, version, about = "HLPLL / HLPPL Bubble Backtester (Rust)")]
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

    let cfg = BacktestConfig {
        lookback_days: args.window,
        refit_every: args.refit_every,
        long_threshold: args.long_thresh,
        short_threshold: args.short_thresh,
        cost_bps: args.cost_bps as f64,
        max_position: 1.0,
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
            "Running backtest for {} (window={}, refit_every={}) ...",
            ticker, cfg.lookback_days, cfg.refit_every
        );
        match run_backtest(ticker, &bars, &cfg) {
            Ok(res) => {
                print_summary(&res);

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

                // Save detailed signals for diagnostics
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

    info!("All done. Results in {}", args.outdir);
    Ok(())
}


