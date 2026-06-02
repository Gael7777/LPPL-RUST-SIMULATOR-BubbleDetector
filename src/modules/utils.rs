use chrono::NaiveDate;
use polars::prelude::{Column, CsvWriter, DataFrame, SerWriter};
use std::error::Error;

use crate::modules::backtest::BacktestResult;

/// Save a vector of (date, value) as CSV via Polars for convenience.
pub fn save_series_csv(
    path: &str,
    dates: &[NaiveDate],
    values: &[f64],
    col_name: &str,
) -> Result<(), Box<dyn Error>> {
    let epochs: Vec<i32> = dates
        .iter()
        .map(|d| (*d - NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days() as i32)
        .collect();

    let df = DataFrame::new(vec![
        Column::new("date_epoch".into(), epochs),
        Column::new(col_name.into(), values.to_vec()),
    ])?;
    let mut file = std::fs::File::create(path)?;
    CsvWriter::new(&mut file).finish(&mut df.clone())?;
    Ok(())
}

/// Save detailed daily signals for easier diagnostics/analysis.
pub fn save_signals_csv(
    path: &str,
    signals: &[crate::modules::backtest::DailySignal],
) -> Result<(), Box<dyn Error>> {
    let epoch0 = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();

    let date_str: Vec<String> = signals.iter().map(|s| s.date.to_string()).collect();
    let date_epoch: Vec<i32> = signals
        .iter()
        .map(|s| (s.date - epoch0).num_days() as i32)
        .collect();
    let close: Vec<f64> = signals.iter().map(|s| s.close).collect();
    let daily_return: Vec<f64> = signals.iter().map(|s| s.daily_return).collect();
    let volume: Vec<f64> = signals.iter().map(|s| s.volume).collect();
    let eps_norm: Vec<f64> = signals.iter().map(|s| s.eps_norm).collect();
    let hype_volume: Vec<f64> = signals.iter().map(|s| s.hype_volume).collect();
    let sentiment: Vec<f64> = signals.iter().map(|s| s.sentiment).collect();
    let bubble_score: Vec<f64> = signals.iter().map(|s| s.bubble_score).collect();
    let position: Vec<f64> = signals.iter().map(|s| s.position).collect();
    let trade: Vec<i32> = signals.iter().map(|s| if s.trade { 1 } else { 0 }).collect();

    let mut df = DataFrame::new(vec![
        Column::new("date".into(), date_str),
        Column::new("date_epoch".into(), date_epoch),
        Column::new("close".into(), close),
        Column::new("daily_return".into(), daily_return),
        Column::new("volume".into(), volume),
        Column::new("eps_norm".into(), eps_norm),
        Column::new("hype_volume".into(), hype_volume),
        Column::new("sentiment".into(), sentiment),
        Column::new("bubble_score".into(), bubble_score),
        Column::new("position".into(), position),
        Column::new("trade".into(), trade),
    ])?;
    let mut file = std::fs::File::create(path)?;
    CsvWriter::new(&mut file).finish(&mut df)?;
    Ok(())
}

/// Basic print for backtest summary
pub fn print_summary(result: &crate::modules::backtest::BacktestResult) {
    println!("\n=== Backtest Summary: {} ===", result.ticker);
    println!("Period: {} -> {} ({} days)", result.start_date, result.end_date, result.n_days);
    println!("Strategy Total Return : {:.2}%", result.total_return * 100.0);
    println!("Strategy Ann. Return  : {:.2}%", result.annualized_return * 100.0);
    println!("Strategy Sharpe       : {:.3}", result.sharpe);
    println!("Strategy Max DD       : {:.2}%", result.max_drawdown * 100.0);
    println!("Num Trades            : {}", result.num_trades);
    println!("Buy & Hold Return     : {:.2}%", result.buy_hold_return * 100.0);
    println!("Buy & Hold Sharpe     : {:.3}", result.buy_hold_sharpe);
    if let Some(last) = result.signals.last() {
        let rec = if last.position > 0.5 {
            "BUY / GO LONG (bullish)"
        } else if last.position < -0.5 {
            "SELL / GO SHORT (bearish/risk)"
        } else {
            "HOLD / NEUTRAL (flat)"
        };
        println!("Final recommendation at {}: {}", last.date, rec);
    }
}

/// Simple equity curve plot using plotters (used by CLI and explorer export).
pub fn plot_equity_curve(
    dates: &[NaiveDate],
    equity: &[f64],
    out_path: &str,
    ticker: &str,
) -> Result<(), Box<dyn Error>> {
    use plotters::prelude::*;

    if dates.len() < 2 {
        return Ok(());
    }

    let root = BitMapBackend::new(out_path, (960, 540)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_eq = equity.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_eq = equity.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_range = (min_eq * 0.95)..(max_eq * 1.05);

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("{} Equity Curve (HLPLL Strategy)", ticker), ("sans-serif", 28))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(dates[0]..*dates.last().unwrap(), y_range)?;

    chart.configure_mesh().draw()?;

    let series_data: Vec<(NaiveDate, f64)> =
        dates.iter().zip(equity.iter()).map(|(d, e)| (*d, *e)).collect();

    chart
        .draw_series(LineSeries::new(series_data, &BLUE))
        .unwrap()
        .label("Strategy Equity")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Export both normalized and dollar-scaled equity + calls the simple plot.
/// Also saves a full signals CSV. This is used by the interactive explorer for "full user control" exports.
pub fn export_backtest_artifacts(
    result: &BacktestResult,
    outdir: &str,
    initial_capital: f64,
) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(outdir)?;

    let ticker = &result.ticker;

    // Signals (detailed)
    let sig_path = format!("{}/{}_signals.csv", outdir, ticker);
    save_signals_csv(&sig_path, &result.signals)?;

    // Equity normalized (for compatibility)
    let eq_dates: Vec<_> = result.signals.iter().map(|s| s.date).collect();
    let eq_norm: Vec<f64> = if result.equity.len() == result.signals.len() + 1 {
        result.equity[1..].to_vec()
    } else {
        result.equity.clone()
    };
    let eq_path = format!("{}/{}_equity.csv", outdir, ticker);
    save_series_csv(&eq_path, &eq_dates, &eq_norm, "equity")?;

    // Dollar equity for the 10k (or custom) simulation
    let eq_dollar: Vec<f64> = eq_norm.iter().map(|e| e * initial_capital).collect();
    let dol_path = format!("{}/{}_equity_{:.0}k.csv", outdir, ticker, initial_capital / 1000.0);
    save_series_csv(&dol_path, &eq_dates, &eq_dollar, "equity_usd")?;

    // PNG equity
    let png_path = format!("{}/{}_equity.png", outdir, ticker);
    plot_equity_curve(&eq_dates, &eq_norm, &png_path, ticker)?;

    Ok(())
}

// ============================================================================
// Shared visualization helpers (used by TUI explorer and native GUI explorer)
// These turn DailySignal slices into simple (f64, f64) series for plotting libs.
// ============================================================================

use crate::modules::backtest::DailySignal;

/// Build three separate series for price, split by the strategy position at each bar.
/// This lets UIs draw the price line in different colors for LONG / SHORT / FLAT
/// regimes — a very effective "bubble indicator" visualization.
pub fn build_regime_price_segments(
    signals: &[DailySignal],
    start: usize,
    len: usize,
) -> (Vec<(f64, f64)>, Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let mut longs = vec![];
    let mut shorts = vec![];
    let mut flats = vec![];

    let end = (start + len).min(signals.len());
    let mut prev_pos = 0.0;
    let mut curr: &mut Vec<(f64, f64)> = &mut flats;

    for (i, sig) in signals[start..end].iter().enumerate() {
        let x = i as f64;
        let y = sig.close;
        let p = sig.position;

        if (p - prev_pos).abs() > 0.1 {
            if p > 0.5 {
                curr = &mut longs;
            } else if p < -0.5 {
                curr = &mut shorts;
            } else {
                curr = &mut flats;
            }
            prev_pos = p;
        }
        curr.push((x, y));
    }
    (longs, shorts, flats)
}

/// Simple (relative_x, bubble_score) series for the current view window.
pub fn build_score_series(signals: &[DailySignal], start: usize, len: usize) -> Vec<(f64, f64)> {
    signals[start..(start + len).min(signals.len())]
        .iter()
        .enumerate()
        .map(|(i, s)| (i as f64, s.bubble_score))
        .collect()
}

/// (relative_x, equity_in_usd) series for the current view, scaled to a capital base.
pub fn build_equity_series(
    res: &BacktestResult,
    start: usize,
    len: usize,
    capital: f64,
) -> Vec<(f64, f64)> {
    let sigs = &res.signals;
    let eq = &res.equity;
    let end = (start + len).min(sigs.len());
    (start..end)
        .map(|k| {
            let x = (k - start) as f64;
            let usd = eq.get(k + 1).copied().unwrap_or(1.0) * capital;
            (x, usd)
        })
        .collect()
}
