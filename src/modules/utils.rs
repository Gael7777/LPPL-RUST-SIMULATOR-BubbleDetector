use chrono::NaiveDate;
use polars::prelude::{Column, CsvWriter, DataFrame, SerWriter};
use std::error::Error;

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
}
