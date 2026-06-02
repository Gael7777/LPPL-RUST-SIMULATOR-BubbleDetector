use chrono::{NaiveDate, TimeZone, Utc};
use polars::prelude::{col, lit, Column, DataFrame, LazyFrame, NamedFrom, ScanArgsParquet, Series};
use reqwest::blocking::Client;
use serde::Deserialize;
use std::error::Error;
use std::time::Duration;

/// Price bar data structure (preferred for LPPL fitting - no Polars overhead)
#[derive(Debug, Clone)]
pub struct PriceBar {
    pub date: NaiveDate,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub adj_close: f64,
    pub volume: f64,
}

/// Fetch daily OHLCV from Yahoo Finance public API (no API key).
/// Uses the v8 chart endpoint. Returns sorted ascending by date.
pub fn fetch_yahoo_history(
    ticker: &str,
    start: NaiveDate,
    end: NaiveDate,
) -> Result<Vec<PriceBar>, Box<dyn Error>> {
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent("Mozilla/5.0 (compatible; HLPLL-Backtester/0.1)")
        .build()?;

    let period1 = start
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp();
    let period2 = end
        .and_hms_opt(23, 59, 59)
        .unwrap()
        .and_utc()
        .timestamp();

    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval=1d&indicators=quote&includeAdjustedClose=true",
        ticker, period1, period2
    );

    let resp = client.get(&url).send()?;
    if !resp.status().is_success() {
        return Err(format!("Yahoo API error for {}: HTTP {}", ticker, resp.status()).into());
    }

    let json: YahooChartResponse = resp.json()?;

    let result = json
        .chart
        .result
        .into_iter()
        .next()
        .ok_or("No result in Yahoo response")?;

    let timestamps = result.timestamp.unwrap_or_default();
    let quote = result.indicators.quote.into_iter().next().ok_or("No quote data")?;
    let adjclose = result
        .indicators
        .adjclose
        .and_then(|a| a.into_iter().next())
        .map(|a| a.adjclose)
        .unwrap_or_default();

    let mut bars = Vec::with_capacity(timestamps.len());

    for (i, &ts) in timestamps.iter().enumerate() {
        // Yahoo returns unix seconds; convert safely
        let date = match Utc.timestamp_opt(ts, 0) {
            chrono::LocalResult::Single(dt) => dt.date_naive(),
            _ => continue,
        };

        if date < start || date > end {
            continue;
        }

        let open = quote.open.get(i).copied().flatten().unwrap_or(f64::NAN);
        let high = quote.high.get(i).copied().flatten().unwrap_or(f64::NAN);
        let low = quote.low.get(i).copied().flatten().unwrap_or(f64::NAN);
        let close = quote.close.get(i).copied().flatten().unwrap_or(f64::NAN);
        let volume = quote.volume.get(i).copied().flatten().unwrap_or(0.0);
        let adj = adjclose.get(i).copied().flatten().unwrap_or(close);

        if close.is_finite() && close > 0.0 {
            bars.push(PriceBar {
                date,
                open,
                high,
                low,
                close,
                adj_close: adj,
                volume,
            });
        }
    }

    bars.sort_by_key(|b| b.date);
    if bars.is_empty() {
        return Err(format!("No valid price data returned for {}", ticker).into());
    }
    Ok(bars)
}

/// Convert Vec<PriceBar> to a simple Polars DataFrame for export / analysis.
pub fn bars_to_dataframe(bars: &[PriceBar]) -> Result<DataFrame, Box<dyn Error>> {
    let dates: Vec<i32> = bars.iter().map(|b| (b.date - NaiveDate::from_ymd_opt(1970,1,1).unwrap()).num_days() as i32).collect();
    let opens: Vec<f64> = bars.iter().map(|b| b.open).collect();
    let highs: Vec<f64> = bars.iter().map(|b| b.high).collect();
    let lows: Vec<f64> = bars.iter().map(|b| b.low).collect();
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let adjcloses: Vec<f64> = bars.iter().map(|b| b.adj_close).collect();
    let vols: Vec<f64> = bars.iter().map(|b| b.volume).collect();

    let df = DataFrame::new(vec![
        Column::new("date_epoch".into(), dates),
        Column::new("open".into(), opens),
        Column::new("high".into(), highs),
        Column::new("low".into(), lows),
        Column::new("close".into(), closes),
        Column::new("adj_close".into(), adjcloses),
        Column::new("volume".into(), vols),
    ])?;
    Ok(df)
}

/// Load from local parquet (optional path for large datasets from Step 3).
/// Expects columns including "ticker", "date", "close", "volume" etc.
pub fn load_prices_parquet(path: &str, tickers: &[&str]) -> Result<DataFrame, Box<dyn Error>> {
    let lf = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?;
    let df = lf
        .filter(col("ticker").is_in(lit(Series::new("tickers".into(), tickers))))
        .collect()?;
    Ok(df)
}

// --- Serde structs for Yahoo JSON ---

#[derive(Deserialize, Debug)]
struct YahooChartResponse {
    chart: YahooChart,
}

#[derive(Deserialize, Debug)]
struct YahooChart {
    result: Vec<YahooResult>,
}

#[derive(Deserialize, Debug)]
struct YahooResult {
    timestamp: Option<Vec<i64>>,
    indicators: YahooIndicators,
}

#[derive(Deserialize, Debug)]
struct YahooIndicators {
    quote: Vec<YahooQuote>,
    #[serde(default)]
    adjclose: Option<Vec<YahooAdjClose>>,
}

#[derive(Deserialize, Debug)]
struct YahooQuote {
    open: Vec<Option<f64>>,
    high: Vec<Option<f64>>,
    low: Vec<Option<f64>>,
    close: Vec<Option<f64>>,
    volume: Vec<Option<f64>>,
}

#[derive(Deserialize, Debug)]
struct YahooAdjClose {
    adjclose: Vec<Option<f64>>,
}
