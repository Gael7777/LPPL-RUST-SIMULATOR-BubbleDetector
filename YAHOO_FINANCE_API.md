# Yahoo Finance data (as used in this repo)

This project fetches **daily OHLCV** from Yahoo’s **public chart API** (v8). There is **no API key**, no OAuth, and no official SDK — just one HTTP GET and JSON parsing.

Implementation lives in `src/modules/data.rs` (`fetch_yahoo_history`). The same function is re-exported from the library crate as `hlpll_backtester::fetch_yahoo_history`.

---

## Endpoint

```
GET https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}
```

### Query parameters


| Parameter              | Value in this repo            | Meaning                                       |
| ---------------------- | ----------------------------- | --------------------------------------------- |
| `period1`              | Unix timestamp (seconds), UTC | Start of range — midnight UTC on `start` date |
| `period2`              | Unix timestamp (seconds), UTC | End of range — 23:59:59 UTC on `end` date     |
| `interval`             | `1d`                          | Daily bars                                    |
| `indicators`           | `quote`                       | OHLCV arrays in response                      |
| `includeAdjustedClose` | `true`                        | Split/dividend-adjusted close                 |


### Example URL

```
https://query1.finance.yahoo.com/v8/finance/chart/AAPL?period1=1609459200&period2=1640995199&interval=1d&indicators=quote&includeAdjustedClose=true
```

(`period1` / `period2` are computed from `chrono::NaiveDate` → UTC datetime → `.timestamp()`.)

---

## HTTP client settings


| Setting    | Value                                            |
| ---------- | ------------------------------------------------ |
| Library    | `reqwest` **blocking** client                    |
| Timeout    | 30 seconds                                       |
| User-Agent | `Mozilla/5.0 (compatible; HLPLL-Backtester/0.1)` |


Yahoo often blocks or throttles requests with a missing or bot-like User-Agent. Keep a descriptive UA string.

### Rust dependencies (minimal copy)

```toml
reqwest = { version = "0.12", features = ["blocking", "json", "rustls-tls"] }
serde = { version = "1.0", features = ["derive"] }
chrono = { version = "0.4", features = ["serde"] }
```

---

## Ticker symbols

Pass Yahoo-style symbols as in the URL path (uppercase is fine):

- US equities: `AAPL`, `MSFT`
- Indices: `^GSPC`
- Crypto (Yahoo symbols): `BTC-USD`
- Class shares: `BRK-B` (hyphen, not dot)

Invalid or delisted symbols typically yield HTTP 200 with an empty `result` or no usable quotes — handle empty bars as an error.

---

## JSON response shape (what we parse)

Top-level:

```json
{
  "chart": {
    "result": [
      {
        "timestamp": [1609459200, ...],
        "indicators": {
          "quote": [
            {
              "open": [null, 130.5, ...],
              "high": [...],
              "low": [...],
              "close": [...],
              "volume": [...]
            }
          ],
          "adjclose": [
            { "adjclose": [null, 129.8, ...] }
          ]
        }
      }
    ]
  }
}
```

Notes:

- `timestamp` entries are **Unix seconds** (UTC).
- OHLCV fields are **parallel arrays** aligned by index with `timestamp`.
- Values can be `null` (missing session / holiday gap) — we use `Option<f64>` in Serde.
- We take the **first** element of `chart.result`, first `quote`, first `adjclose` block.

---

## Parsing logic (behavior to replicate)

1. **GET** the URL; fail on non-2xx HTTP status.
2. Deserialize into structs matching the tree above (`serde`).
3. Zip `timestamp[i]` with `quote.*[i]` and `adjclose[i]`.
4. Convert each timestamp → `NaiveDate` (UTC); skip invalid timestamps.
5. **Filter** bars where `start <= date <= end` (Yahoo may return slightly wider range).
6. For each bar:
  - `open`, `high`, `low`, `close`, `volume` from `quote` (default volume `0`, missing OHLC → `NaN`).
  - `adj_close` from `adjclose`, or **fallback to `close`** if adjusted series missing.
7. **Keep** only rows where `close` is finite and `close > 0`.
8. **Sort** ascending by date.
9. Error if zero bars remain.

### Output struct (`PriceBar`)

```rust
pub struct PriceBar {
    pub date: NaiveDate,   // calendar date (UTC-derived)
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,        // raw close from quote
    pub adj_close: f64,    // preferred for returns / backtests
    pub volume: f64,
}
```

In this HLPLL project, **strategy and LPPL fitting use `adj_close`** for price levels and returns; raw `close` is still stored for export.

---

## Function signature (copy into another crate)

```rust
pub fn fetch_yahoo_history(
    ticker: &str,
    start: chrono::NaiveDate,
    end: chrono::NaiveDate,
) -> Result<Vec<PriceBar>, Box<dyn std::error::Error>>
```

Returns `Vec<PriceBar>` sorted oldest → newest.

Optional helper in this repo: `bars_to_dataframe(&[PriceBar])` → Polars `DataFrame` with columns `date_epoch`, `open`, `high`, `low`, `close`, `adj_close`, `volume`.

---

## Where it is called in this repo


| Location              | Usage                             |
| --------------------- | --------------------------------- |
| `src/modules/data.rs` | Implementation                    |
| `src/engine.rs`       | `HlpplEngine::fetch()`            |
| `src/main.rs`         | CLI batch backtest per ticker     |
| `src/bin/gui.rs`      | “Test Yahoo API” + run simulation |
| `src/bin/explorer.rs` | TUI **[F]** fetch                 |


Alternative data path: `load_prices_parquet()` for local Parquet files (not Yahoo).

---

## Minimal standalone example

```rust
use chrono::NaiveDate;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap();

    // Copy `fetch_yahoo_history` + Serde structs from src/modules/data.rs,
    // or depend on this crate with feature "core" and call:
    // let bars = hlpll_backtester::fetch_yahoo_history("AAPL", start, end)?;

    for b in bars.iter().take(3) {
        println!("{} adj_close={}", b.date, b.adj_close);
    }
    Ok(())
}
```

To reuse without copying code, add to `Cargo.toml`:

```toml
hlpll-backtester = { path = "../grok-lppl-rust", default-features = false, features = ["core"] }
```

Then: `use hlpll_backtester::fetch_yahoo_history;`

---

## Caveats

- **Unofficial API** — Yahoo can change URL, JSON shape, or rate limits without notice.
- **Not for production market data** — use a licensed vendor for compliance-critical systems.
- **US/session calendar** — daily bars follow Yahoo’s trading calendar; gaps are normal.
- **Rate limiting** — batch many tickers with delays; avoid hammering from one IP.
- **No `yahoo_finance` crate here** — older docs (`instructions_for_ai.md`) mention a crate; **this repo uses raw `reqwest` only**.

---

## Quick checklist for a new Rust project

1. Add `reqwest` (blocking + json), `serde`, `chrono`.
2. Build URL with `period1` / `period2` from UTC midnight / end-of-day.
3. Set a browser-like `User-Agent`.
4. Deserialize `chart.result[0]` + parallel arrays.
5. Filter, drop invalid closes, sort by date.
6. Prefer `**adj_close`** for return series and backtests.

Source of truth: `src/modules/data.rs` lines 20–109 (fetch) and 143–180 (Serde types).