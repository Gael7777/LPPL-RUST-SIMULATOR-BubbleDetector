# HLPLL / HLPPL Rust Backtester

Rust implementation for **walk-forward backtesting** of the Hyped Log-Periodic Power Law (HLPPL / HLPLL) bubble-detection framework, inspired by the Johns Hopkins preprint [arXiv:2510.10878](https://arxiv.org/abs/2510.10878).

This repo is a **practical open-source approximation**: it uses Yahoo daily OHLCV, a volume-based hype proxy, and a price-return sentiment proxy. The full paper pipeline (proprietary news corpus + dual-stream Transformer) is not replicated here.

---

## Table of contents

1. [Overview](#1-overview)
2. [Data requirements](#2-data-requirements)
3. [How the pipeline works](#3-how-the-pipeline-works)
4. [Mathematical model](#4-mathematical-model)
5. [CLI reference](#5-cli-reference)
6. [Output files](#6-output-files)
7. [Reading `*_signals.csv` columns](#7-reading-_signalscsv-columns)
8. [Console backtest summary](#8-console-backtest-summary)
9. [Project layout](#9-project-layout)
10. [Limitations vs. the paper](#10-limitations-vs-the-paper)
11. [Quick start](#11-quick-start)
12. [Extensions](#12-extensions)
13. [License & citation](#13-license--citation)

---

## 1. Overview

For each ticker and date range, the program:

1. Downloads daily OHLCV from Yahoo Finance.
2. Fits a **7-parameter LPPL** model on a rolling window of log adjusted closes.
3. Builds a **bubble score** from LPPL residuals + volume hype + sentiment proxy.
4. Runs a simple **long / short / flat** strategy from score thresholds.
5. Writes CSVs and an equity PNG under `results/`.

Typical use: research and signal diagnostics on volatile, hype-sensitive names—not production trading without further validation.

---

## 2. Data requirements

### Minimum (what runs today)

| Layer | Fields used | Source in this repo |
|--------|-------------|---------------------|
| Prices | `date`, `adj_close`, `volume` (also `open/high/low/close` fetched) | Yahoo Finance v8 chart API |
| Hype proxy | `volume` | Rolling z-score in `hype.rs` |
| Sentiment proxy | derived from daily returns | `sentiment.rs` (not news NLP) |

You need at least **`window + 30`** trading days of history per ticker (see `--window`).

### Recommended for paper-like behavior

| Layer | Ideal fields | Notes |
|--------|--------------|--------|
| Prices | long history, split-adjusted | Parquet bulk data via `load_prices_parquet()` in `data.rs` |
| News hype | daily `ticker`, `date`, mention count | Replace volume proxy |
| News sentiment | daily mean sentiment per ticker | FinBERT / lexicon on headlines |
| Regime context | VIX, rates, sector ETF | Optional filters |

Helpers exist for Parquet (`load_prices_parquet`, `bars_to_dataframe`); the CLI still fetches Yahoo by default.

---

## 3. How the pipeline works

```text
CLI args
   │
   ▼
fetch_yahoo_history(ticker, start, end)  →  Vec<PriceBar>
   │
   ▼
run_backtest (walk-forward loop, day i = lookback .. end)
   │
   ├─ Every refit_every days (from first tradable day):
   │     • Window = bars[i - window .. i)
   │     • fit_lppl_on_bars → LPPL params + residuals
   │     • paper: overlapping-window ε, running-max ε_norm ∈ [-1,1]
   │     • hype_volume = last volume z-score (60-day roll inside window)
   │     • sentiment = last value from return-based proxy
   │     • bubble_score = piecewise formula (α₁=0.7, α₂=0.3)
   │
   ├─ Other days: reuse previous bubble_score / components
   │
   ├─ position:
   │     score >  long_threshold  → +1 (long)
   │     score < -short_threshold → -1 (short)
   │     else                     →  0 (flat)
   │
   └─ PnL: daily_ret = position × asset_return − cost (if position changed)
         equity *= (1 + daily_ret)
   │
   ▼
results/<TICKER>_signals.csv
results/<TICKER>_equity.csv
results/<TICKER>_equity.png
```

**Timing details**

- **Time axis for LPPL:** trading-day index `0 .. window-1` inside each fit window (not calendar dates), to avoid weekend/holiday gaps.
- **Prices for LPPL:** `ln(adj_close)`.
- **First signal day:** index `lookback_days` (first `window` bars are warm-up only).
- **Paper mode (default):** signals are precomputed for every bar (overlapping LPPL windows + causal running-max norm).
- **Fast mode:** between refits, LPPL curve is projected forward with updated hype/sentiment.

---

## 4. Mathematical model

> **Math in Preview:** Equations use `$$ ... $$` (display) and `$ ... $` (inline), which render on **GitHub** and in VS Code/Cursor when Markdown math is enabled (e.g. built-in preview or “Markdown+Math”). If you still see raw `\ln` / `\tau`, use the **plain-text lines** under each equation—they are equivalent.

### 4.1 LPPL log-price equation

The fitted model (Johansen–Sornette–style LPPL) is:

$$
\ln p_t = A + B\,\tau_t^{m} + C\,\tau_t^{m}\cos(\omega \ln \tau_t + \phi)
$$

**Plain text:**

```text
ln(p_t) = A + B * tau^m + C * tau^m * cos(omega * ln(tau) + phi)
where tau = t_c - t
```

**Symbols**

| Symbol | Meaning |
|--------|---------|
| $p_t$ | Adjusted close on trading-day index $t$ inside the fit window |
| $\tau_t$ | Time to critical point: $t_c - t$ (clipped to a small positive value in code) |
| $t_c, m, \omega, \phi$ | Nonlinear parameters (random search) |
| $A, B, C$ | Linear parameters (OLS for each nonlinear trial) |

**Validity filters** on accepted fits (`lppl.rs`):

- $0 < m < 1$
- $0.5 < \omega < 30$
- Damping (approx.): $|C| < |B|\sqrt{1 + m^2}$

Fitting uses **1200 random multi-starts** over $(t_c, m, \omega, \phi)$; each trial solves $(A,B,C)$ by least squares (`nalgebra`).

### 4.2 LPPL residual and `eps_norm`

Residual on each day in the window:

$$
\varepsilon_t = \ln p_t - \widehat{\ln p}_t
$$

Normalized signal (last day of the window):

$$
\varepsilon_{\text{norm}}(t) = \frac{\varepsilon(t)}{\max_{s \le t} |\varepsilon(s)|}
$$

(bounded in \([-1, 1]\); paper Eq. 8)

**Plain text:** `eps_norm(t) = epsilon(t) / running_max_abs_epsilon_up_to_t`

**Interpretation**

- **Positive `eps_norm`:** price is **above** the LPPL curve → overextension / bubble-like in log space.  
- **Negative `eps_norm`:** price is **below** the LPPL curve → depressed vs. the fitted path.

### 4.3 Volume hype (`hype_volume`)

Rolling z-score of volume (last **60** bars inside the LPPL window):

$$
z_t = \frac{V_t - \mu_{t,w}}{\sigma_{t,w}}
\qquad
\text{hype}_t = \tanh(0.6 \cdot z_t)
$$

**Plain text:** `hype_volume = tanh(0.6 * volume_z_score)`

**Interpretation:** unusually high volume vs. recent history → positive hype. This is a **volume attention proxy**, not news mention count.

### 4.4 Sentiment proxy (`sentiment`)

From each daily return $r_t$ in the window:

$$
\text{sent}_t = \mathrm{sign}(r_t)\cdot 0.15 + \mathrm{clip}(0.3\, r_t,\,-0.4,\,0.4)
$$

The CSV column is the **last** $\text{sent}_t$ in the window. **Not** headline or FinBERT sentiment.

### 4.5 Bubble score (piecewise)

Weights in code: $\alpha_1 = 0.7$ (hype), $\alpha_2 = 0.3$ (sentiment).

**If** $\varepsilon_{\text{norm}} \ge 0$:

$$
\text{BubbleScore} = \varepsilon_{\text{norm}} + \alpha_1 \cdot \text{hype} + \alpha_2 \cdot \text{sent}
$$

**If** $\varepsilon_{\text{norm}} < 0$:

$$
\text{BubbleScore} = \varepsilon_{\text{norm}} - \alpha_1 \cdot \text{hype} - \alpha_2 \cdot \text{sent}
$$

**Plain text:**

```text
if eps_norm >= 0:
    bubble_score = eps_norm + 0.7*hype_volume + 0.3*sentiment
else:
    bubble_score = eps_norm - 0.7*hype_volume - 0.3*sentiment
```

When price is high vs. the LPPL fit, hype and sentiment **raise** the score; when price is low vs. the fit, they **lower** it further.

### 4.6 Trading rules

Let $L$ = `--long-thresh`, $S$ = `--short-thresh` (short side uses $-S$):

| Condition | Position |
|-----------|----------|
| `bubble_score > L` | `+1` (full long; `max_position = 1.0`) |
| `bubble_score < -S` | `-1` (full short) |
| otherwise | `0` (flat) |

**Daily strategy return** on day $i$:

$$
r^{\text{strat}}_i = \text{position}_{i-1} \cdot r^{\text{asset}}_i - \mathbf{1}_{\text{trade}} \cdot \frac{\text{cost\_bps}}{10000}
$$

- $r^{\text{asset}}_i = \dfrac{\text{adj\_close}_i}{\text{adj\_close}_{i-1}} - 1$  
- Cost is deducted **once** when position changes (one-way bps).  
- Equity: $E_i = E_{i-1}(1 + r^{\text{strat}}_i)$, $E_0 = 1$.

### 4.7 Is `bubble_score > 0.75` “good” or “bad”?

**Short answer:** A score above **0.75** is neither universally good nor bad. It means the model sees a **strong positive bubble-style reading** (price above LPPL, with hype/sentiment aligned in that direction). Whether that is desirable depends on **what you are trying to do**.

#### What the number means (signal level)

| `bubble_score` (rough guide) | Reading |
|------------------------------|---------|
| Near `0` | Weak / neutral composite signal |
| `0.75` to `1.5` | Elevated (default long threshold lives here) |
| `> 1.5` (often with high `eps_norm`) | Very strong “extended + attention” episode |
| `< -0.75` | Strong negative reading → simulator goes **short** |
| Between `-L` and `+L` | **Flat** (no position) |

`0.75` is only the **default CLI cutoff** (`--long-thresh`). It is not a fixed “pass/fail” grade for the stock or the LPPL fit. Tune it per ticker after looking at the score distribution in `*_signals.csv`.

#### Risk / bubble-detection lens (fundamentals)

For **detecting bubble risk** (will price correct?), a **high positive** score is usually a **warning**, not a buy signal:

- High `eps_norm` → price has run **above** the fitted LPPL path.  
- Positive `hype_volume` → volume spike / attention.  
- Together → “heated” conditions; **crash or drawdown risk can be higher**, not lower.

So for a long-term holder or risk manager: **`bubble_score > 0.75` is often “bad” (risky)**, not “good (safe).”

#### What **this backtester** does (strategy lens)

The simulator uses thresholds as **trading rules**, not as moral “good/bad” labels:

- `bubble_score > 0.75` → **`position = +1` (long)**  
- `bubble_score < -0.75` → **`position = -1` (short)**  
- otherwise → **flat**

So in this repo, a high score means “the rule says go long.” That can **disagree** with the risk interpretation above: you might want to **short** bubbles in another strategy, but this code **does not** do that unless you change the rules in `backtest.rs` or invert how you use the CSV.

Whether that long rule is “good” is empirical: check `*_equity.csv` and the console summary vs. buy-and-hold. A high score can still lose money after costs if the move reverses.

#### How to read a day with `bubble_score > 0.75`

1. Open `*_signals.csv` on that `date`.  
2. Check **`eps_norm`** — is the score driven mainly by LPPL overextension?  
3. Check **`hype_volume`** and **`sentiment`** — are they amplifying the score?  
4. Compare **`close`** and **`volume`** to public price action (and news separately, if you have it).  
5. See **`position`** and **`trade`** — did the strategy actually enter long and pay cost?

**Example decomposition (conceptual):**  
`bubble_score = 1.2` might come from `eps_norm = 0.9`, `hype_volume = 0.3`, `sentiment = 0.1` → mostly LPPL stretch, with some volume confirmation.

#### Practical takeaway

| Your goal | Score `> 0.75` is… |
|-----------|---------------------|
| Early warning of overheated price | **Meaningful alert** (investigate; often cautious) |
| Automatic “buy” without more checks | **Not automatically good** — depends on backtest / regime |
| Quality of LPPL fit | **Unrelated** — check fit stability across refits, not score alone |

If you want “high score = danger,” use the CSV for **alerts** and ignore or invert the built-in long/short mapping until the strategy matches your thesis.

---

## 5. CLI reference

Run:

```bash
cargo run --release -- [FLAGS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--tickers` | string | `HOUS,AMTX,CAR` | Comma-separated Yahoo symbols (spaces after commas are trimmed). |
| `--start` | `YYYY-MM-DD` | `2018-01-01` | First calendar date for price download (inclusive). |
| `--end` | `YYYY-MM-DD` | `2024-12-31` | Last calendar date for price download (inclusive). |
| `--window` | usize | `250` | LPPL window **W** (trading days), notebook default. |
| `--window-stride` | usize | `5` | Paper mode: fit every Nth overlapping window end (`1` = full notebook). |
| `--signal-mode` | str | `paper` | `paper` (overlapping windows) or `fast` (single window + refit). |
| `--refit-every` | usize | `25` | Refit LPPL and recompute score every this many days (from first signal day). Between refits, score is unchanged. |
| `--long-thresh` | f64 | `0.55` | Go **long** when `bubble_score >` this value (paper scale). |
| `--short-thresh` | f64 | `0.55` | Go **short** when `bubble_score < -` this value. |
| `--cost-bps` | f64 | `10.0` | One-way transaction cost in **basis points** (10 = 0.10%) deducted on each position change. |
| `--outdir` | path | `results` | Directory for `*_signals.csv`, `*_equity.csv`, `*_equity.png`. |

**Not exposed on CLI (hardcoded)**

| Parameter | Value | Location |
|-----------|-------|----------|
| `max_position` | `1.0` | `main.rs` → `BacktestConfig` |
| Hype rolling window | `60` bars | `backtest.rs` |
| Bubble weights $\alpha_1$, $\alpha_2$ | `0.7`, `0.3` | `backtest.rs` |
| LPPL multi-start samples | `1200` (+ `300` fallback) | `lppl.rs` |

**Example (tighter window, more frequent refit)**

```bash
cargo run --release -- \
  --tickers CAR,AMTX \
  --start 2021-06-01 --end 2024-12-31 \
  --window 180 --refit-every 40 \
  --long-thresh 0.65 --short-thresh 0.65 \
  --cost-bps 8
```

**Threshold tuning (rule of thumb)**

- **Lower** thresholds → more trades, more exposure to noise.  
- **Higher** thresholds → fewer trades, only extreme scores.  
- Compare `bubble_score` distribution in `*_signals.csv` before picking $L$ and $S$.

---

## 6. Output files

Per ticker `TICKER` in `--outdir`:

| File | Contents |
|------|----------|
| `TICKER_signals.csv` | Daily diagnostics: price, components, score, position, trade flag |
| `TICKER_equity.csv` | Cumulative strategy equity (starts near 1.0) |
| `TICKER_equity.png` | Plot of strategy equity vs. date |

If a ticker fails to download or has insufficient history, it is skipped with a message; other tickers still run.

---

## 7. Reading `*_signals.csv` columns

Header:

```text
date,date_epoch,close,daily_return,volume,eps_norm,hype_volume,sentiment,bubble_score,position,trade
```

| Column | Meaning | How to use / compare |
|--------|---------|----------------------|
| `date` | Calendar date (`YYYY-MM-DD`) | Join to external data on this key. |
| `date_epoch` | Days since 1970-01-01 | Legacy/plotting; prefer `date` for merges. |
| `close` | **Adjusted close** from Yahoo | Compare bubble episodes to actual price level and trends. |
| `daily_return` | **Strategy** daily return that day (position × asset return − cost if traded) | Not the raw stock return; use for PnL alignment with `equity.csv`. |
| `volume` | Raw share volume | Spot liquidity spikes; relates to `hype_volume`. |
| `eps_norm` | Normalized LPPL residual at end of fit window | Core “mispricing vs. LPPL” signal; positive ≈ price above fitted bubble path. |
| `hype_volume` | $\tanh$-scaled volume z-score | Attention proxy; high with rising price may align with retail hype. |
| `sentiment` | Return-based proxy in $[-0.55, 0.55]$ roughly | Momentum tint only—not news sentiment. |
| `bubble_score` | Combined score (Section 4.5) | Compare to `--long-thresh` / `--short-thresh`. |
| `position` | Target position: `-1`, `0`, or `+1` | What the simulator holds **after** that day’s decision. |
| `trade` | `1` if position changed, else `0` | Count turnover; align with cost days. |

**Reading a row**

1. Check `close` and `volume` for market reality.  
2. Decompose `bubble_score` via `eps_norm`, `hype_volume`, `sentiment`.  
3. See if `position` matches thresholds: long if `bubble_score > long_thresh`, etc.  
4. On `trade = 1`, expect `daily_return` to include a cost drag.

**Stale score blocks**

Equal `bubble_score` for many consecutive rows is normal: score updates only on refit days (`--refit-every`).

**External validation (manual)**

- Overlay `close` with high `|bubble_score|`.  
- Pull news for same `date` + ticker separately (not in CSV yet).  
- Compare strategy `equity` to buy-and-hold using console summary.

---

## 8. Console backtest summary

Printed after each ticker:

| Line | Meaning |
|------|---------|
| Strategy Total Return | $E_{\text{final}} - 1$ over backtest period |
| Strategy Ann. Return | Compound annualization from mean daily strategy return (252 days) |
| Strategy Sharpe | Mean / std of daily strategy returns × $\sqrt{252}$ |
| Strategy Max DD | Largest peak-to-trough drop on equity curve |
| Num Trades | Count of days with `trade = 1` |
| Buy & Hold Return | Long-only hold of adjusted close over same days |
| Buy & Hold Sharpe | Sharpe of raw asset daily returns |

Backtest **starts** on the first day after the warm-up window (`--window` bars), not on `--start` if history is too short.

---

## 9. Project layout

```text
src/
├── main.rs              # CLI, orchestration, equity PNG
└── modules/
    ├── data.rs          # Yahoo fetch, PriceBar, optional Parquet
    ├── lppl.rs          # LPPL fit + residuals
    ├── hype.rs          # Volume hype proxy
    ├── sentiment.rs     # Return-based sentiment proxy
    ├── bubble_score.rs  # eps_norm + piecewise score
    ├── backtest.rs      # Walk-forward loop + metrics
    └── utils.rs         # CSV export + summary print
```

See also `instructions.md` for the original build guide and dataset ideas.

---

## 10. Limitations vs. the paper

- No proprietary WSJ corpus or Transformer fusion.  
- Hype = volume z-score, not news mention index.  
- Sentiment = price-return heuristic, not FinBERT.  
- Random-search LPPL instead of industrial-grade global optimization.  
- Single-asset, full notional long/short, no borrow constraints.  
- Costs are a simple one-way bps on trade days.

Results are best treated as **research signals**, not validated alpha, until tuned and validated on your universe and costs.

---

## 11. Quick start

```bash
cargo build --release

cargo run --release -- \
  --tickers CAR,AMTX \
  --start 2021-06-01 --end 2024-12-31 \
  --window 180 --refit-every 40 \
  --long-thresh 0.65 --short-thresh 0.65 \
  --cost-bps 8
```

Inspect `results/CAR_signals.csv` and compare `bubble_score` to `close` and `volume`.

---

## 11.5 Interactive Explorer (TUI)

A separate **modular** binary for live tweaking, Yahoo API validation, bubble indicator visualization, and $10k (or custom capital) trade simulation that **strictly re-uses the same strategy code**.

```bash
cargo run --release --bin hlpll-explorer
# or cargo build --release --bin hlpll-explorer ; .\target\release\hlpll-explorer.exe
```

**Features / full user control**
- Edit any param live (ticker, dates, window, refit, long/short thresh, cost, capital) — Tab to focus, type or +/- to nudge.
- **[F]** — Test Yahoo Finance API for the exact ticker+time range you chose. Shows bar count, date span, last price. Explicit success/failure.
- **[R]** — Run the *exact* same `run_backtest` + LPPL + bubble_score + position logic as the main app (and the paper strategy). 10k (or your capital) equity is derived by simple scaling of the multiplier curve.
- Live **three stacked terminal charts**:
  - Price colored green (long), red (short), gray (flat) — visual "bubble regime" indicator.
  - Bubble score with horizontal threshold lines. Clearly see when it crosses into long/short.
  - Dollar equity curve for your simulated investment.
- Pan (← → h l), zoom ([ ]), cursor (j k), reset view (0). Cursor panel shows live decomposition (eps_norm, hype, sentiment, decision).
- Trade log with per-leg $PnL computed from the equity curve segments (strictly follows every position change + cost).
- **[E]** exports the full signals CSV, normalized + dollar equity CSVs, and the PNG equity plot (via shared `export_backtest_artifacts`).
- Press `?` for help overlay. `q` to quit.

The explorer is intentionally separated (own `src/bin/`) so the core backtester stays a fast headless CLI while giving researchers an interactive lab for "what if I change the threshold / window / ticker / period?" with immediate visual feedback.

### 11.6 Native GUI Explorer (proper Windows .exe + modern UI)

A third, fully separate frontend using **egui + eframe** (immediate-mode native GUI). This produces a real double-clickable Windows executable with high-quality interactive charts (zoom, pan, hover, multiple synchronized plots, regime-colored price, live bubble-score thresholds, $ equity, cursor scrubbing, trade log, etc.).

```bash
# Clean build — only pulls egui/wgpu/etc when you ask for the GUI feature
cargo run --release --bin hlpll-gui --no-default-features --features gui
```

- All heavy lifting (Yahoo fetch, LPPL multi-start, bubble score, strategy, trade extraction, exports) goes through the single `HlpplEngine` in the library.
- Left panel: all tweakable parameters with sliders + text fields + preset buttons.
- "Test Yahoo API" and "Run Simulation" never block the UI (background worker thread + channels).
- Central area: three live `egui_plot` charts (price by position color, bubble score + threshold lines, dollar equity).
- Full cursor control, export button that re-uses the engine's artifact writer.
- Result: a proper, accurate, nice-looking desktop application for detailed what-if analysis.

The three frontends (CLI, TUI `hlpll-explorer`, native GUI `hlpll-gui`) + any future consumers all share the exact same isolated logic engine.

---

## Modularity & Isolating the HLPPL Logic Engine

If you only ever want the pure backtesting engine (no UIs):

```toml
[dependencies]
hlpll-backtester = { version = "...", default-features = false }
```

Then just use:

```rust
use hlpll_backtester::{HlpplEngine, BacktestConfig, fetch_yahoo_history};
```

`HlpplEngine` (in `src/engine.rs`) + the re-exports at the crate root are the blessed surface. The `modules/` directory contains the implementation details and is public mainly for the in-tree frontends.

Feature flags in `Cargo.toml` ensure that `ratatui`, `egui`, `clap`, etc. are only compiled when you actually build the corresponding binary.

This structure makes it straightforward to:
- Use the engine from Python (via PyO3) or another Rust service
- Write a web version later (axum + egui_web or leptos)
- Swap the visualization layer completely

---



## 12. Extensions

1. **News parquet** → daily mention counts → replace `compute_volume_hype`.  
2. **FinBERT (ONNX)** → real `sentiment` column from headlines.  
3. **Parquet price bulk** → wire `load_prices_parquet` into CLI (`--data-source`).  
4. **Export** `news_mentions`, `news_sentiment` columns in `save_signals_csv`.  
5. **Live mode** → daily cron: fetch latest bar, append, refit, alert if `|bubble_score| > threshold`.

---

## 13. License & citation

MIT / Apache-2.0 (your choice).

If you use this for research, cite:

> Johns Hopkins preprint (arXiv:2510.10878) — *Detecting Financial Bubbles with the Hyped Log-Periodic Power Law model*

---

## 14. Extensive Run Modes, LPPLS Confidence (C1), Future Bubble Prediction & Live Sentiment (2026 updates)

The project was evolved into the **most complete Rust LPPL research+experimentation platform** after analyzing the new `grok-build-assets/gemini-data-LPPLS.md` (multi-window JLS strict filters, Bubble Confidence Index as % valid fits, tc calendar projection, risk levels LOW/MODERATE/HIGH/CRITICAL, trading dev notes) + independent research (literature from Sornette/JLS, ETH FCO, papers on 2 centuries S&P, Chinese/ crypto bubbles, Imperial thesis backtests of conf+trust strategies, QuantConnect LPPLS examples, GSADF comparisons).

**Is it a good idea?** Yes. LPPLS C1 (fraction of rolling windows meeting 0.1≤m≤0.9, 4.5≤ω≤13, B<0, tc>t+margin + damping) is the literature-standard robust "bubble index". Single fits are noisy; multi-window + ensemble gives reliable early warnings and tc clusters. Papers show ex-ante capture of known bubbles with fewer false positives than alternatives in many cases; trading strategies using C1 (long on positive bubble conf or short risk) have been backtested successfully on assets (with costs/drawdowns). Our "HLPPL" (LPPL + volume hype + return sent) + live daily eps projection + directional bias/invert + C1 risk filter/sizing is a strong practical hybrid addressing gaps noted in the docs (pure LPPL under-emphasizes behavioral). **Caveats from research**: not infallible (regime dependent, window sensitive, past performance != future, high DD possible); best as *one indicator* + risk mgmt. Never sole basis for live capital. The tool now lets you rigorously test all of it reproducibly.

### New capabilities
- **RunMode** (in BacktestConfig, CLI --mode, GUI radio, TUI):
  - `historical` (default): classic walk-forward + 10k equity sim + B&H + full signals (C1 per bar if enabled).
  - `prediction`: pure future bubble prediction at end of data — C1%, risk level, list+median of predicted critical dates (tc), days-to, prob-within-horizon from the valid fits distro. No (or hybrid) equity.
  - `live`: snapshot "current sentiment" at latest bar for live trading decisions (score + C1 + synthesized BUY/SELL/HOLD rec that respects bias/invert/conf flat/sizing + actionable note on risk).
  - `hybrid`: both for validation (backtest history of the signal you would have used live).
- **LPPLS Confidence (C1) & strict filters**: `enable_bubble_analysis`, analysis_lookback min/max/step, filter_m_* , filter_omega_*, filter_b_negative, filter_tc_offset. Matches the gemini Python "enhanced multi-window engine" + JLS table.
- **Ensemble seeds**: --ensemble-seeds "42,43,44" (or GUI/TUI) for mean/std C1 across independent multi-start searches — much more stable than single seed.
- **Proportional sizing + risk override**: use_confidence_for_flat (force 0 if C1 high), use_confidence_for_sizing (scale pos by C1/100 clamped).
- **Prediction horizon**: controls "prob tc in next N days".
- **Cursor / query recs now include C1**: in all UIs and --query-date, see score + confidence + future peak if avail.
- Exports: signals still have bubble_confidence; prediction runs print rich report + (in hybrid) the CSVs.

### CLI examples (all features)
```powershell
# Historical backtest (with C1 overlay + B&H)
cargo run --release -- --tickers CAR --start 2022-01-01 --end 2024-12-31 --mode historical --enable-bubble-analysis

# Pure future bubble prediction (C1 + tc cluster) on latest data
cargo run --release -- --tickers ^NDX --start 2024-01-01 --end 2026-06-03 --mode prediction --ensemble-seeds "42,43,44" --predict-horizon 90 --filter-m-min 0.1 --filter-omega-min 4.5

# Live sentiment for "should I trade this now?"
cargo run --release -- --tickers AMD --start 2025-01-01 --end 2026-06-03 --mode live --invert --position-bias longonly --use-conf-flat --conf-flat-thresh 65

# Full hybrid (validate the live rule historically + see current pred)
cargo run --release -- --tickers CAR --mode hybrid --ensemble-seeds 42,43
```

GUI: left panel now has Run Mode radios + ensemble text + horizon + full JLS m/omega/B<0/tc filters surfaced. Big "PREDICTION" or "LIVE SENTIMENT" callouts above charts. "Run Simulation" respects the chosen mode (uses run_with_mode on engine). Dedicated bubble analysis button still there.

TUI/Explorer: supports via strings (run_mode_str etc); [R]un uses current. Use GUI or CLI for easiest access to prediction/live.

### Using for live trading on current sentiment
1. Set recent window (e.g. 1-2y data).
2. Run in `live` or `prediction` mode (or hybrid then look at last).
3. Read the synthesized RECOMMENDATION (respects your bias/invert/C1 filter) + actionable note + median tc if any.
4. Cross with fundamentals/news/VIX. The C1 high = "many scales show herding signature" → often risk-off or short candidate per literature.

**This is research tooling, not advice.** Backtests (even reproducible) and C1 forecasts can and do fail. Use for exploration, parameter studies, robustness across seeds/modes/windows. Always paper trade first, size small, have stops.

See `grok-build-assets/gemini-data-LPPLS.md` (the source doc) for the Python blueprint we aligned the Rust C1 to, risk assessment text, and why rolling multi-window + strict filters > single brittle fit.

Enjoy the most extensive LPPL Rust platform! Tweak, predict, backtest live rules, export, repeat.
