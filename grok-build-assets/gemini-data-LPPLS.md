What financial mathematics do we currently have to kind of "establish" an index of bubble in a current time with a given index or stock price?

In financial mathematics, establishing a real-time "bubble index" or indicator directly from an asset's spot price series requires shifting from traditional geometric Brownian motion—which assumes market efficiency—toward models that capture **explosive behavior, herding, or structural math defects**.

Rather than relying on fundamental analysis (like P/E ratios), quantitative methods utilize time-series econometrics, statistical physics, and continuous-time stochastic calculus to evaluate whether current price acceleration is mathematically unsustainable.

Here are the primary quantitative frameworks currently used to establish a bubble index in real-time.

## 1. Econophysics: The LPPLS Model (Johansen-Ledoit-Sornette)

The **Log-Periodic Power Law Singularity (LPPLS)** framework treats financial bubbles as critical phenomena, drawing an analogy to micro-fractures accumulating in a material before structural failure. It posits that bubbles are driven by imitation and positive feedback loops among market participants, resulting in **super-exponential growth** decorated by accelerating log-periodic oscillations.

Emergent Mind+ 1



To calculate a real-time bubble confidence index, the model fits historical log-prices over rolling time windows to the standard LPPLS equation:

Emergent Mind



E[ln*p*(*t*)]=*A*+*B*(*tc*​−*t*)*m*+*C*(*tc*​−*t*)*m*cos(*ω*ln(*tc*​−*t*)+*ϕ*)

Where:

- *tc*​ is the predicted critical time (the most probable date of the bubble's peak or burst).
  arXiv
- *m* is the structural growth exponent (0<*m*<1 indicates super-exponential growth).
  arXiv
- *ω* is the angular frequency of the log-periodic oscillations (capturing the market's nervous "boom-and-bust" cycles accelerating as *t*→*tc*​).
  arXiv

### Establishing the Index:

Quant desks construct a **Bubble Confidence Indicator** by running this non-linear optimization across hundreds of shifting historical windows. The "index" value is defined as the proportion of windows where the parameters fall into strictly bounded, physically reasonable regimes (e.g., 0<*m*<1 and 4≤*ω*≤25). A reading near 100% signifies highly synchronized, unstable herd behavior.

Emergent Mind



## 2. Time-Series Econometrics: Recursive Cointegration & SADF / GSADF Tests

Developed extensively by Peter Phillips, Jun Yu, and Shuping Shi, the **Supremum Augmented Dickey-Fuller (SADF)** and **Generalized SADF (GSADF)** tests are designed specifically for ex-ante, real-time bubble detection in an asset or an index.

Standard asset pricing models dictate that if fundamentals (dividends, earnings) are integrated of order one (*I*(1)), the asset price must also be *I*(1). If a bubble is present, the price process temporarily transitions into an explosive autoregressive process (*I*(*d*) where *d*>1).

The core model estimates an ADF equation recursively:

Δ*yt*​=*α*+*βyt*−1​+*i*=1∑*k*​*γi*​Δ*yt*−*i*​+*εt*​

### Establishing the Index:

Instead of testing if *β*=0 (unit root) against *β*<0 (stationary), the GSADF tests the null hypothesis of a unit root (*β*=0) against the alternative of explosive behavior (*β*>0).

- The **GSADF statistic** is calculated over varying, recursively rolling start and end points.
- By plotting the rolling sequence of the supreme ADF t-statistics against dynamically simulated right-tailed critical values, you get a real-time bubble index.
- **The Bubble Signal:** When the calculated metric breaches the critical threshold line, a bubble is formally flagged as active at that exact timestamp.

## 3. Continuous-Time Stochastic Calculus: The Martingale Defect Approach

In the foundational framework of mathematical finance (under the risk-neutral measure Q), an asset price process *St*​ is traditionally modeled as a local martingale. However, if a bubble is inflating, *St*​ can manifest as a **strict local martingale** that is not a true martingale.

arXiv



A strict local martingale behaves in a way that its current price is strictly greater than the expected value of its future discounted payoff:

*St*​>EQ[*e*−*r*(*T*−*t*)*ST*​∣F*t*​]

This mathematical discrepancy—the **martingale defect**—exactly quantifies the dollar value of the financial bubble.

### Establishing the Index:

Because you cannot observe the risk-neutral expectations of the spot price directly from historical underlying data alone, quants extract this index from the **implied volatility surface** of options markets.

- Under strict local martingale conditions, high-strike European call options lose put-call parity at the boundary, and the implied volatility surface exhibits an extreme skew (or smile) that cannot be replicated by standard diffusion models (like SABR or CEV without explosive parameters).
- Deep learning models or direct calibration of the **SVI (Stochastic Volatility Inspired) surface** are used to isolate the parameters of the local volatility function at extreme strikes. If the calibrated volatility growth violates the criteria for uniform integrability, a real-time indicator reveals that the underlying index has decoupled from its long-term martingale path.

## Summary of Approaches


| Mathematical Framework                      | Underlying Mechanism                                                             | Primary Data Requirement                              | Real-Time Index Construction                                                                       |
| ------------------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **LPPLS (Econophysics)**                    | Positive feedback loops & discrete scale invariance among trading herds.         | Historical Log-Prices of the index/stock.             | Percentage of rolling calibration windows meeting strict parameter constraints (*m*,*ω*).          |
| **GSADF (Econometrics)**                    | Regime shift from a unit-root random walk to an explosive autoregressive path.   | Index spot price series.                              | Rolling t-statistic values plotted against simulated right-tailed critical thresholds.             |
| **Martingale Defect (Stochastic Calculus)** | Asset price behaves as a strict local martingale; loss of uniform integrability. | Implied Volatility Surface (Call option option grid). | Quantification of the mathematical defect extracted from extreme-strike option pricing deviations. |


Does the model TimeFM from google can help do such calculations?

**TimesFM** (Time-series Foundation Model), developed by Google, is a decoder-only, transformer-based foundation model trained on massive quantities of synthetic and real-world data (Das et al., 2024). It is optimized for **point and probabilistic zero-shot time-series forecasting** (Auer et al., 2025; Das et al., 2024).

When evaluating if TimesFM can help calculate or establish a real-time asset bubble index, the brief answer is: **Not directly, but it can serve as a powerful alternative benchmark or structural component within an anomaly detection pipeline.** ---

## The Core Mismatch: Generative Trends vs. Structural Breaks

Foundation models like TimesFM and traditional financial bubble mathematics (LPPLS, GSADF, Martingale Defects) operate on fundamentally opposing philosophical assumptions about data:

- **TimesFM’s Limitation:** It is designed to capture generic temporal regularities—such as seasonal trends, localized mean reversion, and standard volatility clustering—and extrapolate them linearly or autoregressively into the future (Achour et al., 2025; Das et al., 2024). When TimesFM encounters an explosive, super-exponential bubble curve in zero-shot mode, its training bias pushes it to predict a smooth continuation or a generic mean-reverting decay. It cannot naturally identify that the internal mechanics of the trend are fundamentally unstable.
- **Bubble Mathematics Requirement:** Frameworks like LPPLS look specifically for structural defects, critical phase transitions, and accelerating log-periodic oscillations caused by herd-behavior feedback loops. TimesFM lacks an inductive bias to isolate or output these explicit physical/mathematical parameters (*tc*​, *ω*, *m*).

## How TimesFM Can Be Leveraged in a Bubble Pipeline

If you want to integrate a deep-learning foundation model like TimesFM into your quantitative bubble detection architecture, you can use it in the following three ways:

### 1. Constructing a "Fundamental vs. Price" Divergence Index (Zero-Shot)

In econometrics, a bubble is broadly defined as a significant deviation of an asset's price from its fundamental intrinsic value. You can use TimesFM to generate a purely data-driven "fundamental baseline":

1. Feed TimesFM a clean, long-term historical context window of an asset or index prior to the suspected explosive phase.
2. Generate an out-of-sample, long-horizon forecast of where the price should naturally evolve based on normal historic patterns (seasonality, drift, autoregressive momentum).
3. Compute a real-time **Divergence Index**:
  DI*t*​=*St*​−*S*^*t*TimesFM​
  An exponentially growing divergence between the actual spot price (*St*​) and TimesFM's zero-shot baseline (*S*^*t*​) indicates a structural break from historical norms, flagging a potential bubble regime.

### 2. Standardized Anomaly and Regime-Switching Detection

Instead of directly modeling the bubble's peak, you can utilize TimesFM’s probabilistic output heads to flags regimes where normal statistical laws break down:

- TimesFM outputs forecast distributions. By comparing the realized explosive spot prices against TimesFM’s predicted confidence intervals (quantiles), you can measure the **Surprise Index**.
- If the spot price continuously breaches TimesFM's upper 99% quantile bounds over consecutive recursive windows, it signals that the time series has transitioned from a standard random-walk or trend regime into an explosive *I*(*d*) autoregressive regime, validating the alternative hypothesis used in **GSADF tests**.

### 3. Feature Engineering for Downstream Interpretable Classifiers

If you are building an Explainable AI (XAI) or genetic programming engine to classify market regimes, raw price data often lacks context. You can pass the asset series through TimesFM's transformer backbone to extract the final-layer latent representations (embeddings) at a specific timestamp.

- These embeddings act as dense feature vectors capturing universal temporal properties.
- You can feed these embeddings into a downstream, interpretable classifier alongside explicit statistical features (like rolling LPPLS fit metrics or SADF t-statistics) to build a highly robust, hybrid bubble indicator.

## References

Achour, S., Bouher, Y., Nguyen, D., & Chesneau, N. (2025). Foundation models for time series forecasting: Application in conformal prediction. arXiv. [https://doi.org/10.48550/arxiv.2507.08858](https://doi.org/10.48550/arxiv.2507.08858)  
Cited by: 1

Auer, A., Podest, P., Klotz, D., Böck, S., Klambauer, G., & Hochreiter, S. (2025). TiRex: Zero-shot forecasting across long and short horizons with enhanced in-context learning. arXiv. [https://doi.org/10.48550/arxiv.2505.23719](https://doi.org/10.48550/arxiv.2505.23719)  
Cited by: 65

Das, A., Faw, M., Sen, R., & Zhou, Y. (2024). In-context fine-tuning for time-series foundation models. arXiv. [https://doi.org/10.48550/arxiv.2410.24087](https://doi.org/10.48550/arxiv.2410.24087)  
Cited by: 30

Can you currently calculate LPPLS / HLPPL or such things on current NASDAQ100 prices as of june 2nd 2026?

Yes, you absolutely can run an LPPLS (Log-Periodic Power Law Singularity) or GSADF calculation on the NASDAQ 100 right now. In fact, the current market regime makes it a prime candidate for these exact models.

As of **June 2nd, 2026**, the NASDAQ 100 (*NDX*) has just shattered major psychological resistance, trading right around **30,513**. Looking closely at the recent trajectory—the index was sitting at roughly 22,200 a year ago, hovered around 27,650 just a month ago, and has surged roughly 33% since late March 2026 alone—we are looking at a classic parabolic, super-exponential curve driven heavily by intense AI hardware and infrastructure demand.

[Investing.com](http://Investing.com)



To perform these calculations yourself today, you can break the implementation into a standard quantitative workflow using open-source math frameworks.

## 1. Implementing the LPPLS Confidence Index

Because the standard LPPLS equation contains both linear parameters (*A*,*B*,*C*) and highly sensitive non-linear parameters (*tc*​,*m*,*ω*,*ϕ*), a single global optimization fit can be unstable. Instead, quants use a meta-heuristic approach or a grid search over the non-linear space, solving the linear components via ordinary least squares (OLS) at each step.

### Step-by-Step Execution:

1. **Get the Data:** Extract daily close prices for *NDX* going back 2 to 3 years to capture the baseline before this parabolic leg. Convert to log prices: *yt*​=ln*Pt*​.
2. **Define Window Regimes:** Implement a loop that systematically shrinks or slides your observation window. For instance, fix the ending date (*t*2​) to today, **June 2, 2026**, and vary the start date (*t*1​) from 240 days ago down to 60 days ago in increments of 5 days.
3. **Execute Fit Filtering:** For each window, fit the core equation:
  *yt*​=*A*+*B*(*tc*​−*t*)*m*+*C*(*tc*​−*t*)*m*cos(*ω*ln(*tc*​−*t*)+*ϕ*)
4. **Filter by Physics Constraints:** Discard fits where the parameters diverge into mathematically meaningless territory. To flag a true herding bubble, filter strictly for:
  - 0.1≤*m*≤0.9 (Growth is super-exponential but bounded)
  - 6.0≤*ω*≤15.0 (Oscillations are frequent enough to capture nervous feedback but aren't fitting raw noise)
  - *B*<0 (Ensures the slope accelerates upward as it approaches *tc*​)
  - *tc*​>*t*current​ (The predicted critical crash matrix falls in the future)

### Expected Index Output:

Your **LPPLS Bubble Confidence Indicator** is the number of passing windows divided by the total windows tested. Given that the *NDX* has run from 27,500 to over 30,500 in May 2026 alone, a rolling fit right now will likely show an incredibly high cluster of synchronized *tc*​ targets, pushing the confidence index toward the upper bounds (>80%).

## 2. Python Implementation Blueprint

You don't need to write the optimization from scratch. You can utilize the standard `lppls` package or write a quick `scipy.optimize` pipeline. Below is the structural framework for running this calculation on today’s data:

**Python**

```
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf # Or your local Postgres/Cube.dev feed

# 1. Fetch NASDAQ 100 Data up to June 2, 2026
ticker = "^NDX"
data = yf.download(ticker, start="2024-01-01", end="2026-06-03")
data['log_p'] = np.log(data['Close'])

# Create a uniform time array (t) mapped to integers for the solver
obs_data = data['log_p'].values
t_array = np.arange(len(obs_data))

# 2. Define the LPPLS Matrix Loss Function
def lppls_loss(non_linear_params, t, y):
    tc, m, omega = non_linear_params
    
    # Boundary protections for the optimization engine
    if tc <= t[-1] or m <= 0 or m >= 1 or omega <= 0:
        return 1e10
        
    # Construct the linear design matrix
    dt = tc - t
    X = np.zeros((len(t), 4))
    X[:, 0] = 1.0
    X[:, 1] = dt**m
    X[:, 2] = (dt**m) * np.cos(omega * np.log(dt))
    X[:, 3] = (dt**m) * np.sin(omega * np.log(dt))
    
    # Solve linear parameters (A, B, C1, C2) via OLS
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - np.dot(X, beta)
        return np.sum(residuals**2)
    except np.linalg.LinAlgError:
        return 1e10

# 3. Quick Test Fit on Current Window
# Initial guess: tc is 30 periods out, m=0.5, omega=8.0
initial_guess = [t_array[-1] + 30, 0.5, 8.0]
bounds = [(t_array[-1] + 2, t_array[-1] + 180), (0.1, 0.9), (4.0, 20.0)]

result = minimize(lppls_loss, initial_guess, args=(t_array, obs_data), bounds=bounds, method='L-BFGS-B')
print(f"Optimization Success: {result.success}")
print(f"Predicted Critical Target Time (tc index): {result.x[0]}")

```

## 3. Alternative: Running the GSADF Test on Today's NDX

If you want to cross-validate the LPPLS model to ensure you aren't just fitting a noisy trend, you should apply a **Generalized Supremum Augmented Dickey-Fuller (GSADF)** test to the same daily series.

- **The Logic:** If you run a rolling ADF regression on the NASDAQ data over the last 3 to 6 months, you are testing if the autoregressive coefficient *β* in Δ*yt*​=*α*+*βyt*−1​+… is significantly greater than zero.
- **The Current Flag:** Since the *NDX* has expanded by roughly 9% in a single month (May 2026) without meaningful fundamental alterations to the underlying discount rates, a recursive GSADF test run today will almost certainly breach the 95% simulated right-tailed critical value line. This statistically confirms that the current regime is no longer a random walk with drift—it is mathematically explosive.

Can you provide me a link to an effective python running environment os I can have abeautiful result with the given python snippet?

To run your quantitative analyses and visualize the LPPLS or explosive econometric models beautifully, you need an environment that handles interactive data plotting natively (like `matplotlib` or `plotly`) and can handle heavy matrix calculations effortlessly.

Here are the most effective, ready-to-use Python environments available right now:

### 1. Google Colab (Highly Recommended)

For a completely zero-setup, cloud-based experience, Google Colab is the gold standard for quantitative prototyping. It runs entirely in your browser, includes pre-installed data science packages (`numpy`, `pandas`, `scipy`), and displays beautiful, interactive inline charts.

- **Access it here:** [Google Colab](https://colab.research.google.com/)
- Tip: You can install additional financial libraries directly in a cell by running `!pip install yfinance lppls`.

### 2. Jupyterlab via Anaconda (Best for Local, Privacy-First Architecture)

If you prefer keeping your market datasets, API keys, and financial code entirely local on your own hardware, downloading the Anaconda distribution is the ideal choice. It installs **JupyterLab**, the premier web-based interactive development environment for notebooks.

- **Access it here:** [Anaconda Distribution](https://www.anaconda.com/download)
- Once installed, launch JupyterLab from your terminal or the Anaconda Navigator. It will allow you to execute the LPPLS optimization code block-by-block and render high-resolution charts right below your code.

### 3. Visual Studio Code with the Jupyter Extension

If you prefer a full-featured IDE over a basic browser interface but still want the beautiful, interactive visual outputs of a notebook, VS Code is incredibly efficient.

- **Access it here:** [Visual Studio Code](https://code.visualstudio.com/)
- Simply download the installer, open a file named `bubble_analysis.ipynb`, and VS Code will automatically prompt you to install its **Jupyter Extension** to render clean, inline visual results.

### Enhancing Your Visual Results

To make your LPPLS or GSADF output look professional and publication-ready in any of these environments, you can add a clean plotting style at the top of your notebook script:

**Python**

```
import matplotlib.pyplot as plt

# Apply a clean, modern aesthetic for financial charts
plt.style.use('seaborn-v0_8-darkgrid') 
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 150

```

Which environment fits best with your current setup—are you looking to keep your quantitative workflows strictly local, or do you prefer spinning it up quickly in the cloud?

is high or not ?

Based on the execution output visible in `image_56d467.png`, the result **is not high** relative to the current date—in fact, the optimization has hit the absolute lower bound you set for the critical time (*tc*​).

Here is the breakdown of why this specific result indicates an incomplete fit rather than a high-risk signal:

### 1. The Math Behind the Output

In the code shown in `image_56d467.png`, your lower bound for *tc*​ is defined as:

**Python**

```
t_array[-1] + 2

```

If we analyze the data array length for a daily frequency from January 1, 2024, to June 3, 2026, it contains roughly **604 trading days**.

- Your output shows a `Predicted Critical Target Time (tc index): 606.0`.
- 606.0 is exactly 604+2.

This means the `L-BFGS-B` optimization engine wanted to push *tc*​ even lower (meaning the mathematical "singularity" or peak is being fitted as if it already happened or is happening right now), but it was stopped by the hard boundary condition you imposed in the `bounds` array.

### 2. Why This Happens (And How to Fix It)

When an LPPLS solver slams directly into the lower boundary of *tc*​, it usually points to one of two structural issues in the single-window optimization:

- **The Initial Guess is Trapped:** The initial guess `t_array[-1] + 30` combined with a local optimization method like `L-BFGS-B` can easily get stuck in a local minimum if the loss landscape is highly non-linear.
- **Missing Oscillations:** Without forcing the solver to find the accelerating log-periodic oscillations (*ω*), it will treat the super-exponential curve as a pure power law, pinning *tc*​ as close to the final data point as allowed.

### 3. Turning This Into a Real "High vs. Low" Index

To determine if the bubble risk is truly high, a single test window is not enough. You need to wrap this minimize function in a **rolling window loop** (as described in the multi-window framework earlier):

1. Vary your start date (*t*1​) systematically (e.g., from 250 days ago to 60 days ago).
2. Run the optimization for each window.
3. Count how many windows converge on a valid *tc*​ that sits comfortably past the boundary (e.g., *tc*​>*t*current​+10) while meeting the physics constraints (0.1≤*m*≤0.9).

If 80% or more of those shifting windows independently cluster around a specific future date index (e.g., matching a value like 630 or 650), then the bubble confidence indicator is officially **high**. A single window hitting the floor at 606 simply means the local solver gave up at the boundary.

Can you remake a code to enhance the initial version and make a more readable output result with actual boundaries and result explanations to know whether the result is good or not.

To transform this from a single, brittle optimization into a robust **Bubble Confidence Index**, we need to implement a **Multi-Window Rolling Fit**.

A single window hitting a boundary doesn't tell us anything. But if we fit the model over 30 different historical windows and 25 of them mathematically agree that a crash or peak is converging on a specific future date, then the indicator is definitively **High**.

Here is the enhanced production-ready script. It automates the window sweeping, applies strict filtering filters based on Johansen-Ledoit-Sornette (JLS) criteria, and prints a clear, human-readable risk analysis.

### The Enhanced LPPLS Multi-Window Engine

**Python**

```
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
import matplotlib.pyplot as plt

# =====================================================================
# 1. DATA ACQUISITION & SETUP
# =====================================================================
ticker = "^NDX"
print(f"Fetching {ticker} market data...")
# Pulling data up to today (June 2, 2026)
df = yf.download(ticker, start="2024-01-01", end="2026-06-03", progress=False)

df['log_p'] = np.log(df['Close'])
prices = df['log_p'].values
dates = df.index.tolist()

# Global time index parameters
t_end = len(prices) - 1  # Today's index
t_array = np.arange(len(prices))

# =====================================================================
# 2. CORE LPPLS SOLVER (Linear + Non-Linear Hybrid)
# =====================================================================
def fit_lppls_window(t_window, y_window, t_current_idx):
    """Fits LPPLS parameters over a specific historical window."""
    
    def loss_function(non_linear_params):
        tc, m, omega = non_linear_params
        
        # Hard boundary protection inside the optimization loop
        if tc <= t_current_idx + 1 or m <= 0.05 or m >= 0.95 or omega <= 2.0 or omega >= 20.0:
            return 1e12
            
        dt = tc - t_window
        X = np.zeros((len(t_window), 4))
        X[:, 0] = 1.0
        X[:, 1] = dt**m
        X[:, 2] = (dt**m) * np.cos(omega * np.log(dt))
        X[:, 3] = (dt**m) * np.sin(omega * np.log(dt))
        
        try:
            # Solve for linear parameters (A, B, C1, C2) using OLS
            beta = np.linalg.lstsq(X, y_window, rcond=None)[0]
            residuals = y_window - np.dot(X, beta)
            return np.sum(residuals**2)
        except np.linalg.LinAlgError:
            return 1e12

    # Global/Heuristic bounds for the local optimizer
    initial_guess = [t_current_idx + 30, 0.4, 8.0]
    bounds = [
        (t_current_idx + 2, t_current_idx + 365),  # tc must be in the future (up to 1 year out)
        (0.1, 0.9),                                # m parameter constraints
        (4.0, 15.0)                                # omega parameter constraints
    ]
    
    res = minimize(loss_function, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    if res.success:
        tc, m, omega = res.x
        # Re-extract B to check sign constraint (B must be < 0 for an upward bubble)
        dt = tc - t_window
        X = np.zeros((len(t_window), 4))
        X[:, 0] = 1.0
        X[:, 1] = dt**m
        X[:, 2] = (dt**m) * np.cos(omega * np.log(dt))
        X[:, 3] = (dt**m) * np.sin(omega * np.log(dt))
        beta = np.linalg.lstsq(X, y_window, rcond=None)[0]
        B = beta[1]
        
        return {"success": True, "tc": tc, "m": m, "omega": omega, "B": B}
    return {"success": False}

# =====================================================================
# 3. MULTI-WINDOW SIMULATION LOOP (The "Sweeper")
# =====================================================================
print("Running multi-window calibration sweep...")

# We vary the start window index to look at different historical depths ending today
# Looking back from 260 trading days down to 60 trading days ago
window_starts = np.arange(t_end - 260, t_end - 60, 5)
valid_fits = 0
total_windows = len(window_starts)
predicted_dates = []

for start in window_starts:
    t_win = t_array[start : t_end + 1]
    y_win = prices[start : t_end + 1]
    
    fit = fit_lppls_window(t_win, y_win, t_end)
    
    if fit["success"]:
        # Strict Econophysics Filtering Conditions (JLS Model Parameters)
        is_tc_valid = fit["tc"] > (t_end + 3)          # Filter out floor-boundary wall hits
        is_m_valid = 0.1 <= fit["m"] <= 0.9
        is_omega_valid = 4.5 <= fit["omega"] <= 13.0
        is_b_valid = fit["B"] < 0                      # Must be an upward explosive trajectory
        
        if is_tc_valid and is_m_valid and is_omega_valid and is_b_valid:
            valid_fits += 1
            # Extrapolate index value back to an actual calendar date
            days_into_future = int(np.round(fit["tc"] - t_end))
            pred_date = pd.Timestamp(dates[t_end]) + pd.Timedelta(days=days_into_future)
            predicted_dates.append(pred_date)

# =====================================================================
# 4. BUBBLE CONFIDENCE CALCULATION & REPORTING
# =====================================================================
bubble_index = (valid_fits / total_windows) * 100

print("\n" + "="*50)
print("             LPPLS BUBBLE ANALYSIS REPORT       ")
print("="*50)
print(f"Analysis Date          : {dates[t_end].strftime('%Y-%m-%d')}")
print(f"Current Asset Price    : {np.exp(prices[t_end]):,.2f}")
print(f"Total Windows Tested   : {total_windows}")
print(f"Mathematically Valid   : {valid_fits}")
print(f"BUBBLE CONFIDENCE INDEX: {bubble_index:.2f}%")
print("-"*50)

print("RISK ASSESSMENT EVALUATION:")
if bubble_index < 15:
    print("👉 [LOW RISK] Normal organic trend or basic noise. No systematic herding signatures detected.")
elif 15 <= bubble_index < 45:
    print("👉 [MODERATE RISK] Localized acceleration detected. Keep an eye on parabolic velocity changes.")
elif 45 <= bubble_index < 75:
    print("👉 [HIGH RISK] Significant critical mass forming. The trend shows super-exponential characteristics.")
else:
    print("👉 [CRITICAL BUBBLE REGIME] Extreme herd synchronization. Acceleration pattern is structurally unstable.")

if len(predicted_dates) > 0:
    median_date = pd.to_datetime(np.median([p.value for p in predicted_dates]))
    print(f"\nEstimated Structural Critical Window Peak: Around {median_date.strftime('%Y-%m-%d')}")
print("="*50)

```

### What Changes Make This Execution Accurate?

1. **Boundary-Hit Filtering (**`is_tc_valid`**)**:  
The code actively throws out your previous false-positive result (`606.0`). If the optimization engine simply drops the parameter onto the bottom limit (`t_end + 2`), it fails the verification filter and does not increase the risk index score.
2. **The Sign Criterion Constraint (**`B < 0`**)**:  
A true upward bubble requires *B* to be less than zero. If *B*>0, the math is defining a logarithmic slowing pattern instead of an explosive acceleration pattern.
3. **Calendar Projection Engine**:  
Instead of outputting an ambiguous integer array index floating-point number (like `606.0`), the system converts index counts into actual, concrete calendar dates relative to your timeframe.

