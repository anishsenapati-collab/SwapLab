# Interest Rate Swap Valuation
###  Financial Modelling and Engineering

interest rate swap valuation using the **Vasicek stochastic interest rate model**, calibrated from real US Treasury bill data (FRED DTB3, 2006–2026).

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Data | FRED DTB3 — 5,219 daily observations (2006–2026) |
| Vasicek κ (mean reversion) | 0.09611 — half-life ≈ 7.2 years |
| Vasicek θ (long-run mean) | 1.189% |
| Vasicek σ (volatility) | 0.702% |
| Current rate r₀ | 3.630% (March 2026) |
| Par swap rate K* | 3.137% (5yr quarterly, $1M notional) |
| DV01 | $371.38 per basis point |
| Monte Carlo convergence error | 2.0% (2,000 paths) |

---

## Project Structure

```
es418_group13/
│
├── DTB3.csv                   ← FRED data (download separately — see below)
│
├── data_loader.py             ← Phase 1a: load & clean rate data
├── vasicek_simulator.py       ← Phase 1b: calibrate, simulate, validate
├── phase2_bonds.py            ← Phase 2:  bond prices & yield curve
├── phase3_swap.py             ← Phase 3:  swap cash flows & NPV
├── phase4_sensitivity.py      ← Phase 4:  DV01, scenarios, sensitivity
├── phase5_montecarlo.py       ← Phase 5:  Monte Carlo validation
├── phase6_summary.py          ← Phase 6:  master runner (run this)

```

---



## Output Files

Running `phase6_summary.py` produces the following files:

| File | Description | Report Figure |
|------|-------------|---------------|
| `data_overview.png` | Rate history + Δr histogram | Figure 1 |
| `vasicek_paths.png` | Simulated rate fan plot | Figure 2 |
| `vasicek_validation.png` | Analytical vs simulated mean/var | Figure 3 |
| `vasicek_terminal.png` | Terminal rate distribution | Figure 4 |
| `yield_curve.png` | Bond prices, yields, forward rates | Figure 5 |
| `yield_curve_shapes.png` | Normal / flat / inverted curves | Figure 6 |
| `yield_sensitivity.png` | Parameter sensitivity analysis | Figure 7 |
| `discount_factors.png` | Quarterly discount factors | Figure 8 |
| `swap_cashflows.png` | Fixed vs floating cash flows | Figure 9 |
| `swap_npv_vs_rate.png` | NPV as function of fixed rate K | Figure 10 |
| `swap_sensitivity.png` | NPV sensitivity to all params | Figure 11 |
| `swap_scenarios.png` | NPV under rate shift scenarios | Figure 12 |
| `mc_npv_distribution.png` | Monte Carlo NPV histogram | Figure 13 |
| `mc_convergence.png` | MC convergence to closed-form | Figure 14 |
| `summary_results.txt` | All key numbers in one file | — |

All figures are saved as both `.png` (150 dpi) and `.pdf` (vector, for the report).

---

## Theory Summary

### Vasicek Model

The short rate follows:

```
dr(t) = κ(θ − r(t)) dt + σ dW(t)
```

### Bond Price (closed-form)

```
P(0,T) = A(T) · exp(−B(T) · r₀)

B(T) = (1 − exp(−κT)) / κ
A(T) = exp( (θ − σ²/2κ²)(B(T)−T) − σ²B(T)²/4κ )
```

### Swap Valuation

```
PV(fixed)    = K · Δt · N · Σ P(0, tᵢ)
PV(floating) = (1 − P(0, T)) · N
NPV          = PV(floating) − PV(fixed)
Par rate K*  = (1 − P(0,T)) / (Δt · Σ P(0,tᵢ))
```

---

## Dependencies

| Package | 
|---------|
| numpy |
| pandas |
| matplotlib |
| scipy |

## Key Findings

- The calibrated κ of 0.096 implies a **mean reversion half-life of 7.2 years**, consistent with the persistent low-rate environment of 2009–2021 that dominates the 20-year sample.
- The long-run mean θ = 1.189% is significantly below the current rate r₀ = 3.630%, producing a **downward-sloping yield curve** — the model forecasts rates declining toward their historical average over the next decade.
- The yield curve's long-run limit of 0.922% lies **below θ** due to Jensen's inequality convexity adjustment (−σ²/2κ²).
- The par swap rate of **K* = 3.137%** lies below the current short rate, reflecting the inverted term structure.
- The **DV01 of $371** means the fixed-rate payer gains approximately $371 for every basis point rise in rates — a natural long-duration position.
- The Vasicek model produced **0.307% negative rate observations** in simulation — a known theoretical limitation that is acceptable for this application.

---

## Limitations

- **Negative rates**: Vasicek allows negative rates with positive probability. The CIR model eliminates this at the cost of analytical tractability.
- **Constant parameters**: κ, θ, σ are assumed constant over the 20-year calibration window. In reality, these shift across rate regimes.
- **Single-factor**: The model uses one factor for the entire yield curve. Two-factor extensions (e.g., G2++) provide richer term structure dynamics.
- **No credit risk**: The swap is valued assuming no counterparty default risk. CVA adjustments would be needed for real-world pricing.

---

## References

- Vasicek, O. (1977). *An Equilibrium Characterization of the Term Structure*. Journal of Financial Economics, 5(2), 177–188.
- Board of Governors of the Federal Reserve System (2026). *3-Month Treasury Bill Secondary Market Rate (DTB3)*. FRED. https://fred.stlouisfed.org/series/DTB3
- Hull, J.C. (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
- Brigo, D., & Mercurio, F. (2006). *Interest Rate Models — Theory and Practice* (2nd ed.). Springer Finance.

---


