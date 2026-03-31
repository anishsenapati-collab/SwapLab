"""
vasicek_simulator.py
ES418 Group 13 — Interest Rate Swap Valuation
Phase 1: Vasicek model — calibration, simulation, validation, plots.

CALIBRATED VALUES FROM YOUR DTB3.CSV (2006–2026):
  κ  = 0.09611   (half-life ≈ 7.2 years — slow mean reversion)
  θ  = 0.01189   (long-run mean ≈ 1.189%)
  σ  = 0.00702   (annual volatility ≈ 0.702%)
  r₀ = 0.03630   (current rate ≈ 3.630% as of Mar 2026)

STRUCTURE:
  VasicekConfig        — single dataclass holding all parameters + settings
  calibrate_vasicek()  — OLS estimation of κ, θ, σ from real rate data
  vasicek_simulate()   — Euler-Maruyama simulation with antithetic variates
  print_summary()      — analytical vs simulated statistics table
  plot_paths()         — fan plot with mean, percentile bands, annotations
  plot_validation()    — mean and variance comparison plots
  plot_terminal()      — terminal rate distribution at horizon T
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIG DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VasicekConfig:
    """
    All parameters and settings for the Vasicek simulator in one place.
    Change a value here and every function picks it up automatically.

    Default values are pre-filled with OLS calibration results
    from your DTB3.csv (2006-03-27 → 2026-03-26).

    Model parameters
    ----------------
    kappa  : speed of mean reversion (κ > 0)
              Your data → 0.09611  (very slow, half-life ≈ 7.2 years)
    theta  : long-run mean rate (θ), in DECIMAL (0.05 = 5%)
              Your data → 0.01189  (≈ 1.189%, pulled low by post-2008 era)
    sigma  : annual volatility (σ > 0), in DECIMAL
              Your data → 0.00702  (≈ 0.702%)
    r0     : initial short rate, in DECIMAL
              Your data → 0.03630  (≈ 3.630%, last observation Mar 2026)

    Simulation settings
    -------------------
    T        : time horizon in years
    N        : number of time steps (252*T gives daily steps)
    n_paths  : number of Monte Carlo paths
    seed     : random seed for reproducibility

    Output settings
    ---------------
    save_plots : if True, saves PNG + PDF of every figure
    dpi        : resolution of saved PNG files
    """

    # ── model parameters (pre-filled from your DTB3 calibration) ─────
    kappa  : float = 0.09611    # mean reversion speed
    theta  : float = 0.01189    # long-run mean
    sigma  : float = 0.00702    # annual volatility
    r0     : float = 0.03630    # starting rate (last observed)

    # ── simulation settings ───────────────────────────────────────────
    T       : float = 5.0       # horizon in years
    N       : int   = 1260      # steps (252 trading days × 5 years)
    n_paths : int   = 2000      # Monte Carlo paths
    seed    : int   = 42        # random seed for reproducibility

    # ── output settings ───────────────────────────────────────────────
    save_plots : bool = True
    dpi        : int  = 150

    # ── derived properties (read-only) ────────────────────────────────

    @property
    def dt(self) -> float:
        """Time step size in years."""
        return self.T / self.N

    @property
    def half_life(self) -> float:
        """
        Time (years) for gap between r0 and theta to halve.
        Formula: ln(2) / kappa
        """
        return np.log(2) / self.kappa

    @property
    def long_run_std(self) -> float:
        """
        Stationary standard deviation of r(t) as T → ∞.
        Formula: sigma / sqrt(2 * kappa)
        """
        return self.sigma / np.sqrt(2 * self.kappa)

    def summary(self):
        print("\n" + "=" * 52)
        print("  Vasicek Configuration")
        print("=" * 52)
        print(f"  κ  (mean reversion speed) : {self.kappa:.5f}")
        print(f"  θ  (long-run mean)        : {self.theta:.5f}"
              f"  ({self.theta:.3%})")
        print(f"  σ  (annual volatility)    : {self.sigma:.5f}"
              f"  ({self.sigma:.3%})")
        print(f"  r₀ (initial rate)         : {self.r0:.5f}"
              f"  ({self.r0:.3%})")
        print(f"  Half-life of reversion    : {self.half_life:.3f} years")
        print(f"  Long-run std of r(t)      : {self.long_run_std:.5f}"
              f"  ({self.long_run_std:.3%})")
        print(f"  Horizon T                 : {self.T} years")
        print(f"  Steps N                   : {self.N}"
              f"  (dt = {self.dt:.5f} yr)")
        print(f"  Paths                     : {self.n_paths:,}")
        print(f"  Random seed               : {self.seed}")
        print("=" * 52 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CALIBRATION  —  OLS on discretised Vasicek SDE
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_vasicek(rate_series, dt=1/252) -> VasicekConfig:
    """
    Estimate Vasicek parameters from a historical rate series using OLS.

    The discretised Vasicek SDE is:
        Δr = a + b·r + ε
    where  a = κ·θ·Δt   and   b = −κ·Δt.

    Solving for structural parameters:
        κ = −b / Δt
        θ = a / (κ · Δt)
        σ = std(residuals) / sqrt(Δt)

    Parameters
    ----------
    rate_series : array-like — observed short rates in DECIMAL form
    dt          : float      — time between observations (default: 1/252)

    Returns
    -------
    VasicekConfig with calibrated parameters
    """

    r       = np.asarray(rate_series, dtype=float)
    delta_r = np.diff(r)      # Δr_t = r_{t+1} - r_t
    r_lag   = r[:-1]          # r_t  (regressor)

    # OLS: Δr = intercept + slope * r_lag + residual
    slope, intercept, r_val, p_val, _ = stats.linregress(r_lag, delta_r)

    # recover structural parameters
    kappa_raw = -slope / dt
    theta_raw = (intercept / (kappa_raw * dt)
                 if kappa_raw > 0 else 0.03)
    residuals = delta_r - (intercept + slope * r_lag)
    sigma_raw = np.std(residuals) / np.sqrt(dt)

    # safety clamps — prevent degenerate values
    kappa = float(np.clip(kappa_raw, 0.01, 20.0))
    theta = float(np.clip(theta_raw, 0.001, 0.30))
    sigma = float(np.clip(sigma_raw, 0.001, 0.50))
    r0    = float(r[-1])

    print("\n" + "=" * 52)
    print("  OLS Calibration Results")
    print("=" * 52)
    print(f"  Observations  : {len(r):,}")
    print(f"  R²            : {r_val**2:.6f}")
    print(f"  p-value       : {p_val:.3e}")
    print(f"  κ (raw)       : {kappa_raw:.5f}  → used: {kappa:.5f}")
    print(f"  θ (raw)       : {theta_raw:.5f}  → used: {theta:.5f}")
    print(f"  σ (raw)       : {sigma_raw:.5f}  → used: {sigma:.5f}")
    print(f"  r₀            : {r0:.5f}  ({r0:.3%})")
    print("=" * 52)

    return VasicekConfig(kappa=kappa, theta=theta, sigma=sigma, r0=r0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SIMULATION  —  Euler-Maruyama with antithetic variates
# ═══════════════════════════════════════════════════════════════════════════════

def vasicek_simulate(cfg: VasicekConfig, antithetic: bool = True):
    """
    Simulate Vasicek interest rate paths using Euler-Maruyama.

    Update rule at each step i:
        r[i+1] = r[i]  +  κ·(θ − r[i])·Δt  +  σ·√Δt·ε

    Antithetic variates (antithetic=True):
        For every ε drawn, also simulate −ε.
        Halves Monte Carlo variance at no extra cost.

    Parameters
    ----------
    cfg        : VasicekConfig
    antithetic : bool — use antithetic variance reduction (recommended)

    Returns
    -------
    r : np.ndarray, shape (n_paths, N+1)
        Each ROW is one complete simulated path of r(t).
    t : np.ndarray, shape (N+1,)
        Time grid from 0 to T.
    """

    np.random.seed(cfg.seed)

    dt   = cfg.dt
    t    = np.linspace(0, cfg.T, cfg.N + 1)
    half = cfg.n_paths // 2

    # draw all random shocks up front (vectorised — much faster than per-step)
    if antithetic:
        eps_pos = np.random.randn(half, cfg.N)
        eps     = np.vstack([eps_pos, -eps_pos])
    else:
        eps = np.random.randn(cfg.n_paths, cfg.N)

    # Euler-Maruyama loop
    r       = np.zeros((cfg.n_paths, cfg.N + 1))
    r[:, 0] = cfg.r0

    for i in range(cfg.N):
        drift     = cfg.kappa * (cfg.theta - r[:, i]) * dt
        noise     = cfg.sigma * np.sqrt(dt) * eps[:, i]
        r[:, i+1] = r[:, i] + drift + noise

    # report negative rates (known Vasicek limitation)
    n_neg   = (r < 0).sum()
    pct_neg = n_neg / r.size * 100
    method  = "antithetic variates" if antithetic else "standard Monte Carlo"

    print(f"\n[vasicek] Simulated {cfg.n_paths:,} paths | "
          f"{cfg.N} steps | {method}")
    if n_neg > 0:
        print(f"[vasicek] Negative rate obs: {n_neg:,} ({pct_neg:.3f}%) "
              f"— known Vasicek limitation, kept as-is")
    else:
        print(f"[vasicek] No negative rates observed.")

    return r, t


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  VALIDATION  —  compare simulation against analytical formulas
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(r: np.ndarray, t: np.ndarray, cfg: VasicekConfig):
    """
    Print table comparing simulated statistics against Vasicek
    closed-form formulas for E[r(t)] and Std[r(t)].

    Vasicek analytical mean : E[r(t)]   = r0·e^(-κt) + θ·(1 − e^(-κt))
    Vasicek analytical std  : Std[r(t)] = σ·sqrt((1 − e^(-2κt)) / 2κ)

    Error% < 1% confirms the simulator is correct.
    """

    def ana_mean(s):
        return (cfg.r0 * np.exp(-cfg.kappa * s)
                + cfg.theta * (1 - np.exp(-cfg.kappa * s)))

    def ana_std(s):
        return cfg.sigma * np.sqrt(
            (1 - np.exp(-2 * cfg.kappa * s)) / (2 * cfg.kappa))

    checkpoints = [c for c in [0.5, 1.0, 2.0, 3.0, 5.0] if c <= cfg.T]

    print("\n" + "=" * 62)
    print("  Simulation Validation — Analytical vs Simulated")
    print("=" * 62)
    print(f"  {'Mat':>5} | {'Ana.Mean':>9} {'Sim.Mean':>9} | "
          f"{'Ana.Std':>8} {'Sim.Std':>8} | {'Err%':>6}")
    print("  " + "─" * 58)

    for cp in checkpoints:
        idx  = int(round(cp / cfg.T * cfg.N))
        am   = ana_mean(cp);  sm  = r[:, idx].mean()
        asd  = ana_std(cp);   ssd = r[:, idx].std()
        err  = abs(sm - am) / abs(am) * 100 if am != 0 else 0
        print(f"  {cp:>5.1f} | {am:>9.5f} {sm:>9.5f} | "
              f"{asd:>8.5f} {ssd:>8.5f} | {err:>5.2f}%")

    print("=" * 62)
    pct_neg = (r < 0).sum() / r.size * 100
    print(f"  Negative rate obs  : {pct_neg:.4f}%")
    print(f"  Half-life          : {cfg.half_life:.4f} years")
    print(f"  Long-run std       : {cfg.long_run_std:.5f}"
          f"  ({cfg.long_run_std:.3%})\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_paths(r: np.ndarray, t: np.ndarray, cfg: VasicekConfig,
               data_start_year=2006, data_end_year=2026):
    """
    Fan plot of simulated rate paths.

    Layers (bottom to top):
      1. Individual paths         — faint blue lines
      2. 5th–95th percentile band — light blue shading
      3. 25th–75th percentile band — medium blue shading (IQR)
      4. Mean path                — solid navy line
      5. Long-run mean θ          — dashed red reference line

    Parameter annotation box embedded top-left — plot is self-documenting.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    # individual paths (random subset to avoid overplotting)
    n_show   = min(300, cfg.n_paths)
    idx_show = np.random.RandomState(0).choice(
                   cfg.n_paths, n_show, replace=False)
    for i in idx_show:
        ax.plot(t, r[i] * 100, color='steelblue',
                alpha=0.04, linewidth=0.5, zorder=1)

    # percentile bands
    p05 = np.percentile(r,  5, axis=0) * 100
    p25 = np.percentile(r, 25, axis=0) * 100
    p75 = np.percentile(r, 75, axis=0) * 100
    p95 = np.percentile(r, 95, axis=0) * 100

    ax.fill_between(t, p05, p95, alpha=0.12, color='steelblue',
                    label='5th–95th percentile', zorder=2)
    ax.fill_between(t, p25, p75, alpha=0.22, color='steelblue',
                    label='25th–75th percentile (IQR)', zorder=3)

    # mean path
    ax.plot(t, r.mean(axis=0) * 100, color='navy',
            linewidth=2.2, label='Simulated mean', zorder=5)

    # long-run mean reference
    ax.axhline(cfg.theta * 100, color='crimson', linewidth=1.3,
               linestyle='--', label=f'θ = {cfg.theta:.3%}', zorder=4)

    # starting rate dot
    ax.plot(0, cfg.r0 * 100, 'o', color='navy', markersize=7,
            zorder=6, label=f'r₀ = {cfg.r0:.3%}')

    # parameter annotation box
    ann = (f"κ = {cfg.kappa:.5f}  |  θ = {cfg.theta:.3%}  |  "
           f"σ = {cfg.sigma:.3%}  |  r₀ = {cfg.r0:.3%}\n"
           f"Half-life = {cfg.half_life:.2f} yr  |  "
           f"{cfg.n_paths:,} paths  |  T = {cfg.T} yr  |  "
           f"Calibrated from DTB3 "
           f"{data_start_year}–{data_end_year}")
    ax.text(0.02, 0.97, ann,
            transform=ax.transAxes, fontsize=8.5, va='top',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='white', alpha=0.88, ec='#cccccc'))

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.2f}%'))
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Short rate r(t)', fontsize=12)
    ax.set_title(
        f'Vasicek model — simulated interest rate paths  '
        f'(calibrated from DTB3 {data_start_year}–{data_end_year})',
        fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.22)
    plt.tight_layout()

    if cfg.save_plots:
        fig.savefig('vasicek_paths.png', dpi=cfg.dpi, bbox_inches='tight')
        fig.savefig('vasicek_paths.pdf',              bbox_inches='tight')
        print("[plot] Saved vasicek_paths.png + .pdf")

    plt.close()
    return fig


def plot_validation(r: np.ndarray, t: np.ndarray, cfg: VasicekConfig):
    """
    Two-panel validation figure:
      Left  — analytical mean vs simulated mean over time
      Right — analytical variance vs simulated variance over time

    If the blue (simulated) and red (analytical) lines overlap,
    the simulator is correct. Include in your report's methodology section.
    """
    ana_mean = (cfg.r0 * np.exp(-cfg.kappa * t)
                + cfg.theta * (1 - np.exp(-cfg.kappa * t)))
    ana_var  = ((cfg.sigma ** 2) / (2 * cfg.kappa)
                * (1 - np.exp(-2 * cfg.kappa * t)))

    sim_mean = r.mean(axis=0)
    sim_var  = r.var(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, ana, sim, ylabel, title in [
        (ax1, ana_mean, sim_mean, 'E[r(t)]',   'Mean of r(t)'),
        (ax2, ana_var,  sim_var,  'Var[r(t)]', 'Variance of r(t)'),
    ]:
        ax.plot(t, ana, 'r--', linewidth=2.0, label='Analytical (exact)')
        ax.plot(t, sim, 'b-',  linewidth=1.5,
                label='Simulated (E-M)', alpha=0.85)
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel(ylabel,         fontsize=11)
        ax.set_title(title,           fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.22)

    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y*100:.3f}%'))

    plt.suptitle(
        f'Validation: Euler-Maruyama vs Vasicek analytical formulas  '
        f'({cfg.n_paths:,} paths, antithetic variates)',
        fontsize=13)
    plt.tight_layout()

    if cfg.save_plots:
        fig.savefig('vasicek_validation.png', dpi=cfg.dpi, bbox_inches='tight')
        fig.savefig('vasicek_validation.pdf',              bbox_inches='tight')
        print("[plot] Saved vasicek_validation.png + .pdf")

    plt.close()
    return fig


def plot_terminal(r: np.ndarray, cfg: VasicekConfig):
    """
    Histogram of r(T) — the terminal rate distribution.
    Overlays the theoretical normal distribution.

    Shows that Vasicek produces Gaussian rates — good supplementary figure.
    """
    r_T    = r[:, -1]
    ana_mu = (cfg.r0 * np.exp(-cfg.kappa * cfg.T)
              + cfg.theta * (1 - np.exp(-cfg.kappa * cfg.T)))
    ana_sd = cfg.sigma * np.sqrt(
        (1 - np.exp(-2 * cfg.kappa * cfg.T)) / (2 * cfg.kappa))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(r_T * 100, bins=70, density=True, color='steelblue',
            alpha=0.65, edgecolor='white', linewidth=0.3,
            label='Simulated r(T)')

    x   = np.linspace(r_T.min() * 100, r_T.max() * 100, 300)
    pdf = (1 / (ana_sd * 100 * np.sqrt(2 * np.pi))
           * np.exp(-0.5 * ((x - ana_mu * 100) / (ana_sd * 100)) ** 2))
    ax.plot(x, pdf, 'r-', linewidth=2.0,
            label=f'Analytical  N({ana_mu*100:.3f}%,  {ana_sd*100:.3f}%²)')

    ax.axvline(ana_mu * 100, color='crimson', ls='--', lw=1.3,
               label=f'Ana. mean = {ana_mu:.3%}')
    ax.axvline(r_T.mean() * 100, color='navy', ls=':', lw=1.3,
               label=f'Sim. mean = {r_T.mean():.3%}')

    ax.set_xlabel(f'r(T={cfg.T}) — terminal rate (%)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Terminal rate distribution at T = {cfg.T} years',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.22)
    plt.tight_layout()

    if cfg.save_plots:
        fig.savefig('vasicek_terminal.png', dpi=cfg.dpi, bbox_inches='tight')
        fig.savefig('vasicek_terminal.pdf',              bbox_inches='tight')
        print("[plot] Saved vasicek_terminal.png + .pdf")

    plt.close()
    return fig
