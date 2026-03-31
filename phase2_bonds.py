"""
phase2_bonds.py
ES418 Group 13 — Interest Rate Swap Valuation
Phase 2: Vasicek closed-form bond pricing and yield curve construction.

WHAT THIS FILE DOES:
  1. Computes zero-coupon bond prices P(0,T) using the Vasicek
     closed-form formula — no simulation needed here.
  2. Converts bond prices to yields: y(T) = -ln(P) / T
  3. Computes forward rates: f(t,T) = -d(lnP)/dT
  4. Shows how the yield curve shape changes with r0 vs theta
  5. Produces a sensitivity analysis across all four parameters
  6. Saves all figures ready for your report

VASICEK BOND PRICE FORMULA:
  P(0,T) = A(T) * exp(-B(T) * r0)

  where:
    B(T) = (1 - exp(-kappa*T)) / kappa
    A(T) = exp( (theta - sigma^2/(2*kappa^2)) * (B(T)-T)
                - sigma^2*B(T)^2 / (4*kappa) )

CALIBRATED PARAMETERS FROM YOUR DTB3.CSV:
  kappa = 0.09611   theta = 0.01189
  sigma = 0.00702   r0    = 0.03630
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CORE FORMULA FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def vasicek_B(kappa, T):
    """
    B(T) = (1 - exp(-kappa*T)) / kappa

    This is the sensitivity of log bond price to the current rate r0.
    - For small T : B(T) ≈ T   (current rate dominates)
    - For large T : B(T) → 1/kappa  (saturates — mean reversion kicks in)

    Parameters
    ----------
    kappa : float       — mean reversion speed
    T     : float or np.ndarray — maturity (years)
    """
    return (1 - np.exp(-kappa * T)) / kappa


def vasicek_lnA(kappa, theta, sigma, T):
    """
    ln A(T) = (theta - sigma^2/(2*kappa^2)) * (B(T) - T)
              - sigma^2 * B(T)^2 / (4*kappa)

    This captures the long-run mean level and the Jensen's inequality
    correction from the convexity of exp(-integral of r).
    Note: the long-run yield is BELOW theta due to this correction.

    Parameters
    ----------
    kappa, theta, sigma : floats
    T                   : float or np.ndarray
    """
    B   = vasicek_B(kappa, T)
    return ((theta - sigma**2 / (2 * kappa**2)) * (B - T)
            - (sigma**2 * B**2) / (4 * kappa))


def bond_price(kappa, theta, sigma, r0, T):
    """
    Vasicek zero-coupon bond price.
    P(0,T) = A(T) * exp(-B(T) * r0)

    Parameters
    ----------
    kappa, theta, sigma, r0 : floats — Vasicek parameters
    T : float or np.ndarray — maturity or array of maturities

    Returns
    -------
    P : same shape as T — bond prices in [0,1]
    """
    T   = np.asarray(T, dtype=float)
    B   = vasicek_B(kappa, T)
    lnA = vasicek_lnA(kappa, theta, sigma, T)
    return np.exp(lnA - B * r0)


def yield_curve(kappa, theta, sigma, r0, T):
    """
    Continuously compounded yield:
    y(T) = -ln( P(0,T) ) / T

    Parameters
    ----------
    T : float or np.ndarray — must be > 0

    Returns
    -------
    y : same shape as T — yields in decimal (0.03 = 3%)
    """
    T = np.asarray(T, dtype=float)
    P = bond_price(kappa, theta, sigma, r0, T)
    return -np.log(P) / T


def forward_rate(kappa, theta, sigma, r0, T):
    """
    Instantaneous forward rate:
    f(0,T) = -d(ln P(0,T)) / dT

    Analytically:
    f(0,T) = (theta - sigma^2/(2*kappa^2))
             + (r0 - theta + sigma^2/kappa^2) * exp(-kappa*T)
             - sigma^2 / (2*kappa^2) * exp(-2*kappa*T)

    Parameters
    ----------
    T : float or np.ndarray

    Returns
    -------
    f : same shape as T — forward rates in decimal
    """
    T = np.asarray(T, dtype=float)
    term1 = theta - sigma**2 / (2 * kappa**2)
    term2 = (r0 - theta + sigma**2 / kappa**2) * np.exp(-kappa * T)
    term3 = -(sigma**2 / (2 * kappa**2)) * np.exp(-2 * kappa * T)
    return term1 + term2 + term3


def long_run_yield(kappa, theta, sigma):
    """
    Theoretical yield as T → infinity.
    lim(T→∞) y(T) = theta - sigma^2 / (2*kappa^2)

    This is BELOW theta due to the Jensen's inequality / convexity adjustment.
    """
    return theta - sigma**2 / (2 * kappa**2)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_bond_table(kappa, theta, sigma, r0,
                     maturities=None, label=""):
    """
    Print a table of bond prices, yields, and forward rates
    at key maturities. Include this in your report's results section.
    """
    if maturities is None:
        maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20]

    T   = np.array(maturities, dtype=float)
    P   = bond_price(kappa, theta, sigma, r0, T)
    y   = yield_curve(kappa, theta, sigma, r0, T)
    f   = forward_rate(kappa, theta, sigma, r0, T)
    lry = long_run_yield(kappa, theta, sigma)

    title = f"Vasicek Bond Prices & Yields{' — ' + label if label else ''}"
    print("\n" + "=" * 68)
    print(f"  {title}")
    print(f"  κ={kappa:.5f} | θ={theta:.3%} | σ={sigma:.3%} | r₀={r0:.3%}")
    print("=" * 68)
    print(f"  {'Mat(yr)':>8} │ {'Price P':>10} │ "
          f"{'Yield y(T)':>10} │ {'Fwd f(T)':>10}")
    print("  " + "─" * 50)
    for i, t in enumerate(maturities):
        print(f"  {t:>8.2f} │ {P[i]:>10.6f} │ "
              f"{y[i]:>10.4%} │ {f[i]:>10.4%}")
    print("  " + "─" * 50)
    print(f"  {'∞':>8} │ {'—':>10} │ {lry:>10.4%} │ {lry:>10.4%}")
    print("=" * 68 + "\n")

    return P, y, f


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_yield_curve(kappa, theta, sigma, r0,
                     save=True, dpi=150):
    """
    Three-panel yield curve figure:
      Top left  — bond prices P(0,T) vs maturity
      Top right — yield curve y(T) vs maturity
      Bottom    — forward rate curve f(0,T) vs maturity

    Include as Figure 2 in your report.
    """
    T   = np.linspace(0.01, 20, 500)
    P   = bond_price(kappa,  theta, sigma, r0, T)
    y   = yield_curve(kappa, theta, sigma, r0, T)
    f   = forward_rate(kappa, theta, sigma, r0, T)
    lry = long_run_yield(kappa, theta, sigma)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── panel 1: bond prices ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(T, P, color='steelblue', lw=2)
    ax.axhline(1.0, color='crimson', lw=1, ls='--', alpha=0.6,
               label='P = 1.0 (face value)')
    ax.set_xlabel('Maturity T (years)', fontsize=11)
    ax.set_ylabel('Bond price P(0,T)', fontsize=11)
    ax.set_title('Zero-coupon bond prices', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # ── panel 2: yield curve ──────────────────────────────────────────
    ax = axes[1]
    ax.plot(T, y * 100, color='steelblue', lw=2, label='Yield y(T)')
    ax.axhline(theta * 100, color='crimson', lw=1.2, ls='--',
               label=f'θ = {theta:.3%}')
    ax.axhline(lry * 100, color='darkorange', lw=1.2, ls=':',
               label=f'y(∞) = {lry:.3%}')
    ax.axhline(r0 * 100, color='navy', lw=1, ls='-.',
               alpha=0.7, label=f'r₀ = {r0:.3%}')
    ax.set_xlabel('Maturity T (years)', fontsize=11)
    ax.set_ylabel('Yield (%)', fontsize=11)
    ax.set_title('Yield curve y(T)', fontsize=12)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    # ── panel 3: forward rates ────────────────────────────────────────
    ax = axes[2]
    ax.plot(T, f * 100, color='darkorange', lw=2, label='Forward rate f(0,T)')
    ax.axhline(theta * 100, color='crimson', lw=1.2, ls='--',
               label=f'θ = {theta:.3%}')
    ax.axhline(lry * 100, color='gray', lw=1, ls=':',
               label=f'f(∞) = {lry:.3%}')
    ax.set_xlabel('Maturity T (years)', fontsize=11)
    ax.set_ylabel('Forward rate (%)', fontsize=11)
    ax.set_title('Forward rate curve f(0,T)', fontsize=12)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    # shared annotation
    fig.suptitle(
        f'Vasicek term structure  |  '
        f'κ={kappa:.5f}  θ={theta:.3%}  σ={sigma:.3%}  r₀={r0:.3%}',
        fontsize=13)
    plt.tight_layout()

    if save:
        fig.savefig('yield_curve.png', dpi=dpi, bbox_inches='tight')
        fig.savefig('yield_curve.pdf',           bbox_inches='tight')
        print("[plot] Saved yield_curve.png + .pdf")

    plt.close()
    return fig


def plot_curve_shapes(kappa, theta, sigma, save=True, dpi=150):
    """
    Shows how yield curve shape depends on r0 vs theta.

    Three cases plotted together:
      r0 > theta  → downward sloping (inverted)
      r0 = theta  → nearly flat (slight hump from convexity)
      r0 < theta  → upward sloping (normal)

    Include as Figure 3 in your report.
    """
    T     = np.linspace(0.01, 20, 500)
    cases = [
        (theta * 2.5, 'crimson',    f'r₀ = {theta*2.5:.2%}  (r₀ > θ, inverted)'),
        (theta,       'gray',       f'r₀ = θ = {theta:.2%}  (r₀ = θ, flat)'),
        (theta * 0.2, 'steelblue',  f'r₀ = {theta*0.2:.2%}  (r₀ < θ, normal)'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for r0_case, color, label in cases:
        y = yield_curve(kappa, theta, sigma, r0_case, T)
        f = forward_rate(kappa, theta, sigma, r0_case, T)
        ax1.plot(T, y * 100, color=color, lw=2, label=label)
        ax2.plot(T, f * 100, color=color, lw=2, label=label)

    for ax in [ax1, ax2]:
        ax.axhline(theta * 100, color='black', lw=1, ls='--',
                   alpha=0.5, label=f'θ = {theta:.3%}')
        ax.axhline(long_run_yield(kappa, theta, sigma) * 100,
                   color='black', lw=1, ls=':', alpha=0.4,
                   label=f'y(∞) = {long_run_yield(kappa,theta,sigma):.3%}')
        ax.set_xlabel('Maturity T (years)', fontsize=11)
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25)

    ax1.set_ylabel('Yield (%)', fontsize=11)
    ax1.set_title('Yield curve shapes — varying r₀', fontsize=12)
    ax2.set_ylabel('Forward rate (%)', fontsize=11)
    ax2.set_title('Forward rate shapes — varying r₀', fontsize=12)

    fig.suptitle(
        f'Vasicek yield curve shapes  |  '
        f'κ={kappa:.5f}  θ={theta:.3%}  σ={sigma:.3%}',
        fontsize=13)
    plt.tight_layout()

    if save:
        fig.savefig('yield_curve_shapes.png', dpi=dpi, bbox_inches='tight')
        fig.savefig('yield_curve_shapes.pdf',           bbox_inches='tight')
        print("[plot] Saved yield_curve_shapes.png + .pdf")

    plt.close()
    return fig


def plot_sensitivity(kappa, theta, sigma, r0, save=True, dpi=150):
    """
    Four-panel sensitivity analysis.
    Each panel varies ONE parameter while keeping the others fixed.

    Shows how each parameter shapes the yield curve — good for your
    report's analysis section.
    """
    T    = np.linspace(0.01, 20, 500)
    base = yield_curve(kappa, theta, sigma, r0, T) * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    # ── panel 1: vary kappa ───────────────────────────────────────────
    ax = axes[0]
    for k, col in [(0.02, '#1a6eb5'), (0.05, '#3d9be9'),
                   (kappa, 'navy'), (0.30, '#f07030'), (1.00, '#c0392b')]:
        lbl = f'κ={k:.3f}' + (' ← calibrated' if k == kappa else '')
        ax.plot(T, yield_curve(k, theta, sigma, r0, T)*100,
                color=col, lw=2 if k==kappa else 1.2,
                ls='-' if k==kappa else '--', label=lbl)
    ax.set_title('Effect of κ (mean reversion speed)', fontsize=11)
    ax.set_ylabel('Yield (%)', fontsize=10)

    # ── panel 2: vary theta ───────────────────────────────────────────
    ax = axes[1]
    for th, col in [(0.005, '#1a6eb5'), (0.01, '#3d9be9'),
                    (theta, 'navy'), (0.03, '#f07030'), (0.05, '#c0392b')]:
        lbl = f'θ={th:.3%}' + (' ← calibrated' if th == theta else '')
        ax.plot(T, yield_curve(kappa, th, sigma, r0, T)*100,
                color=col, lw=2 if th==theta else 1.2,
                ls='-' if th==theta else '--', label=lbl)
    ax.set_title('Effect of θ (long-run mean)', fontsize=11)

    # ── panel 3: vary sigma ───────────────────────────────────────────
    ax = axes[2]
    for sg, col in [(0.002, '#1a6eb5'), (0.005, '#3d9be9'),
                    (sigma, 'navy'), (0.015, '#f07030'), (0.025, '#c0392b')]:
        lbl = f'σ={sg:.3%}' + (' ← calibrated' if sg == sigma else '')
        ax.plot(T, yield_curve(kappa, theta, sg, r0, T)*100,
                color=col, lw=2 if sg==sigma else 1.2,
                ls='-' if sg==sigma else '--', label=lbl)
    ax.set_title('Effect of σ (volatility)', fontsize=11)
    ax.set_ylabel('Yield (%)', fontsize=10)
    ax.set_xlabel('Maturity T (years)', fontsize=10)

    # ── panel 4: vary r0 ──────────────────────────────────────────────
    ax = axes[3]
    for r, col in [(0.005, '#1a6eb5'), (0.01, '#3d9be9'),
                   (r0, 'navy'), (0.05, '#f07030'), (0.08, '#c0392b')]:
        lbl = f'r₀={r:.3%}' + (' ← calibrated' if r == r0 else '')
        ax.plot(T, yield_curve(kappa, theta, sigma, r, T)*100,
                color=col, lw=2 if r==r0 else 1.2,
                ls='-' if r==r0 else '--', label=lbl)
    ax.set_title('Effect of r₀ (initial rate)', fontsize=11)
    ax.set_xlabel('Maturity T (years)', fontsize=10)

    for ax in axes:
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.22)

    fig.suptitle(
        'Vasicek yield curve — parameter sensitivity analysis',
        fontsize=13)
    plt.tight_layout()

    if save:
        fig.savefig('yield_sensitivity.png', dpi=dpi, bbox_inches='tight')
        fig.savefig('yield_sensitivity.pdf',           bbox_inches='tight')
        print("[plot] Saved yield_sensitivity.png + .pdf")

    plt.close()
    return fig


def plot_discount_factors(kappa, theta, sigma, r0, save=True, dpi=150):
    """
    Plot discount factors P(0,T) for a set of swap payment dates.
    This is the direct input your Phase 3 swap valuation needs.

    Shows the discount factors at quarterly intervals (T = 0.25, 0.5, ..., 5)
    — exactly the dates a 5-year quarterly swap would use.
    """
    # continuous curve
    T_cont = np.linspace(0.01, 5.5, 300)
    P_cont = bond_price(kappa, theta, sigma, r0, T_cont)

    # quarterly payment dates (as a swap would use)
    T_qtly = np.arange(0.25, 5.25, 0.25)
    P_qtly = bond_price(kappa, theta, sigma, r0, T_qtly)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(T_cont, P_cont, color='steelblue', lw=2,
            label='Discount factor P(0,T)')
    ax.scatter(T_qtly, P_qtly, color='navy', s=50, zorder=5,
               label='Quarterly payment dates')

    # annotate a few
    for i, (t, p) in enumerate(zip(T_qtly, P_qtly)):
        if t in [1.0, 2.0, 3.0, 5.0]:
            ax.annotate(f'P(0,{t:.0f})={p:.4f}',
                        xy=(t, p), xytext=(t+0.15, p+0.005),
                        fontsize=8, color='navy')

    ax.set_xlabel('Maturity T (years)', fontsize=11)
    ax.set_ylabel('Discount factor P(0,T)', fontsize=11)
    ax.set_title(
        'Vasicek discount factors at quarterly swap payment dates',
        fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    # param box
    ax.text(0.02, 0.05,
            f"κ={kappa:.5f}  θ={theta:.3%}  σ={sigma:.3%}  r₀={r0:.3%}",
            transform=ax.transAxes, fontsize=8.5,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4',
                      fc='white', alpha=0.85, ec='#cccccc'))
    plt.tight_layout()

    if save:
        fig.savefig('discount_factors.png', dpi=dpi, bbox_inches='tight')
        fig.savefig('discount_factors.pdf',           bbox_inches='tight')
        print("[plot] Saved discount_factors.png + .pdf")

    plt.close()
    return fig, T_qtly, P_qtly


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Run the full Phase 2 pipeline using calibrated DTB3 parameters.
    Returns (T_swap, P_swap) for use in Phase 3.
    """

    # ── calibrated parameters from Phase 1 ───────────────────────────
    kappa = 0.09611
    theta = 0.01189
    sigma = 0.00702
    r0    = 0.03630

    print("\n" + "█" * 54)
    print("  ES418 Group 13  —  Phase 2: Bond Pricing & Yield Curve")
    print("█" * 54)
    print(f"\n  Using calibrated parameters from DTB3 (2006–2026):")
    print(f"  κ={kappa:.5f} | θ={theta:.3%} | σ={sigma:.3%} | r₀={r0:.3%}")
    print(f"  Long-run yield y(∞) = {long_run_yield(kappa,theta,sigma):.4%}")

    # ── step 1: bond table ────────────────────────────────────────────
    print("\n[Step 1] Bond price / yield / forward rate table ...")
    P, y, f = print_bond_table(kappa, theta, sigma, r0,
                                label="Calibrated DTB3 parameters")

    # ── step 2: main yield curve plot ─────────────────────────────────
    print("[Step 2] Plotting yield curve ...")
    plot_yield_curve(kappa, theta, sigma, r0)

    # ── step 3: curve shape diagram ───────────────────────────────────
    print("[Step 3] Plotting curve shapes (normal / flat / inverted) ...")
    plot_curve_shapes(kappa, theta, sigma)

    # ── step 4: sensitivity analysis ─────────────────────────────────
    print("[Step 4] Plotting parameter sensitivity ...")
    plot_sensitivity(kappa, theta, sigma, r0)

    # ── step 5: discount factors for swap ────────────────────────────
    print("[Step 5] Plotting discount factors at quarterly dates ...")
    _, T_swap, P_swap = plot_discount_factors(kappa, theta, sigma, r0)

    # ── summary ───────────────────────────────────────────────────────
    print("\n" + "█" * 54)
    print("  Phase 2 complete.")
    print()
    print("  Saved files:")
    for fname in ['yield_curve', 'yield_curve_shapes',
                  'yield_sensitivity', 'discount_factors']:
        for ext in ['png', 'pdf']:
            path = f'{fname}.{ext}'
            if os.path.exists(path):
                kb = os.path.getsize(path) / 1024
                print(f"    {path:35s}  {kb:.1f} KB")
    print()
    print("  Next step: run phase3_swap.py")
    print("█" * 54 + "\n")

    return T_swap, P_swap, kappa, theta, sigma, r0


if __name__ == '__main__':
    T_swap, P_swap, kappa, theta, sigma, r0 = main()
