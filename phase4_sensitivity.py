"""
phase4_sensitivity.py
ES418 Group 13 — Interest Rate Swap Valuation
Phase 4: Swap value sensitivity — how NPV changes when rates shift.

WHAT THIS FILE DOES:
  1. DV01 — change in swap NPV per 1 basis point parallel shift in r0
  2. Rate sensitivity — NPV as a function of r0 (current rate level)
  3. Parameter sensitivity — NPV response to shifts in κ, θ, σ
  4. Duration approximation of the swap
  5. Saves all sensitivity plots for the report

KEY CONCEPTS:
  DV01  = NPV(r0 + 1bp) - NPV(r0)           (dollar value of 1bp)
  BPV   = |DV01|                             (basis point value)
  Dur   = -dNPV/dr0 / NPV                   (duration approximation)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from phase2.phase2_bonds import bond_price, yield_curve, forward_rate
from phase3.phase3_swap  import build_swap


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DV01 AND DURATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dv01(kappa, theta, sigma, r0,
                 tenor=5.0, freq=4, notional=1_000_000,
                 fixed_rate=None, bump=0.0001):
    """
    Compute DV01 — change in NPV per 1 basis point move in r0.

    Uses a central difference (bump up / bump down) for accuracy:
        DV01 = (NPV(r0+bump) - NPV(r0-bump)) / 2

    Parameters
    ----------
    bump : float — shift size (default 0.0001 = 1 bp)

    Returns
    -------
    dv01   : float — NPV change per 1bp (negative for fixed payer)
    npv    : float — base NPV
    npv_up : float — NPV after +1bp shift
    npv_dn : float — NPV after -1bp shift
    """
    swap_base = build_swap(kappa, theta, sigma, r0,
                           tenor, freq, notional, fixed_rate)
    K = swap_base['par_rate'] if fixed_rate is None else fixed_rate

    swap_up = build_swap(kappa, theta, sigma, r0 + bump,
                         tenor, freq, notional, K)
    swap_dn = build_swap(kappa, theta, sigma, r0 - bump,
                         tenor, freq, notional, K)

    dv01 = (swap_up['npv'] - swap_dn['npv']) / 2

    print(f"\n  DV01 Analysis (fixed rate = {K:.4%})")
    print(f"  Base NPV       : {swap_base['npv']:>14,.2f}")
    print(f"  NPV (+1bp)     : {swap_up['npv']:>14,.2f}")
    print(f"  NPV (-1bp)     : {swap_dn['npv']:>14,.2f}")
    print(f"  DV01           : {dv01:>14,.2f}  per 1bp shift in r₀")
    print(f"  BPV (|DV01|)   : {abs(dv01):>14,.2f}")

    return dv01, swap_base['npv'], swap_up['npv'], swap_dn['npv']


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SENSITIVITY SWEEPS
# ═══════════════════════════════════════════════════════════════════════════════

def npv_vs_r0(kappa, theta, sigma, fixed_rate,
              tenor=5.0, freq=4, notional=1_000_000,
              r0_range=None):
    """NPV as r0 varies — shows rate sensitivity profile."""
    if r0_range is None:
        r0_range = np.linspace(0.001, 0.12, 200)
    npvs = [build_swap(kappa, theta, sigma, r,
                       tenor, freq, notional,
                       fixed_rate=fixed_rate)['npv']
            for r in r0_range]
    return r0_range, np.array(npvs)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_rate_sensitivity(kappa, theta, sigma, r0,
                          tenor=5.0, freq=4, notional=1_000_000,
                          save=True, dpi=150):
    """
    Four-panel sensitivity figure:
      Top left  — NPV vs current rate r0
      Top right — NPV vs long-run mean theta
      Bot left  — NPV vs mean-reversion speed kappa
      Bot right — DV01 across different r0 levels
    """
    swap_base = build_swap(kappa, theta, sigma, r0,
                           tenor, freq, notional)
    K = swap_base['par_rate']   # keep fixed rate constant

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # ── panel 1: NPV vs r0 ────────────────────────────────────────────
    r0_grid = np.linspace(0.001, 0.12, 200)
    npv_r0  = [build_swap(kappa, theta, sigma, r,
                          tenor, freq, notional, K)['npv']
               for r in r0_grid]
    ax = axes[0]
    ax.plot(r0_grid*100, np.array(npv_r0)/1000, color='steelblue', lw=2)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axvline(r0*100, color='crimson', lw=1.2, ls='--',
               label=f'Current r₀ = {r0:.3%}')
    ax.axvline(K*100, color='darkorange', lw=1.2, ls=':',
               label=f'Fixed rate K = {K:.3%}')
    ax.fill_between(r0_grid*100, np.array(npv_r0)/1000, 0,
                    where=np.array(npv_r0)>0,
                    alpha=0.1, color='green')
    ax.fill_between(r0_grid*100, np.array(npv_r0)/1000, 0,
                    where=np.array(npv_r0)<0,
                    alpha=0.1, color='red')
    ax.set_xlabel('Current rate r₀ (%)', fontsize=11)
    ax.set_ylabel('Swap NPV (thousands)', fontsize=11)
    ax.set_title('NPV vs current short rate r₀', fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # ── panel 2: NPV vs theta ─────────────────────────────────────────
    th_grid = np.linspace(0.001, 0.08, 200)
    npv_th  = [build_swap(kappa, th, sigma, r0,
                          tenor, freq, notional, K)['npv']
               for th in th_grid]
    ax = axes[1]
    ax.plot(th_grid*100, np.array(npv_th)/1000, color='darkorange', lw=2)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axvline(theta*100, color='crimson', lw=1.2, ls='--',
               label=f'Calibrated θ = {theta:.3%}')
    ax.set_xlabel('Long-run mean θ (%)', fontsize=11)
    ax.set_ylabel('Swap NPV (thousands)', fontsize=11)
    ax.set_title('NPV vs long-run mean θ', fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # ── panel 3: NPV vs kappa ─────────────────────────────────────────
    k_grid  = np.linspace(0.01, 2.0, 200)
    npv_k   = [build_swap(k, theta, sigma, r0,
                          tenor, freq, notional, K)['npv']
               for k in k_grid]
    ax = axes[2]
    ax.plot(k_grid, np.array(npv_k)/1000, color='teal', lw=2)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axvline(kappa, color='crimson', lw=1.2, ls='--',
               label=f'Calibrated κ = {kappa:.4f}')
    ax.set_xlabel('Mean reversion speed κ', fontsize=11)
    ax.set_ylabel('Swap NPV (thousands)', fontsize=11)
    ax.set_title('NPV vs mean reversion speed κ', fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # ── panel 4: DV01 across r0 levels ────────────────────────────────
    bump    = 0.0001
    dv01_arr = []
    for r in r0_grid:
        s_up = build_swap(kappa, theta, sigma, r+bump,
                          tenor, freq, notional, K)['npv']
        s_dn = build_swap(kappa, theta, sigma, r-bump,
                          tenor, freq, notional, K)['npv']
        dv01_arr.append((s_up - s_dn) / 2)
    ax = axes[3]
    ax.plot(r0_grid*100, np.array(dv01_arr),
            color='purple', lw=2)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axvline(r0*100, color='crimson', lw=1.2, ls='--',
               label=f'Current r₀ = {r0:.3%}')
    ax.set_xlabel('Current rate r₀ (%)', fontsize=11)
    ax.set_ylabel('DV01 (per 1bp)', fontsize=11)
    ax.set_title('DV01 across rate levels', fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    fig.suptitle(
        f'Swap Sensitivity Analysis  |  '
        f'N={notional:,.0f}  K={K:.3%}  Tenor={tenor}yr',
        fontsize=13)
    plt.tight_layout()

    if save:
        fig.savefig('swap_sensitivity.png', dpi=dpi, bbox_inches='tight')
        fig.savefig('swap_sensitivity.pdf',           bbox_inches='tight')
        print("[plot] Saved swap_sensitivity.png + .pdf")
    plt.close()


def plot_rate_shift_scenarios(kappa, theta, sigma, r0,
                              tenor=5.0, freq=4, notional=1_000_000,
                              save=True, dpi=150):
    """
    Bar chart showing NPV under parallel rate shift scenarios:
    -200bp, -100bp, -50bp, 0, +50bp, +100bp, +200bp
    """
    swap_base = build_swap(kappa, theta, sigma, r0,
                           tenor, freq, notional)
    K = swap_base['par_rate']

    shifts  = [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]
    labels  = ['-200bp', '-100bp', '-50bp', '0bp',
               '+50bp', '+100bp', '+200bp']
    npvs    = [build_swap(kappa, theta, sigma, max(r0+s, 0.0001),
                          tenor, freq, notional, K)['npv']
               for s in shifts]
    colors  = ['#1a6eb5' if v >= 0 else '#c0392b' for v in npvs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, [v/1000 for v in npvs],
                  color=colors, alpha=0.8, edgecolor='white')
    ax.axhline(0, color='black', lw=0.8)
    for bar, val in zip(bars, npvs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (max(npvs)/1000*0.02),
                f'{val/1000:,.1f}k',
                ha='center', va='bottom', fontsize=8.5)
    ax.set_xlabel('Parallel rate shift', fontsize=11)
    ax.set_ylabel('Swap NPV (thousands)', fontsize=11)
    ax.set_title(
        f'Swap NPV under parallel rate shifts  |  '
        f'K={K:.3%}  r₀={r0:.3%}  N={notional:,.0f}',
        fontsize=12)
    ax.grid(alpha=0.25, axis='y')
    plt.tight_layout()

    if save:
        fig.savefig('swap_scenarios.png', dpi=dpi, bbox_inches='tight')
        fig.savefig('swap_scenarios.pdf',           bbox_inches='tight')
        print("[plot] Saved swap_scenarios.png + .pdf")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    kappa    = 0.09611
    theta    = 0.01189
    sigma    = 0.00702
    r0       = 0.03630
    notional = 1_000_000

    print("\n" + "█" * 54)
    print("  ES418 Group 13  —  Phase 4: Sensitivity Analysis")
    print("█" * 54)

    print("\n[Step 1] DV01 computation ...")
    dv01, npv0, npv_up, npv_dn = compute_dv01(
        kappa, theta, sigma, r0, notional=notional)

    print("\n[Step 2] Rate shift scenarios ...")
    swap_par = build_swap(kappa, theta, sigma, r0, notional=notional)
    K = swap_par['par_rate']
    shifts = [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]
    labels = ['-200bp', '-100bp', '-50bp', '0', '+50bp', '+100bp', '+200bp']
    print(f"\n  {'Shift':>8} │ {'r₀_new':>8} │ {'NPV':>14}")
    print("  " + "─" * 36)
    for s, lb in zip(shifts, labels):
        r_new = max(r0+s, 0.0001)
        npv_s = build_swap(kappa, theta, sigma, r_new,
                           notional=notional, fixed_rate=K)['npv']
        print(f"  {lb:>8} │ {r_new:>8.4%} │ {npv_s:>14,.2f}")

    print("\n[Step 3] Plotting sensitivity ...")
    plot_rate_sensitivity(kappa, theta, sigma, r0, notional=notional)

    print("[Step 4] Plotting rate shift scenarios ...")
    plot_rate_shift_scenarios(kappa, theta, sigma, r0, notional=notional)

    print("\n" + "█" * 54)
    print("  Phase 4 complete.")
    for fname in ['swap_sensitivity', 'swap_scenarios']:
        for ext in ['png', 'pdf']:
            path = f'{fname}.{ext}'
            if os.path.exists(path):
                kb = os.path.getsize(path)/1024
                print(f"    {path:35s}  {kb:.1f} KB")
    print("  Next step: run phase5_montecarlo.py")
    print("█" * 54 + "\n")


if __name__ == '__main__':
    main()
