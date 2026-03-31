"""
phase5_montecarlo.py
ES418 Group 13 — Interest Rate Swap Valuation
Phase 5: Monte Carlo valuation of the swap using simulated rate paths.

WHAT THIS FILE DOES:
  1. Uses Vasicek simulated paths from Phase 1 to value the swap stochastically
  2. At each simulated path, computes floating cash flows from realised rates
  3. Discounts using path-consistent discount factors
  4. Produces a distribution of swap NPVs across all paths
  5. Compares Monte Carlo NPV vs closed-form NPV from Phase 3

WHY THIS MATTERS:
  Phase 3 gave a single NPV using expected (risk-neutral) values.
  Phase 5 shows the distribution of possible outcomes — crucial for
  understanding risk, not just expected value. The mean of the Monte Carlo
  distribution should match the closed-form result (convergence check).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from phase2.phase2_bonds import bond_price
from phase3_swap  import build_swap
from phase1.vasicek_simulator import VasicekConfig, vasicek_simulate


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  MONTE CARLO SWAP VALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def mc_swap_value(r_paths, t_grid, kappa, theta, sigma,
                  fixed_rate, tenor=5.0, freq=4,
                  notional=1_000_000):
    """
    Value the swap on each simulated path.

    For each path:
      - Floating CF at t_i = r(t_{i-1}) * dt * N   (rate set at period start)
      - Discount each CF by the path-average discount:
            D(t_i) = exp( -mean(r) * t_i )  [Euler approximation]
      - NPV_path = PV(float) - PV(fixed)

    Parameters
    ----------
    r_paths    : ndarray (n_paths, N+1)  — simulated rates from Phase 1
    t_grid     : ndarray (N+1,)          — time grid
    fixed_rate : float                   — fixed leg coupon K
    notional   : float

    Returns
    -------
    npv_paths : ndarray (n_paths,)  — NPV on each path
    """
    dt          = 1.0 / freq
    n_payments  = int(tenor * freq)
    t_payments  = np.arange(1, n_payments+1) * dt   # t1 ... tn
    n_paths     = r_paths.shape[0]
    npv_paths   = np.zeros(n_paths)

    for p in range(n_paths):
        path     = r_paths[p]
        pv_fix   = 0.0
        pv_flt   = 0.0

        for i, t_i in enumerate(t_payments):
            # find index in time grid closest to t_i
            idx_i    = np.argmin(np.abs(t_grid - t_i))
            idx_prev = max(0, np.argmin(np.abs(t_grid - (t_i - dt))))

            # floating rate = rate at START of period (set-in-advance)
            r_set    = path[idx_prev]

            # path-average discount factor (trapezoidal rule on log)
            idx_end  = idx_i
            avg_r    = np.mean(path[:idx_end+1])
            D_i      = np.exp(-avg_r * t_i)

            cf_flt   = r_set * dt * notional
            cf_fix   = fixed_rate * dt * notional

            pv_flt  += cf_flt * D_i
            pv_fix  += cf_fix * D_i

        npv_paths[p] = pv_flt - pv_fix

    return npv_paths


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mc_distribution(npv_paths, npv_closedform,
                         fixed_rate, notional, save=True, dpi=150):
    """Histogram of Monte Carlo NPVs with closed-form benchmark."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(npv_paths/1000, bins=60, color='steelblue',
            alpha=0.65, edgecolor='white', lw=0.3,
            density=True, label='MC NPV distribution')

    ax.axvline(npv_paths.mean()/1000, color='navy', lw=2, ls='-',
               label=f'MC mean = {npv_paths.mean()/1000:,.1f}k')
    ax.axvline(npv_closedform/1000, color='crimson', lw=2, ls='--',
               label=f'Closed-form = {npv_closedform/1000:,.1f}k')

    pct5  = np.percentile(npv_paths, 5)
    pct95 = np.percentile(npv_paths, 95)
    ax.axvline(pct5/1000,  color='darkorange', lw=1.2, ls=':',
               label=f'5th pct = {pct5/1000:,.1f}k')
    ax.axvline(pct95/1000, color='darkgreen', lw=1.2, ls=':',
               label=f'95th pct = {pct95/1000:,.1f}k')

    ax.set_xlabel('Swap NPV (thousands)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(
        f'Monte Carlo swap NPV distribution  |  '
        f'K={fixed_rate:.3%}  N={notional:,.0f}  '
        f'n_paths={len(npv_paths):,}',
        fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if save:
        fig.savefig('mc_npv_distribution.png', dpi=dpi, bbox_inches='tight')
        fig.savefig('mc_npv_distribution.pdf',           bbox_inches='tight')
        print("[plot] Saved mc_npv_distribution.png + .pdf")
    plt.close()


def plot_mc_convergence(npv_paths, npv_closedform,
                        save=True, dpi=150):
    """Running mean of MC NPV — shows convergence to closed-form value."""
    running_mean = np.cumsum(npv_paths) / np.arange(1, len(npv_paths)+1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(running_mean/1000, color='steelblue', lw=1.2,
            label='Running MC mean')
    ax.axhline(npv_closedform/1000, color='crimson', lw=1.5, ls='--',
               label=f'Closed-form = {npv_closedform/1000:,.2f}k')
    ax.set_xlabel('Number of paths', fontsize=11)
    ax.set_ylabel('Running mean NPV (thousands)', fontsize=11)
    ax.set_title('Monte Carlo convergence to closed-form NPV', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if save:
        fig.savefig('mc_convergence.png', dpi=dpi, bbox_inches='tight')
        fig.savefig('mc_convergence.pdf',           bbox_inches='tight')
        print("[plot] Saved mc_convergence.png + .pdf")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    kappa    = 0.09611
    theta    = 0.01189
    sigma    = 0.00702
    r0       = 0.03630
    notional = 1_000_000

    print("\n" + "█" * 54)
    print("  ES418 Group 13  —  Phase 5: Monte Carlo Valuation")
    print("█" * 54)

    # ── closed-form benchmark ─────────────────────────────────────────
    swap_par = build_swap(kappa, theta, sigma, r0,
                          tenor=5.0, freq=4, notional=notional)
    K        = swap_par['par_rate']
    npv_cf   = swap_par['npv']
    print(f"\n  Closed-form par rate K* = {K:.5%}")
    print(f"  Closed-form NPV         = {npv_cf:,.2f}  (should be ~0)")

    # ── simulate paths ────────────────────────────────────────────────
    print("\n[Step 1] Simulating rate paths ...")
    cfg = VasicekConfig(
        kappa=kappa, theta=theta, sigma=sigma, r0=r0,
        T=5.0, N=1260, n_paths=2000, seed=42
    )
    r_paths, t_grid = vasicek_simulate(cfg, antithetic=True)

    # ── Monte Carlo valuation ─────────────────────────────────────────
    print("\n[Step 2] Valuing swap on each path ...")
    # use a slightly off-market rate to get non-zero NPV
    K_offmarket = K + 0.005   # K* + 50bp
    npv_paths   = mc_swap_value(
        r_paths, t_grid, kappa, theta, sigma,
        fixed_rate=K_offmarket, notional=notional
    )
    npv_cf_off = build_swap(kappa, theta, sigma, r0,
                            notional=notional,
                            fixed_rate=K_offmarket)['npv']

    print(f"\n  Fixed rate used     : {K_offmarket:.4%}  (K* + 50bp)")
    print(f"  Closed-form NPV     : {npv_cf_off:>12,.2f}")
    print(f"  MC mean NPV         : {npv_paths.mean():>12,.2f}")
    print(f"  MC std NPV          : {npv_paths.std():>12,.2f}")
    print(f"  MC 5th percentile   : {np.percentile(npv_paths,5):>12,.2f}")
    print(f"  MC 95th percentile  : {np.percentile(npv_paths,95):>12,.2f}")
    print(f"  Convergence error   : "
          f"{abs(npv_paths.mean()-npv_cf_off)/abs(npv_cf_off)*100:.3f}%")

    # ── plots ─────────────────────────────────────────────────────────
    print("\n[Step 3] Plotting MC distribution ...")
    plot_mc_distribution(npv_paths, npv_cf_off, K_offmarket, notional)

    print("[Step 4] Plotting MC convergence ...")
    plot_mc_convergence(npv_paths, npv_cf_off)

    print("\n" + "█" * 54)
    print("  Phase 5 complete.")
    for fname in ['mc_npv_distribution', 'mc_convergence']:
        for ext in ['png', 'pdf']:
            path = f'{fname}.{ext}'
            if os.path.exists(path):
                kb = os.path.getsize(path)/1024
                print(f"    {path:35s}  {kb:.1f} KB")
    print("  Next step: run phase6_summary.py")
    print("█" * 54 + "\n")

    return npv_paths, npv_cf_off


if __name__ == '__main__':
    npv_paths, npv_cf = main()
