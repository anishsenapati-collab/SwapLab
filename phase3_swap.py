"""
phase3_swap.py
ES418 Group 13 — Interest Rate Swap Valuation
Phase 3: Plain vanilla interest rate swap — cash flows and NPV valuation.

WHAT THIS FILE DOES:
  1. Constructs fixed and floating leg cash flows for a plain vanilla IRS
  2. Values each leg using Vasicek discount factors from Phase 2
  3. Computes swap NPV = PV(floating) - PV(fixed)
  4. Finds the par swap rate K* where NPV = 0
  5. Prints a full cash flow table and saves valuation plots

SWAP STRUCTURE:
  - Notional         : N (default 1,000,000)
  - Tenor            : 5 years
  - Payment frequency: quarterly (every 0.25 years)
  - Fixed leg pays   : K * dt * N  at each payment date
  - Floating leg pays: r(t_i) * dt * N  (set at start of each period)
  - Day count        : Actual/Actual (approximated as dt = 0.25)

KEY FORMULAS:
  PV(fixed)    = K * sum( dt * P(0, t_i) ) * N
  PV(floating) = ( P(0, t_0) - P(0, t_n) ) * N
  NPV          = PV(floating) - PV(fixed)
  Par rate K*  = ( P(0,t_0) - P(0,t_n) ) / sum( dt * P(0,t_i) )

CALIBRATED PARAMETERS FROM DTB3 (2006–2026):
  kappa=0.09611  theta=0.01189  sigma=0.00702  r0=0.03630
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from phase2.phase2_bonds import bond_price, yield_curve, forward_rate, long_run_yield


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SWAP CONSTRUCTOR
# ═══════════════════════════════════════════════════════════════════════════════

def build_swap(kappa, theta, sigma, r0,
               tenor=5.0, freq=4, notional=1_000_000, fixed_rate=None):
    """
    Build a plain vanilla fixed-for-floating interest rate swap.

    Parameters
    ----------
    kappa, theta, sigma, r0 : Vasicek parameters
    tenor     : float — swap tenor in years (default 5)
    freq      : int   — payments per year (4 = quarterly)
    notional  : float — notional principal
    fixed_rate: float — fixed coupon rate K (if None, uses par rate)

    Returns
    -------
    dict with all swap details and cash flows
    """
    dt         = 1.0 / freq                         # length of each period
    n_payments = int(tenor * freq)                   # total number of payments
    t_payments = np.arange(1, n_payments+1) * dt     # payment dates t1,...,tn
    t0         = 0.0                                 # swap start

    # ── discount factors at each payment date ─────────────────────────
    P = bond_price(kappa, theta, sigma, r0, t_payments)
    P0 = bond_price(kappa, theta, sigma, r0, np.array([t0 + dt]))[0]
    # P(0,t0) = 1 when t0=0 (today)
    P_start = 1.0   # P(0, 0) = 1 always

    # ── floating leg ──────────────────────────────────────────────────
    # PV(floating) = P(0,t0) - P(0,tn)  [replication argument]
    pv_floating = (P_start - P[-1]) * notional

    # floating cash flows: implied forward rate for each period
    fwd_rates = forward_rate(kappa, theta, sigma, r0, t_payments)
    cf_floating = fwd_rates * dt * notional
    pv_cf_floating = cf_floating * P    # discounted

    # ── fixed leg ─────────────────────────────────────────────────────
    annuity = np.sum(dt * P)            # A = sum of dt * P(0,ti)

    # par swap rate: K* = (P(0,t0) - P(0,tn)) / annuity
    par_rate = (P_start - P[-1]) / annuity

    # use provided fixed rate or par rate
    K = fixed_rate if fixed_rate is not None else par_rate

    cf_fixed    = K * dt * notional * np.ones(n_payments)
    pv_cf_fixed = cf_fixed * P          # discounted
    pv_fixed    = np.sum(pv_cf_fixed)

    # ── NPV ───────────────────────────────────────────────────────────
    # from the fixed-rate payer's perspective:
    # NPV = PV(floating received) - PV(fixed paid)
    npv = pv_floating - pv_fixed

    return {
        # schedule
        't_payments'    : t_payments,
        'dt'            : dt,
        'n_payments'    : n_payments,
        'tenor'         : tenor,
        'freq'          : freq,
        'notional'      : notional,
        # rates
        'fixed_rate'    : K,
        'par_rate'      : par_rate,
        'fwd_rates'     : fwd_rates,
        # discount factors
        'P'             : P,
        'annuity'       : annuity,
        # cash flows
        'cf_fixed'      : cf_fixed,
        'cf_floating'   : cf_floating,
        'pv_cf_fixed'   : pv_cf_fixed,
        'pv_cf_floating': pv_cf_floating,
        # PVs
        'pv_fixed'      : pv_fixed,
        'pv_floating'   : pv_floating,
        'npv'           : npv,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_swap_table(swap):
    """Print a detailed cash flow table for the swap."""
    t   = swap['t_payments']
    P   = swap['P']
    K   = swap['fixed_rate']
    fwd = swap['fwd_rates']
    cf_fix = swap['cf_fixed']
    cf_flt = swap['cf_floating']
    pv_fix = swap['pv_cf_fixed']
    pv_flt = swap['pv_cf_floating']
    N   = swap['notional']
    dt  = swap['dt']

    print("\n" + "=" * 90)
    print(f"  Interest Rate Swap — Cash Flow Schedule")
    print(f"  Notional={N:,.0f} | Fixed rate={K:.4%} | "
          f"Par rate={swap['par_rate']:.4%} | Tenor={swap['tenor']}yr | "
          f"Freq={swap['freq']}/yr")
    print("=" * 90)
    print(f"  {'Date':>6} │ {'P(0,t)':>8} │ {'Fwd rate':>9} │ "
          f"{'CF Fixed':>12} │ {'CF Float':>12} │ "
          f"{'PV Fixed':>12} │ {'PV Float':>12} │ {'Net PV':>12}")
    print("  " + "─" * 87)

    for i in range(len(t)):
        net_pv = pv_flt[i] - pv_fix[i]
        print(f"  {t[i]:>6.2f} │ {P[i]:>8.6f} │ {fwd[i]:>9.4%} │ "
              f"{cf_fix[i]:>12,.2f} │ {cf_flt[i]:>12,.2f} │ "
              f"{pv_fix[i]:>12,.2f} │ {pv_flt[i]:>12,.2f} │ "
              f"{net_pv:>12,.2f}")

    print("  " + "─" * 87)
    print(f"  {'TOTAL':>6} │ {'':>8} │ {'':>9} │ "
          f"{swap['cf_fixed'].sum():>12,.2f} │ "
          f"{swap['cf_floating'].sum():>12,.2f} │ "
          f"{swap['pv_fixed']:>12,.2f} │ "
          f"{swap['pv_floating']:>12,.2f} │ "
          f"{swap['npv']:>12,.2f}")
    print("=" * 90)
    print(f"\n  PV (fixed leg)    : {swap['pv_fixed']:>14,.2f}")
    print(f"  PV (floating leg) : {swap['pv_floating']:>14,.2f}")
    print(f"  Swap NPV          : {swap['npv']:>14,.2f}  "
          f"({'payer gains' if swap['npv']>0 else 'payer loses'})")
    print(f"  Par swap rate K*  : {swap['par_rate']:.6%}")
    print(f"  Annuity factor    : {swap['annuity']:.6f}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cashflows(swap, save=True, dpi=150):
    """Bar chart of fixed vs floating cash flows and their present values."""
    t       = swap['t_payments']
    cf_fix  = swap['cf_fixed']
    cf_flt  = swap['cf_floating']
    pv_fix  = swap['pv_cf_fixed']
    pv_flt  = swap['pv_cf_floating']
    width   = 0.09

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── left: raw cash flows ──────────────────────────────────────────
    ax1.bar(t - width/2, cf_fix/1000, width, color='steelblue',
            alpha=0.8, label='Fixed leg')
    ax1.bar(t + width/2, cf_flt/1000, width, color='darkorange',
            alpha=0.8, label='Floating leg')
    ax1.set_xlabel('Payment date (years)', fontsize=11)
    ax1.set_ylabel('Cash flow (thousands)', fontsize=11)
    ax1.set_title('Swap cash flows — fixed vs floating', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.25, axis='y')
    ax1.axhline(0, color='black', lw=0.8)

    # ── right: present values ─────────────────────────────────────────
    ax2.bar(t - width/2, pv_fix/1000, width, color='steelblue',
            alpha=0.8, label='PV fixed')
    ax2.bar(t + width/2, pv_flt/1000, width, color='darkorange',
            alpha=0.8, label='PV floating')
    ax2.set_xlabel('Payment date (years)', fontsize=11)
    ax2.set_ylabel('Present value (thousands)', fontsize=11)
    ax2.set_title('Present values of cash flows', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.25, axis='y')
    ax2.axhline(0, color='black', lw=0.8)

    K  = swap['fixed_rate']
    K_ = swap['par_rate']
    N  = swap['notional']
    fig.suptitle(
        f"IRS Cash Flows  |  N={N:,.0f}  K={K:.3%}  "
        f"K*={K_:.3%}  NPV={swap['npv']:,.0f}",
        fontsize=13)
    plt.tight_layout()

    if save:
        fig.savefig('swap_cashflows.png', dpi=dpi, bbox_inches='tight')
        fig.savefig('swap_cashflows.pdf',           bbox_inches='tight')
        print("[plot] Saved swap_cashflows.png + .pdf")
    plt.close()


def plot_npv_vs_rate(kappa, theta, sigma, r0,
                     tenor=5.0, freq=4, notional=1_000_000,
                     save=True, dpi=150):
    """
    Plot swap NPV as a function of the fixed rate K.
    The zero crossing is the par swap rate K*.
    """
    swap_par  = build_swap(kappa, theta, sigma, r0, tenor, freq, notional)
    K_star    = swap_par['par_rate']
    K_range   = np.linspace(K_star * 0.3, K_star * 1.8, 200)
    npvs      = [build_swap(kappa, theta, sigma, r0,
                            tenor, freq, notional,
                            fixed_rate=k)['npv'] for k in K_range]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(K_range * 100, np.array(npvs) / 1000,
            color='steelblue', lw=2)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axvline(K_star * 100, color='crimson', lw=1.3, ls='--',
               label=f'Par rate K* = {K_star:.4%}')
    ax.fill_between(K_range*100, np.array(npvs)/1000, 0,
                    where=np.array(npvs) > 0,
                    alpha=0.12, color='green', label='Payer gains')
    ax.fill_between(K_range*100, np.array(npvs)/1000, 0,
                    where=np.array(npvs) < 0,
                    alpha=0.12, color='crimson', label='Payer loses')
    ax.set_xlabel('Fixed rate K (%)', fontsize=11)
    ax.set_ylabel('NPV (thousands)', fontsize=11)
    ax.set_title('Swap NPV vs fixed rate K  (fixed-rate payer perspective)',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if save:
        fig.savefig('swap_npv_vs_rate.png', dpi=dpi, bbox_inches='tight')
        fig.savefig('swap_npv_vs_rate.pdf',           bbox_inches='tight')
        print("[plot] Saved swap_npv_vs_rate.png + .pdf")
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
    print("  ES418 Group 13  —  Phase 3: Swap Valuation")
    print("█" * 54)

    # ── at-market swap (K = par rate) ─────────────────────────────────
    print("\n[Step 1] Building at-market swap (K = K*) ...")
    swap_par = build_swap(kappa, theta, sigma, r0,
                          tenor=5.0, freq=4, notional=notional)
    print_swap_table(swap_par)

    # ── off-market swap (K fixed at 4%) ───────────────────────────────
    print("[Step 2] Building off-market swap (K = 4.00%) ...")
    swap_off = build_swap(kappa, theta, sigma, r0,
                          tenor=5.0, freq=4, notional=notional,
                          fixed_rate=0.04)
    print_swap_table(swap_off)

    # ── plots ─────────────────────────────────────────────────────────
    print("[Step 3] Plotting cash flows ...")
    plot_cashflows(swap_off)

    print("[Step 4] Plotting NPV vs fixed rate ...")
    plot_npv_vs_rate(kappa, theta, sigma, r0)

    print("\n" + "█" * 54)
    print("  Phase 3 complete.")
    for fname in ['swap_cashflows', 'swap_npv_vs_rate']:
        for ext in ['png', 'pdf']:
            path = f'{fname}.{ext}'
            if os.path.exists(path):
                kb = os.path.getsize(path) / 1024
                print(f"    {path:35s}  {kb:.1f} KB")
    print("  Next step: run phase4_sensitivity.py")
    print("█" * 54 + "\n")

    return swap_par, swap_off


if __name__ == '__main__':
    swap_par, swap_off = main()
