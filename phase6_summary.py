"""
phase6_summary.py
ES418 Group 13 — Interest Rate Swap Valuation
Phase 6: Full pipeline summary — runs all phases and produces final outputs.

RUN THIS FILE to execute the complete project end-to-end:
    python phase6_summary.py

OUTPUT:
  All figures from Phases 1–5 saved as PNG + PDF
  Final summary table printed to console
  summary_results.txt — machine-readable results for the report
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, time

# ── imports from all phases ───────────────────────────────────────────────────
from phase1.data_loader        import load_dtb3, describe_data, plot_raw_data
from phase1.vasicek_simulator  import (VasicekConfig, calibrate_vasicek,
                                 vasicek_simulate, print_summary,
                                 plot_paths, plot_validation, plot_terminal)
from phase2.phase2_bonds       import (bond_price, yield_curve, forward_rate,
                                 long_run_yield, print_bond_table,
                                 plot_yield_curve, plot_curve_shapes,
                                 plot_sensitivity, plot_discount_factors)
from phase3_swap        import (build_swap, print_swap_table,
                                 plot_cashflows, plot_npv_vs_rate)
from phase4_sensitivity import (compute_dv01, plot_rate_sensitivity,
                                 plot_rate_shift_scenarios)
from phase5_montecarlo  import (mc_swap_value, plot_mc_distribution,
                                 plot_mc_convergence)


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_all(data_file='DTB3.csv', notional=1_000_000):

    start_time = time.time()

    print("\n" + "█" * 60)
    print("  ES418 Group 13  —  Interest Rate Swap Valuation")
    print("  Full Pipeline — Phases 1 through 5")
    print("█" * 60)

    results = {}   # collect all key numbers for final summary

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1 — Data & Simulation
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("  PHASE 1: Vasicek Model Calibration & Simulation")
    print("─"*60)

    if not os.path.exists(data_file):
        print(f"  ERROR: {data_file} not found.")
        sys.exit(1)

    rates, dates = load_dtb3(data_file)
    describe_data(rates)
    plot_raw_data(rates, dates, save=True)

    cfg = calibrate_vasicek(rates, dt=1/252)
    cfg.T       = 5.0
    cfg.N       = 1260
    cfg.n_paths = 2000
    cfg.seed    = 42
    cfg.save_plots = True
    cfg.summary()

    r, t = vasicek_simulate(cfg, antithetic=True)
    print_summary(r, t, cfg)
    plot_paths(r, t, cfg,
               data_start_year=dates[0].year,
               data_end_year=dates[-1].year)
    plot_validation(r, t, cfg)
    plot_terminal(r, cfg)

    results['kappa'] = cfg.kappa
    results['theta'] = cfg.theta
    results['sigma'] = cfg.sigma
    results['r0']    = cfg.r0
    results['half_life'] = cfg.half_life
    results['n_obs'] = len(rates)
    results['date_start'] = str(dates[0].date())
    results['date_end']   = str(dates[-1].date())

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2 — Bond Pricing & Yield Curve
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("  PHASE 2: Bond Pricing & Yield Curve")
    print("─"*60)

    kappa = cfg.kappa
    theta = cfg.theta
    sigma = cfg.sigma
    r0    = cfg.r0

    P_arr, y_arr, f_arr = print_bond_table(
        kappa, theta, sigma, r0,
        label="Calibrated DTB3 parameters")
    plot_yield_curve(kappa, theta, sigma, r0)
    plot_curve_shapes(kappa, theta, sigma)
    plot_sensitivity(kappa, theta, sigma, r0)
    _, T_swap, P_swap = plot_discount_factors(kappa, theta, sigma, r0)

    lry = long_run_yield(kappa, theta, sigma)
    results['lry']        = lry
    results['P_1yr']      = float(bond_price(kappa, theta, sigma, r0, 1.0))
    results['P_5yr']      = float(bond_price(kappa, theta, sigma, r0, 5.0))
    results['yield_1yr']  = float(yield_curve(kappa, theta, sigma, r0, 1.0))
    results['yield_5yr']  = float(yield_curve(kappa, theta, sigma, r0, 5.0))

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3 — Swap Valuation
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("  PHASE 3: Interest Rate Swap Valuation")
    print("─"*60)

    swap_par = build_swap(kappa, theta, sigma, r0,
                          tenor=5.0, freq=4, notional=notional)
    print_swap_table(swap_par)

    swap_off = build_swap(kappa, theta, sigma, r0,
                          tenor=5.0, freq=4, notional=notional,
                          fixed_rate=swap_par['par_rate'] + 0.005)
    print_swap_table(swap_off)

    plot_cashflows(swap_off)
    plot_npv_vs_rate(kappa, theta, sigma, r0, notional=notional)

    results['par_rate']   = swap_par['par_rate']
    results['pv_fixed']   = swap_par['pv_fixed']
    results['pv_float']   = swap_par['pv_floating']
    results['npv_par']    = swap_par['npv']
    results['npv_off']    = swap_off['npv']
    results['annuity']    = swap_par['annuity']

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4 — Sensitivity Analysis
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("  PHASE 4: Sensitivity Analysis")
    print("─"*60)

    dv01, npv0, npv_up, npv_dn = compute_dv01(
        kappa, theta, sigma, r0, notional=notional)
    plot_rate_sensitivity(kappa, theta, sigma, r0, notional=notional)
    plot_rate_shift_scenarios(kappa, theta, sigma, r0, notional=notional)

    results['dv01'] = dv01

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5 — Monte Carlo Valuation
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("  PHASE 5: Monte Carlo Swap Valuation")
    print("─"*60)

    K_off   = swap_par['par_rate'] + 0.005
    npv_cf  = build_swap(kappa, theta, sigma, r0,
                         notional=notional,
                         fixed_rate=K_off)['npv']
    npv_mc  = mc_swap_value(r, t, kappa, theta, sigma,
                            fixed_rate=K_off, notional=notional)

    print(f"\n  MC mean NPV       : {npv_mc.mean():>12,.2f}")
    print(f"  Closed-form NPV   : {npv_cf:>12,.2f}")
    print(f"  Convergence error : "
          f"{abs(npv_mc.mean()-npv_cf)/abs(npv_cf)*100:.3f}%")

    plot_mc_distribution(npv_mc, npv_cf, K_off, notional)
    plot_mc_convergence(npv_mc, npv_cf)

    results['mc_mean']   = float(npv_mc.mean())
    results['mc_std']    = float(npv_mc.std())
    results['mc_p5']     = float(np.percentile(npv_mc, 5))
    results['mc_p95']    = float(np.percentile(npv_mc, 95))
    results['mc_cf']     = float(npv_cf)

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    _print_final_summary(results, notional, elapsed)
    _save_results_txt(results, notional)
    _list_all_outputs()


def _print_final_summary(res, notional, elapsed):
    print("\n" + "█" * 60)
    print("  FINAL SUMMARY — ES418 Group 13")
    print("█" * 60)

    print(f"\n  ── Data ──────────────────────────────────────────────")
    print(f"  Source            : FRED DTB3 (3-Month T-Bill)")
    print(f"  Period            : {res['date_start']} → {res['date_end']}")
    print(f"  Observations      : {res['n_obs']:,}")

    print(f"\n  ── Vasicek Parameters (OLS calibrated) ───────────────")
    print(f"  κ  (mean reversion): {res['kappa']:.5f}"
          f"   half-life = {res['half_life']:.2f} yr")
    print(f"  θ  (long-run mean) : {res['theta']:.5f}"
          f"   ({res['theta']:.3%})")
    print(f"  σ  (volatility)    : {res['sigma']:.5f}"
          f"   ({res['sigma']:.3%})")
    print(f"  r₀ (current rate)  : {res['r0']:.5f}"
          f"   ({res['r0']:.3%})")

    print(f"\n  ── Bond Prices & Yields ──────────────────────────────")
    print(f"  P(0,1)            : {res['P_1yr']:.6f}"
          f"   yield = {res['yield_1yr']:.4%}")
    print(f"  P(0,5)            : {res['P_5yr']:.6f}"
          f"   yield = {res['yield_5yr']:.4%}")
    print(f"  Long-run yield    : {res['lry']:.4%}")

    print(f"\n  ── Swap Valuation (N={notional:,.0f}) ──────────────────")
    print(f"  Par swap rate K*  : {res['par_rate']:.5%}")
    print(f"  PV (fixed leg)    : {res['pv_fixed']:>12,.2f}")
    print(f"  PV (float leg)    : {res['pv_float']:>12,.2f}")
    print(f"  NPV (at-market)   : {res['npv_par']:>12,.2f}  (≈ 0 ✓)")
    print(f"  NPV (K*+50bp)     : {res['npv_off']:>12,.2f}")
    print(f"  DV01              : {res['dv01']:>12,.2f}  per 1bp")

    print(f"\n  ── Monte Carlo (2,000 paths) ─────────────────────────")
    print(f"  MC mean NPV       : {res['mc_mean']:>12,.2f}")
    print(f"  Closed-form NPV   : {res['mc_cf']:>12,.2f}")
    print(f"  MC std NPV        : {res['mc_std']:>12,.2f}")
    print(f"  MC 5th–95th pct   : [{res['mc_p5']:,.0f}, {res['mc_p95']:,.0f}]")
    print(f"  Convergence error : "
          f"{abs(res['mc_mean']-res['mc_cf'])/abs(res['mc_cf'])*100:.3f}%")

    print(f"\n  Pipeline completed in {elapsed:.1f} seconds.")
    print("█" * 60 + "\n")


def _save_results_txt(res, notional):
    """Save key numbers to a text file for easy report writing."""
    lines = [
        "ES418 Group 13 — Interest Rate Swap Valuation",
        "Key Results",
        "=" * 50,
        "",
        "DATA",
        f"Source          : FRED DTB3",
        f"Period          : {res['date_start']} to {res['date_end']}",
        f"Observations    : {res['n_obs']}",
        "",
        "VASICEK PARAMETERS",
        f"kappa           : {res['kappa']:.5f}",
        f"theta           : {res['theta']:.5f}",
        f"sigma           : {res['sigma']:.5f}",
        f"r0              : {res['r0']:.5f}",
        f"half_life       : {res['half_life']:.4f} years",
        "",
        "BOND PRICES",
        f"P(0,1yr)        : {res['P_1yr']:.6f}",
        f"P(0,5yr)        : {res['P_5yr']:.6f}",
        f"yield_1yr       : {res['yield_1yr']:.6f}",
        f"yield_5yr       : {res['yield_5yr']:.6f}",
        f"long_run_yield  : {res['lry']:.6f}",
        "",
        "SWAP VALUATION",
        f"notional        : {notional:,.0f}",
        f"par_rate        : {res['par_rate']:.6f}",
        f"pv_fixed        : {res['pv_fixed']:.2f}",
        f"pv_floating     : {res['pv_float']:.2f}",
        f"npv_atmarket    : {res['npv_par']:.2f}",
        f"dv01            : {res['dv01']:.2f}",
        "",
        "MONTE CARLO",
        f"mc_mean         : {res['mc_mean']:.2f}",
        f"mc_std          : {res['mc_std']:.2f}",
        f"mc_p5           : {res['mc_p5']:.2f}",
        f"mc_p95          : {res['mc_p95']:.2f}",
        f"closed_form     : {res['mc_cf']:.2f}",
    ]
    with open('summary_results.txt', 'w') as f:
        f.write('\n'.join(lines))
    print("[saved] summary_results.txt")


def _list_all_outputs():
    """List every output file with size."""
    all_files = [
        'data_overview',
        'vasicek_paths', 'vasicek_validation', 'vasicek_terminal',
        'yield_curve', 'yield_curve_shapes',
        'yield_sensitivity', 'discount_factors',
        'swap_cashflows', 'swap_npv_vs_rate',
        'swap_sensitivity', 'swap_scenarios',
        'mc_npv_distribution', 'mc_convergence',
    ]
    print("\n  All output files:")
    total_kb = 0
    for fname in all_files:
        for ext in ['png', 'pdf']:
            path = f'{fname}.{ext}'
            if os.path.exists(path):
                kb = os.path.getsize(path) / 1024
                total_kb += kb
                print(f"    {path:40s}  {kb:7.1f} KB")
    if os.path.exists('summary_results.txt'):
        kb = os.path.getsize('summary_results.txt') / 1024
        print(f"    {'summary_results.txt':40s}  {kb:7.1f} KB")
        total_kb += kb
    print(f"\n    Total: {total_kb/1024:.2f} MB")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    run_all(data_file='DTB3.csv', notional=1_000_000)
