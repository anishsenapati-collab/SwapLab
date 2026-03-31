"""
main.py
ES418 Group 13 — Interest Rate Swap Valuation
Phase 1 entry point — run this file to execute the full pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  QUICK START — LOCAL (VS Code / PyCharm / Terminal)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Place all four files + DTB3.csv in the same folder
  2. pip install -r requirements.txt
  3. python main.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  QUICK START — GOOGLE COLAB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Cell 1:  !pip install -q numpy pandas matplotlib scipy
  Cell 2:  Upload all files via the Files panel (left sidebar)
  Cell 3:  exec(open('main.py').read())
  Cell 4:  from IPython.display import display, Image
           for f in ['data_overview.png','vasicek_paths.png',
                     'vasicek_validation.png','vasicek_terminal.png']:
               display(Image(f))

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OUTPUT FILES (saved automatically)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  data_overview.png / .pdf       — raw rate history + Δr histogram
  vasicek_paths.png / .pdf       — simulated rate fan plot
  vasicek_validation.png / .pdf  — analytical vs simulated mean & variance
  vasicek_terminal.png / .pdf    — terminal rate distribution at T
  rates_clean.npy                — cached cleaned rates (auto-generated)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — works on Colab + local

from data_loader import (load_dtb3, describe_data, plot_raw_data)
from vasicek_simulator import (VasicekConfig, calibrate_vasicek,
                                vasicek_simulate, print_summary,
                                plot_paths, plot_validation, plot_terminal)


# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS — only edit this section
# ══════════════════════════════════════════════════════════════════════════════

# path to your downloaded CSV — must match exactly
DATA_FILE = 'DTB3.csv'

# simulation horizon and paths — change freely
SIM_T       = 5.0    # years to simulate forward
SIM_N       = 1260   # time steps (252 × 5 = daily over 5 years)
SIM_PATHS   = 2000   # Monte Carlo paths (higher = smoother, slower)
SIM_SEED    = 42     # fix this for reproducible report figures

# set False to skip calibration and use hardcoded Vasicek defaults
# (useful if you want to experiment with different κ, θ, σ values)
USE_CALIBRATION = True

# manual override — only used when USE_CALIBRATION = False
# these are the values from your DTB3 calibration for reference
MANUAL_CFG = VasicekConfig(
    kappa   = 0.09611,
    theta   = 0.01189,
    sigma   = 0.00702,
    r0      = 0.03630,
    T       = SIM_T,
    N       = SIM_N,
    n_paths = SIM_PATHS,
    seed    = SIM_SEED,
)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE  —  do not edit below this line
# ══════════════════════════════════════════════════════════════════════════════

def main():

    print("\n" + "█" * 54)
    print("  ES418 Group 13  —  Phase 1: Vasicek Simulator")
    print("█" * 54)

    # ── STEP 1: load data ─────────────────────────────────────────────
    print(f"\n[Step 1] Loading data from '{DATA_FILE}' ...")

    if not os.path.exists(DATA_FILE):
        print(f"\n  ERROR: '{DATA_FILE}' not found in current directory.")
        print("  Download from: https://fred.stlouisfed.org/series/DTB3")
        print("  Click Download → CSV, rename file to DTB3.csv\n")
        return None, None, None

    rates, dates = load_dtb3(DATA_FILE)
    describe_data(rates)

    # ── STEP 2: raw data plot ─────────────────────────────────────────
    print("\n[Step 2] Plotting raw data ...")
    plot_raw_data(rates, dates, save=True)

    # ── STEP 3: calibrate or use manual params ────────────────────────
    if USE_CALIBRATION:
        print("\n[Step 3] Calibrating Vasicek parameters via OLS ...")
        cfg = calibrate_vasicek(rates, dt=1/252)
        # apply simulation settings on top of calibrated model params
        cfg.T          = SIM_T
        cfg.N          = SIM_N
        cfg.n_paths    = SIM_PATHS
        cfg.seed       = SIM_SEED
        cfg.save_plots = True
    else:
        print("\n[Step 3] Using manual parameters (calibration skipped) ...")
        cfg = MANUAL_CFG

    cfg.summary()

    # ── STEP 4: simulate ──────────────────────────────────────────────
    print("[Step 4] Running Euler-Maruyama simulation ...")
    r, t = vasicek_simulate(cfg, antithetic=True)

    # ── STEP 5: validate ──────────────────────────────────────────────
    print("\n[Step 5] Validation table (analytical vs simulated) ...")
    print_summary(r, t, cfg)

    # ── STEP 6: plots ─────────────────────────────────────────────────
    print("[Step 6] Generating and saving plots ...")
    plot_paths(r, t, cfg,
               data_start_year=dates[0].year,
               data_end_year=dates[-1].year)
    plot_validation(r, t, cfg)
    plot_terminal(r, cfg)

    # ── DONE ──────────────────────────────────────────────────────────
    print("\n" + "█" * 54)
    print("  Phase 1 complete.")
    print()
    print("  Saved files:")
    for fname in ['data_overview', 'vasicek_paths',
                  'vasicek_validation', 'vasicek_terminal']:
        for ext in ['png', 'pdf']:
            path = f'{fname}.{ext}'
            if os.path.exists(path):
                kb = os.path.getsize(path) / 1024
                print(f"    {path:35s}  {kb:.1f} KB")
    print()
    print("  Next step: run phase2_bonds.py")
    print("█" * 54 + "\n")

    # return objects so Phase 2 can import directly
    return cfg, r, t


if __name__ == '__main__':
    cfg, r, t = main()
